"""LLM judge for the Sheldon persona: pairwise preference and rubric scoring.

Samples N held-out ids (seed 0, same convention as scripts/make_read_me_v3b.py: sorted ids,
shuffled, first N) and pulls the gold, base, and v3b responses for those ids. Judge is
claude-sonnet-4-6 at temperature 0 with sheldon_style_guide.md in the system prompt and an
instruction to ignore factual correctness.

Three runs:
  1. pairwise base vs v3b, each pair asked twice with the order swapped -> v3b win rate, Wilson 95% CI
  2. pairwise base vs gold, same protocol, as a control
  3. rubric scoring of every response alone on the six section-5 dimensions (0-2 each), JSON out
     -> mean total for gold / base / v3b with bootstrap CIs, per-dimension table

Every raw judge response (or API failure) is appended to results/judge_raw.jsonl. API and parse
failures are counted and reported; no row is dropped silently.

Usage (from repo root, venv active):
    python persona_eval/judge.py [--n 50] [--seed 0] [--workers 8]
"""
import argparse
import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np

HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "data" / "heldout_gold.jsonl"
BASE_PATH = HERE / "gens" / "heldout_base_eval.jsonl"
V3B_PATH = HERE / "gens" / "heldout_v3b_eval.jsonl"
GUIDE_PATH = HERE / "sheldon_style_guide.md"
RAW_LOG = HERE / "results" / "judge_raw.jsonl"
REPORT = HERE / "results" / "judge_eval.md"

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
N_BOOT = 2000

DIMS = [
    ("pedantic_precision", "Pedantic precision"),
    ("superiority_as_fact", "Superiority as fact"),
    ("rules_and_routines", "Rules and routines"),
    ("formal_register", "Formal register"),
    ("tangent_or_reference", "Tangent or reference"),
    ("social_obliviousness", "Social obliviousness"),
]
PENALTIES = ["penalty_bazinga", "penalty_name_drops", "penalty_section3"]
SYSTEMS = ["gold", "base", "v3b"]
LABELS = {"gold": "GOLD (dataset reference)", "base": "BASE (Qwen2.5-3B-Instruct)",
          "v3b": "V3B (Sheldon SFT)"}

SYSTEM_PROMPT_TEMPLATE = """You are a judge evaluating how well a written response matches the persona of Dr. Sheldon Cooper. The style guide below is your only reference. Follow its rubric exactly.

IMPORTANT: Ignore factual correctness entirely. Do not reward or penalize a response for whether its facts, math, or advice are right or wrong, complete or incomplete. Judge only whether the writing is in character. Responses may have been cut off at a fixed length; do not penalize truncation.

Answer only with the JSON object requested. No preamble, no markdown fences.

<style_guide>
{guide}
</style_guide>"""

PAIRWISE_TEMPLATE = """Below is a user prompt and two responses to it. Decide which response is more in character as Sheldon Cooper according to the style guide. Ignore factual correctness. You must pick one; ties are not allowed.

<prompt>
{prompt}
</prompt>

<response_A>
{a}
</response_A>

<response_B>
{b}
</response_B>

Return JSON: {{"winner": "A" or "B", "reason": "<one sentence>"}}"""

RUBRIC_TEMPLATE = """Below is a user prompt and one response to it. Score the response on the six dimensions of the style guide's judge rubric (section 5), each 0 (absent), 1 (present but weak or generic), or 2 (clearly present and specific to the text). Then report the penalties: how many times "Bazinga" appears beyond the first (penalty_bazinga: 1 if it appears more than once, else 0), whether there are more than two friend/family name-drops that do nothing in the sentence (penalty_name_drops: 1 or 0), and how many distinct section-3 "Never" items the response violates (penalty_section3: integer >= 0). Ignore factual correctness.

<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

Return JSON with exactly these keys:
{{"pedantic_precision": 0-2, "superiority_as_fact": 0-2, "rules_and_routines": 0-2, "formal_register": 0-2, "tangent_or_reference": 0-2, "social_obliviousness": 0-2, "penalty_bazinga": 0 or 1, "penalty_name_drops": 0 or 1, "penalty_section3": integer, "note": "<one sentence>"}}"""


def log(msg):
    print(msg, flush=True)


def load(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    log(f"read {len(rows)} rows from {path.relative_to(HERE.parent)}")
    return rows


def sample_ids(gold, n, seed):
    ids = sorted(gold)
    random.Random(seed).shuffle(ids)
    ids = ids[:n]
    log(f"sampled {len(ids)} ids (seed={seed})")
    return ids


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


class Judge:
    def __init__(self, guide_text, raw_log_path):
        self.client = anthropic.Anthropic(max_retries=4, timeout=120.0)
        self.system = [{"type": "text", "text": SYSTEM_PROMPT_TEMPLATE.format(guide=guide_text),
                        "cache_control": {"type": "ephemeral"}}]
        self.raw_log = raw_log_path.open("w", encoding="utf-8")
        self.lock = threading.Lock()
        self.api_failures = 0
        self.parse_failures = 0
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}

    def close(self):
        self.raw_log.close()

    def _write(self, record):
        with self.lock:
            self.raw_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.raw_log.flush()

    def call(self, meta, user_text):
        """One judge call. Returns (parsed_json_or_None, error_kind_or_None). Always logs."""
        record = dict(meta)
        record.update({"model": MODEL, "temperature": TEMPERATURE, "ts": time.time()})
        try:
            resp = self.client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, extra_body={"temperature": TEMPERATURE},
                system=self.system, messages=[{"role": "user", "content": user_text}],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            with self.lock:
                self.api_failures += 1
                self.calls += 1
            record.update({"error": "api", "error_type": type(e).__name__, "error_msg": str(e)})
            self._write(record)
            return None, "api"
        text = "".join(b.text for b in resp.content if b.type == "text")
        record.update({"request_id": resp._request_id, "stop_reason": resp.stop_reason,
                       "usage": resp.usage.to_dict(), "raw": text})
        with self.lock:
            self.calls += 1
            for k in self.usage:
                self.usage[k] += getattr(resp.usage, k, 0) or 0
        if resp.stop_reason == "refusal":
            with self.lock:
                self.api_failures += 1
            record["error"] = "refusal"
            self._write(record)
            return None, "api"
        try:
            parsed = parse_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            with self.lock:
                self.parse_failures += 1
            record.update({"error": "parse", "error_msg": str(e)})
            self._write(record)
            return None, "parse"
        record["parsed"] = parsed
        self._write(record)
        return parsed, None


# ----------------------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def boot_ci(values, rng, n_boot=N_BOOT):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot_ci(a, b, rng, n_boot=N_BOOT):
    """CI for mean(a - b) with a and b aligned on the same ids."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------------- tasks

def run_pairwise(judge, ids, prompts, left, right, left_name, right_name, workers):
    """Each id judged twice: (left as A, right as B) and swapped. Returns per-id results."""
    jobs = []
    for i in ids:
        for order in ("lr", "rl"):
            a, b = (left[i], right[i]) if order == "lr" else (right[i], left[i])
            a_name, b_name = (left_name, right_name) if order == "lr" else (right_name, left_name)
            meta = {"task": "pairwise", "comparison": f"{left_name}_vs_{right_name}", "id": i,
                    "order": order, "A": a_name, "B": b_name}
            jobs.append((meta, PAIRWISE_TEMPLATE.format(prompt=prompts[i], a=a, b=b)))
    results = {}  # (id, order) -> winner system name or error kind
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge.call, meta, text): meta for meta, text in jobs}
        for fut in as_completed(futs):
            meta = futs[fut]
            parsed, err = fut.result()
            key = (meta["id"], meta["order"])
            if err:
                results[key] = ("error", err)
                continue
            w = str(parsed.get("winner", "")).strip().upper()
            if w not in ("A", "B"):
                judge.parse_failures += 1
                results[key] = ("error", "parse")
                continue
            results[key] = ("ok", meta[w])
    return results


def summarize_pairwise(results, ids, target, other):
    """Win counts for `target` over all valid judgments, plus per-id agreement."""
    wins = valid = 0
    both = split = neither = failed = 0
    for i in ids:
        outs = []
        for order in ("lr", "rl"):
            status, val = results[(i, order)]
            if status == "ok":
                valid += 1
                wins += val == target
                outs.append(val)
        if len(outs) < 2:
            failed += 1
        elif outs[0] == target and outs[1] == target:
            both += 1
        elif outs[0] != outs[1]:
            split += 1
        else:
            neither += 1
    p, lo, hi = wilson(wins, valid)
    return {"target": target, "other": other, "wins": wins, "valid": valid, "total": 2 * len(ids),
            "rate": p, "ci_lo": lo, "ci_hi": hi, "both": both, "split": split, "neither": neither,
            "failed_ids": failed}


def run_rubric(judge, ids, prompts, texts, workers):
    """texts: {system: {id: response}}. Returns {system: {id: parsed_scores}} (missing on failure)."""
    jobs = []
    for sysname in SYSTEMS:
        for i in ids:
            meta = {"task": "rubric", "system": sysname, "id": i}
            jobs.append((meta, RUBRIC_TEMPLATE.format(prompt=prompts[i], response=texts[sysname][i])))
    scores = {s: {} for s in SYSTEMS}
    failures = {s: 0 for s in SYSTEMS}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge.call, meta, text): meta for meta, text in jobs}
        for fut in as_completed(futs):
            meta = futs[fut]
            parsed, err = fut.result()
            if err:
                failures[meta["system"]] += 1
                continue
            try:
                dims = {k: int(parsed[k]) for k, _ in DIMS}
                pens = {k: int(parsed.get(k, 0)) for k in PENALTIES}
            except (KeyError, TypeError, ValueError):
                judge.parse_failures += 1
                failures[meta["system"]] += 1
                continue
            if any(not 0 <= v <= 2 for v in dims.values()) or any(v < 0 for v in pens.values()):
                judge.parse_failures += 1
                failures[meta["system"]] += 1
                continue
            raw_total = sum(dims.values())
            total = max(0, raw_total - sum(pens.values()))
            scores[meta["system"]][meta["id"]] = {**dims, **pens, "raw_total": raw_total,
                                                   "total": total}
    return scores, failures


# ----------------------------------------------------------------------------- report

def fmt_ci(m, lo, hi, nd=2):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_report(args, ids, pw_main, pw_ctrl, scores, rub_fail, judge, elapsed):
    rng = np.random.default_rng(args.seed)
    n = len(ids)
    L = []
    L.append("# LLM-judge persona evaluation (held-out chat prompts)\n")
    L.append(f"Judge: `{MODEL}`, temperature {TEMPERATURE}, system prompt = `sheldon_style_guide.md` "
             f"plus an instruction to ignore factual correctness. {n} held-out ids sampled with seed "
             f"{args.seed} from `data/heldout_gold.jsonl` (sorted ids, shuffled, first {n}). "
             f"Responses: gold = dataset reference, base = Qwen2.5-3B-Instruct, v3b = Sheldon SFT "
             f"(greedy, max_new_tokens 400; gold is untruncated). Raw judge outputs: "
             f"`results/judge_raw.jsonl`. Script: `persona_eval/judge.py`.\n")
    L.append(f"API calls: {judge.calls}. API failures (exceptions after SDK retries, or refusals): "
             f"{judge.api_failures}. Unparseable/invalid judge outputs: {judge.parse_failures}. "
             f"Wall time {elapsed/60:.1f} min. Tokens: input {judge.usage['input_tokens']}, "
             f"cache read {judge.usage['cache_read_input_tokens']}, cache write "
             f"{judge.usage['cache_creation_input_tokens']}, output {judge.usage['output_tokens']}.\n")

    L.append("## 1. Pairwise: base vs v3b\n")
    L.append("Each pair is judged twice with A/B order swapped; the judge must pick one. Win rate is over "
             "all valid judgments (2 per prompt), Wilson 95% CI. The per-prompt row shows how often the "
             "two orderings agreed.\n")
    for title, s in (("base vs v3b (main)", pw_main), ("base vs gold (control)", pw_ctrl)):
        L.append(f"**{title}** - {s['target']} wins {s['wins']}/{s['valid']} judgments "
                 f"({s['total']} attempted): win rate **{s['rate']:.3f}** "
                 f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]. Per prompt: {s['target']} wins both orderings "
                 f"{s['both']}, split {s['split']}, {s['other']} wins both {s['neither']}, "
                 f"prompts with a failed judgment {s['failed_ids']}.\n")
    L.append("| Comparison | Judgments (valid/attempted) | Target wins | Win rate | Wilson 95% CI | Both orders | Split | Neither |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in (pw_main, pw_ctrl):
        L.append(f"| {s['target']} vs {s['other']} | {s['valid']}/{s['total']} | {s['wins']} | "
                 f"{s['rate']:.3f} | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {s['both']} | {s['split']} | {s['neither']} |")
    L.append("")

    L.append("## 2. Rubric scores (each response judged alone)\n")
    L.append("Six dimensions from section 5 of the style guide, 0-2 each; penalties (Bazinga more than "
             "once, more than two idle name-drops, each section-3 violation) subtract 1 each, floor 0. "
             "Total is computed from the judge's per-dimension integers, max 12. Mean with 95% "
             f"bootstrap CI ({N_BOOT} resamples, seed {args.seed}).\n")
    L.append("| System | n scored | failed | Mean total (95% CI) | Mean raw total (no penalties) | Mean penalties |")
    L.append("|---|---|---|---|---|---|")
    stats = {}
    for s in SYSTEMS:
        sc = scores[s]
        totals = [sc[i]["total"] for i in ids if i in sc]
        raws = [sc[i]["raw_total"] for i in ids if i in sc]
        pens = [sum(sc[i][k] for k in PENALTIES) for i in ids if i in sc]
        m, lo, hi = boot_ci(totals, rng)
        stats[s] = (m, lo, hi)
        L.append(f"| {s} | {len(totals)} | {rub_fail[s]} | {fmt_ci(m, lo, hi)} | "
                 f"{np.mean(raws) if raws else float('nan'):.2f} | "
                 f"{np.mean(pens) if pens else float('nan'):.2f} |")
    L.append("")
    L.append("Paired differences in mean total (bootstrap over prompts scored for both systems):\n")
    L.append("| Difference | Mean | 95% CI |")
    L.append("|---|---|---|")
    for a, b in (("v3b", "base"), ("v3b", "gold"), ("gold", "base")):
        common = [i for i in ids if i in scores[a] and i in scores[b]]
        m, lo, hi = paired_boot_ci([scores[a][i]["total"] for i in common],
                                   [scores[b][i]["total"] for i in common], rng)
        L.append(f"| {a} - {b} (n={len(common)}) | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    L.append("")
    L.append("Per-dimension means (0-2):\n")
    L.append("| # | Dimension | gold | base | v3b |")
    L.append("|---|---|---|---|---|")
    for k, (key, name) in enumerate(DIMS, 1):
        cells = []
        for s in SYSTEMS:
            vals = [scores[s][i][key] for i in ids if i in scores[s]]
            cells.append(f"{np.mean(vals):.2f}" if vals else "nan")
        L.append(f"| {k} | {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Penalty rates (fraction of responses with the penalty; section-3 is mean count):\n")
    L.append("| Penalty | gold | base | v3b |")
    L.append("|---|---|---|---|")
    for key in PENALTIES:
        cells = []
        for s in SYSTEMS:
            vals = [scores[s][i][key] for i in ids if i in scores[s]]
            cells.append(f"{np.mean(vals):.2f}" if vals else "nan")
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Score distribution (count of responses at each total):\n")
    L.append("| Total | " + " | ".join(SYSTEMS) + " |")
    L.append("|---|" + "---|" * len(SYSTEMS))
    for t in range(13):
        row = [str(sum(1 for i in ids if i in scores[s] and scores[s][i]["total"] == t)) for s in SYSTEMS]
        L.append(f"| {t} | " + " | ".join(row) + " |")
    L.append("")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {REPORT.relative_to(HERE.parent)}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gold = load(GOLD_PATH)
    base = load(BASE_PATH)
    v3b = load(V3B_PATH)
    assert set(gold) == set(base) == set(v3b), "id sets differ between gold/base/v3b files"
    ids = sample_ids(gold, args.n, args.seed)
    for i in ids:
        assert gold[i]["prompt"] == base[i]["prompt"] == v3b[i]["prompt"], f"prompt mismatch for {i}"
    prompts = {i: gold[i]["prompt"] for i in ids}
    texts = {"gold": {i: gold[i]["gold"] for i in ids},
             "base": {i: base[i]["response"] for i in ids},
             "v3b": {i: v3b[i]["response"] for i in ids}}
    log(f"loaded responses for {len(ids)} ids x 3 systems")

    guide = GUIDE_PATH.read_text(encoding="utf-8")
    RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
    judge = Judge(guide, RAW_LOG)
    t0 = time.time()
    try:
        log(f"[1/3] pairwise base vs v3b: {2*len(ids)} calls")
        r_main = run_pairwise(judge, ids, prompts, texts["base"], texts["v3b"], "base", "v3b", args.workers)
        pw_main = summarize_pairwise(r_main, ids, "v3b", "base")
        log(f"  v3b wins {pw_main['wins']}/{pw_main['valid']} = {pw_main['rate']:.3f} "
            f"[{pw_main['ci_lo']:.3f}, {pw_main['ci_hi']:.3f}]; api fail {judge.api_failures}, parse fail {judge.parse_failures}")

        log(f"[2/3] pairwise base vs gold (control): {2*len(ids)} calls")
        r_ctrl = run_pairwise(judge, ids, prompts, texts["base"], texts["gold"], "base", "gold", args.workers)
        pw_ctrl = summarize_pairwise(r_ctrl, ids, "gold", "base")
        log(f"  gold wins {pw_ctrl['wins']}/{pw_ctrl['valid']} = {pw_ctrl['rate']:.3f} "
            f"[{pw_ctrl['ci_lo']:.3f}, {pw_ctrl['ci_hi']:.3f}]; api fail {judge.api_failures}, parse fail {judge.parse_failures}")

        log(f"[3/3] rubric: {3*len(ids)} calls")
        scores, rub_fail = run_rubric(judge, ids, prompts, texts, args.workers)
        for s in SYSTEMS:
            log(f"  {s}: scored {len(scores[s])}/{len(ids)}, failed {rub_fail[s]}")
    finally:
        judge.close()
    elapsed = time.time() - t0
    log(f"total calls {judge.calls}, API failures {judge.api_failures}, parse failures {judge.parse_failures}, "
        f"{elapsed/60:.1f} min")
    stats = write_report(args, ids, pw_main, pw_ctrl, scores, rub_fail, judge, elapsed)
    for s in SYSTEMS:
        log(f"  rubric mean total {s}: {fmt_ci(*stats[s])}")


if __name__ == "__main__":
    main()
