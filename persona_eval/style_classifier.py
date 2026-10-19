"""Sheldon style classifier: logistic regression on TF-IDF features.

Class 1 = gold (SFT dataset Sheldon response), class 0 = base model response, paired on the same
prompts. Three runs: raw text; masked text where Big Bang Theory names/catchphrases -> NAME;
strict text where markdown syntax is stripped and whitespace collapsed before the NAME mask;
no-names text, which is strict with the NAME token deleted rather than kept.

Every text is truncated to MAX_TOKENS with the Qwen2.5-3B-Instruct tokenizer before featurizing so
gold responses are cut the same way generation cut model outputs (max_new_tokens 400).

No eval files:   5-fold CV (folds grouped by prompt id) -> results/style_classifier_cv.md
With eval files: train on all train pairs, report eval accuracy gold-vs-base, mean P(Sheldon) with
bootstrap CI for gold/base/ft, coefficients -> results/style_classifier_eval.md
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 400
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
N_FOLDS = 5
SEED = 0
TOP_K = 30
N_BOOT = 1000

# Names and catchphrases from sheldon_style_guide.md, plus the extras requested (Bazinga, Meemaw,
# Moon Pie, roommate agreement, Fun with Flags, Dr., Cooper). Cast surnames are added because the
# guide's first names imply them. Longer phrases first so they win over their sub-strings.
MASK_TERMS = [
    "Fun with Flags", "roommate agreement", "Moon Pie", "Star Trek", "East Texas",
    "Farrah Fowler", "Good Lord", "Oh, dear", "Excuse me", "For the record", "there, there",
    "Sheldon", "Cooper", "Leonard", "Hofstadter", "Howard", "Wolowitz", "Rajesh", "Raj",
    "Koothrappali", "Penny", "Amy", "Fowler", "Bernadette", "Mary", "Meemaw", "Caltech", "Spock",
    "Texas", "Bazinga", "Incidentally",
]
MASK_RE = re.compile(
    r"\bDr\.|" + "|".join(r"\b" + re.escape(t) + r"\b" for t in MASK_TERMS),
    flags=re.IGNORECASE,
)


NAME_RUN_RE = re.compile(r"NAME(?:\s+NAME)+")


def mask(text):
    """Replace mask terms with NAME, then collapse whitespace-separated runs of NAME into one."""
    return NAME_RUN_RE.sub("NAME", MASK_RE.sub("NAME", text))


# Markdown syntax removed in the "strict" run. Order matters: fences and links before inline markers.
MD_PATTERNS = [
    (re.compile(r"```[^\n]*\n?"), " "),                            # code fence lines (keep the code text)
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M), ""),            # headers
    (re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.M), " "),       # horizontal rules
    (re.compile(r"^[ \t]*[-*+][ \t]+", re.M), ""),                 # list bullets
    (re.compile(r"^[ \t]*\d+[.)][ \t]+", re.M), ""),               # numbered-list markers
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),               # [text](url) / ![alt](src) -> text
    (re.compile(r"\*\*|__"), ""),                                  # bold markers
    (re.compile(r"(?<![\w*])\*(?!\s)|(?<!\s)\*(?![\w*])"), ""),    # italic *...* markers
    (re.compile(r"(?<!\w)_(?=\S)|(?<=\S)_(?!\w)"), ""),            # italic _..._ markers
    (re.compile(r"`+"), ""),                                       # backticks / inline code
    (re.compile(r"\s+"), " "),                                     # collapse whitespace
]


def strip_markdown(text):
    for pat, rep in MD_PATTERNS:
        text = pat.sub(rep, text)
    return text.strip()


def strict(text):
    """Strip markdown, collapse whitespace, then apply the NAME mask."""
    return mask(strip_markdown(text))


def no_names(text):
    """Same as strict, but the NAME token is deleted instead of kept."""
    return re.sub(r"\s+", " ", NAME_RUN_RE.sub("NAME", MASK_RE.sub("NAME", strip_markdown(text))).replace("NAME", "")).strip()


def read_jsonl(path):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    print(f"read {len(rows)} rows from {path}")
    return rows


def text_of(row):
    return row["gold"] if "gold" in row else row["response"]


def truncate_all(tokenizer, texts, label):
    """Cut each text to MAX_TOKENS tokens. Returns (texts, n_truncated, token_counts)."""
    out, counts, n_trunc = [], [], 0
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        counts.append(len(ids))
        if len(ids) > MAX_TOKENS:
            n_trunc += 1
            t = tokenizer.decode(ids[:MAX_TOKENS])
        out.append(t)
    print(f"{label}: {len(out)} texts, {n_trunc} truncated to {MAX_TOKENS} tokens")
    return out, n_trunc, np.array(counts)


def pair(gold_rows, base_rows, name):
    """Align base rows to gold rows by id; every id must appear in both, prompts must match."""
    base_by_id = {r["id"]: r for r in base_rows}
    missing = [r["id"] for r in gold_rows if r["id"] not in base_by_id]
    extra = set(base_by_id) - {r["id"] for r in gold_rows}
    if missing or extra:
        sys.exit(f"{name}: id mismatch (missing in base: {missing[:5]}, extra in base: {list(extra)[:5]})")
    for g in gold_rows:
        if g["prompt"] != base_by_id[g["id"]]["prompt"]:
            sys.exit(f"{name}: prompt mismatch at id {g['id']}")
    print(f"{name}: {len(gold_rows)} paired prompts")
    return [base_by_id[g["id"]] for g in gold_rows]


def make_pipeline(C=1.0):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=C, max_iter=5000))])


def cv_select(X, y, groups):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(), {"clf__C": C_GRID}, cv=cv, scoring="accuracy", n_jobs=-1, refit=False)
    gs.fit(X, y, groups=groups)
    best_C = gs.best_params_["clf__C"]
    grid = {C: (m, s) for C, m, s in zip(C_GRID, gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"])}
    # out-of-fold predictions at the best C on the same folds, for subset accuracies
    oof = cross_val_predict(make_pipeline(best_C), X, y, cv=cv, groups=groups, n_jobs=-1)
    return best_C, grid, oof


def top_coefs(model):
    names = model.named_steps["feats"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    pos = [(names[i], coef[i]) for i in order[::-1][:TOP_K]]
    neg = [(names[i], coef[i]) for i in order[:TOP_K]]
    return pos, neg


def coef_table(items):
    lines = ["| feature | coef |", "|---|---|"]
    for n, c in items:
        safe = n.replace("|", "\\|")
        lines.append(f"| `{safe}` | {c:+.3f} |")
    return "\n".join(lines)


def boot_ci(p, rng):
    means = [rng.choice(p, size=len(p), replace=True).mean() for _ in range(N_BOOT)]
    return p.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_gold", default="data/train_gold.jsonl")
    ap.add_argument("--train_base", default="gens/base_train.jsonl")
    ap.add_argument("--heldout_gold", default="data/heldout_gold.jsonl",
                    help="partner held-out set; training pairs whose source_id is in it are dropped")
    ap.add_argument("--eval_gold")
    ap.add_argument("--eval_base")
    ap.add_argument("--eval_ft")
    ap.add_argument("--ood_base", help="optional OOD responses (id, prompt, response); scored with the strict run only")
    ap.add_argument("--ood_ft")
    ap.add_argument("--out", default=None, help="results markdown path")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    eval_mode = bool(args.eval_gold)
    if eval_mode and not (args.eval_base and args.eval_ft):
        sys.exit("--eval_gold requires --eval_base and --eval_ft")
    out_path = Path(args.out or ("results/style_classifier_eval.md" if eval_mode else "results/style_classifier_cv.md"))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    # ---- training pairs
    tr_gold = read_jsonl(args.train_gold)
    tr_base_all = read_jsonl(args.train_base)
    # never train on the 502 held-out prompts: drop training pairs whose dataset id is in the held-out set
    heldout_ids = {r["id"] for r in read_jsonl(args.heldout_gold)}
    n_before = len(tr_gold)
    tr_gold = [r for r in tr_gold if r["source_id"] not in heldout_ids]
    tr_base_all = [r for r in tr_base_all if r["source_id"] not in heldout_ids]
    n_dropped = n_before - len(tr_gold)
    print(f"train gold: dropped {n_dropped} rows whose source_id is in the {len(heldout_ids)}-row held-out set; {len(tr_gold)} remain")
    tr_base = pair(tr_gold, tr_base_all, "train")
    gold_txt, gold_trunc, _ = truncate_all(tok, [text_of(r) for r in tr_gold], "train gold")
    base_txt, base_trunc, base_ntok = truncate_all(tok, [text_of(r) for r in tr_base], "train base")
    base_capped = base_ntok >= MAX_TOKENS
    print(f"train base responses at the {MAX_TOKENS}-token cap: {int(base_capped.sum())} of {len(base_txt)}")

    X_raw = gold_txt + base_txt
    y = np.array([1] * len(gold_txt) + [0] * len(base_txt))
    groups = np.array([r["id"] for r in tr_gold] * 2)
    capped_row = np.concatenate([base_capped, base_capped])  # pair-level flag on both members
    print(f"training rows: {len(X_raw)} ({int(y.sum())} gold, {int((y == 0).sum())} base)")

    # ---- eval pairs
    if eval_mode:
        ev_gold = read_jsonl(args.eval_gold)
        ev_base = pair(ev_gold, read_jsonl(args.eval_base), "eval base")
        ev_ft = pair(ev_gold, read_jsonl(args.eval_ft), "eval ft")
        train_prompts = {r["prompt"] for r in tr_gold}
        train_ids = {r["source_id"] for r in tr_gold}
        overlap = sum(r["prompt"] in train_prompts or r["id"] in train_ids for r in ev_gold)
        print(f"eval prompts also in train (by prompt text or dataset id): {overlap}")
        if overlap:
            sys.exit("eval prompts overlap training prompts; refusing to continue")
        ev_txt = {}
        for k, rows in [("gold", ev_gold), ("base", ev_base), ("ft", ev_ft)]:
            ev_txt[k], _, _ = truncate_all(tok, [text_of(r) for r in rows], f"eval {k}")
        ood_txt = {}
        if args.ood_base or args.ood_ft:
            if not (args.ood_base and args.ood_ft):
                sys.exit("--ood_base and --ood_ft must be given together")
            ood_base = read_jsonl(args.ood_base)
            ood_ft = pair(ood_base, read_jsonl(args.ood_ft), "ood ft")
            for k, rows in [("base", ood_base), ("ft", ood_ft)]:
                ood_txt[k], _, _ = truncate_all(tok, [text_of(r) for r in rows], f"ood {k}")
    summary = []

    md = []
    md.append(f"# Style classifier ({'eval' if eval_mode else '5-fold CV'})\n")
    md.append(f"Train: {args.train_gold} vs {args.train_base}, {len(tr_gold)} paired prompts "
              f"({n_before} sampled, {n_dropped} dropped because their dataset id is in the {len(heldout_ids)}-row held-out set {args.heldout_gold}), "
              f"{len(X_raw)} rows. Texts truncated to {MAX_TOKENS} tokens with {TOKENIZER}: "
              f"{gold_trunc} gold and {base_trunc} base texts truncated; "
              f"{int(base_capped.sum())} base responses at the generation cap.\n")
    md.append(f"Features: TF-IDF char_wb 3-5 grams + word 1-2 grams (min_df 2, sublinear tf). "
              f"Logistic regression, C from {C_GRID} by {N_FOLDS}-fold CV with folds grouped by prompt id.\n")
    md.append(f"Masked run replaces these with `NAME` (case-insensitive, word-bounded; consecutive NAMEs collapsed to one): "
              f"{', '.join(MASK_TERMS)}, Dr.\n")
    md.append("Strict run first strips markdown (bold/italic markers, headers, list bullets and numbered-list "
              "markers, code fences and backticks, horizontal rules, link syntax), collapses whitespace, "
              "then applies the same NAME mask.\n")
    md.append("No-names run is the strict run with the NAME token deleted rather than kept.\n")

    for run_name, fn in [("raw", lambda t: t), ("masked", mask), ("strict", strict), ("no-names", no_names)]:
        print(f"\n=== run: {run_name} ===")
        X = [fn(t) for t in X_raw]
        best_C, grid, oof = cv_select(X, y, groups)
        cv_acc = grid[best_C][0]
        oof_acc = (oof == y).mean()
        uncapped_acc = (oof[~capped_row] == y[~capped_row]).mean()
        n_uncapped_pairs = int((~base_capped).sum())
        print(f"best C = {best_C}; CV accuracy = {cv_acc:.4f} (oof {oof_acc:.4f}); "
              f"uncapped pairs ({n_uncapped_pairs}): {uncapped_acc:.4f}")
        for C in C_GRID:
            print(f"  C={C:<6} acc={grid[C][0]:.4f} +/- {grid[C][1]:.4f}")

        model = make_pipeline(best_C).fit(X, y)
        n_feats = len(model.named_steps["feats"].get_feature_names_out())
        pos, neg = top_coefs(model)

        md.append(f"## Run: {run_name}\n")
        md.append(f"Best C = {best_C}, features = {n_feats}.\n")
        md.append("| C | CV accuracy | std |\n|---|---|---|")
        for C in C_GRID:
            md.append(f"| {C} | {grid[C][0]:.4f} | {grid[C][1]:.4f} |")
        md.append("")
        md.append(f"- {N_FOLDS}-fold CV accuracy (best C): **{cv_acc:.4f}** (out-of-fold {oof_acc:.4f}, {len(y)} rows)")
        md.append(f"- CV accuracy on pairs whose base response did not hit the {MAX_TOKENS}-token cap: "
                  f"**{uncapped_acc:.4f}** ({n_uncapped_pairs} pairs, {2 * n_uncapped_pairs} rows)\n")

        if eval_mode:
            rng = np.random.default_rng(SEED)
            P = {k: model.predict_proba([fn(t) for t in ev_txt[k]])[:, 1] for k in ("gold", "base", "ft")}
            ev_acc = np.mean(np.concatenate([P["gold"] >= 0.5, P["base"] < 0.5]))
            print(f"eval accuracy gold-vs-base: {ev_acc:.4f} ({2 * len(ev_gold)} rows)")
            md.append(f"- Eval accuracy gold-vs-base: **{ev_acc:.4f}** ({len(ev_gold)} pairs)\n")
            md.append("| set | n | mean P(Sheldon) | 95% bootstrap CI | frac P >= 0.5 |\n|---|---|---|---|---|")
            ci_run = {}
            for k in ("gold", "base", "ft"):
                m, lo, hi = boot_ci(P[k], rng)
                frac = float((P[k] >= 0.5).mean())
                ci_run[k] = (m, lo, hi, frac)
                print(f"mean P(Sheldon) {k}: {m:.4f} [{lo:.4f}, {hi:.4f}] (n={len(P[k])}); frac >= 0.5: {frac:.4f}")
                md.append(f"| {k} | {len(P[k])} | {m:.4f} | [{lo:.4f}, {hi:.4f}] | {frac:.4f} |")
            md.append("")
            summary.append((run_name, ev_acc, ci_run))
            if run_name == "strict" and ood_txt:
                rng = np.random.default_rng(SEED)
                md.append(f"### OOD prompts (strict run): {args.ood_base} vs {args.ood_ft}\n")
                md.append("| set | n | mean P(Sheldon) | 95% bootstrap CI | P >= 0.5 |\n|---|---|---|---|---|")
                for k in ("base", "ft"):
                    po = model.predict_proba([fn(t) for t in ood_txt[k]])[:, 1]
                    m, lo, hi = boot_ci(po, rng)
                    print(f"OOD mean P(Sheldon) {k}: {m:.4f} [{lo:.4f}, {hi:.4f}] (n={len(po)}, >=0.5: {int((po >= 0.5).sum())})")
                    md.append(f"| {k} | {len(po)} | {m:.4f} | [{lo:.4f}, {hi:.4f}] | {int((po >= 0.5).sum())} |")
                md.append("")

        md.append(f"### Top {TOP_K} positive coefficients (toward gold)\n")
        md.append(coef_table(pos) + "\n")
        md.append(f"### Top {TOP_K} negative coefficients (toward base)\n")
        md.append(coef_table(neg) + "\n")
        print("top positive:", ", ".join(repr(n) for n, _ in pos[:10]))
        print("top negative:", ", ".join(repr(n) for n, _ in neg[:10]))

    if eval_mode:
        head = [f"## Summary: eval on {args.eval_gold} ({len(ev_gold)} held-out prompts)\n",
                f"Base = {args.eval_base}, ft = {args.eval_ft}. Mean P(Sheldon) with 95% bootstrap CI, then the fraction of rows scored >= 0.5.\n",
                "| run | gold-vs-base accuracy | gold | base | ft | gold >= 0.5 | base >= 0.5 | ft >= 0.5 |\n|---|---|---|---|---|---|---|---|"]
        for run_name, ev_acc, ci in summary:
            cells = " | ".join(f"{ci[k][0]:.4f} [{ci[k][1]:.4f}, {ci[k][2]:.4f}]" for k in ("gold", "base", "ft"))
            cells += " | " + " | ".join(f"{ci[k][3]:.4f}" for k in ("gold", "base", "ft"))
            head.append(f"| {run_name} | {ev_acc:.4f} | {cells} |")
        head.append("")
        i = next(j for j, line in enumerate(md) if line.startswith("## Run:"))
        md[i:i] = head
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
