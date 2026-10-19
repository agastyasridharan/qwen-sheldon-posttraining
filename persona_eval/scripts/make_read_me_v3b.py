"""Write results/read_me_v3b.txt: 20 held-out prompts (seed 0), each with gold, base, and v3b responses."""
import json
import random
from pathlib import Path

SEED, N = 0, 20
OUT = Path("results/read_me_v3b.txt")

def load(path):
    rows = {json.loads(l)["id"]: json.loads(l) for l in Path(path).open(encoding="utf-8")}
    print(f"read {len(rows)} rows from {path}")
    return rows

gold = load("data/heldout_gold.jsonl")
base = load("gens/heldout_base_eval.jsonl")
v3b = load("gens/heldout_v3b_eval.jsonl")
assert set(gold) == set(base) == set(v3b)

ids = sorted(gold)
random.Random(SEED).shuffle(ids)
ids = ids[:N]
print(f"sampled {len(ids)} ids (seed={SEED})")

bar = "=" * 100
sub = "-" * 100
lines = [f"{N} held-out prompts sampled with seed {SEED} from data/heldout_gold.jsonl ({len(gold)} rows).",
         "GOLD = dataset reference (tbooy/sheldon-cooper-sft-20k), BASE = Qwen/Qwen2.5-3B-Instruct,",
         "V3B = Qwen2.5-3B-Instruct + agastyasridharan/Qwen2.5-3B-Instruct-Sheldon-SFT-v3b-LoRA.",
         "Greedy decoding, max_new_tokens 400, from partner/sft/generate_heldout.py.", ""]
for k, i in enumerate(ids, 1):
    assert gold[i]["prompt"] == base[i]["prompt"] == v3b[i]["prompt"]
    lines += [bar, f"[{k}/{N}]  id={i}", bar, "", "### PROMPT", "", gold[i]["prompt"], "",
              sub, "### GOLD", sub, "", gold[i]["gold"], "",
              sub, "### BASE (Qwen2.5-3B-Instruct)", sub, "", base[i]["response"], "",
              sub, "### V3B (Sheldon SFT LoRA)", sub, "", v3b[i]["response"], "", ""]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}: {len(ids)} prompts, {len(lines)} lines")
