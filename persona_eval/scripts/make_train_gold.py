"""Sample 1,000 training rows from tbooy/sheldon-cooper-sft-20k (seed 0).

Writes data/train_gold.jsonl with fields: id (position in sample), source_id (dataset id column),
prompt (first user turn), gold (first assistant turn).
"""
import json
from pathlib import Path

from datasets import load_dataset

SEED = 0
N = 1000
OUT = Path("data/train_gold.jsonl")

ds = load_dataset("tbooy/sheldon-cooper-sft-20k", split="train")
print(f"full train split: {len(ds)} rows")

sample = ds.shuffle(seed=SEED).select(range(N))
print(f"sampled: {len(sample)} rows (seed={SEED})")

rows = []
skipped = 0
for i, row in enumerate(sample):
    msgs = row["messages"]
    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if user is None or asst is None:
        skipped += 1
        print(f"WARNING: row {i} missing user or assistant turn; roles={[m['role'] for m in msgs]}")
        continue
    rows.append({"id": i, "source_id": row["id"], "prompt": user, "gold": asst})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(rows)} rows to {OUT} (skipped {skipped})")
for r in rows[:3]:
    print(json.dumps(r, ensure_ascii=False))
