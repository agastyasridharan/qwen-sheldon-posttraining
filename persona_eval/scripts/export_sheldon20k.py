"""Export tbooy/sheldon-cooper-sft-20k (train split) to data/sheldon20k.jsonl.

One row per line, in dataset order, fields: id, messages, label, meta.
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/sheldon20k.jsonl")

ds = load_dataset("tbooy/sheldon-cooper-sft-20k", split="train")
print(f"full train split: {len(ds)} rows; columns: {ds.column_names}")

n = 0
with OUT.open("w", encoding="utf-8") as f:
    for r in ds:
        f.write(json.dumps({"id": r["id"], "messages": r["messages"], "label": r["label"], "meta": r["meta"]}, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} rows to {OUT}")
