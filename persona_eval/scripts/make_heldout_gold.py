"""Write data/heldout_gold.jsonl from the partner's data/partner_v2/heldout_prompts.jsonl.

Fields: id (dataset id), prompt (first user turn), gold (partner's reference = first assistant turn).
Also reports overlap with data/train_gold.jsonl (source_id) -- must be zero.
"""
import json
from pathlib import Path

SRC = Path("data/partner_v2/heldout_prompts.jsonl")
OUT = Path("data/heldout_gold.jsonl")
TRAIN = Path("data/train_gold.jsonl")

rows = [json.loads(l) for l in SRC.open(encoding="utf-8")]
print(f"read {len(rows)} rows from {SRC}")

out = []
for r in rows:
    users = [m["content"] for m in r["messages"] if m["role"] == "user"]
    assert users, r["id"]
    out.append({"id": r["id"], "prompt": users[0], "gold": r["reference"]})

with OUT.open("w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} rows to {OUT}")

if TRAIN.exists():
    train_ids = {json.loads(l)["source_id"] for l in TRAIN.open(encoding="utf-8")}
    overlap = [o["id"] for o in out if o["id"] in train_ids]
    print(f"train_gold rows: {len(train_ids)}; heldout ids overlapping train_gold: {len(overlap)}")
