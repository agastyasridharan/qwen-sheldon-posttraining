"""Convert the partner-format generation files (gens/heldout_{base,v3b}.jsonl) to the classifier's
format: id, prompt (last user turn), response. Held-out rows go to gens/heldout_<m>_eval.jsonl,
the 40 ood- rows to gens/ood_<m>.jsonl.
"""
import json
from pathlib import Path

GOLD = Path("data/heldout_gold.jsonl")
gold = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in GOLD.open(encoding="utf-8")}
print(f"gold ids: {len(gold)}")

def write(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {path}")

for m in ["base", "v3b"]:
    src = Path(f"gens/heldout_{m}.jsonl")
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    print(f"read {len(rows)} rows from {src}")
    conv = [{"id": r["id"], "prompt": r["messages"][-1]["content"], "response": r["response"]} for r in rows]
    held = [r for r in conv if not r["id"].startswith("ood-")]
    ood = [r for r in conv if r["id"].startswith("ood-")]
    assert len(held) + len(ood) == len(rows)
    assert {r["id"] for r in held} == set(gold), "held-out ids differ from gold"
    assert all(r["prompt"] == gold[r["id"]] for r in held), "prompt mismatch vs gold"
    write(f"gens/heldout_{m}_eval.jsonl", held)
    write(f"gens/ood_{m}.jsonl", ood)
