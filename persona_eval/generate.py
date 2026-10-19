"""Generate responses for a jsonl of prompts with a HF causal LM (optionally + a peft adapter).

Settings (fixed, same for every model): chat template with no system prompt, greedy decoding,
max_new_tokens 400. Output rows keep the input order and carry id, source_id, prompt, response.

Usage:
  python generate.py --model Qwen/Qwen2.5-3B-Instruct --input data/train_gold.jsonl \
      --output gens/smoke_base.jsonl --limit 5 --batch_size 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hub id of the base model")
    p.add_argument("--adapter", default=None, help="Optional Hub id of a peft adapter")
    p.add_argument("--input", required=True, help="Input jsonl with id, source_id, prompt")
    p.add_argument("--output", required=True, help="Output jsonl")
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"],
                   help="auto = bfloat16 on both CUDA and CPU")
    return p.parse_args()


def load_rows(path, limit):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    print(f"input rows: {len(rows)}")
    if limit is not None:
        rows = rows[:limit]
        print(f"after --limit {limit}: {len(rows)} rows")
    return rows


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.bfloat16  # float32 3B weights (12 GB) swap on a 16 GB laptop
    else:
        dtype = getattr(torch, args.dtype)
    print(f"device: {device}, dtype: {dtype}")

    rows = load_rows(args.input, args.limit)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # decoder-only batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"adapter loaded and merged: {args.adapter}")
    model.to(device).eval()
    print(f"model loaded in {time.time() - t0:.1f}s: {args.model}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    gen_start = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            tb = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            elapsed = time.time() - tb
            for r, resp, toks in zip(batch, responses, new_tokens):
                n_new = int((toks != tokenizer.pad_token_id).sum())
                f.write(json.dumps(
                    {"id": r["id"], "source_id": r.get("source_id"), "prompt": r["prompt"], "response": resp},
                    ensure_ascii=False,
                ) + "\n")
                f.flush()
                n_written += 1
                print(f"[{n_written}/{len(rows)}] id={r['id']} new_tokens={n_new} "
                      f"batch_time={elapsed:.1f}s ({elapsed / len(batch):.1f}s/prompt)")

    total = time.time() - gen_start
    print(f"wrote {n_written} rows to {out_path}")
    print(f"generation time: {total:.1f}s total, {total / max(n_written, 1):.1f}s per prompt")


if __name__ == "__main__":
    main()
