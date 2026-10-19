# Sheldon Cooper persona on Qwen2.5-3B-Instruct (checkpoint 1 SFT, checkpoint 2 RLAIF, checkpoint 3 RLVR + combined reward)

Post-training `Qwen/Qwen2.5-3B-Instruct` to answer in the voice of Dr. Sheldon Cooper while tracking a verifiable STEM capability, in four stages:
LoRA SFT (checkpoint 1), GRPO with an LLM-judge reward (RLAIF, checkpoint 2), CISPO with a verifiable math reward (RLVR, checkpoint 3), and a
combined judge + verifier reward (checkpoint 3). Persona is measured with a style classifier, deterministic monitors and a pairwise LLM judge;
capability with GSM8K, MATH-500 and AIME. Team: Sridharan, Huang, Wu. Write-ups: [`CHECKPOINT1.md`](CHECKPOINT1.md) (SFT),
[`CHECKPOINT2.md`](CHECKPOINT2.md) (RLAIF), [`CHECKPOINT3.md`](CHECKPOINT3.md) (RLVR and combined reward), and the final report
[`Final_Report_Sridharan_Huang_Wu_Revised.pdf`](Final_Report_Sridharan_Huang_Wu_Revised.pdf) (source `main.tex`).

## Deliverable map, checkpoint 3 (RLVR and combined reward)

| Deliverable | Section of `CHECKPOINT3.md` | Where in this repo |
|---|---|---|
| Code and models | §1 | `rlvr/` (grader, rewards, trainer subclass, optimizers, training and evaluation scripts, data pipeline, `runpod/` launchers and sequencers, tests, plans, `ALGORITHM.md`, model cards in `cards/`); trainer states in `runs/{pilot-*,rlvr-main,rlvr-3b-s1,rlvr-3b-s2,rlvr-3c-fullft,rlvr-4-combined}/`; weights on the Hugging Face Hub: [RLVR-math-v1](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLVR-math-v1) (+ [LoRA checkpoints](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLVR-math-v1-LoRA)), [RLVR-math-3b-LoRA](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLVR-math-3b-LoRA), **[RLVR-math-v2-fullft](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLVR-math-v2-fullft)** (best persona-plus-math model), **[Combined-v1](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-Combined-v1)** (final-stage model) |
| Task choice and baselines | §2 | `results/rlvr/baselines.md`, `evals/rlvr/{base,sft-touchup-v4,grpo-v2}/`, `rlvr/PLAN.md` |
| Method: recipe, difficulty schedule, pilots | §3 | `rlvr/ALGORITHM.md`, `rlvr/trainer.py`, `rlvr/reward.py`, `rlvr/data/build_schedule.py`, `rlvr/data/passrate_*.jsonl`, `results/rlvr/difficulty_*.md`, `results/rlvr/pilots.md` |
| Runs and training dynamics | §4 | `results/rlvr/{rlvr-main,rlvr-3b,rlvr-3c-fullft,rlvr-4-combined}.md`, `runs/*/trainer_state.json`, `results/rlvr/NOTES.md` |
| STEM capability: MATH-500, GSM8K, AIME | §5 | `evals/rlvr/<run>/checkpoint-N/{avg4,greedy}/*.summary.json`, `evals/rlvr/<run>/{avg4,greedy,avg16}/` |
| Persona: outputs, monitors, judge, reward-hacking analysis | §6 | `persona_eval/gens/{heldout,ood,short_heldout}_{rlvr_main,rlvr_fullft,combined}*.jsonl`, `results/rlvr/persona_{rlvr-main,rlvr-3c-fullft,rlvr-4-combined}.md`, `results/rlvr/judge_h2h_*.json`, `rlvr/reward_combined.py` |
| Worklog, what did not work, next steps | §7, §8 | `CHECKPOINT3.md`; `results/rlvr/DECISION_LOG.md` |

## Deliverable map, checkpoint 2 (RLAIF)

| Deliverable | Section of `CHECKPOINT2.md` | Where in this repo |
|---|---|---|
| Code and models | §1 | `rlaif/` (data pipeline, judge, reward, `train_grpo.py`, `runpod/` launchers); trainer states in `runs/{sft-touchup-v4,grpo-v2,grpo-v1-lr1e-6}/`; weights on the Hugging Face Hub (links in §1) |
| Reward design and judge calibration | §2, §3 | `rlaif/reward/`, `rlaif/judge/`, `rlaif/judge/calibration_luna_v2.md`, `rlaif/persona_audit.md`, `rlaif/sheldon_style_guide_v2.md` |
| Runs and training dynamics | §4 | `runs/*/trainer_state.json`, `results/rlaif/stage2_grpo_v2.md` |
| Persona: outputs and metrics | §5 | `persona_eval/gens/{heldout,ood,short_heldout}_{touchup,grpo_v2}*.jsonl`, `persona_eval/results/style_classifier_eval_{touchup,grpo_v2}.md`, `results/rlaif/judge_h2h_grpo_v2.json` |
| STEM capability: GSM8K | §6 | `evals/gsm8k/{base,v3b,sft-touchup-v4,grpo-v2,grpo-v1-lr1e-6}/*.summary.json` |
| Worklog, what did not work, next steps | §7, §8 | `CHECKPOINT2.md` |

## Deliverable map, checkpoint 1 (SFT)

| Deliverable | Section of `CHECKPOINT1.md` | Where in this repo |
|---|---|---|
| Repository, training code, and trained model | §1 | `sft/train_sft.py`, `sft/prepare_data.py`, `sft/prepare_data_v3a.py`, `sft/gen_v3b/`, model cards in `sft/cards/`; adapters and merged weights on the Hugging Face Hub (links in §1) |
| Persona: model outputs and similarity metric | §2 | `persona_eval/` — classifier `style_classifier.py`, generation `generate.py`, helper `scripts/`, outputs `gens/`, results `results/style_classifier_eval.md`, 20-prompt side-by-side sample `results/read_me_v3b.txt`, pipeline notes `README_pipeline.md` |
| STEM capability: GSM8K | §3 | `evals/`, per-run trajectories `results/<run>/`, cross-arm overlay `results/overlay_all_sft.png` |
| Worklog and what did not work | §4 | `CHECKPOINT1.md` §4; v3 plan in `sft/PLAN_v3_mixed_sft.md` |

## Layout

- `sft/` — stage 1: data preparation and filtering, the v3b math-rewrite pipeline, LoRA SFT trainer, GSM8K evaluator, model cards, the v3 plan.
- `rlaif/` — stage 2: data patch and RL prompt set, OpenRouter judge client and calibration, the 17-term rule reward and batch tax, GRPO
  trainer, RunPod launchers, the v3b persona audit and the style guide.
- `rlvr/` — stages 3 and 4: math grader, verifiable and combined rewards, the TRL trainer subclass (prompt-level aggregation, masking),
  Muon/Muon^p optimizers, training and vLLM evaluation scripts, the data pipeline (pools, contamination filter, pass-rate labels,
  difficulty schedules), pod launchers and stage sequencers, CPU tests, plans, `ALGORITHM.md`, and model cards.
- `persona_eval/` — the persona similarity classifier and the LLM-judge pipeline with their data, generations to the held-out, OOD and
  short prompt sets for every model (base, v3b, touch-up, grpo-v2, rlvr_main, rlvr_fullft, combined), and results.
- `data/` — the GSM8K test set; the MATH-500 and AIME evaluation sets live in `rlvr/data/`.
- `evals/` — per-checkpoint and final evaluation summaries: `gsm8k/<model>/` (stages 1–2) and `rlvr/<run>/` (MATH-500 by level, GSM8K, AIME; stages 3–4).
- `runs/<run>/` — trainer states, arguments and per-step traces for every SFT, RLAIF, RLVR pilot, RLVR and combined run (weights are on the Hub).
- `results/` — SFT trajectories and overlays, `rlaif/` (stage-2 tables and judge outputs), `rlvr/` (baselines, pilots, per-run tables,
  persona reports, judge outputs, `NOTES.md`, `DECISION_LOG.md`).
- `CHECKPOINT1.md`, `CHECKPOINT2.md`, `CHECKPOINT3.md`, `main.tex` / `Final_Report_Sridharan_Huang_Wu_Revised.pdf` — the write-ups and the final report.

All training and evaluation code was developed in https://github.com/TBOO-Y/cs2881r-sheldon-sft, which has the full commit history.
