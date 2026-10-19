# Checkpoint 2 (RLAIF) — Sheldon Cooper persona on Qwen2.5-3B-Instruct

**Character:** Dr. Sheldon Cooper. **STEM task:** GSM8K (exact numeric match on `\boxed{}`). **Base model:** `Qwen/Qwen2.5-3B-Instruct`.
**Starting point:** the checkpoint-1 SFT model v3b. **Submitted RLAIF model:** `grpo-v2` (GRPO with an LLM-judge reward, 109 steps, merged bf16).

**Headline.** The RLAIF stage is built end to end and ran twice on an 8×H100 node, but at the step budget we could afford it did not
produce a persona change that a calibrated judge can detect: `grpo-v2` beats its own seed in exactly 50.0% of pairwise judgments
(95% CI 0.45–0.55), while GSM8K stays at 64.0 (seed 63.9) at every checkpoint. The measurable persona gain of this stage came from a
one-epoch SFT touch-up on a patched data set that precedes RL (touch-up beats v3b 0.57, CI 0.52–0.62). The write-up explains why the RL
step was too small (learning rate), what the judge calibration found, and what the next run should change.

## 1. Repository, code and models

- Code: https://github.com/TBOO-Y/cs2881r-sheldon-sft (full history). The stage-2 code is copied into `rlaif/` here:
  `rlaif/data/` (data patch, short-reply set, touch-up set, RL prompt set), `rlaif/judge/` (OpenRouter client, gate and pairwise judge
  prompts, calibration), `rlaif/reward/` (17 deterministic rule terms, batch diversity tax, group reward), `rlaif/train_grpo.py`
  (TRL 1.13 `GRPOTrainer`), `rlaif/runpod/` (node launchers, per-checkpoint evaluator, sequencer), `rlaif/persona_audit.md` (the v3b
  failure audit the reward is built from) and `rlaif/sheldon_style_guide_v2.md` (the canon table used by the judge and the rules).
- Trained models. Seed for RL: `sft-touchup-v4` (v3b + 1 epoch on the patched set). RLAIF: `grpo-v2` (LoRA r32 adapter, checkpoints
  every 25 steps, merged bf16). Trainer states are in `runs/{sft-touchup-v4,grpo-v2,grpo-v1-lr1e-6}/trainer_state.json`; the adapters
  (120 MB each) and merged weights (6 GB) exceed GitHub's file limit and are on the Hugging Face Hub (public, Qwen research license):
  - touch-up (RL seed): [merged](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-SFT-touchup-v4) · [LoRA + 5 checkpoints](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-SFT-touchup-v4-LoRA)
  - **grpo-v2 (submitted)**: [merged](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLAIF-grpo-v2) · [LoRA + 5 checkpoints with per-step reward traces](https://huggingface.co/tbooy/Qwen2.5-3B-Instruct-Sheldon-RLAIF-grpo-v2-LoRA)
  Model cards are in `rlaif/cards/`; `rlaif/hf_upload_stage2.py` did the upload.
- Data. RL prompts: `rlaif/data/rl_prompts.jsonl`, 5,714 prompts, no references (GRPO needs none): 3,299 persona prompts re-weighted
  toward feedback/advice/creative, 1,000 short prompts (3–12 words), 500 with an explicit format constraint, 400 two-turn, 300 verdict
  ("pick a side"), 120 AI-identity probes, 95 sensitive-but-legitimate. Touch-up set: `rlaif/data/v4/` (4,612 rows: 813 new short-prompt
  replies, 2,999 opener-flattened replays of the canon-patched v3b persona rows, 800 math rows; 17% math vs 35% in v3b).

## 2. Method

**Why not persona-only RL.** A 2026-09-19 audit of v3b (`rlaif/persona_audit.md`; 233 probes, 8 reviewed slices of the held-out
generations, n-gram statistics of the training set) found that its failures are mostly not "too little persona": opener template
collapse (97% of replies start with one of six templates), an announced-and-explained non-joke in 18% of replies, canon errors amplified
from the data (Thai food on Tuesday), fabricated "corrections" of words the user never wrote, false self-counts, self-contradiction, and
task failures on constrained prompts. A judge that rewards "more Sheldon" would reward all of these. The reward is therefore gated and
multi-term, and a small data patch plus SFT touch-up runs first so that RL starts from a policy that can sample the target behaviours at all.

**Reward** (per completion *i* in a GRPO group of 8 completions of the same prompt):

```
R_i = G_i · (2·P_i + F_i) − RulePenalty_i − 0.5·FalseClaim_i − BatchTax_i
```

- `G_i` task gate, a persona-blind judge call: lists the user's asks with status done/partial/not (task score = mean), flags `refused`
  (→ G = 0), `worse_off` (→ ×0.25), `contradiction` (→ ×0.5) and `false_claim_about_user`. A zero gate zeroes all persona credit.
- `P_i` persona score, relative: each completion is compared against 2 sibling completions of the same prompt (ring 1), in both
  presentation orders, on 9 items (voice, answer-first, validity of any correction, template reliance, whether a cited rule binds,
  register held to the last sentence, specificity, superiority aimed at ideas not the person, humour/canon); `P` = fraction of the 4
  verdicts won. Within-group comparison matches GRPO's relative advantages and avoids absolute-score drift.
- `F_i` form bonus: +0.25 each for length inside the prompt-class band without truncation, no preamble, single paragraph for phatic prompts.
- Rule penalties (`rlaif/reward/rules.py`): 17 deterministic terms from the audit's catalogue: false claims about the prompt (quoted spans
  must occur in it), wrong self-counts, fabricated rewrite claims, violated numeric/format constraints, fence-sitting on verdict prompts,
  truncation, repetition loops (skip the judge, zero persona credit), preamble ratio, over-length per prompt class, markdown/`\boxed`
  bleed, tool voice, canon table violations, anti-canon behaviour (drives, drinks, accepts being an AI), repeated Bazingas, credential
  stacking, idle name-dropping, arithmetic errors, AI-framing.
- Batch tax: a rolling window over the last 512 completions taxes repeated 3-word openers, 4-word closers, verbatim long sentences,
  cast names above their training marginal and a stock-phrase list refreshed every 25 steps from the policy's own repeated 5-grams.
  This targets mode collapse, which per-sample rewards cannot see.

**Judge.** `openai/gpt-5.6-luna` via OpenRouter, reasoning effort low, JSON mode, 96 concurrent requests, disk-cached. Per step:
128 gate calls + 256 pairwise calls; measured $0.33 per step and 4.9 s median latency (50–60 s of a 65–75 s step is the judge).

**Training.** TRL 1.13 `GRPOTrainer`, LoRA r32/α64 on all projections, 16 prompts × 8 completions per step, 400-token rollouts at
T 0.8 / top-p 0.95 / repetition penalty 1.05 (vLLM colocated), DAPO token-level loss, KL β 0.04 to the seed, clip ε 0.2, constant LR after
10 warm-up steps, checkpoints every 25 steps, each scored on the full GSM8K test and the persona prompts on a spare GPU while training continues.

## 3. Judge calibration (before any RL spend)

`rlaif/judge/calibrate.py` runs the judge on committed probe files: gold vs v3b (40 held-out prompts), v3b vs base (120 probe prompts),
v3b sibling pairs at T 0.7 (80, what RL actually compares), and the gate on gold, v3b and base. Reports: `rlaif/judge/calibration_openai_gpt_5_6_luna.md` (first prompts) and `rlaif/judge/calibration_luna_v2.md` (final).

| check | gpt-4.1-nano | Luna, first prompts | Luna, final prompts |
|---|---|---|---|
| v3b beats base (should be ≈1) | 0.52 | 0.60 | **0.98** (117/120 order-consistent) |
| gold beats v3b | 0.52 | 0.64–0.70 | 0.70 |
| picked the first-shown reply (0.50 = no bias) | 0.91 | 0.60–0.71 | 0.50 on v3b/base; 0.65–0.72 on close pairs |
| gate: gold replies flagged "worse off than no reply" | 0/40 (flags nothing) | 6/40 | 2/40 |
| gate: v3b replies flagged worse-off | 0/40 | 25/40 | 12/40, 8 of them real fabrications |

Two prompt defects were found and fixed. (1) The pairwise rubric had only avoid-the-defect items, so a bland assistant reply "won"
template, correction and humour by doing nothing; in RL that is a gradient toward dropping the persona. A positive `voice` item and the
rule that a reply without the voice can only beat an unusable reply fixed it. (2) `worse_off` fired on pedantic framing around correct
answers (a correct imply/infer explanation, correct autumn-leaf chemistry); it now requires a wrong main answer or concrete harm, with a
quote. Residual first-shown bias on close pairs is why both presentation orders are always judged; single-order judging would have
halved the cost and was rejected on this evidence. Calibration cost $2.20.

## 4. Runs and training dynamics

| run | learning rate | steps | judge cost | outcome |
|---|---|---|---|---|
| `grpo-v1` | 1e-6 (inherited default) | 90, stopped | $31 | reward flat (slope +0.09 per 100 steps, corr 0.09); every monitor flat |
| `grpo-v2` | 1e-5 | 109, 4-hour wall-clock cap | $37 | reward flat (first-20 mean −0.59, last-20 −0.58); judge sees no change vs seed |

Per-step reward has std 0.20 across steps against a within-batch std of 1.2, so trends below ±0.1 are invisible; neither run has one.
The logged KL sits at a floor of 0.0015 from bf16 noise between the policy and reference forward passes (it is 0.001 at step 1, when the
adapter is exactly zero), so it is not informative at this scale. The adapter weights are:

| run | step 25 | step 50 | step 75 | largest entry at step 75 |
|---|---|---|---|---|
| `grpo-v1` (1e-6), ‖B‖_F | 0.020 | 0.034 | 0.047 | 4.4e-5 |
| `grpo-v2` (1e-5), ‖B‖_F | 0.201 | – | – | – |

The optimizer moved exactly as the learning rate dictates (10× at 1e-5), and it was not enough: at 1e-6 the relative weight change after
75 steps is about 0.05%, at 1e-5 about 0.5%. For comparison, the touch-up SFT reshaped the openers visibly in 73 steps at 5e-5 with a
clean gradient. For a 100–300-step LoRA GRPO run with a noisy judge reward, 3e-5 to 5e-5 is the range to use, and the time we had
(a hard 4-hour cap, two restarts) did not allow a third attempt. Checkpoint 2 therefore reports a validated pipeline and a null RL result.

## 5. Persona: outputs and metrics

Outputs: `persona_eval/gens/heldout_{touchup,grpo_v2,grpo_v2_ck75}_eval.jsonl` (502 held-out prompts, greedy, 400 tokens),
`persona_eval/gens/ood_{touchup,grpo_v2}.jsonl` (40 OOD prompts), `persona_eval/gens/short_heldout_{v3b,touchup,grpo_v2}.jsonl`
(60 short prompts never trained on), all with base and v3b outputs to the same prompts from checkpoint 1.

**5a. Style classifier (the checkpoint-1 metric).** Same classifier, retrained from the same data, applied to the new outputs
(`persona_eval/results/style_classifier_eval_{touchup,grpo_v2}.md`). Mean P(Sheldon) on the 502 held-out prompts:

| variant | gold (dataset) | base | v3b (checkpoint 1) | touch-up | grpo-v2 |
|---|---|---|---|---|---|
| raw | 0.806 | 0.226 | 0.825 | 0.772 [0.764, 0.780] | 0.774 [0.767, 0.781] |
| masked (names → NAME) | 0.900 | 0.129 | 0.912 | 0.861 | 0.865 |
| strict (markdown stripped + masked) | 0.802 | 0.232 | 0.820 | 0.766 | 0.769 |
| no-names | 0.798 | 0.238 | 0.818 | 0.772 | 0.775 |
| fraction scored ≥ 0.5 (strict) | 1.000 | 0.004 | 1.000 | 0.988 | 0.994 |

Gold-vs-base accuracy is unchanged at 0.998 (same training pairs). Both new models sit about 0.05 below v3b and gold on every variant,
and level with each other. This is the checkpoint-1 caveat playing out: the classifier measures "written like the SFT dataset", and the
touch-up deliberately removed the dataset's most stereotyped surface (template openers 97% → 66%), so the score drops even though the
judge prefers the result (5b). It should be read as a drift meter, not a quality score, from here on.

**5b. LLM judge, head to head.** The calibrated Luna judge, 200 held-out prompts, both orders (`results/rlaif/judge_h2h_grpo_v2.json`):

| comparison | win rate of first system | Wilson 95% CI | order-consistent pairs |
|---|---|---|---|
| grpo-v2 (final) vs touch-up | 0.500 | [0.45, 0.55] | 110/200 |
| grpo-v2 (step 75) vs touch-up | 0.535 | [0.49, 0.58] | 96/200 |
| touch-up vs v3b | **0.570** | [0.52, 0.62] | 100/200 |

Persona-blind gate flags on the same 200 replies (rates): refused v3b 0.115 → touch-up 0.060 → grpo-v2 0.070; worse-off 0.340 → 0.300
→ 0.285; false claim about the user 0.450 → 0.365 → 0.410; mean task score 0.77 for all three.

**5c. Deterministic monitors** (`rlaif/audit/quant/compare.py`, 502 held-out prompts; gold = the dataset's own replies):

| metric | v3b | touch-up | grpo-v2 | gold |
|---|---|---|---|---|
| rule penalty (17 terms, mean) | 1.131 | 1.003 | 0.870 | 0.653 |
| repetition-loop term | 0.167 | 0.191 | 0.131 | 0.006 |
| replies hitting the 400-token cap | 21.7% | 18.9% | 17.5% | 0% |
| template-opener share | 96.6% | 65.7% | 63.7% | 34.5% |
| opener entropy (bits) | 4.27 | 5.98 | 5.89 | 7.50 |
| relent block ("Amy asked me to practise kindness…") | 13.1% | 4.8% | 3.4% | 5.0% |
| Tuesday/Thai canon error | 9.4% | 2.0% | 1.6% | 1.2% |
| announced or explained joke | 18.5% | 27.9% | 30.3% | 3.0% |
| Bazinga | 1.8% | 1.2% | 1.2% | 11.2% |
| any cast name | 57% | 40% | 39% | 70% |
| mean words | 235 | 216 | 213 | 262 |
| short prompts (60): mean words / rule penalty | 142 / 1.28 | 78 / 0.46 | 70 / 0.36 | – |

**Reading 5a–5c together.** The touch-up is where the measurable change is: template openers 97% → 66%, opener entropy +1.7 bits, canon
errors and the relent block cut by 3–4×, short prompts answered at half the length, and a significant judge preference over v3b. The
RL run moved the deterministic monitors a little further (rule penalty −0.13, loops and cap hits down; 367 of 502 replies keep the same
first 60 characters and change only in the continuation, where loops and over-runs live), but the judge, which reads the whole reply,
rates it level with the seed, and the classifier rates it level with the seed (0.774 vs 0.772, overlapping CIs). Two audit targets did not move at all in either step: the
announced-joke opener ("I am about to make a joke…", 28–30%) and Bazinga, which the touch-up data still under-represents.

## 6. STEM capability: GSM8K

GSM8K test, 1,319 problems, zero-shot greedy, strict scoring; re-runs move a checkpoint by about ±1 point. Baselines were re-run on the
same node (`evals/gsm8k/*/final.jsonl.summary.json`).

| model | GSM8K strict | boxed |
|---|---|---|
| base | 85.7 | 99.8 |
| v3b (checkpoint 1) | 63.7 | 99.9 |
| sft-touchup-v4 (RL seed) | 63.9 | 99.8 |
| grpo-v2 steps 25 / 50 / 75 / 100 / 109 | 64.7 / 64.8 / 63.9 / 64.9 / 64.1 | 99.8–99.9 |
| **grpo-v2 final (merged)** | **64.0** | 99.9 |

Nothing in the RL prompt mix is a GSM8K-style problem (54 one-line arithmetic prompts out of 5,714), so GSM8K was protected by the 17%
math replay in the touch-up, the KL anchor, and, mostly, by the small total movement of the policy: `runs/grpo-v2/trainer_state.json`
carries the per-step traces and `rlaif/runpod/follow_eval.sh` produced the per-checkpoint scores. This is a weak guarantee: a run at the
learning rate recommended above needs an explicit protection, and the natural one is a ~10% slice of GSM8K-train problems in the RL mix
rewarded by boxed-answer correctness with the persona judge skipped (the repo already holds 4,800 test-deduplicated problems with
references from checkpoint 1). That is the RLVR stage's mechanism and we kept it out of this stage deliberately so the RLAIF numbers stand alone.

## 7. Worklog

- **Sep 19** — v3b persona audit (`rlaif/persona_audit.md`): 8 reviewed slices of the held-out generations, a 233-probe battery
  (constraints, identity, canon, emotional, adversarial), training-set n-gram statistics; design decisions: gated multi-term reward, data
  patch + SFT touch-up before RL, seed from v3b, sibling-pairwise judge, batch-level diversity tax.
- **Sep 20** — data patch (545 canon drops, template and opener caps, sentence dedup → 8,930 persona rows), 813 short-prompt
  replies with a deterministic filter and QA gate, v4 touch-up set, 5,714 RL prompts, judge client, gate and pairwise prompts, 17 rule
  terms, batch tax, reward, `GRPOTrainer` script, cluster launchers. First calibration: gpt-4.1-nano picks the first-shown reply 91% of
  the time (unusable); the OpenAI account was then terminated mid-calibration, the Anthropic keys 401'd, and the cluster node crashed. Handover.
- **Sep 20** — repo audit; the whole data chain regenerated bit-identically from the public dataset; judge client rewritten for
  OpenRouter (reported-cost accounting, reasoning effort, stop-on-spend-cap); calibration rewritten to run on committed probe files;
  Luna calibrated twice with two prompt fixes in between; mock judge + CPU dry run of the full GRPO loop; RunPod launchers, per-checkpoint
  evaluator, sequencer; runtime and cost model (the inherited 32-worker, serial-order judging would have taken 8–17 h per 300 steps).
- **Sep 20–21 (pod)** — node setup 1 min, touch-up 4 min, baselines; smoke test found a hung request costing a 180 s timeout per step
  (cut to 45 s); `grpo-v1` at 1e-6 for 90 steps, diagnosed flat from the reward trace and adapter norms; restarted as `grpo-v2` at 1e-5,
  109 steps to the 4-hour cap; final evaluations; head-to-head judge; classifier. OpenRouter spend for everything: about $73.

**Things that did not work / were abandoned.** Learning rate 1e-6 (no movement) and, at our step budget, 1e-5 (movement too small to
matter). gpt-4.1-nano as a judge. The first gate prompt (flagged 15% of gold as harmful) and the first pairwise rubric (a bland reply won
by avoiding defects). Single-order pairwise judging (position bias 0.65–0.72 on close pairs). A locally served Qwen-32B judge (planned
for a cluster that no longer existed). Running anything on RunPod's `/workspace` volume (S3-backed FUSE: no chmod/utime/hard links,
10× slower; everything ran on the local disk with a mirror). The inherited 180 s request timeout. Two of the audit's items were
implemented but had no effect in 109 steps: the stock-phrase refresh of the batch tax, and the announced-joke pressure.

## 8. What the next run changes

1. Learning rate 3e-5 to 5e-5, at least 200 steps, same reward; the per-checkpoint GSM8K monitor and best-checkpoint selection stay.
2. A GSM8K-train anchor slice (~10%) with a verifiable boxed-answer reward and no persona judge, since the KL anchor will no longer
   be doing the STEM protection on its own.
3. A dedicated rule term for the announced/explained joke (currently only taxed as a stock phrase) and a sign flip on the cast-breadth
   pressure: the audit wants more of the cast, the current idle-name penalty and name tax push the other way.
4. Two judge samples for close sibling pairs, since 45% of them are order-inconsistent and contribute noise rather than signal.
