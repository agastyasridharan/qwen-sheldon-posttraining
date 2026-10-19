# LLM-judge persona evaluation (held-out chat prompts)

Judge: `claude-sonnet-4-6`, temperature 0.0, system prompt = `sheldon_style_guide.md` plus an instruction to ignore factual correctness. 50 held-out ids sampled with seed 0 from `data/heldout_gold.jsonl` (sorted ids, shuffled, first 50). Responses: gold = dataset reference, base = Qwen2.5-3B-Instruct, v3b = Sheldon SFT (greedy, max_new_tokens 400; gold is untruncated). Raw judge outputs: `results/judge_raw.jsonl`. Script: `persona_eval/judge.py`.

API calls: 350. API failures (exceptions after SDK retries, or refusals): 0. Unparseable/invalid judge outputs: 0. Wall time 2.7 min. Tokens: input 277714, cache read 562100, cache write 0, output 43137.

## 1. Pairwise: base vs v3b

Each pair is judged twice with A/B order swapped; the judge must pick one. Win rate is over all valid judgments (2 per prompt), Wilson 95% CI. The per-prompt row shows how often the two orderings agreed.

**base vs v3b (main)** - v3b wins 98/100 judgments (100 attempted): win rate **0.980** [0.930, 0.994]. Per prompt: v3b wins both orderings 48, split 2, base wins both 0, prompts with a failed judgment 0.

**base vs gold (control)** - gold wins 100/100 judgments (100 attempted): win rate **1.000** [0.963, 1.000]. Per prompt: gold wins both orderings 50, split 0, base wins both 0, prompts with a failed judgment 0.

| Comparison | Judgments (valid/attempted) | Target wins | Win rate | Wilson 95% CI | Both orders | Split | Neither |
|---|---|---|---|---|---|---|---|
| v3b vs base | 100/100 | 98 | 0.980 | [0.930, 0.994] | 48 | 2 | 0 |
| gold vs base | 100/100 | 100 | 1.000 | [0.963, 1.000] | 50 | 0 | 0 |

## 2. Rubric scores (each response judged alone)

Six dimensions from section 5 of the style guide, 0-2 each; penalties (Bazinga more than once, more than two idle name-drops, each section-3 violation) subtract 1 each, floor 0. Total is computed from the judge's per-dimension integers, max 12. Mean with 95% bootstrap CI (2000 resamples, seed 0).

| System | n scored | failed | Mean total (95% CI) | Mean raw total (no penalties) | Mean penalties |
|---|---|---|---|---|---|
| gold | 50 | 0 | 8.00 [7.26, 8.68] | 8.72 | 0.72 |
| base | 50 | 0 | 0.04 [0.00, 0.10] | 0.58 | 2.12 |
| v3b | 50 | 0 | 6.50 [5.86, 7.12] | 7.26 | 0.76 |

Paired differences in mean total (bootstrap over prompts scored for both systems):

| Difference | Mean | 95% CI |
|---|---|---|
| v3b - base (n=50) | +6.46 | [+5.84, +7.06] |
| v3b - gold (n=50) | -1.50 | [-2.28, -0.72] |
| gold - base (n=50) | +7.96 | [+7.26, +8.66] |

Per-dimension means (0-2):

| # | Dimension | gold | base | v3b |
|---|---|---|---|---|
| 1 | Pedantic precision | 1.72 | 0.02 | 1.48 |
| 2 | Superiority as fact | 1.56 | 0.00 | 1.20 |
| 3 | Rules and routines | 0.94 | 0.02 | 0.92 |
| 4 | Formal register | 1.58 | 0.54 | 1.54 |
| 5 | Tangent or reference | 1.62 | 0.00 | 0.94 |
| 6 | Social obliviousness | 1.30 | 0.00 | 1.18 |

Penalty rates (fraction of responses with the penalty; section-3 is mean count):

| Penalty | gold | base | v3b |
|---|---|---|---|
| penalty_bazinga | 0.00 | 0.00 | 0.00 |
| penalty_name_drops | 0.18 | 0.00 | 0.12 |
| penalty_section3 | 0.54 | 2.12 | 0.64 |

Score distribution (count of responses at each total):

| Total | gold | base | v3b |
|---|---|---|---|
| 0 | 0 | 48 | 0 |
| 1 | 1 | 2 | 0 |
| 2 | 1 | 0 | 1 |
| 3 | 0 | 0 | 3 |
| 4 | 2 | 0 | 9 |
| 5 | 3 | 0 | 6 |
| 6 | 6 | 0 | 4 |
| 7 | 8 | 0 | 10 |
| 8 | 7 | 0 | 8 |
| 9 | 7 | 0 | 3 |
| 10 | 7 | 0 | 4 |
| 11 | 3 | 0 | 1 |
| 12 | 5 | 0 | 1 |

