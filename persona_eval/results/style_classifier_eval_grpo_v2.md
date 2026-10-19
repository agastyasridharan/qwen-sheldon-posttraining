# Style classifier (eval)

Train: data/train_gold.jsonl vs gens/base_train.jsonl, 972 paired prompts (1000 sampled, 28 dropped because their dataset id is in the 502-row held-out set data/heldout_gold.jsonl), 1944 rows. Texts truncated to 400 tokens with Qwen/Qwen2.5-3B-Instruct: 219 gold and 0 base texts truncated; 361 base responses at the generation cap.

Features: TF-IDF char_wb 3-5 grams + word 1-2 grams (min_df 2, sublinear tf). Logistic regression, C from [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] by 5-fold CV with folds grouped by prompt id.

Masked run replaces these with `NAME` (case-insensitive, word-bounded; consecutive NAMEs collapsed to one): Fun with Flags, roommate agreement, Moon Pie, Star Trek, East Texas, Farrah Fowler, Good Lord, Oh, dear, Excuse me, For the record, there, there, Sheldon, Cooper, Leonard, Hofstadter, Howard, Wolowitz, Rajesh, Raj, Koothrappali, Penny, Amy, Fowler, Bernadette, Mary, Meemaw, Caltech, Spock, Texas, Bazinga, Incidentally, Dr.

Strict run first strips markdown (bold/italic markers, headers, list bullets and numbered-list markers, code fences and backticks, horizontal rules, link syntax), collapses whitespace, then applies the same NAME mask.

No-names run is the strict run with the NAME token deleted rather than kept.

## Summary: eval on data/heldout_gold.jsonl (502 held-out prompts)

Base = gens/heldout_base_eval.jsonl, ft = gens/heldout_grpo_v2_eval.jsonl. Mean P(Sheldon) with 95% bootstrap CI, then the fraction of rows scored >= 0.5.

| run | gold-vs-base accuracy | gold | base | ft | gold >= 0.5 | base >= 0.5 | ft >= 0.5 |
|---|---|---|---|---|---|---|---|
| raw | 0.9980 | 0.8063 [0.8005, 0.8117] | 0.2256 [0.2189, 0.2319] | 0.7741 [0.7669, 0.7814] | 1.0000 | 0.0040 | 0.9960 |
| masked | 0.9980 | 0.8995 [0.8942, 0.9046] | 0.1285 [0.1222, 0.1350] | 0.8647 [0.8564, 0.8728] | 1.0000 | 0.0040 | 0.9940 |
| strict | 0.9980 | 0.8022 [0.7964, 0.8075] | 0.2316 [0.2257, 0.2377] | 0.7690 [0.7618, 0.7765] | 1.0000 | 0.0040 | 0.9940 |
| no-names | 0.9980 | 0.7984 [0.7926, 0.8037] | 0.2378 [0.2316, 0.2440] | 0.7753 [0.7684, 0.7821] | 1.0000 | 0.0040 | 0.9960 |

## Run: raw

Best C = 0.3, features = 135999.

| C | CV accuracy | std |
|---|---|---|
| 0.01 | 0.9990 | 0.0013 |
| 0.03 | 0.9990 | 0.0013 |
| 0.1 | 0.9995 | 0.0010 |
| 0.3 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 0.0000 |
| 10 | 1.0000 | 0.0000 |
| 30 | 1.0000 | 0.0000 |
| 100 | 1.0000 | 0.0000 |

- 5-fold CV accuracy (best C): **1.0000** (out-of-fold 1.0000, 1944 rows)
- CV accuracy on pairs whose base response did not hit the 400-token cap: **1.0000** (611 pairs, 1222 rows)

- Eval accuracy gold-vs-base: **0.9980** (502 pairs)

| set | n | mean P(Sheldon) | 95% bootstrap CI | frac P >= 0.5 |
|---|---|---|---|---|
| gold | 502 | 0.8063 | [0.8005, 0.8117] | 1.0000 |
| base | 502 | 0.2256 | [0.2189, 0.2319] | 0.0040 |
| ft | 502 | 0.7741 | [0.7669, 0.7814] | 0.9960 |

### Top 30 positive coefficients (toward gold)

| feature | coef |
|---|---|
| `word__which` | +1.198 |
| `word__not` | +1.083 |
| `word__is` | +1.074 |
| `word__which is` | +1.035 |
| `word__my` | +0.966 |
| `word__because` | +0.946 |
| `word__me` | +0.895 |
| `char__ i ` | +0.804 |
| `word__leonard` | +0.705 |
| `word__is not` | +0.652 |
| `word__is the` | +0.627 |
| `char__ whic` | +0.610 |
| `word__no` | +0.608 |
| `word__but` | +0.606 |
| `word__that` | +0.603 |
| `char__hich` | +0.602 |
| `char__which` | +0.602 |
| `char__whic` | +0.602 |
| `char__hich ` | +0.593 |
| `char__hic` | +0.589 |
| `word__you` | +0.587 |
| `char__ich ` | +0.584 |
| `char__ich` | +0.584 |
| `word__have` | +0.580 |
| `word__it is` | +0.579 |
| `word__one` | +0.577 |
| `word__two` | +0.569 |
| `char__ is ` | +0.541 |
| `char__not ` | +0.537 |
| `word__though` | +0.531 |

### Top 30 negative coefficients (toward base)

| feature | coef |
|---|---|
| `char__ **` | -0.795 |
| `char__ - ` | -0.724 |
| `word__sure` | -0.720 |
| `word__might` | -0.644 |
| `word__can` | -0.620 |
| `char__** ` | -0.541 |
| `word__or` | -0.503 |
| `word__some` | -0.502 |
| `char__**: ` | -0.480 |
| `char__**:` | -0.480 |
| `word__can be` | -0.479 |
| `char__*: ` | -0.478 |
| `word__these` | -0.453 |
| `char__###` | -0.413 |
| `word__we` | -0.403 |
| `char__ ###` | -0.403 |
| `char__## ` | -0.403 |
| `char__ ##` | -0.403 |
| `char__### ` | -0.403 |
| `char__ ### ` | -0.380 |
| `char__ sure` | -0.379 |
| `word__let` | -0.374 |
| `char__:** ` | -0.372 |
| `char__:**` | -0.372 |
| `word__help` | -0.367 |
| `word__this` | -0.361 |
| `char__’s ` | -0.348 |
| `char__ can ` | -0.344 |
| `char__e! ` | -0.341 |
| `word__certainly` | -0.341 |

## Run: masked

Best C = 1, features = 135739.

| C | CV accuracy | std |
|---|---|---|
| 0.01 | 0.9990 | 0.0013 |
| 0.03 | 0.9990 | 0.0013 |
| 0.1 | 0.9995 | 0.0010 |
| 0.3 | 0.9995 | 0.0010 |
| 1 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 0.0000 |
| 10 | 1.0000 | 0.0000 |
| 30 | 1.0000 | 0.0000 |
| 100 | 0.9995 | 0.0010 |

- 5-fold CV accuracy (best C): **1.0000** (out-of-fold 1.0000, 1944 rows)
- CV accuracy on pairs whose base response did not hit the 400-token cap: **1.0000** (611 pairs, 1222 rows)

- Eval accuracy gold-vs-base: **0.9980** (502 pairs)

| set | n | mean P(Sheldon) | 95% bootstrap CI | frac P >= 0.5 |
|---|---|---|---|---|
| gold | 502 | 0.8995 | [0.8942, 0.9046] | 1.0000 |
| base | 502 | 0.1285 | [0.1222, 0.1350] | 0.0040 |
| ft | 502 | 0.8647 | [0.8564, 0.8728] | 0.9940 |

### Top 30 positive coefficients (toward gold)

| feature | coef |
|---|---|
| `word__name` | +2.380 |
| `word__which` | +1.948 |
| `word__is` | +1.761 |
| `word__not` | +1.735 |
| `word__which is` | +1.677 |
| `word__my` | +1.520 |
| `word__because` | +1.487 |
| `char__ i ` | +1.315 |
| `word__me` | +1.262 |
| `char__name` | +1.147 |
| `char__nam` | +1.136 |
| `char__ nam` | +1.135 |
| `char__ name` | +1.134 |
| `word__is not` | +1.042 |
| `word__is the` | +1.025 |
| `word__no` | +0.999 |
| `char__ame` | +0.996 |
| `char__ na` | +0.994 |
| `char__ whic` | +0.990 |
| `word__you` | +0.981 |
| `char__hich` | +0.977 |
| `char__which` | +0.977 |
| `char__whic` | +0.977 |
| `word__that` | +0.970 |
| `word__but` | +0.963 |
| `char__hich ` | +0.961 |
| `char__hic` | +0.955 |
| `char__ich` | +0.948 |
| `char__ich ` | +0.947 |
| `word__have` | +0.938 |

### Top 30 negative coefficients (toward base)

| feature | coef |
|---|---|
| `word__sure` | -1.220 |
| `char__ **` | -1.220 |
| `char__ - ` | -1.122 |
| `word__might` | -1.023 |
| `word__can` | -0.958 |
| `char__** ` | -0.837 |
| `word__or` | -0.780 |
| `word__some` | -0.777 |
| `char__**:` | -0.727 |
| `char__**: ` | -0.727 |
| `char__*: ` | -0.724 |
| `word__can be` | -0.722 |
| `word__these` | -0.717 |
| `word__we` | -0.665 |
| `char__ sure` | -0.649 |
| `char__###` | -0.625 |
| `char__## ` | -0.610 |
| `char__### ` | -0.610 |
| `char__ ##` | -0.610 |
| `char__ ###` | -0.610 |
| `word__this` | -0.603 |
| `char__e! ` | -0.594 |
| `word__let` | -0.592 |
| `word__certainly` | -0.581 |
| `char__’s ` | -0.579 |
| `char__ ### ` | -0.575 |
| `char__:** ` | -0.558 |
| `char__:**` | -0.558 |
| `word__help` | -0.544 |
| `char__ migh` | -0.537 |

## Run: strict

Best C = 0.3, features = 131626.

| C | CV accuracy | std |
|---|---|---|
| 0.01 | 0.9990 | 0.0013 |
| 0.03 | 0.9995 | 0.0010 |
| 0.1 | 0.9995 | 0.0010 |
| 0.3 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 0.0000 |
| 10 | 1.0000 | 0.0000 |
| 30 | 1.0000 | 0.0000 |
| 100 | 1.0000 | 0.0000 |

- 5-fold CV accuracy (best C): **1.0000** (out-of-fold 1.0000, 1944 rows)
- CV accuracy on pairs whose base response did not hit the 400-token cap: **1.0000** (611 pairs, 1222 rows)

- Eval accuracy gold-vs-base: **0.9980** (502 pairs)

| set | n | mean P(Sheldon) | 95% bootstrap CI | frac P >= 0.5 |
|---|---|---|---|---|
| gold | 502 | 0.8022 | [0.7964, 0.8075] | 1.0000 |
| base | 502 | 0.2316 | [0.2257, 0.2377] | 0.0040 |
| ft | 502 | 0.7690 | [0.7618, 0.7765] | 0.9940 |

### OOD prompts (strict run): gens/ood_base.jsonl vs gens/ood_grpo_v2.jsonl

| set | n | mean P(Sheldon) | 95% bootstrap CI | P >= 0.5 |
|---|---|---|---|---|
| base | 40 | 0.2339 | [0.2146, 0.2527] | 0 |
| ft | 40 | 0.6215 | [0.5733, 0.6674] | 31 |

### Top 30 positive coefficients (toward gold)

| feature | coef |
|---|---|
| `word__name` | +1.486 |
| `word__which` | +1.229 |
| `word__not` | +1.111 |
| `word__is` | +1.105 |
| `word__which is` | +1.060 |
| `word__my` | +1.004 |
| `word__because` | +0.969 |
| `char__ i ` | +0.831 |
| `word__me` | +0.812 |
| `char__name` | +0.721 |
| `char__ nam` | +0.713 |
| `char__ name` | +0.713 |
| `char__nam` | +0.713 |
| `word__is not` | +0.667 |
| `word__is the` | +0.638 |
| `word__but` | +0.628 |
| `char__ whic` | +0.626 |
| `word__that` | +0.624 |
| `word__no` | +0.623 |
| `char__ na` | +0.622 |
| `char__ame` | +0.622 |
| `char__whic` | +0.618 |
| `char__hich` | +0.618 |
| `char__which` | +0.618 |
| `char__hich ` | +0.608 |
| `char__hic` | +0.603 |
| `word__have` | +0.600 |
| `char__ich ` | +0.599 |
| `char__ich` | +0.599 |
| `word__you` | +0.597 |

### Top 30 negative coefficients (toward base)

| feature | coef |
|---|---|
| `word__sure` | -0.735 |
| `word__might` | -0.657 |
| `word__can` | -0.638 |
| `word__some` | -0.527 |
| `word__or` | -0.519 |
| `word__can be` | -0.502 |
| `word__these` | -0.473 |
| `char__ sure` | -0.395 |
| `word__we` | -0.394 |
| `word__help` | -0.391 |
| `word__let` | -0.383 |
| `word__this` | -0.373 |
| `char__’s ` | -0.371 |
| `char__ can ` | -0.364 |
| `char__can ` | -0.353 |
| `char__ migh` | -0.350 |
| `char__might` | -0.350 |
| `char__migh` | -0.350 |
| `char__ mig` | -0.349 |
| `char__e! ` | -0.346 |
| `char__mig` | -0.342 |
| `word__certainly` | -0.339 |
| `word__example` | -0.339 |
| `word__break` | -0.332 |
| `word__great` | -0.330 |
| `word__often` | -0.325 |
| `char__s: ` | -0.321 |
| `word__let break` | -0.315 |
| `word__sure let` | -0.311 |
| `char__ so, ` | -0.307 |

## Run: no-names

Best C = 0.3, features = 131500.

| C | CV accuracy | std |
|---|---|---|
| 0.01 | 0.9979 | 0.0030 |
| 0.03 | 0.9990 | 0.0013 |
| 0.1 | 0.9995 | 0.0010 |
| 0.3 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 0.0000 |
| 10 | 1.0000 | 0.0000 |
| 30 | 1.0000 | 0.0000 |
| 100 | 1.0000 | 0.0000 |

- 5-fold CV accuracy (best C): **1.0000** (out-of-fold 1.0000, 1944 rows)
- CV accuracy on pairs whose base response did not hit the 400-token cap: **1.0000** (611 pairs, 1222 rows)

- Eval accuracy gold-vs-base: **0.9980** (502 pairs)

| set | n | mean P(Sheldon) | 95% bootstrap CI | frac P >= 0.5 |
|---|---|---|---|---|
| gold | 502 | 0.7984 | [0.7926, 0.8037] | 1.0000 |
| base | 502 | 0.2378 | [0.2316, 0.2440] | 0.0040 |
| ft | 502 | 0.7753 | [0.7684, 0.7821] | 0.9960 |

### Top 30 positive coefficients (toward gold)

| feature | coef |
|---|---|
| `word__which` | +1.261 |
| `word__is` | +1.137 |
| `word__not` | +1.136 |
| `word__which is` | +1.085 |
| `word__my` | +1.037 |
| `word__because` | +0.992 |
| `char__ i ` | +0.853 |
| `word__me` | +0.845 |
| `word__is not` | +0.681 |
| `word__is the` | +0.650 |
| `word__but` | +0.647 |
| `word__that` | +0.644 |
| `char__ whic` | +0.641 |
| `word__no` | +0.636 |
| `char__whic` | +0.632 |
| `char__hich` | +0.632 |
| `char__which` | +0.632 |
| `char__hich ` | +0.623 |
| `word__have` | +0.617 |
| `char__hic` | +0.617 |
| `word__you` | +0.617 |
| `char__ich ` | +0.614 |
| `char__ich` | +0.613 |
| `word__it is` | +0.606 |
| `word__one` | +0.605 |
| `word__two` | +0.595 |
| `word__he` | +0.572 |
| `char__ is ` | +0.566 |
| `char__not ` | +0.562 |
| `word__so` | +0.559 |

### Top 30 negative coefficients (toward base)

| feature | coef |
|---|---|
| `word__sure` | -0.753 |
| `word__might` | -0.668 |
| `word__can` | -0.651 |
| `word__some` | -0.537 |
| `word__or` | -0.531 |
| `word__can be` | -0.513 |
| `word__these` | -0.484 |
| `char__ sure` | -0.405 |
| `word__we` | -0.403 |
| `word__help` | -0.399 |
| `word__let` | -0.392 |
| `word__this` | -0.379 |
| `char__’s ` | -0.378 |
| `char__ can ` | -0.372 |
| `char__can ` | -0.361 |
| `char__migh` | -0.357 |
| `char__might` | -0.357 |
| `char__ migh` | -0.357 |
| `char__ mig` | -0.356 |
| `char__e! ` | -0.353 |
| `char__mig` | -0.348 |
| `word__example` | -0.346 |
| `word__certainly` | -0.345 |
| `word__break` | -0.341 |
| `word__great` | -0.336 |
| `word__often` | -0.330 |
| `char__s: ` | -0.329 |
| `word__let break` | -0.323 |
| `word__sure let` | -0.319 |
| `char__ so, ` | -0.315 |
