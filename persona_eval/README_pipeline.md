Task: style classifier for the Sheldon SFT assignment (checkpoint 1).
Goal: logistic regression that scores how much a response looks like the SFT dataset's Sheldon responses.

Data:
- Dataset: tbooy/sheldon-cooper-sft-20k, split train. Each row: messages (list of role/content), label, meta.
- Training questions: 1,000 rows sampled with seed 0. Use the first user turn as the prompt and the first assistant turn as gold.
- Held-out evaluation: 502 rows my partner will send later, plus base and v3b responses to them. Not available yet.
- Base responses to the 1,000 training prompts are generated separately (remote GPU). This laptop is CPU only; only run generation here on tiny test sets.

Classifier: sklearn logistic regression on TF-IDF features (char_wb 3-5 grams + word 1-2 grams), C by 5-fold CV. Class 1 = gold, class 0 = base response, always paired on the same prompts.
Run twice: raw text, and with Big Bang Theory names/catchphrases (from sheldon_style_guide.md, plus Bazinga, Meemaw, Moon Pie, roommate agreement, Fun with Flags) replaced by the token NAME.
Report: accuracy gold-vs-base on the eval set, mean P(Sheldon) with bootstrap CI for gold, base, v3b, and the 30 largest positive and negative coefficients.

Generation settings (for any model): chat template, no system prompt, greedy, max_new_tokens 400. Same for every model.
Never train on the 502 held-out prompts. Report row counts after every step. Do not silently skip rows.