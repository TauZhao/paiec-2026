# PAIEC 2026 entry: E1 plus a relative Gemma difficulty rating

This directory holds the program we selected for the NeurIPS 2026 Predictive AI
Evaluation Competition (Codabench competition 17828) and a small offline test.

- Selected Codabench submission ID: **1034506** (development score 0.1452 on that draw).
  Submissions 1024989 and 1034503 ran byte-identical ZIPs.
- ZIP SHA-256: `43b547eb723b015c811095e298877e34c65237ef82b1e4a6688259f95009a724`.
  `submission/` is that ZIP unpacked, unchanged.

## Method

`submission/e1_base.py` is the public E1 predictor (Beta shrinkage of the pair's own
labels toward a peer-mixed benchmark mean, plus same-item peer residuals). On top of
it, `submission/model.py` asks `google/gemma-4-31B-it` how hard each visible item is,
reading the probabilities of the digits 1–9 after a fixed prompt and taking their
expectation. Only differences between items on the same benchmark are used:

    s    = (5 - rating) * 1.2 / 2.218
    x    = clip(m / (m + 3) * (s - mean_s), -1.5, 1.5)
    beta = clip((11 * 0.6 + g) / (11 + H), 0, 1.2)
    p    = sigmoid(logit(p_E1) + beta * x)

`mean_s` and `m` come from the rated labeled items of the same benchmark. `g` and `H`
are the score and information of a logistic slope at zero, computed from those labels
with each subject's smoothed success rate as the offset, so the slope shrinks toward
0.6 when labels are few and moves toward zero when the rating does not track the labels.
With no labels on the benchmark (budget 0) the output is exactly E1 and the model is
not loaded. If the GPU, the model or a rating fails, that item falls back to E1.

## Running

The platform calls `predict([subject, item], labeled)` from `submission/model.py`.
`models.txt` asks the platform to provide Gemma 4 31B; `prediction_config.json` sets
concurrency 1. A GPU with about 80 GB is needed for bfloat16 inference. No API key
and no network access are used.

Offline check of the rule and the E1 fallbacks (no GPU needed, ratings stubbed):

    python tests/test_v4_rule.py

## Attribution and licences

- `e1_base.py`: unchanged MIT code from Agnuxo1/AIMS-PAIEC-2026, commit
  8e0a4ac27b2bd288db8b3572b71edbd5357463c7 (`submission/UPSTREAM_LICENSE`).
- The difficulty prompt and expected-digit rating follow veseluhha/paiec-2026,
  commit 63e7725, MIT (`submission/LICENSE_veseluhha`).
- Gemma 4 31B (Apache-2.0) is used as released, without fine-tuning.
- Our own code is MIT (`submission/LICENSE`).
- Development used the official measurement-db sources and the public Gemma scores
  released by veseluhha (CC BY-SA 4.0). No data files are included here.
