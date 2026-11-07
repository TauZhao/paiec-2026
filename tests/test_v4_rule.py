"""Offline checks of submission/model.py with no GPU, network or data.

Run from the release directory:  python tests/test_v4_rule.py
Ratings are stubbed, so this checks the rule and the E1 fallbacks, not Gemma.
"""
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PAIEC_V4_CACHE"] = tempfile.mkdtemp(prefix="v4_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
import e1_base  # noqa: E402
import model  # noqa: E402

rng = random.Random(0)
items = [dict(benchmark_id="b1", item_content=f"task {i}", item_features="") for i in range(40)]
truth = {it["item_content"]: rng.uniform(2, 8) for it in items}
subjects = [dict(model=f"m{j}") for j in range(3)]
labeled = []
for s in subjects:
    for it in items[:25]:
        p = 1 / (1 + math.exp(truth[it["item_content"]] - 5))
        labeled.append(([s, it], int(rng.random() < p)))
target = [subjects[0], items[30]]

# 1. Budget zero equals E1 and never loads the model.
assert model.predict(target, []) == e1_base.predict(target, [])
assert not model._started

# 2. With labels but no engine (model cannot load), the result equals E1.
model.LOAD_WAIT = 0.01
model._engine.clear()
model._ready.set()
model._started = True
assert model.predict(target, labeled) == e1_base.predict(target, labeled)

# 3. With stubbed ratings the rule matches a direct implementation.
model._engine.update(stub=True)
model._rate_batch = lambda batch: [truth[i["item_content"]] for i in batch]
got = model.predict(target, labeled)
rows = [(repr(o), r, y) for (o, r), y in labeled]
score = {r["item_content"]: (5 - truth[r["item_content"]]) * model.SCALE for _, r, _ in rows}
mean = sum(score.values()) / len(score)
m = len(score)
tot = {}
for n, _, y in rows:
    k, c = tot.get(n, (0, 0))
    tot[n] = (k + y, c + 1)
g = h = 0.0
for n, r, y in rows:
    k, c = tot[n]
    rate = (k + 1.5) / (c + 3)
    x = min(max(score[r["item_content"]] - mean, -model.CAP), model.CAP)
    g += (y - rate) * x
    h += rate * (1 - rate) * x * x
beta = min(max((model.LAM * model.BETA0 + g) / (model.LAM + h), 0), model.BETA_MAX)
s = (5 - truth[target[1]["item_content"]]) * model.SCALE
x = min(max(m / (m + 3) * (s - mean), -model.CAP), model.CAP)
base = min(max(e1_base.predict(target, labeled), 1e-4), 1 - 1e-4)
want = 1 / (1 + math.exp(-(math.log(base / (1 - base)) + beta * x)))
assert abs(got - want) < 1e-9, (got, want)
print("OK: budget 0 = E1, no model = E1, rule matches (diff %.1e)" % abs(got - want))
