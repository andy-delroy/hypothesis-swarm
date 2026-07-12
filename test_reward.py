import random
from reward import Dataset, score_hypothesis

random.seed(0)
# Planted latent rule: label = 1 iff (a * b) > 0.25, plus a little noise.
def make_rows(n):
    rows = []
    for _ in range(n):
        a, b, c = random.random(), random.random(), random.random()  # c is a distractor
        label = 1 if (a * b) > 0.25 else 0
        if random.random() < 0.05:            # 5% label noise
            label = 1 - label
        rows.append({"a": round(a,3), "b": round(b,3), "c": round(c,3), "label": label})
    return rows

data = Dataset(task_type="classification", target="label",
               train=make_rows(200), test=make_rows(200))

base_rate = sum(r["label"] for r in data.test) / len(data.test)
print(f"held-out positive rate: {base_rate:.2f}  (majority baseline acc ~ {max(base_rate,1-base_rate):.2f})\n")

cases = {
  # Named feature(s) — no literal coefficients. reward.py fits an OLS linear
  # probability model on data.train and thresholds at 0.5 for the class call.
  "GOOD (a*b interaction)":     "def features(row):\n    return {'ab': row['a'] * row['b']}\n",
  "PARTIAL (a alone, half the signal)": "def features(row):\n    return {'a': row['a']}\n",
  "DISTRACTOR (uses c only)":   "def features(row):\n    return {'c': row['c']}\n",
  "CONSTANT (no signal)":       "def features(row):\n    return {'k': 1.0}\n",
  "BROKEN (won't compile)":     "def features(row)\n    return {'x': 1}\n",
  "IMPORT ATTEMPT (blocked)":   "import os\ndef features(row):\n    return {'x': 1}\n",
}

for name, code in cases.items():
    b = score_hypothesis(code, data)
    print(f"{name:36s} total={b.total:5.2f}  acc={b.raw_metric:.2f}  skill={b.skill:.2f}  {b.note}")

# ── Genuinely computable assertions ─────────────────────────────────────────
# GOOD captures the real interaction term the labels were planted on; it must
# clearly beat PARTIAL (half the signal) and DISTRACTOR/CONSTANT (no real
# signal). CONSTANT must land at (or near) 0.0 total, confirming the
# reward-floor fix: a syntactically valid, running hypothesis with zero skill
# no longer earns the old guaranteed 0.10 (format) + 0.10 (coverage) floor.
results = {name: score_hypothesis(code, data) for name, code in cases.items()}
assert results["GOOD (a*b interaction)"].total > results["PARTIAL (a alone, half the signal)"].total, \
    "the true interaction feature must beat a weaker single-feature signal"
assert results["PARTIAL (a alone, half the signal)"].total > results["DISTRACTOR (uses c only)"].total, \
    "a genuinely correlated feature must beat an unrelated distractor"
assert results["CONSTANT (no signal)"].total == 0.0, \
    "a constant feature must score 0.0 total, not the old format+coverage floor"
assert results["BROKEN (won't compile)"].total == 0.0
assert results["IMPORT ATTEMPT (blocked)"].total == 0.0
print("\nAll assertions passed.")
