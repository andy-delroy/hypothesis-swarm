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
  "GOOD (captures rule)": "def predict(row):\n    return 1 if row['a']*row['b'] > 0.25 else 0\n",
  "PARTIAL (wrong threshold)": "def predict(row):\n    return 1 if row['a']*row['b'] > 0.5 else 0\n",
  "DISTRACTOR (uses c only)": "def predict(row):\n    return 1 if row['c'] > 0.5 else 0\n",
  "CONSTANT (majority cheese)": "def predict(row):\n    return 0\n",
  "BROKEN (won't compile)":     "def predict(row)\n    return 1\n",
  "IMPORT ATTEMPT (blocked)":   "import os\ndef predict(row):\n    return 1\n",
}

for name, code in cases.items():
    b = score_hypothesis(code, data)
    print(f"{name:30s} total={b.total:5.2f}  acc={b.raw_metric:.2f}  skill={b.skill:.2f}  {b.note}")
