"""
verl_reward_adapter.py — verl-compatible reward function for the Concrete Hypothesis Swarm.

Contract (verl custom_reward_function interface):
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Intentional design notes
------------------------
* Does NOT import agents.py: agents.py already imports reward.py, and it lazily
  imports the openai client on any call path — both create coupling that breaks in
  a verl training pod that has no Fireworks credentials and no openai package.
  The JSON/code extraction helpers below are duplicated from agents._extract_json
  and reward._extract_code rather than imported. This is an intentional adapter
  pattern: training-time scoring is a different call context than frozen-swarm
  inference, and the two should not share a dependency chain.

* Return 0.0 on any failure, never raise: a single malformed rollout in a GRPO
  group must not crash the whole batch. The frozen-swarm path (agents.py) correctly
  raises on parse failures because a failure there means no usable hypothesis was
  produced and the run should halt. Here, it just means that rollout gets zero
  reward and the other rollouts in the group proceed normally.

extra_info schema (native dict — parquet nested struct, NOT a JSON string):
  {
    "reward_eval_rows": list[dict],   # full rows (features + "strength") from train
    "train_y":          list[float],  # ALL training-set strength values,
                                      # used as the mean-baseline denominator in
                                      # reward._regression_skill (fmean(train_y))
  }

  Only target values (not full rows) are needed for train_y, keeping extra_info
  compact: reward._regression_skill(preds, truth, train_y) uses train_y only to
  compute statistics.fmean(train_y) for the SSE baseline denominator.
"""

from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Self-contained extraction helpers
# (Duplicated from agents._extract_json and reward._extract_code.
#  See module docstring for why we don't import those directly.)
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Strip ```python ... ``` fences if the model wrapped the code."""
    if "```" in raw:
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
    return raw.strip()


def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction.

    Try a direct json.loads() first so that valid JSON containing literal
    backticks in string values (e.g. predict_code wrapped in ```python fences)
    doesn't get incorrectly treated as a markdown code fence. Only fall back to
    fence-stripping if the raw string isn't valid JSON as-is (e.g. the model
    wrapped its entire response in ```json ... ```).
    """
    # Fast path: well-formed JSON (possibly with backticks inside values)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Slow path: strip outer markdown fences, then find outermost { ... }
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# verl reward contract
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: str | None = None,
) -> float:
    """
    Score one rollout against held-out concrete rows. Returns total reward in [0, 1].

    All failure modes return 0.0 rather than raising, so a bad rollout doesn't
    crash the group. See module docstring for design rationale.
    """
    # ── 1. Parse the model's output ──────────────────────────────────────────
    try:
        parsed      = _extract_json(solution_str)
        predict_code = parsed.get("predict_code", "")
        if not isinstance(predict_code, str) or not predict_code.strip():
            return 0.0
    except Exception:
        return 0.0

    # ── 2. Unpack extra_info ─────────────────────────────────────────────────
    try:
        # extra_info is now a native dict from the parquet nested struct column.
        # Defensive: pyarrow may return a dict-like struct scalar rather than a
        # plain dict (same quirk seen with the "prompt" field) — dict() normalises
        # both. json.loads fallback retained for the legacy JSON-string format.
        if isinstance(extra_info, str):
            ei = json.loads(extra_info)
        else:
            ei = dict(extra_info) if extra_info is not None else {}
        # list() converts numpy/pyarrow arrays to plain Python lists if pyarrow
        # deserialized the list<...> fields as arrays instead of lists.
        reward_eval_rows = list(ei["reward_eval_rows"])   # list[dict]  features + "strength"
        train_y_values   = list(ei["train_y"])            # list[float] target values
    except Exception:
        return 0.0

    # ── 3. Score via reward.py ───────────────────────────────────────────────
    try:
        from reward import Dataset, score_hypothesis

        # Build a minimal train list so Dataset.train_y returns the values that
        # _regression_skill needs for its mean-baseline calc (only the target key
        # is accessed; feature columns are never read from train rows).
        fake_train = [{"strength": y} for y in train_y_values]

        dataset = Dataset(
            task_type="regression",
            target="strength",
            train=fake_train,
            test=reward_eval_rows,
        )
        return score_hypothesis(predict_code, dataset).total
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import random

    rng = random.Random(42)

    def _synth_row(rng: random.Random) -> dict:
        cement = rng.uniform(100, 540)
        slag   = rng.uniform(0, 200)
        fly    = rng.uniform(0, 150)
        water  = rng.uniform(140, 230)
        sp     = rng.uniform(0, 30)
        ca     = rng.uniform(800, 1150)
        fa     = rng.uniform(600, 1000)
        age    = float(rng.choice([1, 3, 7, 14, 28, 56, 90, 180, 365]))
        cem_total = cement + slag + fly
        wc  = water / (cem_total + 1e-6)
        # Synthetic ground truth following approximate Abrams + log-age law
        strength = max(2.0, 80.0 - 100.0 * wc + 3.0 * math.log(age + 1) + rng.gauss(0, 3))
        return {
            "cement": round(cement, 2), "blast_furnace_slag": round(slag, 2),
            "fly_ash": round(fly, 2), "water": round(water, 2),
            "superplasticizer": round(sp, 2), "coarse_aggregate": round(ca, 2),
            "fine_aggregate": round(fa, 2), "age": age,
            "strength": round(strength, 4),
        }

    all_rows  = [_synth_row(rng) for _ in range(200)]
    train_y   = [r["strength"] for r in all_rows[:160]]
    eval_rows = all_rows[160:]

    extra_info = {"reward_eval_rows": eval_rows, "train_y": train_y}

    # ── Test 1: valid w/c-ratio hypothesis (should score > 0) ────────────────
    valid_solution = json.dumps({
        "claim": "Strength decreases as the water-to-cementitious ratio rises.",
        "predict_code": (
            "def predict(row):\n"
            "    cem = row['cement'] + row['blast_furnace_slag'] + row['fly_ash']\n"
            "    wc  = row['water'] / (cem + 1e-6)\n"
            "    return max(5.0, 80.0 - 100.0 * wc)"
        ),
    })
    s1 = compute_score("concrete_swarm", valid_solution, "", extra_info)
    print(f"[1] Valid w/c hypothesis         : {s1:.4f}  (expect > 0.0)")

    # ── Test 2: malformed JSON ────────────────────────────────────────────────
    s2 = compute_score("concrete_swarm", "not json at all {{{{", "", extra_info)
    print(f"[2] Malformed solution_str        : {s2:.4f}  (expect 0.0)")

    # ── Test 3: valid JSON, no predict_code field ─────────────────────────────
    s3 = compute_score("concrete_swarm", json.dumps({"claim": "no code here"}), "", extra_info)
    print(f"[3] Missing predict_code          : {s3:.4f}  (expect 0.0)")

    # ── Test 4: code that raises at runtime ───────────────────────────────────
    s4 = compute_score("concrete_swarm", json.dumps({
        "claim": "This will crash.",
        "predict_code": "def predict(row):\n    return 1 / 0",
    }), "", extra_info)
    print(f"[4] Runtime-crashing predict_code : {s4:.4f}  (expect 0.0, must not raise)")

    # ── Test 5: code wrapped in markdown fences ───────────────────────────────
    fenced_solution = json.dumps({
        "claim": "Age drives strength.",
        "predict_code": (
            "```python\n"
            "def predict(row):\n"
            "    return min(60.0, 10.0 * math.log(row['age'] + 1))\n"
            "```"
        ),
    })
    s5 = compute_score("concrete_swarm", fenced_solution, "", extra_info)
    print(f"[5] Fenced predict_code           : {s5:.4f}  (expect > 0.0, fence stripped)")

    print()
    assert s1 > 0.0,   "valid hypothesis must score > 0"
    assert s2 == 0.0,  "malformed JSON must score 0.0"
    assert s3 == 0.0,  "missing predict_code must score 0.0"
    assert s4 == 0.0,  "crashing code must score 0.0"
    assert s5 > 0.0,   "fenced code must be stripped and scored"
    print("All assertions passed.")
