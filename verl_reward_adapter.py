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
    "reward_eval_rows": list[dict],   # full rows (features + "strength"), held out
                                      # for THIS rollout — scored against the fit
    "train_rows":       list[dict],   # full rows (features + "strength"), sampled
                                      # from the train pool — used to fit the OLS
                                      # coefficients (see reward._fit_ols). Full rows
                                      # are needed (not just target values) because
                                      # the reward now fits real coefficients against
                                      # named features(), not just a mean baseline.
  }

  Both slices are subsamples of the same train pool (see verl_data_prep.py), not
  the full training set — keeps parquet size bounded the same way reward_eval_rows
  already did before this file needed a train slice too.
"""

from __future__ import annotations

import json
import re
import sys


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
            print(f"[reward-debug] predict_code missing/empty. solution_str[:300]="
                  f"{repr(solution_str[:300])}", file=sys.stderr)
            return 0.0
    except Exception as exc:
        print(f"[reward-debug] JSON parse failed: {exc}. solution_str[:300]="
              f"{repr(solution_str[:300])}", file=sys.stderr)
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
        train_rows       = list(ei["train_rows"])         # list[dict]  features + "strength"
    except Exception as exc:
        print(f"[reward-debug] extra_info unpacking failed: {exc}. solution_str[:300]="
              f"{repr(solution_str[:300])}", file=sys.stderr)
        return 0.0

    # ── 3. Score via reward.py ───────────────────────────────────────────────
    try:
        from reward import Dataset, score_hypothesis

        dataset = Dataset(
            task_type="regression",
            target="strength",
            train=train_rows,
            test=reward_eval_rows,
        )
        # NOTE: self-imports of math/statistics (bare or indented inside a
        # function body) are stripped centrally in reward.compile_features(),
        # not here — this file used to carry its own copy of that regex, but
        # agents.py's direct score_hypothesis() calls never got the same fix,
        # so a real-model rollout could still zero out silently on that path.
        # Fixed at the source in reward.py so every caller gets it for free.
        result = score_hypothesis(predict_code, dataset)
        if result.total == 0.0:
            print(f"[reward-debug] score_hypothesis returned 0.0 without raising. "
                  f"coverage={getattr(result, 'coverage', None)} "
                  f"degenerate={getattr(result, 'degenerate', None)} "
                  f"skill={getattr(result, 'skill', None)}", file=sys.stderr)
        return result.total
    except Exception as exc:
        print(f"[reward-debug] execution failed: {exc}. solution_str[:300]="
              f"{repr(solution_str[:300])}", file=sys.stderr)
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

    all_rows   = [_synth_row(rng) for _ in range(200)]
    train_rows = all_rows[:160]    # full rows (features + target) — fits the OLS coefficients
    eval_rows  = all_rows[160:]    # held out for scoring the fit

    extra_info = {"reward_eval_rows": eval_rows, "train_rows": train_rows}

    # ── Test 1: valid w/c-ratio feature (should score > 0) ────────────────────
    valid_solution = json.dumps({
        "claim": "Strength decreases as the water-to-cementitious ratio rises.",
        "predict_code": (
            "def features(row):\n"
            "    cem = row['cement'] + row['blast_furnace_slag'] + row['fly_ash']\n"
            "    return {'water_binder_ratio': row['water'] / (cem + 1e-6)}"
        ),
    })
    s1 = compute_score("concrete_swarm", valid_solution, "", extra_info)
    print(f"[1] Valid w/c feature             : {s1:.4f}  (expect > 0.0)")

    # ── Test 2: malformed JSON ────────────────────────────────────────────────
    s2 = compute_score("concrete_swarm", "not json at all {{{{", "", extra_info)
    print(f"[2] Malformed solution_str        : {s2:.4f}  (expect 0.0)")

    # ── Test 3: valid JSON, no predict_code field ─────────────────────────────
    s3 = compute_score("concrete_swarm", json.dumps({"claim": "no code here"}), "", extra_info)
    print(f"[3] Missing predict_code          : {s3:.4f}  (expect 0.0)")

    # ── Test 4: code that raises at runtime ───────────────────────────────────
    s4 = compute_score("concrete_swarm", json.dumps({
        "claim": "This will crash.",
        "predict_code": "def features(row):\n    return {'x': 1 / 0}",
    }), "", extra_info)
    print(f"[4] Runtime-crashing features()   : {s4:.4f}  (expect 0.0, must not raise)")

    # ── Test 5: code wrapped in markdown fences ───────────────────────────────
    fenced_solution = json.dumps({
        "claim": "Age drives strength.",
        "predict_code": (
            "```python\n"
            "def features(row):\n"
            "    return {'log_age': math.log(row['age'] + 1)}\n"
            "```"
        ),
    })
    s5 = compute_score("concrete_swarm", fenced_solution, "", extra_info)
    print(f"[5] Fenced features()             : {s5:.4f}  (expect > 0.0, fence stripped)")

    # ── Test 6: bare "import math" line (should be stripped, then compile) ────
    import_solution = json.dumps({
        "claim": "Strength grows with the square root of age.",
        "predict_code": (
            "import math\n"
            "def features(row):\n"
            "    return {'sqrt_age': math.sqrt(row['age'])}"
        ),
    })
    s6 = compute_score("concrete_swarm", import_solution, "", extra_info)
    print(f"[6] Bare 'import math' line       : {s6:.4f}  (expect > 0.0, import stripped)")

    # ── Test 7: constant feature — reward-floor fix (Decision Log Fork 29) ────
    # A syntactically valid, running hypothesis that carries zero signal must
    # score ~0.0 total, not the old guaranteed 0.20 format+coverage floor.
    constant_solution = json.dumps({
        "claim": "This feature ignores the row entirely.",
        "predict_code": "def features(row):\n    return {'k': 1.0}",
    })
    s7 = compute_score("concrete_swarm", constant_solution, "", extra_info)
    print(f"[7] Constant feature (floor fix)  : {s7:.4f}  (expect 0.0, not the old 0.20 floor)")

    # ── Test 8: good transform vs. deliberately bad transform ─────────────────
    # Synthetic ground truth is 80 - 100*wc + 3*log(age+1) + noise (see _synth_row).
    # A transform close to that structure should clearly beat one built from an
    # unrelated column (superplasticizer plays no role in the synthetic formula).
    good_solution = json.dumps({
        "claim": "Strength is driven by water/binder ratio and log(age).",
        "predict_code": (
            "def features(row):\n"
            "    cem = row['cement'] + row['blast_furnace_slag'] + row['fly_ash']\n"
            "    return {\n"
            "        'water_binder_ratio': row['water'] / (cem + 1e-6),\n"
            "        'log_age': math.log(row['age'] + 1),\n"
            "    }"
        ),
    })
    bad_solution = json.dumps({
        "claim": "Strength is driven by superplasticizer dosage alone.",
        "predict_code": "def features(row):\n    return {'sp': row['superplasticizer']}",
    })
    s_good = compute_score("concrete_swarm", good_solution, "", extra_info)
    s_bad  = compute_score("concrete_swarm", bad_solution, "", extra_info)
    print(f"[8] Good transform (wc + log age) : {s_good:.4f}")
    print(f"    Bad transform (sp only)       : {s_bad:.4f}  (expect good notably > bad)")

    print()
    assert s1 > 0.0,   "valid hypothesis must score > 0"
    assert s2 == 0.0,  "malformed JSON must score 0.0"
    assert s3 == 0.0,  "missing predict_code must score 0.0"
    assert s4 == 0.0,  "crashing code must score 0.0"
    assert s5 > 0.0,   "fenced code must be stripped and scored"
    assert s6 > 0.0,   "bare 'import math' line must be stripped and scored"
    assert s7 == 0.0,  "constant feature must score 0.0, not the old 0.10+0.10 floor"
    assert s_good > s_bad + 0.1, "a transform close to the true structure must beat an unrelated one"
    print("All assertions passed.")
