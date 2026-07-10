"""
verl_data_prep.py — generate verl-compatible training/validation parquet files.

Produces:
  data/verl_train.parquet   (300 examples, seed=0)
  data/verl_val.parquet     (40 examples,  seed=42)

Parquet schema per row (matches verl's RLHFDataset / GSM8k example format):
  data_source  : str
  prompt       : list[dict]   — [{"role": "system", "content": ...},
                                  {"role": "user",   "content": ...}]
                                verl passes this to tokenizer.apply_chat_template()
  ground_truth : str          — "" (rule-based reward needs no gold answer)
  extra_info   : str          — JSON: {"reward_eval_rows": [...], "train_y": [...]}

Does NOT import agents.py (which lazily imports openai/Fireworks) or touch
data.test (held-out; reserved for final evaluation only).
"""

from __future__ import annotations

import json
import os
import random

# ---------------------------------------------------------------------------
# Prompt templates
# (Adapted from agents.PROPOSER_SYSTEM/PROPOSER_USER. Kept standalone here so
# this script runs offline with zero dependency on agents.py / the Fireworks
# client. The key difference from the frozen-swarm prompts: we request exactly
# ONE hypothesis, not k, since each GRPO rollout is a single generation.)
# ---------------------------------------------------------------------------

PROPOSER_TRAIN_SYSTEM = """\
You are a scientist proposing testable, general hypotheses about a dataset.

A hypothesis is a claim about the UNDERLYING STRUCTURE relating the features to \
the target — not a description of these particular rows. It must generalize to \
data you have never seen.

Output BOTH:
  1. "claim": one plain-English sentence a domain expert could read and judge.
  2. "predict_code": Python defining `predict(row: dict) -> float` that \
operationalizes the claim. It receives a dict of feature values (NO target key) \
and returns the predicted target as a float. You may use `math` and `statistics`. \
No imports, no I/O, no data lookups.

Hard rules:
  - Do NOT return a constant. A hypothesis that ignores the features scores zero.
  - Do NOT hard-code these specific rows; encode a generalizable RULE.
  - The code must faithfully implement the stated claim.

Return ONLY valid JSON (no markdown, no fences):
{"claim": "...", "predict_code": "..."}"""

PROPOSER_TRAIN_USER = """{schema}

Propose ONE hypothesis about what predicts '{target}'.\
 Favor simple, mechanistic rules (e.g. water/cement ratio, curing age) over \
complex ones."""


# ---------------------------------------------------------------------------
# Schema formatter (standalone; mirrors agents.schema_str without importing it)
# ---------------------------------------------------------------------------

def _schema_str(
    task_type: str,
    target: str,
    feature_names: list[str],
    sample_rows: list[dict],
) -> str:
    lines = [
        f"Task type: {task_type}",
        f"Target column to predict: '{target}'",
        f"Feature columns: {feature_names}",
        f"Sample of {len(sample_rows)} training rows (features + target):",
    ]
    for r in sample_rows:
        lines.append("  " + json.dumps(r))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_examples(
    data,                       # reward.Dataset
    n_examples: int,
    prompt_sample_size: int = 12,
    reward_eval_size: int = 180,
    seed: int = 0,
) -> list[dict]:
    """
    Build n_examples training dicts for verl.

    Per example:
      * Sample prompt_sample_size rows from data.train for the user prompt
        (WITH the target column, so the model can see the output scale).
      * Sample reward_eval_size rows from data.train (independently reshuffled)
        for the reward-eval slice passed to verl_reward_adapter.compute_score.
        Overlap with the prompt sample is fine — both come from the train pool.
      * train_y is the FULL training-set target-value list (not resampled per
        example), giving _regression_skill a stable mean-baseline denominator
        across all rollouts within a training run.
    """
    rng        = random.Random(seed)
    train_rows = data.train
    target     = data.target
    train_y    = [r[target] for r in train_rows]   # stable; computed once

    feature_names = [k for k in train_rows[0].keys() if k != target]

    examples: list[dict] = []
    for _ in range(n_examples):
        prompt_rows = rng.sample(train_rows, min(prompt_sample_size, len(train_rows)))
        eval_rows   = rng.sample(train_rows, min(reward_eval_size,   len(train_rows)))

        schema      = _schema_str(data.task_type, target, feature_names, prompt_rows)
        user_text   = PROPOSER_TRAIN_USER.format(schema=schema, target=target)

        examples.append({
            "data_source":  "concrete_swarm",
            "prompt": [
                {"role": "system", "content": PROPOSER_TRAIN_SYSTEM},
                {"role": "user",   "content": user_text},
            ],
            "ground_truth": "",
            "extra_info": {
                "reward_eval_rows": eval_rows,
                "train_y":          train_y,
            },
        })

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    from concrete_data import load_concrete

    print("Loading Concrete Compressive Strength dataset…")
    data = load_concrete()
    print(f"  train={len(data.train)}  test={len(data.test)}  "
          f"(test set never touched in this script)")

    print("\nGenerating examples…")
    train_examples = generate_examples(data, n_examples=300, seed=0)
    val_examples   = generate_examples(data, n_examples=40,  seed=42)
    print(f"  train examples : {len(train_examples)}")
    print(f"  val   examples : {len(val_examples)}")

    os.makedirs("data", exist_ok=True)
    train_path = "data/verl_train.parquet"
    val_path   = "data/verl_val.parquet"

    pd.DataFrame(train_examples).to_parquet(train_path, index=False)
    pd.DataFrame(val_examples).to_parquet(val_path,   index=False)
    print(f"\n  → {train_path}")
    print(f"  → {val_path}")

    # ── Sanity-check: round-trip one row ──────────────────────────────────────
    # pyarrow deserialises list<struct<...>> as numpy object arrays, not Python
    # lists, so we convert before asserting. verl's apply_chat_template accepts
    # both forms when iterating over the messages.
    check = pd.read_parquet(train_path)
    assert len(check) == len(train_examples), "row count mismatch after round-trip"
    first    = check.iloc[0]
    prompt_rt = list(first["prompt"])
    assert first["data_source"] == "concrete_swarm"
    assert len(prompt_rt) == 2
    assert prompt_rt[0]["role"] == "system"
    assert prompt_rt[1]["role"] == "user"
    ei_check = first["extra_info"]   # now a native dict from parquet nested struct
    assert len(ei_check["reward_eval_rows"]) == 180
    assert len(ei_check["train_y"]) == len(data.train)
    print("\n  Parquet round-trip checks passed.")

    # ── Print one full example for visual inspection ──────────────────────────
    ex = train_examples[0]
    ei = ex["extra_info"]   # plain dict — no json.loads needed

    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLE  (train[0])")
    print("=" * 70)
    print("\n── System prompt ──")
    print(ex["prompt"][0]["content"])
    print("\n── User prompt ──")
    print(ex["prompt"][1]["content"])
    print("\n── extra_info summary ──")
    summary = {
        "reward_eval_rows_total": len(ei["reward_eval_rows"]),
        "reward_eval_rows[0]":    ei["reward_eval_rows"][0],
        "reward_eval_rows[1]":    ei["reward_eval_rows"][1],
        "train_y_total":          len(ei["train_y"]),
        "train_y[:5]":            ei["train_y"][:5],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
