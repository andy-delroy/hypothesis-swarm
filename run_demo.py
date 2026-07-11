"""
run_demo.py — end-to-end harness for the hypothesis swarm.

  * No FIREWORKS_API_KEY set  -> runs a deterministic MOCK swarm (offline, free).
                                 Proves the orchestration + reward wiring works.
  * FIREWORKS_API_KEY set      -> runs the real four-agent swarm on Fireworks.

Swap `make_synthetic()` for your real dataset loader (materials / drug-interaction)
once the pipeline is green. Keep the held-out split strict: the agent sees only a
sample of train; test is never shown.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone

from reward import Dataset, score_hypothesis
from concrete_data import load_concrete
import agents

_mock_mode = False  # set to True by install_mock(); used when writing results


# ----------------------------------------------------------------------------
# Toy dataset with a KNOWN latent rule, so you can confirm "discovery" happens.
# Replace with your real loader. Latent truth: label = 1 iff a*b > 0.25.
# ----------------------------------------------------------------------------
def make_synthetic(n_train=300, n_test=300, seed=0) -> Dataset:
    rng = random.Random(seed)

    def rows(n):
        out = []
        for _ in range(n):
            a, b, c = rng.random(), rng.random(), rng.random()  # c = distractor
            y = 1 if a * b > 0.25 else 0
            if rng.random() < 0.05:
                y = 1 - y
            out.append({"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "label": y})
        return out

    return Dataset("classification", "label", train=rows(n_train), test=rows(n_test))


# ----------------------------------------------------------------------------
# Offline mock: canned replies keyed by which system prompt is being used.
# Lets you exercise the full propose->critique->refine->verify graph with no API.
# ----------------------------------------------------------------------------
def install_mock():
    global _mock_mode
    _mock_mode = True
    import json

    def fake_call_llm(model, system, user, temperature=0.7, **kwargs):
        if system == agents.PROPOSER_SYSTEM:
            return json.dumps({"hypotheses": [
                {
                    "claim": (
                        "Concrete strength decreases as the water-to-cement ratio rises "
                        "(Abrams' law): a lower w/c ratio means less pore space and higher "
                        "compressive strength."
                    ),
                    "predict_code": (
                        "def predict(row):\n"
                        "    wc = row['water'] / (row['cement'] + 1e-6)\n"
                        "    return max(5.0, 55.0 - 50.0 * wc)"
                    ),
                },
                {
                    "claim": (
                        "Strength grows with curing age because hydration continues over time; "
                        "the gain is approximately proportional to the logarithm of age in days."
                    ),
                    "predict_code": (
                        "def predict(row):\n"
                        "    return min(60.0, 10.0 * math.log(row['age'] + 1))"
                    ),
                },
                {
                    "claim": (
                        "Coarse aggregate content is the primary driver of strength: "
                        "higher aggregate volume increases the skeletal density of the mix."
                    ),
                    "predict_code": (
                        "def predict(row):\n"
                        "    return max(10.0, 70.0 - 0.03 * row['coarse_aggregate'])"
                    ),
                },
            ]})
        if system == agents.CRITIC_SYSTEM:
            return json.dumps({
                "critique": (
                    "The water/cement hypothesis captures the dominant Abrams'-law trend but "
                    "ignores curing age entirely — the same w/c mix at 3 days vs 90 days "
                    "can differ by 20+ MPa. It will systematically under-predict mature "
                    "concrete and over-predict very young concrete. The 55 MPa intercept "
                    "and slope were also eyeballed, not calibrated, so the absolute scale "
                    "may be off."
                ),
                "most_likely_failure": (
                    "large residuals on rows where age < 7 days or age > 90 days, "
                    "because the age effect dominates w/c at the extremes"
                ),
            })
        if system == agents.REFINER_SYSTEM:
            return json.dumps({
                "claim": (
                    "Concrete strength is jointly determined by the water/cement ratio and "
                    "curing age: low w/c gives higher potential strength, and the "
                    "log-of-age term captures ongoing hydration gains over time."
                ),
                "predict_code": (
                    "def predict(row):\n"
                    "    wc = row['water'] / (row['cement'] + 1e-6)\n"
                    "    age_factor = math.log(row['age'] + 1) / math.log(29)\n"
                    "    return max(5.0, (55.0 - 45.0 * wc) * age_factor)"
                ),
                "what_changed": (
                    "added a log(age+1) multiplier normalized to 28 days (standard curing), "
                    "so early-age specimens are scaled down and mature ones are scaled up"
                ),
            })
        if system == agents.VERIFIER_SYSTEM:
            return json.dumps({
                "faithful": True,
                "verdict": (
                    "The hypothesis predicts MPa strength from the water/cement ratio and "
                    "curing age via an Abrams-style formula with a logarithmic age term. "
                    "On held-out data the code faithfully implements this claim, and the "
                    "measured skill score reflects real predictive signal above the "
                    "mean-predictor baseline."
                ),
            })
        return "{}"

    agents.call_llm = fake_call_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="BEFORE",
                        help="Run label e.g. BEFORE or AFTER (overridden by RUN_LABEL env var)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of swarm runs for best-of-N selection (overridden by RUNS env var)")
    args = parser.parse_args()
    label  = os.environ.get("RUN_LABEL") or args.label
    n_runs = int(os.environ.get("RUNS") or args.runs)

    data = load_concrete()
    strengths = [r["strength"] for r in data.test]
    mean_s = sum(strengths) / len(strengths)
    min_s  = min(strengths)
    max_s  = max(strengths)
    print(f"Dataset: {len(data.train)} train / {len(data.test)} held-out | "
          f"held-out mean strength={mean_s:.1f} MPa | "
          f"min={min_s:.1f}  max={max_s:.1f} MPa\n"
          f"(Mean-predictor baseline always scores skill=0.0 by construction — "
          f"any positive skill here reflects real predictive signal.)\n")

    if not os.environ.get("FIREWORKS_API_KEY"):
        print(">> No FIREWORKS_API_KEY — running OFFLINE MOCK swarm.\n")
        install_mock()

    # Blocks that are constant across runs — build once, embed in every JSON.
    dataset_block = {
        "name": "concrete_compressive_strength",
        "train_size": len(data.train),
        "test_size":  len(data.test),
        "task_type":  data.task_type,
        "mean_strength": round(mean_s, 4),
        "min_strength":  round(min_s, 4),
        "max_strength":  round(max_s, 4),
    }
    models_block = {"mock": True} if _mock_mode else {
        "proposer": agents.MODEL_PROPOSER,
        "critic":   agents.MODEL_CRITIC,
        "refiner":  agents.MODEL_REFINER,
        "verifier": agents.MODEL_VERIFIER,
    }

    os.makedirs("results", exist_ok=True)
    run_records: list[dict] = []

    for i in range(1, n_runs + 1):
        print(f"\n--- run {i}/{n_runs} ---")
        run_ts = datetime.now(timezone.utc)
        w = agents.run_swarm(data, k=4, verbose=True)

        run_dict = {
            "label":     label,
            "timestamp": run_ts.isoformat(),
            "run_index": i,
            "dataset":   dataset_block,
            "models":    models_block,
            "winner":    w.to_dict(),
            "debate":    w.debate,
            "trajectory": [
                {"stage": "propose_best", "reward": round(w.debate["original_proposal"]["reward"], 4)},
                {"stage": "refine",       "reward": round(w.debate["refinement"]["reward"], 4)},
                {"stage": "winner",       "reward": round(w.reward.total, 4)},
            ],
        }
        ts_safe      = run_ts.strftime("%Y%m%dT%H%M%SZ")
        stamped_path = f"results/{label}_{ts_safe}_run{i}.json"
        with open(stamped_path, "w") as f:
            json.dump(run_dict, f, indent=2)

        print(f"[run {i}/{n_runs}] winner reward={w.reward.total:.4f}  saved → {stamped_path}")
        run_records.append({"winner": w, "reward": w.reward.total, "run_dict": run_dict})

    # ------------------------------------------------------------------
    # Best-of-N selection
    # ------------------------------------------------------------------
    all_rewards  = [round(r["reward"], 4) for r in run_records]
    chosen_idx   = max(range(n_runs), key=lambda i: run_records[i]["reward"])
    best_record  = run_records[chosen_idx]
    winner       = best_record["winner"]

    print(f"\n=== BEST-OF-{n_runs} SELECTION ===")
    print(f"Run {chosen_idx + 1}/{n_runs} selected  reward={best_record['reward']:.4f}")
    print(f"All run rewards: {all_rewards}")

    print("\n=== WINNING HYPOTHESIS ===")
    print("claim   :", winner.claim)
    print("reward  :", winner.reward.as_dict())
    print("verdict :", winner.verdict)
    print("\nThis reward.total is your BEFORE number. After GRPO-training the proposer")
    print("on the MI300X, rerun and show the same swarm producing a higher-reward,")
    print("more-predictive hypothesis. That before/after IS the demo.")

    # Write the aggregate latest_<label>.json — best run's data + selection provenance.
    final_results = {
        **best_record["run_dict"],
        "selection": {
            "method":          "best_of_n",
            "n_runs":          n_runs,
            "all_run_rewards": all_rewards,
            "chosen_run_index": chosen_idx,   # 0-based
        },
    }
    latest_path = f"results/latest_{label}.json"
    with open(latest_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved → {latest_path} (best of {n_runs} runs)")


if __name__ == "__main__":
    main()
