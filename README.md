# Hypothesis Swarm

A four-agent scientific hypothesis swarm whose reward comes from **executing
hypotheses against held-out data**, not from an LLM judge. The core thesis:
*graded by reality, not by vibes.*

Built around the [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
dataset (1 030 rows, 8 features → MPa strength). The swarm proposes
physics-inspired `predict(row)` functions, critiques them, refines them, and
scores them against a held-out test set. The score is normalised improvement
over the mean-predictor baseline, so a constant predictor can never cheat its
way to a high reward.

After getting a **BEFORE** baseline with frozen Fireworks models, the Proposer
role is GRPO-trained on an AMD MI300X pod using verl. Re-running the swarm with
the trained Proposer gives the **AFTER** number — that delta, plus the
plain-English discovered hypothesis, is the demo.

---

## Repository layout

| File | Purpose |
|---|---|
| `reward.py` | Authoritative grader. Compiles model-generated `predict(row)` in a restricted namespace, runs it on held-out rows, returns a `RewardBreakdown` with skill / coverage / format components. |
| `agents.py` | Four roles (Proposer → Critic → Refiner → Verifier) and the `run_swarm()` loop. Per-role endpoint overrides let the Proposer point at a local vLLM server post-training while the other three stay on Fireworks. |
| `concrete_data.py` | Downloads and caches the UCI dataset, returns a `reward.Dataset` with an 80/20 train/test split. |
| `run_demo.py` | End-to-end harness. Runs offline (mock LLM) without an API key; runs the real four-agent swarm when `FIREWORKS_API_KEY` is set. Saves per-run JSON and picks the best-of-N result. |
| `dashboard_gen.py` | Reads `results/latest_BEFORE.json` (and optionally `latest_AFTER.json`) and writes a self-contained offline HTML dashboard to `results/dashboard.html`. |
| `verl_reward_adapter.py` | verl-compatible `compute_score()` entry point for GRPO training. Self-contained — no Fireworks dependency. |
| `verl_data_prep.py` | Generates `data/verl_train.parquet` (300 examples) and `data/verl_val.parquet` (40 examples) for verl. |
| `verl_train.sh` | GRPO training launch script (Qwen3-4B + LoRA, 4-hour session budget, auto-resume). |
| `checkpoint_status.sh` | Lists saved `global_step_N` checkpoints and prints the latest + its mtime. |
| `serve_trained_proposer.sh` | Merges a LoRA checkpoint via `verl.model_merger` and serves it with vLLM on port 8001. |
| `test_reward.py` | Unit checks that the reward rewards the true rule, gives partial credit to near-misses, and zeroes constant predictors. |

---

## Setup

```bash
git clone git@github.com:andy-delroy/hypothesis-swarm.git
cd hypothesis-swarm
python3 -m venv venv && source venv/bin/activate
pip install requests "xlrd<2" openai pandas pyarrow
```

The UCI dataset is downloaded automatically on first run and cached at
`data/concrete.csv` (git-ignored).

---

## Run offline — no API key, no cost

```bash
python test_reward.py     # confirm the reward function behaves correctly
python run_demo.py        # mock swarm, 3 runs, prints BEFORE reward
```

The mock swarm returns canned concrete hypotheses and never calls any API.
Use it to verify the full pipeline — data loading, scoring, JSON persistence,
best-of-N selection — without spending any credits.

---

## Run live — Fireworks API

```bash
export FIREWORKS_API_KEY=<your-key>
python run_demo.py --label BEFORE --runs 3
```

This runs three full propose→critique→refine→verify cycles, scores each winner
against the held-out test set, picks the best, and writes:

- `results/latest_BEFORE.json` — the canonical BEFORE result
- `results/BEFORE_<timestamp>_run{N}.json` — individual run records

Model defaults (all overridable via env vars — see `agents.py`):

| Role | Default model |
|---|---|
| Proposer | `accounts/fireworks/models/deepseek-v4-pro` |
| Critic | `accounts/fireworks/models/deepseek-v4-pro` |
| Refiner | `accounts/fireworks/models/deepseek-v4-pro` |
| Verifier | `accounts/fireworks/models/kimi-k2p6` |

---

## Generate the dashboard

```bash
python dashboard_gen.py
open results/dashboard.html     # or xdg-open on Linux
```

After GRPO training, run `python run_demo.py --label AFTER` and then
`python dashboard_gen.py` again to populate the AFTER card. The dashboard
is a single self-contained HTML file (no CDN, works offline).

---

## GRPO training on a MI300X pod

### 1 — Pre-flight (on the pod)

```bash
# Confirm verl is installed
python3 -m verl.trainer.main_ppo --help

# Confirm Ray version (determines which env var to use — see verl_train.sh)
python3 -c "import ray; print(ray.__version__)"

# Generate training data (runs locally, no GPU needed)
python3 verl_data_prep.py
# → data/verl_train.parquet  (300 examples)
# → data/verl_val.parquet    (40 examples)
```

### 2 — Fill in the model path

Edit `verl_train.sh` and set `QWEN3_MODEL_PATH` to the Qwen3-4B directory on
the pod (see the pre-flight checklist comments inside the script).

### 3 — Train

```bash
bash verl_train.sh
```

Checkpoints are saved every 10 steps to `checkpoints/concrete_swarm_proposer/`.
To resume after a session ends, just re-run `bash verl_train.sh` — 
`trainer.resume_mode=auto` picks up from the latest checkpoint automatically.

To check progress mid-session:

```bash
bash checkpoint_status.sh
```

### 4 — Serve the trained Proposer

```bash
# In a tmux session on the pod:
bash serve_trained_proposer.sh 40      # use global_step_40
# or: bash serve_trained_proposer.sh   # auto-detects the latest checkpoint
```

The script merges the LoRA adapter into a standard HF model directory and
starts a vLLM server on port 8001. It prints the exact env vars to set before
you proceed.

### 5 — Run the AFTER swarm

In a separate shell (or locally if the pod port is forwarded):

```bash
export PROPOSER_BASE_URL=http://localhost:8001/v1
export PROPOSER_API_KEY=local
export MODEL_PROPOSER=concrete-swarm-proposer
export FIREWORKS_API_KEY=<your-key>   # still needed for Critic/Refiner/Verifier

python run_demo.py --label AFTER --runs 3
python dashboard_gen.py
```

Critic, Refiner, and Verifier continue routing through Fireworks — only the
Proposer switches to the local checkpoint. The before/after reward delta on
the same held-out test set is the result.

---

## Security notes

- `reward.py` executes model-generated code in a restricted namespace
  (no `__import__`, no `open`, no builtins beyond a curated safe list) with a
  wall-clock timeout. Appropriate for an isolated pod or dev machine; add
  container/gVisor/firejail isolation before any public-facing deployment.
- `FIREWORKS_API_KEY` is read from the environment and never written to any
  file. `data/`, `checkpoints/`, `merged_models/`, and `results/*.json` are
  all git-ignored.
