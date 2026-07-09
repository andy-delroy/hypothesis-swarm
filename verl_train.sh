#!/usr/bin/env bash
# =============================================================================
# verl_train.sh — GRPO training: Qwen3-4B LoRA on concrete-hypothesis-swarm
#
# Wall-clock budget : ~4 hours per pod session
# Resume            : re-run this script unchanged — no flags, no edits needed
# Progress check    : bash checkpoint_status.sh
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PRE-FLIGHT CHECKLIST  (manual — complete before first launch)         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 1. CONFIRM MODEL PATH
#    Find the exact Qwen3-4B directory available on this pod, then fill in
#    QWEN3_MODEL_PATH below.
#
#       ls /workspace/models/ | grep -i qwen3
#    or, if HF Hub is reachable and a cached download is acceptable on first run:
#       python3 -c "from huggingface_hub import snapshot_download; \
#                   print(snapshot_download('Qwen/Qwen3-4B'))"
#
# 2. CONFIRM VERL ENTRY POINT
#    verl's Hydra-based trainer must be importable:
#       python3 -m verl.trainer.main_ppo --help
#    Should print Hydra config options. If it errors, check your verl install.
#
# 3. CONFIRM RAY VERSION — determines which device-isolation env var to use:
#       python3 -c "import ray; print(ray.__version__)"
#
#    >= 2.45.0  →  RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1   ← active below
#    <  2.45.0  →  comment out the HIP line; uncomment ROCR line instead
#    (See the ENVIRONMENT section below for the full explanation.)
#
# 4. CONFIRM REQUIRED FILES exist in the pod's working directory:
#       ls -lh data/verl_train.parquet data/verl_val.parquet verl_reward_adapter.py
#    If missing, regenerate from the project root:
#       python3 verl_data_prep.py    # requires data/concrete.csv cache
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# =============================================================================
# FILL IN AFTER PRE-FLIGHT CHECK 1
# =============================================================================
QWEN3_MODEL_PATH="<PLACEHOLDER_QWEN3_4B_PATH>"
# Example values:
#   QWEN3_MODEL_PATH="/workspace/models/Qwen3-4B"
#   QWEN3_MODEL_PATH="Qwen/Qwen3-4B"    # HF Hub id — downloads on first run

# Sanity-check: fail fast rather than silently start with a wrong path.
if [[ "${QWEN3_MODEL_PATH}" == "<PLACEHOLDER_QWEN3_4B_PATH>" ]]; then
    echo "[verl_train] ERROR: QWEN3_MODEL_PATH has not been filled in." >&2
    echo "  Edit this script and set QWEN3_MODEL_PATH (see pre-flight check 1)." >&2
    exit 1
fi

# =============================================================================
# LOGGING  (change here; don't edit the python command below)
# =============================================================================
# Console-only by default.
# To also log to Weights & Biases: change to "[console,wandb]" and
# export WANDB_API_KEY before running.
TRAINER_LOGGER="[console]"

# =============================================================================
# ENVIRONMENT — ROCm / Ray device-isolation
# =============================================================================
# Without this, Ray's resource manager overwrites HIP_VISIBLE_DEVICES after
# verl has already assigned GPU slices to workers, causing each worker to see
# ALL GPUs instead of only its allocated one. The result is silent resource
# contention and non-deterministic OOMs.
#
# This is a documented verl + vLLM + ROCm + Ray interaction — NOT a generic
# ROCm performance tuning knob. Set it unconditionally for any ROCm pod running
# verl with Ray >= 2.45.0.
# Ref: verl AMD GPU docs; https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# If ray < 2.45.0, the env var name changed — use this instead (pre-flight check 3):
# export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# Optional: enable AITer (AMD Inference Transformer) kernel fusions inside vLLM.
# Provides measurable throughput gains on MI300X with no correctness risk, but
# safe to set to 0 or remove entirely if you encounter vLLM stability issues.
export VLLM_ROCM_USE_AITER=1

# =============================================================================
# LAUNCH
# =============================================================================
python3 -m verl.trainer.main_ppo \
    \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=data/verl_train.parquet \
    data.val_files=data/verl_val.parquet \
    \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    \
    actor_rollout_ref.model.path="${QWEN3_MODEL_PATH}" \
    \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=6 \
    \
    custom_reward_function.path=verl_reward_adapter.py \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=50 \
    trainer.save_freq=10 \
    trainer.default_local_dir=checkpoints/concrete_swarm_proposer \
    trainer.resume_mode=auto \
    "trainer.logger=${TRAINER_LOGGER}"

# Notes on tuning knobs:
#
# data.train_batch_size / ppo_mini_batch_size = 16
#   Conservative starting point for a single GPU + 4B model. If you hit an OOM
#   on the first step, halve both values to 8. With rollout.n=6, the effective
#   rollout batch is train_batch_size × n = 96 sequences — already substantive.
#
# lora_rank=16, lora_alpha=16
#   Standard starting point. Rank 32 is a sensible next step if you see the
#   training reward plateau early and have headroom in VRAM.
#
# rollout.n=6
#   Number of candidate completions sampled per prompt for GRPO. More = lower
#   variance reward signal; fewer = faster wall-clock per step. 6 is a
#   reasonable middle ground; 4 is the minimum before GRPO degenerates.
#
# trainer.save_freq=10
#   Checkpoint every 10 steps. Erring on the frequent side given the hard 4-hour
#   wall-clock cap — the cost is extra I/O, not correctness.
#
# trainer.total_epochs=50
#   A safety cap, not the real stopping condition. The pod's 4-hour wall-clock
#   is what ends a session; resume_mode=auto restarts from the latest checkpoint
#   next session. Increase the epoch cap if a future session would outlast it.

# =============================================================================
# AFTER YOUR SESSION ENDS
# =============================================================================
# Resume:   just re-run this script — trainer.resume_mode=auto finds the latest
#           checkpoint under checkpoints/concrete_swarm_proposer/ automatically.
#           No flags, no edits, no manual step selection needed.
#
# Progress: bash checkpoint_status.sh
#           Lists all saved global_step_N dirs, prints the latest + its mtime.
#           Run this before ending a session to confirm the last save landed.
#
# Retrieve: to pull checkpoints off the pod before it terminates, grab:
#           checkpoints/concrete_swarm_proposer/global_step_<N>/actor/
