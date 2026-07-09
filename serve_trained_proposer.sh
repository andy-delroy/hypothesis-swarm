#!/usr/bin/env bash
# =============================================================================
# serve_trained_proposer.sh — merge a GRPO checkpoint and serve it via vLLM
#
# Usage:
#   bash serve_trained_proposer.sh 40      # use global_step_40
#   bash serve_trained_proposer.sh         # auto-detect the latest checkpoint
#
# What this does:
#   1. Resolve the target checkpoint directory (from $1 or latest found)
#   2. Merge the sharded FSDP + LoRA adapter state into a standard HF model dir
#      (verl.model_merger writes a directory vLLM can load directly)
#   3. Serve the merged model via vLLM on port 8001
#   4. Print the env vars to set in your run_demo.py shell so the Proposer role
#      routes to this local server while Critic/Refiner/Verifier stay on Fireworks
#
# Run this in a tmux/screen session — the vllm serve command is a long-running
# server process, not a one-shot command. Ctrl-C to stop.
# =============================================================================

set -euo pipefail

CKPT_BASE="checkpoints/concrete_swarm_proposer"
SERVED_MODEL_NAME="concrete-swarm-proposer"
SERVE_PORT=8001

# =============================================================================
# STEP 1 — Resolve checkpoint directory
# =============================================================================

if [[ -n "${1-}" ]]; then
    # Explicit step number given, e.g. `bash serve_trained_proposer.sh 40`
    STEP="${1}"
    ACTOR_DIR="${CKPT_BASE}/global_step_${STEP}/actor"
    if [[ ! -d "${ACTOR_DIR}" ]]; then
        echo "[serve] ERROR: actor directory not found: ${ACTOR_DIR}" >&2
        echo "  Available checkpoints:" >&2
        find "${CKPT_BASE}" -maxdepth 1 -type d -name 'global_step_*' \
            | sort -V | sed 's/^/    /' >&2
        exit 1
    fi
else
    # Auto-detect the latest checkpoint (same sort -V logic as checkpoint_status.sh)
    if [[ ! -d "${CKPT_BASE}" ]]; then
        echo "[serve] ERROR: checkpoint base directory not found: ${CKPT_BASE}" >&2
        echo "  Has training started yet? Run verl_train.sh first." >&2
        exit 1
    fi

    mapfile -t step_dirs < <(
        find "${CKPT_BASE}" -maxdepth 1 -type d -name 'global_step_*' | sort -V
    )

    if [[ ${#step_dirs[@]} -eq 0 ]]; then
        echo "[serve] ERROR: no global_step_* directories found under ${CKPT_BASE}." >&2
        echo "  Training may not have reached its first save interval yet." >&2
        exit 1
    fi

    latest_step_dir="${step_dirs[-1]}"
    STEP=$(basename "${latest_step_dir}" | sed 's/global_step_//')
    ACTOR_DIR="${latest_step_dir}/actor"

    if [[ ! -d "${ACTOR_DIR}" ]]; then
        echo "[serve] ERROR: latest checkpoint ${latest_step_dir} has no actor/ subdirectory." >&2
        echo "  The checkpoint may be incomplete (pod ended mid-save?)." >&2
        echo "  Try an earlier step: bash serve_trained_proposer.sh <step_number>" >&2
        exit 1
    fi

    echo "[serve] No step given — using latest: global_step_${STEP}"
fi

TARGET_DIR="merged_models/concrete_swarm_proposer_step${STEP}"

echo "[serve] Actor checkpoint : ${ACTOR_DIR}"
echo "[serve] Merge target     : ${TARGET_DIR}"
echo ""

# =============================================================================
# STEP 2 — Merge FSDP + LoRA shards into a standard HF model directory
# =============================================================================
# verl.model_merger combines the sharded FSDP state (split across GPU workers
# during training) and the LoRA adapter weights into a single merged HF model
# directory that vLLM can load as a plain --load-format safetensors checkpoint.
#
# NOTE: Verify the exact CLI against whatever verl version is on the pod:
#   python -m verl.model_merger --help
# The subcommand and flag names may differ across verl releases.

echo "[serve] Merging checkpoint..."

mkdir -p merged_models

if ! python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ACTOR_DIR}" \
        --target_dir "${TARGET_DIR}"; then
    echo "" >&2
    echo "[serve] ERROR: verl.model_merger exited with a non-zero status." >&2
    echo "  Check the traceback above, then run:" >&2
    echo "    python -m verl.model_merger --help" >&2
    echo "  to confirm the correct subcommand and flag names for your verl version." >&2
    echo "" >&2
    echo "  NOT launching vLLM — serving a broken or empty model dir would give" >&2
    echo "  silent garbage outputs, not an obvious error." >&2
    exit 1
fi

# Confirm the merge produced something before proceeding
if [[ ! -d "${TARGET_DIR}" ]] || [[ -z "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]]; then
    echo "[serve] ERROR: merge completed without error but ${TARGET_DIR} is empty or missing." >&2
    echo "  This is unexpected — check verl.model_merger output above." >&2
    exit 1
fi

n_merged=$(find "${TARGET_DIR}" -type f | wc -l)
echo "[serve] Merge complete: ${n_merged} files in ${TARGET_DIR}"
echo ""

# =============================================================================
# STEP 3 — Print env vars for the run_demo.py shell BEFORE blocking on vllm serve
# =============================================================================
# Print these now so they're visible in the terminal before vllm serve takes over
# stdout. You'll need to set them in whatever shell you run `python run_demo.py`
# from — NOT in this terminal (which will be occupied by the server).

echo "============================================================"
echo "  To route the Proposer role at this server, set these"
echo "  vars in your run_demo.py shell (separate terminal/pane):"
echo ""
echo "    export PROPOSER_BASE_URL=http://localhost:${SERVE_PORT}/v1"
echo "    export PROPOSER_API_KEY=local"
echo "    export MODEL_PROPOSER=${SERVED_MODEL_NAME}"
echo ""
echo "  Then run:  python run_demo.py --label AFTER"
echo ""
echo "  Critic / Refiner / Verifier are unaffected — they still"
echo "  route through Fireworks via FIREWORKS_API_KEY."
echo "============================================================"
echo ""

# =============================================================================
# STEP 4 — Serve via vLLM
# =============================================================================
# This is a long-running foreground process. Run this script inside tmux or
# screen so the server survives after you detach. Example:
#   tmux new -s proposer-server
#   bash serve_trained_proposer.sh 40
#   <Ctrl-B d>  # detach, server keeps running

# Reuse the same AITER flag as verl_train.sh for consistency.
# See verl_train.sh for full explanation; safe to remove if vLLM is unstable.
export VLLM_ROCM_USE_AITER=1

echo "[serve] Launching vLLM server on port ${SERVE_PORT}..."
echo "        Model  : ${TARGET_DIR}"
echo "        Name   : ${SERVED_MODEL_NAME}"
echo "        Press Ctrl-C to stop."
echo ""

exec vllm serve "${TARGET_DIR}" \
    --port "${SERVE_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}"
