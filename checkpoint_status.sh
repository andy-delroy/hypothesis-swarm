#!/usr/bin/env bash
# =============================================================================
# checkpoint_status.sh — sanity-check GRPO training progress
#
# Lists all global_step_N directories under the checkpoint path, sorted
# numerically by step. Prints the latest checkpoint and its modification time
# so you can confirm a save landed before ending a pod session.
#
# Usage:  bash checkpoint_status.sh
# =============================================================================

CKPT_DIR="checkpoints/concrete_swarm_proposer"

# ── Directory existence check ─────────────────────────────────────────────────

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[checkpoint_status] No checkpoint directory found at: ${CKPT_DIR}"
    echo "  Training hasn't started yet, or CKPT_DIR in verl_train.sh was changed."
    exit 0
fi

# ── Collect step directories ───────────────────────────────────────────────────
# sort -V (version sort) handles the numeric suffix correctly:
#   global_step_9 < global_step_10 < global_step_100
# Plain lexicographic sort would put global_step_100 before global_step_9.

mapfile -t step_dirs < <(
    find "${CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' \
        | sort -V
)

if [[ ${#step_dirs[@]} -eq 0 ]]; then
    echo "[checkpoint_status] No global_step_* directories found under ${CKPT_DIR}."
    echo "  Training may be running but hasn't completed its first save interval yet"
    echo "  (trainer.save_freq=10 means the first checkpoint appears after step 10)."
    exit 0
fi

# ── Print all steps ───────────────────────────────────────────────────────────

echo "Checkpoints in ${CKPT_DIR}/  (${#step_dirs[@]} total, sorted by step)"
echo "──────────────────────────────────────────────────────────────────────"

for dir in "${step_dirs[@]}"; do
    # stat -c: GNU/Linux;  stat --format: alternate GNU spelling.
    # The 2>/dev/null fallback keeps the script non-fatal on unusual filesystems.
    mtime=$(stat -c "%y" "${dir}" 2>/dev/null \
            || stat --format="%y" "${dir}" 2>/dev/null \
            || echo "(mtime unavailable)")
    printf "  %-30s  %s\n" "$(basename "${dir}")" "${mtime}"
done

# ── Highlight the latest ──────────────────────────────────────────────────────

latest="${step_dirs[-1]}"
latest_mtime=$(stat -c "%y" "${latest}" 2>/dev/null \
               || stat --format="%y" "${latest}" 2>/dev/null \
               || echo "(mtime unavailable)")

echo "──────────────────────────────────────────────────────────────────────"
echo "Latest checkpoint : $(basename "${latest}")"
echo "  Full path : ${latest}"
echo "  Modified  : ${latest_mtime}"

# ── Actor weights quick-check ─────────────────────────────────────────────────
# verl saves actor weights under global_step_N/actor/. Confirm they're present
# so you know the checkpoint is actually usable and not a partial write.

actor_dir="${latest}/actor"
if [[ -d "${actor_dir}" ]]; then
    n_files=$(find "${actor_dir}" -type f | wc -l)
    du_out=$(du -sh "${actor_dir}" 2>/dev/null || echo "?")
    echo "  Actor dir : ${actor_dir}  (${n_files} files, ${du_out})"
else
    echo "  Actor dir : NOT FOUND at ${actor_dir} — checkpoint may be incomplete."
fi
