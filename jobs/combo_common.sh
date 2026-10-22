# Shared helpers, sourced by job_singles.sh / job_pairs.sh / job_triple.sh (not submitted itself).
# sbatch copies job scripts to a spool dir, so this is sourced by absolute path.

REPO=/mnt/beegfs/spillal2/final_emb_code
DEVICE=${DEVICE:-cuda:0}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-16}
OUT=embeddings_10m_comparison/all_combos
FUSIONS=(early sep_dec late)

cd "$REPO/src"
source "$REPO/.venv/bin/activate"

# embeddings for the given models (no-op for splits that are already cached)
precompute() {
  python precompute_embeddings.py --models "$@" \
      --patches_dir patches/10m --out_dir embeddings/10m --batch_size 8 --device "$DEVICE" \
      || { echo "precompute failed -- aborting"; exit 1; }
}

# run <name> <models...> [--fusion X]; skips finished runs, logs failures and carries on
run() {
  local name=$1; shift
  if ls "$OUT/$name"/results*.json >/dev/null 2>&1; then
    echo "=== skip $name (results exist) ==="; return
  fi
  echo "=== $name  ($(date)) ==="
  time python train_cached.py --models "$@" \
      --embeddings_dir embeddings/10m --patches_dir patches/10m \
      --batch_size "$BATCH" --eval_batch_size "$BATCH" --epochs "$EPOCHS" \
      --save_dir "$OUT/$name" --device "$DEVICE" --num_workers "$NUM_WORKERS" \
    || echo "!!! $name FAILED"
}

# run_fusions <models...>: one run per fusion strategy
run_fusions() {
  local tag; tag=$(IFS=+; echo "$*")
  for f in "${FUSIONS[@]}"; do
    run "${tag}_${f}" "$@" --fusion "$f"
  done
}
