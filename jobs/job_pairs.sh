#!/bin/bash
#SBATCH --job-name=pairs
#SBATCH --output=pairs.out
#SBATCH --error=pairs.err
#SBATCH --time=96:00:00
#SBATCH --partition=max
##SBATCH --nodelist=c58     # uncomment to pin a node

# All 3 pairs x 3 fusion strategies (early, sep_dec, late) = 9 runs.
# Overrides: DEVICE=cuda:1 EPOCHS=20 BATCH=8 NUM_WORKERS=4 sbatch job_pairs.sh

source /mnt/beegfs/spillal2/final_emb_code/jobs/combo_common.sh

precompute croma smarties copernicusfm
run_fusions croma smarties
run_fusions croma copernicusfm
run_fusions smarties copernicusfm
echo "=== done ($(date)) ==="
