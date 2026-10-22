#!/bin/bash
#SBATCH --job-name=triple
#SBATCH --output=triple.out
#SBATCH --error=triple.err
#SBATCH --time=96:00:00
#SBATCH --partition=a5000ada
##SBATCH --nodelist=c32     # uncomment to pin a node

# croma + smarties + copernicusfm x 3 fusion strategies (early, sep_dec, late) = 3 runs.
# Overrides: DEVICE=cuda:1 EPOCHS=20 BATCH=8 NUM_WORKERS=4 sbatch job_triple.sh

source /mnt/beegfs/spillal2/final_emb_code/jobs/combo_common.sh

precompute croma smarties copernicusfm
run_fusions croma smarties copernicusfm
echo "=== done ($(date)) ==="
