#!/bin/bash
#SBATCH --job-name=singles
#SBATCH --output=singles.out
#SBATCH --error=singles.err
#SBATCH --time=96:00:00
#SBATCH --partition=max
##SBATCH --nodelist=c58     # uncomment to pin a node

# Single-model linear probes: croma, smarties, copernicusfm (3 runs).
# Overrides: DEVICE=cuda:1 EPOCHS=20 BATCH=8 NUM_WORKERS=4 sbatch job_singles.sh

source /mnt/beegfs/spillal2/final_emb_code/jobs/combo_common.sh

precompute croma smarties copernicusfm
run croma croma
run smarties smarties
run copernicusfm copernicusfm
echo "=== done ($(date)) ==="
