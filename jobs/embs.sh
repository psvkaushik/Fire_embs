#!/bin/bash
#SBATCH --job-name=get_embs
#SBATCH --output=get_embs.out
#SBATCH --error=get_embs.err
#SBATCH --time=96:00:00
#SBATCH --partition=max
#SBATCH --nodelist=c33



# Load Python environment
cd /mnt/beegfs/spillal2/caburn/Emb_For_Fire/src

source /home/spillal2/micromamba/etc/profile.d/mamba.sh

# Activate the Conda environment
micromamba activate stac

time python precompute_embeddings.py --patches_dir patches/10m --models galileo croma smarties \
    --out_dir embeddings/10m 