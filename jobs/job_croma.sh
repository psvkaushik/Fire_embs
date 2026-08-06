#!/bin/bash
#SBATCH --job-name=croma
#SBATCH --output=croma_lp.out
#SBATCH --error=croma_lp.err
#SBATCH --time=96:00:00
#SBATCH --partition=max
#SBATCH --nodelist=c61


# Load Python environment
cd /mnt/beegfs/spillal2/caburn/Emb_For_Fire/src

source /home/spillal2/micromamba/etc/profile.d/mamba.sh

# Activate the Conda environment
micromamba activate stac



time python train_cached.py --models croma --embeddings_dir embeddings/10m --patches_dir patches/10m --batch_size 16 --eval_batch_size 16 --epochs 40 --save_dir embeddings_10m_comparison/croma --device cuda:0 --num_workers 8