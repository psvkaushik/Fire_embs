#!/bin/bash
#SBATCH --job-name=early_fusion_3
#SBATCH --output=early_fusion_3_lp.out
#SBATCH --error=early_fusion_3_lp.err
#SBATCH --time=96:00:00
#SBATCH --partition=a5000ada
#SBATCH --nodelist=c32


# Load Python environment
cd /mnt/beegfs/spillal2/caburn/Emb_For_Fire/src

source /home/spillal2/micromamba/etc/profile.d/mamba.sh

# Activate the Conda environment
micromamba activate stac


time python train_cached.py --models croma smarties galileo --embeddings_dir embeddings/10m  --patches_dir patches/10m  --batch_size 16 --eval_batch_size 16 --epochs 40 --save_dir embeddings_10m_comparison/early_fusion_3_lp --device cuda:1 --num_workers 8 --fusion early 
