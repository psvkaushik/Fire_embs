#!/bin/bash
#SBATCH --job-name=smart_gali
#SBATCH --output=smart_gali.out
#SBATCH --error=smart_gali.err
#SBATCH --time=96:00:00
#SBATCH --partition=a6000
#SBATCH --nodelist=c31



# Load Python environment
cd /mnt/beegfs/spillal2/caburn/Emb_For_Fire/src

source /home/spillal2/micromamba/etc/profile.d/mamba.sh

# Activate the Conda environment
micromamba activate stac


time python train_cached.py --models croma smarties --embeddings_dir embeddings/10m  --patches_dir patches/10m  --batch_size 16 --eval_batch_size 16 --epochs 40 --save_dir embeddings_10m_comparison/gali_smart_early --device cuda:0 --num_workers 16 --fusion early 

time python train_cached.py --models croma smarties --embeddings_dir embeddings/10m  --patches_dir patches/10m  --batch_size 16 --eval_batch_size 16 --epochs 40 --save_dir embeddings_10m_comparison/gali_smart_late --device cuda:0 --num_workers 8 --fusion late 
