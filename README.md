# final_emb_code

Frozen EO foundation-model embeddings + heads/fusion for CaBuAr burned-area segmentation.
This repo covers exactly **CROMA, SMARTIES and Copernicus-FM** (10m group) -- singles, all pairs and the triple fusion,
each with the `early`, `sep_dec` and `late` fusion strategies.

## 1. Python environment (uv)

Dependencies live in `pyproject.toml` (torch comes from the CUDA 12.1 wheel index; change the
`pytorch-cu121` index URL there for another CUDA version).

```bash
cd /mnt/beegfs/spillal2/final_emb_code
uv sync                      # creates .venv and installs everything from uv.lock
source .venv/bin/activate    # or prefix commands with `uv run`
```

Notable pins and why:

- `datasets<4` -- `src/loader.py` calls `load_dataset(..., trust_remote_code=True)`; script-based datasets were removed in datasets 4.0.
- `numpy<2`, `torch<2.6`, `transformers<5` -- conservative; loosen if needed.
- `h5py` + `hdf5plugin` -- CaBuAr ships as HDF5 + BZip2; required.

A GPU is strongly recommended; ViT-L forward passes over pre+post images are slow on CPU.

The SLURM job scripts (`jobs/*.sh`) activate `.venv` themselves via `jobs/combo_common.sh`.

## 2. Expected layout

```
final_emb_code/
├── use_croma.py                  # CROMA model code (in repo)
├── CROMA_large.pt                # CROMA checkpoint (download, see below)
├── CopernicusFM_repo/            # Copernicus-FM code clone (see below)
├── CopernicusFM_ViT_base_varlang_e100.pth
├── pyproject.toml / uv.lock
├── jobs/
│   ├── combo_common.sh           # shared settings + precompute()/run()/run_fusions() helpers
│   ├── job_singles.sh            # 3 runs:  croma, smarties, copernicusfm
│   ├── job_pairs.sh              # 9 runs:  3 pairs x 3 fusions
│   └── job_triple.sh             # 3 runs:  croma+smarties+copernicusfm x 3 fusions
└── src/                          # run everything from here
    ├── configs.py  band_utils.py  resample.py                     # specs, band/resize utils
    ├── loader.py  prepare_patches.py                              # CaBuAr tiling
    ├── encoders.py  precompute_embeddings.py                      # frozen encoders -> cached embeddings
    └── cached_loader.py  cached_models.py  fusion.py  heads.py  train_cached.py  train_utils.py   # heads/fusion training
```

`src/encoders.py` resolves `use_croma` relative to the repo root (one level above `src/`).

## 3. Manual setup steps

| Model | What to do |
|---|---|
| **croma** | Download `CROMA_large.pt` from the [CROMA repo](https://github.com/antofuller/CROMA#pretrained-models) into the repo root (default path; override with `--croma_ckpt`). |
| **smarties** | Nothing manual. `transformers.AutoModel.from_pretrained("gsumbul/SMARTIES-v1-ViT-L", trust_remote_code=True)` downloads on first use, so it needs internet access or a pre-populated `~/.cache/huggingface` (run once on a login node). |
| **copernicusfm** | (1) `git clone https://github.com/zhu-xlab/Copernicus-FM.git CopernicusFM_repo` in the repo root (code is loaded from `CopernicusFM_repo/Copernicus-FM/src`). (2) `wget https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/CopernicusFM_ViT_base_varlang_e100.pth` into the repo root (default; override with `--copernicusfm_ckpt`). ViT-B/16, takes all 12 S2 bands via per-band wavelength/bandwidth, 14x14 grid at 224px. |

**Input normalization.** Every model receives the same input: CaBuAr's per-band z-score (`CABUAR_S2_MEAN/STD` in `src/configs.py`,
applied in `src/loader.py`). No model-specific normalization is applied. Copernicus-FM is additionally given each band's central
wavelength and bandwidth (nm), and its token output gets a per-token LayerNorm because the checkpoint ships no final norm.

Dataset: nothing to download by hand. `src/loader.py` pulls `DarthReca/california_burned_areas` from the
HF Hub on first run (a few GB, cached under `~/.cache/huggingface`). Needs internet access.

## 4. Run order

Run from `src/`. The cached pipeline (as used by `jobs/job_*.sh`) computes frozen embeddings once, then trains heads on them.

```bash
cd src

# 1. tile CaBuAr once for the 10m group
python prepare_patches.py --group 10m --out_dir patches/10m

# 2. precompute frozen embeddings (once per model)
python precompute_embeddings.py --models croma smarties copernicusfm \
    --croma_ckpt ../CROMA_large.pt \
    --patches_dir patches/10m --out_dir embeddings/10m --batch_size 8 --device cuda:0

# 3. train heads on cached embeddings -- single model
python train_cached.py --models croma --embeddings_dir embeddings/10m --patches_dir patches/10m \
    --save_dir embeddings_10m_comparison/croma --device cuda:0

# ...or fusion over any subset of the three
python train_cached.py --models croma smarties copernicusfm --fusion early \
    --embeddings_dir embeddings/10m --patches_dir patches/10m \
    --save_dir embeddings_10m_comparison/fusion_early --device cuda:0
```

`--fusion` choices are defined in `src/fusion.py` (`early`, `sep_dec`, `late`); see `train_cached.py --help`.

### Training protocol (as in the `jobs/*.out` logs)

Heads are trained on the cached embeddings with: lr 1e-4, weight decay 1e-4, 40 epochs max, early stopping with patience 8
on validation F1 at the best threshold, and the best-validation checkpoint reloaded for testing at the validation-tuned
threshold. No augmentation (embeddings are precomputed once per tile). Fusion heads include the projection/decoder layers from
`src/cached_models.py` (`--proj_dim 256`, `--dec_dim 128`).

### SLURM jobs

The three jobs reproduce the results in `jobs/{singles,pairs,triple}.out`. Each precomputes embeddings (skipped when
already cached), then trains with 40 epochs, batch 16, writing to `src/embeddings_10m_comparison/all_combos/<name>/`;
finished runs (existing `results*.json`) are skipped, so a job can be resubmitted safely.

```bash
cd jobs
sbatch job_singles.sh   # croma, smarties, copernicusfm
sbatch job_pairs.sh     # croma+smarties, croma+copernicusfm, smarties+copernicusfm  x {early,sep_dec,late}
sbatch job_triple.sh    # croma+smarties+copernicusfm  x {early,sep_dec,late}

# overrides
DEVICE=cuda:1 EPOCHS=20 BATCH=8 NUM_WORKERS=4 sbatch job_pairs.sh
```

Note: `job_*.sh` assume `patches/10m` already exists under `src/` (step 1 above).
