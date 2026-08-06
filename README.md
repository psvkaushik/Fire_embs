# fusion_v2 setup checklist

## 1. Python environment

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install transformers datasets scikit-learn tqdm matplotlib numpy einops
pip install h5py hdf5plugin      # CaBuAr ships as HDF5+BZip2 via hdf5plugin -- required, not optional
```

One more thing, only for Galileo:

```bash
git clone https://github.com/nasaharvest/galileo.git    # code written, needs shape verification -- see section 7
```

GPU strongly recommended -- even frozen, a ViT-L encoder forward pass at 224x224, x2 (pre+post) x however
many branches you fuse, adds up fast. CPU will technically run but will be very slow for anything beyond a
single-model smoke test.

## 2. Repo layout fusion_v2 expects

```
your_repo/
├── use_croma.py          # from https://github.com/antofuller/CROMA (unchanged)
├── hf_prithvi.py          # your existing Prithvi wrapper
├── CROMA_large.pt          # checkpoint (see below)
├── galileo/                # git clone https://github.com/nasaharvest/galileo.git here
│   └── data/models/nano/   # downloaded checkpoint folder (config.json + encoder.pt)
└── fusion_v2/
    ├── configs.py
    ├── band_utils.py
    ├── resample.py
    ├── heads.py
    ├── encoders.py
    ├── fusion.py
    ├── models.py
    ├── loader.py
    ├── prepare_patches.py
    ├── run_10m_comparison.py
    ├── sanity_check.py
    ├── galileo_embed_size.py
    └── train.py
```

`encoders.py` does `sys.path.insert(repo_root)` then imports `use_croma`/`hf_prithvi` directly, so those two
must sit one directory above `fusion_v2/` -- matches your v1 layout unchanged. `galileo/` is new (see below;
override the checkpoint subfolder with `--galileo_ckpt_dir` if you put it somewhere else).

## 3. Checkpoints -- what's needed for each model

| Model | Status | What you need |
|---|---|---|
| **croma** | wired | `CROMA_large.pt` locally (default path: repo root, override with `--croma_ckpt`). Get it from the [official CROMA repo](https://github.com/antofuller/CROMA#pretrained-models). |
| **prithvi** | wired | Whatever local checkpoint `hf_prithvi.PrithviViT` already expects in your v1 setup -- unchanged. **Double-check the band order** (configs.py notes flag this -- v1 used B08, the new default here is B8A; confirm which one your checkpoint was actually trained on). |
| **smarties** | wired | Nothing to download manually -- `transformers.AutoModel.from_pretrained("gsumbul/SMARTIES-v1-ViT-L", trust_remote_code=True)` pulls it automatically on first run. Needs internet access and `trust_remote_code=True` (already set). |
| **galileo** | code written, unverified | `git clone https://github.com/nasaharvest/galileo.git` into your repo root (so `<repo_root>/galileo/single_file_galileo.py` exists), then download a checkpoint folder from [huggingface.co/nasaharvest/galileo/tree/main/models](https://huggingface.co/nasaharvest/galileo/tree/main/models) (start with `nano` -- smallest, fastest to sanity-check) to `galileo/data/models/nano/` (needs `config.json` + `encoder.pt`). Run `galileo_embed_size.py` (below) as your first check. |
| **satmae** | not wired | Standalone-only stub, not in a fusion group. Kept in `configs.py` for reference; no `encoders.py` implementation. |

Fusion groups: **10m** = croma + smarties + galileo. **30m** = prithvi only (nothing to fuse there yet --
add a second 30m-native model to `configs.py`/`encoders.py` when you have one).

## 4. Data

Nothing to download by hand -- `loader.py` calls `datasets.load_dataset("DarthReca/california_burned_areas", ...)`
on import, which pulls CaBuAr from the HF Hub automatically (a few GB, first run only, then cached
under `~/.cache/huggingface`). Just needs internet access and the `h5py`/`hdf5plugin` packages from step 1.

## 5. Run order

```bash
cd fusion_v2

# (optional but recommended) pre-tile once per GSD group you'll train on --
# speeds up every subsequent run, see prepare_patches.py's docstring for why
python prepare_patches.py --group 10m --out_dir patches/10m
python prepare_patches.py --group 30m --out_dir patches/30m

# single-model baselines -- croma/prithvi/smarties are fully wired; galileo's
# code is written but not yet shape-verified against a live install (see
# section 7) -- try it standalone first, before trusting it in a fused run
python train.py --models croma --head linear_probe --patches_dir patches/10m
python train.py --models smarties --head upernet --patches_dir patches/10m
python train.py --models galileo --head linear_probe --patches_dir patches/10m
python train.py --models prithvi --head upernet --patches_dir patches/30m

# fusion within the 10m group (any subset of croma/smarties/galileo)
python train.py --models croma smarties galileo --head upernet --fusion early --patches_dir patches/10m

# or use the comparison runner to generate + launch every individual + fusion
# config in the 10m group automatically, and aggregate results into a table:
python run_10m_comparison.py                    # prints the commands
python run_10m_comparison.py --execute           # or actually runs them
python run_10m_comparison.py --compare_only      # aggregate existing results_*.json
```

`--patches_dir` is optional (falls back to on-the-fly random cropping + per-image tiled eval if omitted) but
recommended once you're doing real runs, not just smoke tests.

## 6. Sanity-check the wiring BEFORE chasing checkpoints

`sanity_check.py` runs the entire pipeline -- configs, band selection, resampling, heads, all three fusion
strategies -- with dummy encoders standing in for the real models. No checkpoints, no dataset, no internet
needed. Run it first:

```bash
python sanity_check.py
```

If that passes, any problems you hit afterward are checkpoint/weights issues, not wiring issues -- useful
for isolating what broke.

## 7. Known gaps / things to double check before trusting real numbers

- **Galileo's output shape is traced from source, not run.** `encoders.GalileoEncoder` reads
  `Encoder.forward()`'s return through `apply_linear_projection` -> `apply_attn` -> `split_and_expand_hwtc`
  in their actual `single_file_galileo.py`, giving `[B, H', W', T, 7, D]` -- but this hasn't been executed
  against a live checkpoint. Before trusting it:
  1. Run `python galileo_embed_size.py` (no argument needed -- it defaults to `<repo_root>/galileo/data/models/nano`) to confirm the real embedding size.
  2. Run `python verify_galileo_shape.py` -- loads the real checkpoint, runs one forward pass on a small
     dummy tensor, and confirms whether the actual shape matches `[B, H', W', T, 7, D]`. If it doesn't,
     the mean-pool step in `GalileoEncoder.forward()` is the place to fix.
- **Galileo normalization**: `GalileoEncoder._renormalize()` undoes loader.py's shared CaBuAr-wide
  normalization and reapplies Galileo's own per-band stats (from their `config/normalization.json`,
  hardcoded into `configs.GALILEO_S2_MEAN/STD`), converting CaBuAr's ~0-1 fractional-reflectance scale to
  the raw-DN scale (`*10000`) their stats assume -- confirmed by Galileo's own mean values (~1395 for
  blue, i.e. 0.1395 as a reflectance fraction, an ordinary value). The `*10000` conversion is inferred
  from the standard Sentinel-2 L2A convention, not confirmed against CaBuAr's own documented
  preprocessing -- worth double-checking against a real image if results look off.
- **Prithvi band order** (`B8A` vs `B08`) -- verify against your actual checkpoint.
- `models.py` raises `NotImplementedError` with a pointer to the relevant TODO if you try to use `satmae`
  (no `encoders.py` implementation exists for it).