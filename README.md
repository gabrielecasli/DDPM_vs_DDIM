# ddim-from-ddpm

Reproduction of **DDPM** (Ho et al. 2020, NeurIPS 2020) and **DDIM** (Song et al. 2020, ICLR 2021)
from pretrained DDPM weights. 

---

## Setup

We recommend using **`uv`** for a faster setup.

**Linux / macOS**

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

```

**Windows**

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .

```

Tested with Python 3.10+, PyTorch 2.1+, CUDA 13.0.

---

## Reproducing results

Each notebook can be executed end-to-end:

```bash
# Notebook 1: sanity checks, noise schedule, DDIM vs DDPM at fixed seed
jupyter nbconvert --to notebook --execute notebooks/01_foundations_and_sanity.ipynb

# Notebook 2: η × NFE sweep, FID table, Pareto plots (LONG TO RUN!! check parameters in configs)
jupyter nbconvert --to notebook --execute notebooks/02_eta_nfe_sweep.ipynb

# Notebook 3: DDIM inversion, PSNR curve, SLERP/LERP interpolation
jupyter nbconvert --to notebook --execute notebooks/03_inversion_and_interpolation.ipynb
```

---

## Obtained FID on CIFAR-10

The following results were obtained using **3072 samples** (configurable via `configs/cifar10.yaml: sampling.num_samples_fid`). 

*Note on FID deviation:* Because we evaluate on ~3.1K samples instead of the 50K used in the reference paper (Song et al. 2020), our absolute FID scores are systematically higher due to sample-size bias. However, statistical evaluation (Spearman rank correlation) confirms that the performance dynamics across NFE and η closely match the original paper.

**Our Obtained Results (3.1K samples):**

| NFE | η=0.0 (DDIM) | η=0.2 | η=0.5 | η=1.0 (DDPM) |
|----:|:---:|:---:|:---:|:---:|
|  10 | 22.00 | 22.54 | 24.44 | 49.35 |
|  20 | 15.21 | 15.55 | 15.96 | 25.92 |
|  50 | 12.97 | 12.97 | 13.38 | 16.13 |
| 100 | 12.52 | 12.42 | 12.62 | 13.55 |

**Reference (Song et al. 2020, Table 1, 50K samples):**
* NFE=10:  η=0.0 → 13.36  |  η=1.0 → 41.07
* NFE=20:  η=0.0 → 6.84   |  η=1.0 → 18.36
* NFE=50:  η=0.0 → 4.67   |  η=1.0 → 8.01
* NFE=100: η=0.0 → 4.16   |  η=1.0 → 5.78
---

## Notes on FID computation

We use **`clean-fid`** with `mode="legacy_pytorch"`, not `mode="clean"`.

**Why `legacy_pytorch`?** The DDIM and DDPM papers compute FID using a PyTorch
implementation with bilinear resizing. `clean-fid` in `legacy_pytorch` mode
reproduces this exact resize kernel, making our numbers directly comparable to
Song et al. 2020 Table 1.

---

## Checkpoint provenance

| Dataset      | Source                               | Image size | EMA |
|--------------|--------------------------------------|------------|-----|
| CIFAR-10     | VainF/Diff-Pruning release v0.0.1    | 32×32      | ✓   |
| LSUN Church  | `google/ddpm-ema-church-256` (HF)    | 256×256    | ✓   |
| LSUN Bedroom | `google/ddpm-ema-bedroom-256` (HF)   | 256×256    | ✓   |

All models use a linear β schedule (β_start=1e-4, β_end=0.02, T=1000) and
predict ε (noise).

**CIFAR-10**: The original Ho et al. 2020 EMA checkpoint converted to diffusers
format by [VainF](https://github.com/VainF/Diff-Pruning/issues/3). It is
**downloaded automatically** on first use.

<!-- > **Why not `google/ddpm-cifar10`?**  
> That checkpoint exists and loads correctly, but it is the **non-EMA** model.
> For numbers comparable to Song et al. 2020 Table 1 (which uses EMA weights),
> the VainF conversion is the right choice. -->

### Checkpoint format and the `subfolder` parameter

diffusers stores models in two layouts, which affect how they are loaded:

**Standalone model format** — used by the Google checkpoints
<!-- ```
<repo_root>/config.json
<repo_root>/diffusion_pytorch_model.safetensors
```
Loaded with `UNet2DModel.from_pretrained(hf_id)` — **no subfolder**.  
Passing `subfolder="unet"` here raises `OSError` because there is no `unet/`
directory containing a `config.json`. -->

**Pipeline format** — used by the VainF CIFAR-10 checkpoint
<!-- ```
<repo_root>/model_index.json       ← DDPMPipeline manifest
<repo_root>/unet/config.json
<repo_root>/unet/diffusion_pytorch_model.bin
<repo_root>/scheduler/scheduler_config.json
```
Loaded with `UNet2DModel.from_pretrained(path, subfolder="unet")` — the
subfolder is required to reach `unet/config.json`. -->

`src/models.py` handles both cases automatically; callers never need to specify
a subfolder.

---

## References

1. **Ho et al. 2020** — "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
   https://arxiv.org/abs/2006.11239

2. **Song et al. 2020** — "Denoising Diffusion Implicit Models", ICLR 2021.
   https://arxiv.org/abs/2010.02502

3. **Song et al. 2021** — "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021.
   https://arxiv.org/abs/2011.13456

5. **Parmar et al. 2022** — "On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation", CVPR 2022.
   https://arxiv.org/abs/2104.11222
