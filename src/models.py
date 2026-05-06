"""
Pretrained UNet loaders and a unified ε_θ(x_t, t) wrapper.

All models expose the same eps_fn interface; the rest of the codebase never
needs to know the underlying format or where the weights came from.

Supported model IDs / shorthands
---------------------------------
"cifar10"  → official DDPM EMA weights (Song et al. 2020 Table 1 reference point).
             Auto-downloaded from VainF/Diff-Pruning on first use; cached in
             weights/ddpm_ema_cifar10/.  See _ensure_cifar10_ema() for details.
"church"   → google/ddpm-ema-church-256   (LSUN Church 256×256)
"bedroom"  → google/ddpm-ema-bedroom-256  (LSUN Bedroom 256×256)

Direct Hugging Face model IDs are also accepted.

Checkpoint formats and the subfolder parameter
----------------------------------------------
diffusers stores models in two different layouts on disk / on HuggingFace:

  1. **Standalone model format** (used by Google's checkpoints):
       <repo_root>/config.json          ← UNet2DModel config
       <repo_root>/diffusion_pytorch_model.safetensors
     Load with: UNet2DModel.from_pretrained(hf_id)   # no subfolder

  2. **Pipeline format** (used by the VainF CIFAR-10 EMA checkpoint):
       <repo_root>/model_index.json     ← DDPMPipeline manifest
       <repo_root>/unet/config.json     ← UNet2DModel config
       <repo_root>/unet/diffusion_pytorch_model.bin
       <repo_root>/scheduler/scheduler_config.json
     Load with: UNet2DModel.from_pretrained(path, subfolder="unet")

Passing subfolder="unet" to a standalone model raises:
  OSError: ... does not appear to have a file named config.json
because there is no unet/ subdirectory to look in.

References
----------
Ho et al. 2020, "Denoising Diffusion Probabilistic Models", NeurIPS.
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
VainF, 2023. DDPM EMA CIFAR-10 conversion.
  https://github.com/VainF/Diff-Pruning/releases/tag/v0.0.1
"""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import Tensor

# ── CIFAR-10 EMA checkpoint (VainF conversion of the official Ho et al. weights) ──

_CIFAR10_EMA_URL = (
    "https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip"
)
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_CIFAR10_EMA_DIR = _WEIGHTS_DIR / "ddpm_ema_cifar10"

# Sentinel used in the registry to distinguish local checkpoints from HF IDs
_CIFAR10_EMA_KEY = "local:ddpm-ema-cifar10"


def _ensure_cifar10_ema() -> Path:
    """Return the local path to the CIFAR-10 EMA checkpoint, downloading if absent.

    The checkpoint is stored in weights/ddpm_ema_cifar10/ (pipeline format):
      unet/config.json
      unet/diffusion_pytorch_model.bin
      model_index.json

    The zip from VainF/Diff-Pruning contains a single top-level directory
    'ddpm_ema_cifar10/' which is stripped on extraction.
    """
    ready = _CIFAR10_EMA_DIR / "unet" / "diffusion_pytorch_model.bin"
    if ready.exists():
        return _CIFAR10_EMA_DIR

    _CIFAR10_EMA_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading official DDPM EMA CIFAR-10 checkpoint …\n"
        f"  Source: {_CIFAR10_EMA_URL}\n"
        f"  Target: {_CIFAR10_EMA_DIR}"
    )
    with urlopen(_CIFAR10_EMA_URL) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = zf.namelist()
        # Strip the top-level "ddpm_ema_cifar10/" directory prefix
        prefix = next((m for m in members if m.endswith("/")), "")
        for name in members:
            if name.startswith("__MACOSX") or name == prefix:
                continue
            rel = name[len(prefix):]
            dest = _CIFAR10_EMA_DIR / rel
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))

    print(f"Checkpoint extracted to {_CIFAR10_EMA_DIR}")
    return _CIFAR10_EMA_DIR


# ── Model registry ──────────────────────────────────────────────────────────────

# subfolder=None  → standalone model format (load from HF directly)
# subfolder="unet" → pipeline format (weights live in unet/ subdirectory)
_MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "cifar10":                      (_CIFAR10_EMA_KEY, "unet"),
    "church":                       ("google/ddpm-ema-church-256", None),
    "bedroom":                      ("google/ddpm-ema-bedroom-256", None),
    # Direct HF IDs
    "google/ddpm-ema-church-256":   ("google/ddpm-ema-church-256", None),
    "google/ddpm-ema-bedroom-256":  ("google/ddpm-ema-bedroom-256", None),
    "google/ddpm-cifar10":          ("google/ddpm-cifar10", None),
}


# ── Loader ──────────────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load a pretrained UNet and return an eps_fn callable.

    Parameters
    ----------
    model_id : str
        Shorthand key ("cifar10", "church", "bedroom") or a direct
        Hugging Face model ID.
    device : torch.device or str
    dtype : torch.dtype
        Use torch.float16 for faster inference on GPU.

    Returns
    -------
    eps_fn : callable
        Signature: (x_t: Tensor, t: int) -> ε_θ(x_t, t).
        Output has the same shape and dtype as x_t.
    unet : diffusers.UNet2DModel
        The underlying model (for inspection / debugging).

    Notes
    -----
    Only the UNet is loaded, not the full diffusers pipeline, so the
    scheduler logic stays entirely in our DDIMScheduler (src/scheduler.py).
    """
    try:
        from diffusers import UNet2DModel
    except ImportError as e:
        raise ImportError("diffusers not installed. Run: pip install diffusers") from e

    identifier, subfolder = _MODEL_REGISTRY.get(model_id, (model_id, None))
    device = torch.device(device)

    if identifier == _CIFAR10_EMA_KEY:
        # Local pipeline-format checkpoint; download on first use
        local_path = _ensure_cifar10_ema()
        unet = UNet2DModel.from_pretrained(str(local_path), subfolder="unet")
    elif subfolder is not None:
        unet = UNet2DModel.from_pretrained(identifier, subfolder=subfolder)
    else:
        unet = UNet2DModel.from_pretrained(identifier)

    unet = unet.to(device=device, dtype=dtype)
    unet.eval()

    def eps_fn(x_t: Tensor, t: int) -> Tensor:
        t_batch = torch.full(
            (x_t.shape[0],), t, device=x_t.device, dtype=torch.long
        )
        with torch.no_grad():
            out = unet(x_t, t_batch)
        return out.sample

    return eps_fn, unet


class EpsFnWrapper:
    """Thin wrapper around a UNet that stores alpha_bars for solver dispatch."""

    def __init__(self, unet, alpha_bars: Tensor, device: torch.device, dtype: torch.dtype):
        self.unet = unet
        self.alpha_bars = alpha_bars
        self.device = device
        self.dtype = dtype

    def __call__(self, x_t: Tensor, t: int) -> Tensor:
        t_batch = torch.full(
            (x_t.shape[0],), t, device=x_t.device, dtype=torch.long
        )
        with torch.no_grad():
            out = self.unet(x_t, t_batch)
        return out.sample
