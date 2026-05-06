"""
Utility functions: seeding, image grid saving, device helpers, config loading.
"""

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Set all random seeds and return a device-bound Generator.

    Parameters
    ----------
    seed : int
        Global seed for Python, NumPy, and PyTorch (CPU + CUDA).
    device : torch.device or str
        Device for the returned Generator.

    Returns
    -------
    torch.Generator
        A seeded Generator for the given device.

    Notes
    -----
    torch.Generator(device).manual_seed(seed) is preferred over global
    torch.manual_seed() in this codebase because it avoids hidden global
    state. See sampling.py for usage.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    return generator


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device.

    Parameters
    ----------
    prefer_cuda : bool
        If True and a CUDA GPU is available, use it. MPS (Apple Silicon)
        is intentionally skipped: some diffusion ops are unsupported.

    Returns
    -------
    torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_grid(
    images: Tensor,
    path: Union[str, Path],
    nrow: int = 4,
    value_range: tuple[float, float] = (-1.0, 1.0),
    padding: int = 2,
) -> None:
    """Save a batch of images as a PNG grid.

    Parameters
    ----------
    images : Tensor
        Shape (N, C, H, W). Values in `value_range`.
    path : str or Path
        Output file path. Parent directories are created automatically.
    nrow : int
        Number of images per row in the grid.
    value_range : tuple
        (min, max) of input values. Images are normalised to [0, 1].
    padding : int
        Pixels of padding between images.
    """
    try:
        from torchvision.utils import save_image
    except ImportError as e:
        raise ImportError("torchvision not installed. Run: pip install torchvision") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        images.float().clamp(*value_range),
        str(path),
        nrow=nrow,
        normalize=True,
        value_range=value_range,
        padding=padding,
    )


def images_to_uint8(
    images: Tensor,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Convert a batch of model-output images to uint8 numpy arrays.

    Parameters
    ----------
    images : Tensor
        Shape (N, C, H, W), values in `value_range`.
    value_range : tuple

    Returns
    -------
    np.ndarray
        Shape (N, H, W, C), dtype uint8, values in [0, 255].
        Suitable for passing to PIL or saving to disk with imageio.
    """
    lo, hi = value_range
    imgs = (images.float() - lo) / (hi - lo)   # → [0, 1]
    imgs = (imgs * 255).clamp(0, 255).byte()
    # (N, C, H, W) → (N, H, W, C)
    return imgs.permute(0, 2, 3, 1).cpu().numpy()


def save_samples_to_folder(
    images: Tensor,
    folder: Union[str, Path],
    start_idx: int = 0,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Save individual images as PNG files (for FID computation with clean-fid).

    Parameters
    ----------
    images : Tensor
        Shape (N, C, H, W).
    folder : str or Path
        Output directory. Created if it does not exist.
    start_idx : int
        Filename counter offset (for batched saving in a loop).
    value_range : tuple
        Input value range.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow not installed. Run: pip install Pillow") from e

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    imgs_np = images_to_uint8(images, value_range)
    for i, img in enumerate(imgs_np):
        pil_img = Image.fromarray(img)
        pil_img.save(folder / f"{start_idx + i:06d}.png")


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml") from e

    with open(path, "r") as f:
        return yaml.safe_load(f)
