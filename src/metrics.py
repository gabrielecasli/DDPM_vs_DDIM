"""
Evaluation metrics: FID, (via clean-fid), pairwise LPIPS, and timing.
NOTE
-------------
We use clean-fid with `mode="legacy_pytorch"` rather than `mode="clean"`.

The two modes differ in resizing:
  - "clean"          uses the torchvision Lanczos filter (more accurate).
  - "legacy_pytorch" uses PyTorch's bilinear resize, which is the same
                     implementation used by the original DDPM/DDIM codebases
                     (ermongroup/ddpm, ermongroup/ddim).

Because Song et al. 2020 (DDIM) report FID with the PyTorch bilinear resize, we match their mode to keep numbers comparable.

References
----------
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
"""

import time
from typing import Callable

import torch
from torch import Tensor


def compute_fid(
    folder_generated: str,
    dataset_name: str = "cifar10",
    split: str = "train",
    batch_size: int = 64,
    mode: str = "legacy_pytorch",
    dataset_res: int = 32,
    device: torch.device | str | None = None,
) -> float:
    """Compute Fréchet Inception Distance (lower is better) between a folder of generated PNG/JPEG images and a reference dataset.
    The precomputed reference stats are downloaded automatically on first use.
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError as e:
        raise ImportError("clean-fid not installed. Run: pip install clean-fid") from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    return cleanfid.compute_fid(
        folder_generated,
        dataset_name=dataset_name,
        dataset_split=split,
        batch_size=batch_size,
        mode=mode,
        dataset_res=dataset_res,
        device=device,
    )


import torch
from torch import Tensor
import itertools

def pairwise_lpips(images: Tensor, net: str = "alex", batch_size: int = 64) -> float:
    """Mean pairwise LPIPS diversity score over a mini-batch (Optimized).

    Parameters
    ----------
    images : Tensor
        Shape (N, 3, H, W), values in [0, 1] or [-1, 1]. LPIPS normalises internally.
    net : str
        Backbone for LPIPS: "alex" (fast, recommended) or "vgg".
    batch_size : int

    Returns
    -------
    float
        Mean LPIPS over all N*(N-1)/2 unique pairs. Higher → more diverse.
    """
    try:
        import lpips
    except ImportError as e:
        raise ImportError("lpips not installed. Run: pip install lpips") from e

    N = images.shape[0]
    if N < 2:
        return 0.0

    device = images.device

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    # Unique combinations (N*(N-1)/2)
    # N=4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    indices = list(itertools.combinations(range(N), 2))
    idx1, idx2 = zip(*indices)
    
    # Vectorized indexing
    idx1 = torch.tensor(idx1, device=device)
    idx2 = torch.tensor(idx2, device=device)
    
    num_pairs = len(idx1)
    scores = []

    with torch.no_grad():
        for i in range(0, num_pairs, batch_size):
            batch_idx1 = idx1[i : i + batch_size]
            batch_idx2 = idx2[i : i + batch_size]
            
            img1_batch = images[batch_idx1]
            img2_batch = images[batch_idx2]
            
            score_batch = loss_fn(img1_batch, img2_batch)
            scores.append(score_batch.view(-1))

    if scores:
        mean_score = torch.cat(scores).mean().item()
        return float(mean_score)
    return 0.0



def time_sampling(
    fn: Callable[[], None],
    n_warmup: int = 2,
    n_runs: int = 5,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    """Measure wall-clock time of a sampling function using CUDA events.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    # warm-up
    for _ in range(n_warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_runs):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)

    mean_ms = sum(times) / len(times)
    var_ms = sum((t - mean_ms) ** 2 for t in times) / max(len(times) - 1, 1)
    std_ms = var_ms ** 0.5
    return mean_ms, std_ms
