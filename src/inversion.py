"""
DDIM inversion — encode a real image x_0 into a latent x_T.
"""

import math

import torch
from torch import Tensor

from src.schedule import make_tau


def ddim_invert(
    eps_fn,
    x_0: Tensor,
    alpha: Tensor,
    num_steps: int = 100,
    tau_kind: str = "uniform",
    exponent: float = None,
    final_ratio: float = 0.8,
    T: int = 1000,
) -> Tensor:
    """Invert x_0 to a latent x_T using DDIM forward steps (η=0).
    -----
    The forward DDIM step is Euler on the ODE in the time-increasing direction:
      x_t = sqrt(α_t) · x̂_0_pred(x_{t−1}) + sqrt(1−α_t) · ε_θ(x_{t−1}, t−1)
    where x̂_0_pred uses x_{t−1} and ε_θ(x_{t−1}, t−1) (Song et al. 2020, Eq. 9).
    """
    device = x_0.device
    dtype = x_0.dtype
    tau = make_tau(T, num_steps, tau_kind, exponent=exponent, final_ratio=final_ratio)   # ascending, shape (S,)

    x = x_0.clone()
    a = alpha.to(device)

    with torch.no_grad():
        for i in range(len(tau)):
            t = int(tau[i])
            t_prev = int(tau[i - 1]) if i > 0 else -1

            a_t = a[t].double()
            if t_prev >= 0:
                a_t_prev = a[t_prev].double()
            else:
                a_t_prev = torch.ones(1, device=device, dtype=torch.float64)

            # ε_θ(x_{t−1}, t−1) — model at the previous (less noisy) timestep
            eps = eps_fn(x, max(t_prev, 0)).double()

            # x̂_0 from x_{t−1} (Eq. 9 applied at t_prev)
            x0_pred = (x.double() - (1.0 - a_t_prev).sqrt() * eps) / a_t_prev.sqrt()

            # Forward DDIM step: x_t = sqrt(α_t) · x̂_0 + sqrt(1−α_t) · ε_θ
            x = (a_t.sqrt() * x0_pred + (1.0 - a_t).sqrt() * eps).to(dtype)

    return x   # x_t


def reconstruction_psnr(x_orig: Tensor, x_recon: Tensor) -> float:
    """Peak Signal-to-Noise Ratio between original and reconstructed image.
    Mean PSNR in dB across the batch. Higher is better.
    PSNR > 25 dB is considered good reconstruction; > 30 dB is excellent.
    """
    max_val = x_orig.abs().max().item()
    mse = ((x_orig.float() - x_recon.float()) ** 2).mean().item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / (mse + 1e-12))
