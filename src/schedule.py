"""
Noise schedule and timestep subset selection for DDPM/DDIM.

Forward process (DDPM, Ho et al. 2020, Eq. 4):
    q(x_t | x_0) = N(x_t; sqrt(α_t) x_0, (1 − α_t) I)
with α_t = ∏_{s=1}^{t} (1 − β_s).

The variance schedule β_1, …, β_T is either linear (Ho et al. 2020)
or cosine (Nichol & Dhariwal 2021). Only "linear" is required here.

Tau-subset selection follows Song et al. 2020 (DDIM), Appendix D:
  - uniform:   τ_i = i · ⌊T/S⌋  i = 0, …, S−1
  - quadratic: τ_i = (linspace(0, sqrt(T*0.8), S)[i])^2
    (denser steps near t=0 where the signal changes fastest).

References
----------
Ho et al. 2020, "Denoising Diffusion Probabilistic Models", NeurIPS.
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
"""

import numpy as np
import torch
from torch import Tensor


def make_beta_schedule(
    name: str = "linear",
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tensor:
    """Return the variance schedule β ∈ ℝ^T (Tensor Shape (T,), dtype float64)
    
    Ho et al. 2020, Section 4: linear schedule from β_start to β_end,
    T=1000, β_start=1e-4, β_end=0.02.
    """
    if name == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    raise ValueError(f"Unknown schedule '{name}'. Only 'linear' is supported.")


def compute_alphas(betas: Tensor) -> tuple[Tensor, Tensor]:
    """Compute ᾱ_t_ddpm and α_t from the variance schedule β.
    -----
    Returns: 
    -----
    alphas_ddpm : Tensor
        Shape (T,). ᾱ_t_ddpm = 1 − β_t.
    alphas : Tensor
        Shape (T,). α_t = ∏_{s=1}^{t} ᾱ_t_ddpm  (cumulative product).

    Notes
    -----
    Ho et al. 2020, Eq. 4: q(x_t | x_0) = N(x_t; √α_t x_0, (1−α_t) I).
    """
    alphas_ddpm = 1.0 - betas                           # α_t_ddpm = 1 − β_t
    alphas = torch.cumprod(alphas_ddpm, dim=0)          # α_t = ∏_{s≤t} (1 − β_s)
    return alphas_ddpm, alphas


def make_tau(T: int, S: int, kind: str = "uniform", exponent: float = None, final_ratio: float = 0.8) -> np.ndarray:
    """Select a sub-sequence τ ⊂ {0, …, T−1} of length S.
    Implementation choices match ermongroup/ddim (runners/diffusion.py).
    """
    if kind == "uniform":
        # tau = np.linspace(0, T - 1, S).astype(int)
        skip = T // S
        tau = np.arange(0, T, skip)
        tau = tau[:S]
    elif kind == "quadratic":
        tau = (np.linspace(0, np.sqrt(T * 0.8), S) ** 2).astype(int)
    elif kind == "poly" and exponent is not None:
            if exponent == 0:
                raise ValueError("Check given exponent")
            if final_ratio > 1 or final_ratio <= 0:
                raise ValueError("Check given final_ratio")
            tau = (np.linspace(0, (T * final_ratio) ** (1.0 / exponent), S) ** exponent).astype(int)
    else:
        raise ValueError(f"Unknown tau kind '{kind}'. Choose 'uniform' or 'quadratic'.")
    return tau
