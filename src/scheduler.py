"""
Single DDIM scheduler parametric in η (eta).

Implements the generalized sampling formula from Song et al. 2020
(DDIM, ICLR 2021), Equation 12:

    σ_t(η) = η · sqrt((1 − α_{t−1}) / (1 − α_t)) · sqrt(1 − α_t / α_{t−1})

    x̂_0(x_t) = (x_t − sqrt(1 − α_t) · ε_θ(x_t, t)) / sqrt(α_t)   [Eq. 9]

    x_{t−1} = sqrt(α_{t−1}) · x̂_0
            + sqrt(1 − α_{t−1} − σ_t²) · ε_θ(x_t, t)
            + σ_t · z,      z ~ N(0, I)                               [Eq. 12]

Special cases:
  η = 0  → deterministic DDIM; σ_t = 0, no stochastic term.
  η = 1  → matches DDPM ancestral sampling with variance β̃_t.
  η ∈ (0,1) → interpolation between deterministic and stochastic.

References
----------
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
Ho et al. 2020, "Denoising Diffusion Probabilistic Models", NeurIPS.
"""

import torch
from torch import Tensor


class DDIMScheduler:
    """Generalized DDIM scheduler parametric in η.

    Parameters
    ----------
    alphas : Tensor
        Shape (T,), dtype float64. α_t = ∏_{s≤t}(1−β_s).
    eta : float
        Stochasticity coefficient η ∈ [0, 1].
        0 → deterministic DDIM; 1 → DDPM ancestral sampling.
    """

    def __init__(self, alphas: Tensor, eta: float = 0.0) -> None:
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}.")
        self.alphas = alphas.double()   # keep full precision for schedule math
        self.eta = eta

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        eps: Tensor,
        x_t: Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Perform one reverse step from x_t to x_{t−1}.

        Parameters
        ----------
        eps : Tensor
            ε_θ(x_t, t) — noise predicted by the UNet, shape (B, C, H, W).
        x_t : Tensor
            Noisy image at timestep t, shape (B, C, H, W).
        t : int
            Current timestep index into [0, T−1].
        t_prev : int
            Previous (less noisy) timestep index. −1 means t_prev corresponds
            to t=0 (fully clean), for which α_{t_prev} = 1.
        generator : torch.Generator or None
            For reproducible stochastic noise (η > 0).

        Returns
        -------
        Tensor
            x_{t−1}, shape (B, C, H, W).

        Notes
        -----
        Implements Song et al. 2020, Eq. 12
        """
        device = x_t.device
        dtype = x_t.dtype

        # α_t — cumulative product at current step (Song et al. 2020, Eq. 4)
        alpha_t = self.alphas[t].to(device)

        # α_{t−1} — cumulative product at previous step
        # Convention: for t_prev = -1 (i.e. the step to x_0), α_{t_prev} = 1
        if t_prev < 0:
            alpha_t_prev = torch.ones(1, device=device, dtype=torch.float64)
        else:
            alpha_t_prev = self.alphas[t_prev].to(device)

        # ── Eq. 9: predicted x_0 from the model output ──────────────────
        # x̂_0 = (x_t − sqrt(1 − α_t) · ε_θ) / sqrt(α_t)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
        x0_pred = (x_t.double() - sqrt_one_minus_alpha_t * eps.double()) / sqrt_alpha_t

        # ── Eq. 12: σ_t(η) ──────────────────────────────────────────────
        # σ_t = η · sqrt((1−α_{t−1})/(1−α_t)) · sqrt(1 − α_t/α_{t−1})
        #
        # The second sqrt equals sqrt(β̃_t) when η=1, reproducing DDPM.
        # Note: (1 − α_t/α_{t−1}) = (α_{t−1} − α_t)/α_{t−1}
        ratio = (1.0 - alpha_t_prev) / (1.0 - alpha_t)           # (1−α_{t−1})/(1−α_t)
        sigma_sq = self.eta ** 2 * ratio * (1.0 - alpha_t / alpha_t_prev)  # σ_t²
        sigma = sigma_sq.clamp(min=0.0).sqrt()

        # ── Eq. 12: "direction pointing to x_t" coefficient ─────────────
        # sqrt(1 − α_{t−1} − σ_t²)  (must be ≥ 0; clamp for numerical safety)
        dir_coeff = (1.0 - alpha_t_prev - sigma_sq).clamp(min=0.0).sqrt()

        # ── Eq. 12: assemble x_{t−1} ────────────────────────────────────
        # x_{t−1} = sqrt(α_{t−1}) · x̂_0
        #          + sqrt(1−α_{t−1}−σ_t²) · ε_θ
        #          + σ_t · z
        x_prev = (
            alpha_t_prev.sqrt() * x0_pred           # "predicted x_0" term
            + dir_coeff * eps.double()            # "direction to x_t" term
        )

        if self.eta > 0.0:
            # Sample noise in x_t.dtype so the random stream matches diffusers
            # (both float32 from the same generator → identical noise bytes).
            z = torch.randn(x_t.shape, device=device, dtype=dtype, generator=generator)
            x_prev = x_prev + sigma * z.double()  # stochastic term

        return x_prev.to(dtype)

    def set_eta(self, eta: float) -> None:
        """Update η in-place (useful for sweeps without re-instantiation)."""
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}.")
        self.eta = eta
