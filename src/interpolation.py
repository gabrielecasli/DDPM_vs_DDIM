"""
SLERP interpolation in the latent space of DDIM.

DDIM (η=0) is deterministic: x_0 = f(x_T) where x_T ~ N(0, I) is the
latent. Interpolating between two latents z_1 and z_2 yields a semantic
interpolation between the corresponding images.

SLERP (Spherical Linear Interpolation, Shoemake 1985) is preferred over
LERP because it preserves the norm of the latent vector in high dimensions.
"""

import torch
from torch import Tensor


def slerp(z1: Tensor, z2: Tensor, alpha: float) -> Tensor:
    """Spherical linear interpolation between two latent vectors.

    θ = arccos(⟨ẑ_1, ẑ_2⟩) - angle between unit-normalised vectors.
    SLERP(z_1, z_2, α) = sin((1−α)θ)/sin(θ) · z_1 + sin(αθ)/sin(θ) · z_2

    When θ → 0 (z_1 ≈ z_2), SLERP degenerates to LERP (both are identical
    in the limit). We use LERP as the fallback for numerical stability.
    """
    shape = z1.shape
    z1_flat = z1.reshape(z1.shape[0] if z1.dim() > 1 else 1, -1).double()
    z2_flat = z2.reshape(z2.shape[0] if z2.dim() > 1 else 1, -1).double()

    z1_norm = z1_flat / z1_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z2_norm = z2_flat / z2_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos_theta = (z1_norm * z2_norm).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = cos_theta.acos()   # shape (B,) or (1,)

    sin_theta = theta.sin()

    use_lerp = sin_theta.abs() < 1e-6

    coeff1 = (((1.0 - alpha) * theta).sin() / sin_theta).unsqueeze(-1)  # (B,1)
    coeff2 = ((alpha * theta).sin() / sin_theta).unsqueeze(-1)

    z_slerp = coeff1 * z1_flat + coeff2 * z2_flat

    z_lerp = (1.0 - alpha) * z1_flat + alpha * z2_flat

    mask = use_lerp.unsqueeze(-1)   # (B, 1) or (1, 1)
    z_out = torch.where(mask, z_lerp, z_slerp)

    return z_out.reshape(shape).to(z1.dtype)


def lerp(z1: Tensor, z2: Tensor, alpha: float) -> Tensor:
    """Linear interpolation (baseline comparison for SLERP).
       LERP(z_1, z_2, α) (1−α) z_1 + α z_2. Norm shrinks at the midpoint.
    """
    return ((1.0 - alpha) * z1.double() + alpha * z2.double()).to(z1.dtype)


def interpolation_path(
    z1: Tensor,
    z2: Tensor,
    num_steps: int = 9,
    method: str = "slerp",
) -> list[Tensor]:
    """Generate an interpolation path from z1 to z2.
    Returns
    -------
    list[Tensor]
        Length `num_steps`. Each element has shape (1, C, H, W).
    """
    alphas = [i / (num_steps - 1) for i in range(num_steps)]
    interp_fn = slerp if method == "slerp" else lerp
    return [interp_fn(z1, z2, a) for a in alphas]
