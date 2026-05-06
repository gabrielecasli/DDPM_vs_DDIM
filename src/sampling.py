"""
Generic sampling loops for DDIM/DDPM and alternative ODE solvers.

Two entry points:
  - `sample`              — uses a DDIMScheduler (supports any η ∈ [0, 1]).
  - `sample_with_solver`  — uses a pure step function from solvers.py (η=0).

Timing uses torch.cuda.Event for accurate GPU wall-clock measurement,
with CPU fallback via time.perf_counter.

References
----------
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
Song et al. 2020, Appendix C — τ subset construction.
"""

import time
from typing import Callable

import torch
from torch import Tensor

from src.schedule import make_tau
from src.scheduler import DDIMScheduler


def sample(
    eps_fn: Callable[[Tensor, int], Tensor],
    scheduler: DDIMScheduler,
    shape: tuple[int, ...],
    num_steps: int,
    tau_kind: str = "uniform",
    eta: float = 0.0,
    seed: int = 42,
    sampling_seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    T: int = 1000,
) -> tuple[Tensor, float]:
    """Generate samples using a DDIMScheduler (supports any η ∈ [0, 1]).

    Parameters
    ----------
    eps_fn : callable
        Model: (x_t: Tensor, t: int) -> ε_θ(x_t, t).
    scheduler : DDIMScheduler
        Will have its eta updated to `eta` before sampling.
    shape : tuple
        (B, C, H, W) — shape of the noise tensor.
    num_steps : int
        Number of denoising steps S (= NFE for DDIM/Euler).
    tau_kind : str
        "uniform" or "quadratic".
    eta : float
        Stochasticity coefficient. Forwarded to scheduler.
    seed : int
        Random seed for x_T initialisation.
    sampling_seed : int or None
        Random seed for the stochastic denoising steps (η > 0).
        If None, reuses `seed` — same behaviour as before.
        Separating the two seeds lets you fix x_T while varying the
        per-step noise, which is the correct way to demonstrate that
        DDIM (η=0) is deterministic while DDPM (η=1) is stochastic.
    device : torch.device or str
    dtype : torch.dtype
    T : int
        Total training timesteps.

    Returns
    -------
    x_0 : Tensor
        Generated samples, shape (B, C, H, W).
    elapsed_ms : float
        Wall-clock time in milliseconds (GPU-accurate when CUDA available).

    Notes
    -----
    τ subset: Song et al. 2020, Appendix C.
    Step formula: Song et al. 2020, Eq. 12 (via DDIMScheduler).
    """
    device = torch.device(device)
    scheduler.set_eta(eta)

    init_gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=dtype, generator=init_gen)    # X_T

    step_gen = torch.Generator(device=device).manual_seed(seed if sampling_seed is None else sampling_seed)

    tau = make_tau(T, num_steps, tau_kind)   # ascending indices, shape (S,)

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_ev.record()
    else:
        t0 = time.perf_counter()

    with torch.no_grad():
        for i in reversed(range(len(tau))):        # τ_S → τ_0
            t = int(tau[i])
            t_prev = int(tau[i - 1]) if i > 0 else -1
            eps = eps_fn(x, t)
            x = scheduler.step(eps, x, t, t_prev, generator=step_gen)

    if device.type == "cuda":
        end_ev.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_ev.elapsed_time(end_ev)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1e3

    return x, elapsed_ms


def sample_with_solver(
    eps_fn: Callable[[Tensor, int], Tensor],
    step_fn: Callable,
    alpha_bars: Tensor,
    shape: tuple[int, ...],
    num_steps: int,
    tau_kind: str = "uniform",
    seed: int = 42,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    T: int = 1000,
) -> tuple[Tensor, float]:
    """Generate samples using a pure step function from solvers.py (η=0).

    Parameters
    ----------
    step_fn : callable
        Signature: (eps_fn, x, t, t_prev, alpha_bars) -> x_prev.
        One of: euler_step, heun_step, rk4_step, dpm2_step.
    alpha_bars : Tensor
        Shape (T,). Passed through to step_fn at every iteration.

    All other parameters identical to `sample`.

    Returns
    -------
    x_0 : Tensor
    elapsed_ms : float
    """
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    tau = make_tau(T, num_steps, tau_kind)

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_ev.record()
    else:
        t0 = time.perf_counter()

    with torch.no_grad():
        for i in reversed(range(len(tau))):
            t = int(tau[i])
            t_prev = int(tau[i - 1]) if i > 0 else -1
            x = step_fn(eps_fn, x, t, t_prev, alpha_bars)

    if device.type == "cuda":
        end_ev.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_ev.elapsed_time(end_ev)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1e3

    return x, elapsed_ms
