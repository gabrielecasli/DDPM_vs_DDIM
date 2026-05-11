"""
Generic sampling loops for DDIM/DDPM and alternative ODE solvers.

Two entry points:
  - `sample`              - uses a DDIMScheduler (supports any η ∈ [0, 1]).
  - `sample_with_solver`  - uses a pure step function from solvers.py (η=0).

Timing uses torch.cuda.Event for accurate GPU wall-clock measurement,
with CPU fallback via time.perf_counter.

References
----------
Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR 2021.
Song et al. 2020, Appendix D - τ subset construction.
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
    exponent: float = None,
    final_ratio: float = 0.8,
    eta: float = 0.0,
    seed: int = 42,
    sampling_seed: int | None = None,
    clip_sample: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    T: int = 1000,
) -> tuple[Tensor, float]:
    device = torch.device(device)
    """Generate samples using a DDIMScheduler (supports any η ∈ [0, 1])"""
    init_gen = torch.Generator(device=device).manual_seed(seed)
    x_T = torch.randn(shape, device=device, dtype=dtype, generator=init_gen)

    step_gen = torch.Generator(device=device).manual_seed(
        seed if sampling_seed is None else sampling_seed
    )
    tau = make_tau(T, num_steps, tau_kind, exponent=exponent, final_ratio=final_ratio)

    if device.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_ev.record()
    else:
        t0 = time.perf_counter()

    x_0 = denoise_steps(
        eps_fn=eps_fn,
        scheduler=scheduler,
        x_start=x_T,
        tau=tau,
        eta=eta,
        clip_sample=clip_sample,
        store_intermediates=False,
        generator=step_gen,
    )

    if device.type == "cuda":
        end_ev.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_ev.elapsed_time(end_ev)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1e3

    return x_0, elapsed_ms


def denoise_steps(
    eps_fn: Callable[[Tensor, int], Tensor],
    scheduler: DDIMScheduler,
    x_start: Tensor,
    tau,
    eta: float = 0.0,
    seed: int = 42,
    clip_sample: bool = True,
    store_intermediates: bool = True,
    generator: torch.Generator | None = None,
) -> list[tuple[int, Tensor]] | Tensor:
    """Run the reverse process from x_start.

    If `store_intermediates` is True (default) returns the list of (t, x).
    If False, returns only the final (more efficient).
    """
    scheduler.set_eta(eta)
    if generator is None:
        generator = torch.Generator(device=x_start.device).manual_seed(seed)

    x = x_start.clone() if store_intermediates else x_start

    steps: list[tuple[int, Tensor]] | None = None
    if store_intermediates:
        steps = [(int(tau[-1]), x.detach().clone())]

    with torch.no_grad():
        for i in reversed(range(len(tau))):
            t = int(tau[i])
            t_prev = int(tau[i - 1]) if i > 0 else -1
            x = scheduler.step(
                eps_fn(x, t), x, t, t_prev,
                generator=generator, clip_sample=clip_sample,
            )
            if store_intermediates:
                steps.append((t_prev if t_prev >= 0 else 0, x.detach().clone()))

    return steps if store_intermediates else x


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
