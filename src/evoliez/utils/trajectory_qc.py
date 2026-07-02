"""Small trajectory metadata checks used by v4 preflight tests."""

from __future__ import annotations


def ns_to_steps(simulated_time_ns: float, timestep_fs: float) -> int:
    if simulated_time_ns < 0:
        raise ValueError("simulated_time_ns must be non-negative")
    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    # 1 ns = 1,000,000 fs.
    return int(round((simulated_time_ns * 1_000_000.0) / timestep_fs))


def simulated_time_from_steps(n_steps: int, timestep_fs: float) -> float:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    return (n_steps * timestep_fs) / 1_000_000.0
