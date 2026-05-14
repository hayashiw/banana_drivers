"""Sampling helpers for scan orchestrators.

Reproducibility: pass a numpy.random.Generator constructed from --seed.
hex_id derivation: monotone counter (max(existing)+1 ... +n), zero-padded
to 4 hex chars. This keeps id assignment deterministic given existing
CSV state, so a re-run of the orchestrator with the same seed and same
existing CSV picks up where it left off.
"""
from __future__ import annotations

import math
import numpy as np


HEX_ID_WIDTH = 4   # supports 16^4 = 65536 distinct points per scan dir


def hex_id_from_int(n: int) -> str:
    if n < 0 or n > 16**HEX_ID_WIDTH - 1:
        raise ValueError(f'id index {n} out of [0, {16**HEX_ID_WIDTH - 1}]')
    return f'{n:0{HEX_ID_WIDTH}x}'


def next_id_index(existing_ids: set[str]) -> int:
    """Return the smallest non-negative int whose hex_id is NOT in existing_ids."""
    used = set()
    for hid in existing_ids:
        try:
            used.add(int(hid, 16))
        except ValueError:
            pass
    n = 0
    while n in used:
        n += 1
    return n


def assign_ids(existing_ids: set[str], n_new: int) -> list[str]:
    """Allocate `n_new` hex_ids contiguous past max(existing)+1 (or starting
    at 0 if the CSV is empty). Returns the hex strings.
    """
    used = set()
    max_idx = -1
    for hid in existing_ids:
        try:
            i = int(hid, 16)
            used.add(i)
            if i > max_idx:
                max_idx = i
        except ValueError:
            pass
    start = max_idx + 1
    return [hex_id_from_int(i) for i in range(start, start + n_new)]


# ──────────────────────────────────────────────────────────────────────────
# Distributions
# ──────────────────────────────────────────────────────────────────────────
def sample_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    """Log-uniform in [lo, hi]; lo and hi must be > 0."""
    if lo <= 0 or hi <= 0:
        raise ValueError(f'log_uniform endpoints must be positive (got [{lo}, {hi}])')
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))
