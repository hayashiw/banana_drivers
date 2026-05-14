"""Compute an effective iota at the target plasma surface from a BiotSavart
field. Used by scan_vacuum_curr_iota in --iota-mode trace to pick the
singlestage iota_target from the stage 2 output.

NOT YET IMPLEMENTED — orchestrator currently only supports --iota-mode csv.
The contract is stubbed here so scan.py can import and dispatch by mode
without breaking.

Plan when implemented: load `biotsavart_opt.json`, build the target LCFS
SurfaceRZFourier (s=1.0 of the resized stage1 seed), trace a small number
of fieldlines starting on it, fit dphi/dtheta to extract iota averaged
over the surface. ~150 lines when fully written.
"""
from __future__ import annotations


def trace_iota_from_biotsavart(biotsavart_path: str, *, vmec_s: float = 1.0) -> float:
    raise NotImplementedError(
        'iota-from-biotsavart trace not yet implemented; '
        'use --iota-mode csv instead.'
    )
