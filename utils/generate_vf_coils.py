"""Generate vertical-field (VF) coil BiotSavart for finite-current scenarios.

Creates inside and outside VF coil sets as circular CurveXYZFourier coils,
wraps them in a BiotSavart object, and saves to inputs/vf_biotsavart.json.

The VF coils produce a vertical field to maintain radial equilibrium when
finite plasma current is present. Inside coils carry current in the positive
toroidal direction; outside coils carry current in the negative direction.

Hardware geometry
-----------------
Inside coils : 8 turns (4 top + 4 bottom), R = 0.260350 m
Outside coils: 8 turns (4 top + 4 bottom), R = 1.572133 m, Z = ±0.6505194 m
VF current   : 1 kA per turn (adjustable)

Usage
-----
    python utils/generate_vf_coils.py

Output is always written to inputs/vf_biotsavart.json relative to the
banana_drivers project root.
"""

import os

import numpy as np

from simsopt._core import save
from simsopt.field import BiotSavart, Coil, Current
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveXYZFourier

# ── VF coil parameters ───────────────────────────────────────────────────────

VF_CURRENT = 1e3  # 1 kA per turn
PHI_TO_CCW_DIR = 1

# Inside coils: 4 top + 4 bottom, symmetric about midplane
VF_R_INSIDE = 0.260350
VF_ZS_INSIDE_TOPSIDE = np.array([0.3508375, 0.3635375, 0.3762375, 0.3889375])
VF_ZS_INSIDE = np.sort(np.concatenate([-VF_ZS_INSIDE_TOPSIDE, VF_ZS_INSIDE_TOPSIDE]))
VF_RS_INSIDE = np.full_like(VF_ZS_INSIDE, VF_R_INSIDE)
VF_CURR_INSIDE = np.full_like(VF_RS_INSIDE, PHI_TO_CCW_DIR) * VF_CURRENT

# Outside coils: 4 top + 4 bottom, symmetric about midplane
N_TURNS_VF_OUTSIDE = 4
VF_R_OUTSIDE = 1.572133
VF_Z_OUTSIDE = 0.6505194
VF_ZS_OUTSIDE_TOPSIDE = np.full(N_TURNS_VF_OUTSIDE, VF_Z_OUTSIDE)
VF_ZS_OUTSIDE = np.sort(np.concatenate([-VF_ZS_OUTSIDE_TOPSIDE, VF_ZS_OUTSIDE_TOPSIDE]))
VF_RS_OUTSIDE = np.full_like(VF_ZS_OUTSIDE, VF_R_OUTSIDE)
VF_CURR_OUTSIDE = np.full_like(VF_RS_OUTSIDE, -PHI_TO_CCW_DIR) * VF_CURRENT

# Combined arrays
VF_RS = tuple(np.append(VF_RS_INSIDE, VF_RS_OUTSIDE))
VF_ZS = tuple(np.append(VF_ZS_INSIDE, VF_ZS_OUTSIDE))
VF_CURRENTS = tuple(np.append(VF_CURR_INSIDE, VF_CURR_OUTSIDE))

NQPTS = 128

# ── Output path (relative to project root) ───────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "inputs", "vf_biotsavart.json")


if __name__ == "__main__":
    coils = []
    for R, Z, I in zip(VF_RS, VF_ZS, VF_CURRENTS):
        curve = CurveXYZFourier(NQPTS, order=1)
        curve.set("xc(1)", R)
        curve.set("ys(1)", R)
        curve.set("zc(0)", Z)
        current = ScaledCurrent(Current(1), I)
        coils.append(Coil(curve, current))

    biotsavart = BiotSavart(coils)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save(biotsavart, OUTPUT_FILE)
    print(f"Saved VF BiotSavart ({len(coils)} coils) to {OUTPUT_FILE}")
