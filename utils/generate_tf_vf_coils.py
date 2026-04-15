"""Generate vertical-field (VF) coil BiotSavart for finite-current scenarios.

Creates inside and outside VF coil sets as circular CurveXYZFourier coils,
wraps them in a BiotSavart object, and saves to inputs/vf_biotsavart.json.

The VF coils produce a vertical field to maintain radial equilibrium when
finite plasma current is present. Inside coils carry current in the positive
toroidal direction; outside coils carry current in the negative direction.

Hardware geometry
-----------------
Inside coils : 12 turns (6 top + 6 bottom), R = 0.260350 m
Outside coils:  8 turns (4 top + 4 bottom), R = 1.572133 m, Z = ±0.6505194 m
VF current   : 1 kA per turn (adjustable up to TODO: ? kA per turn)

Usage
-----
    python utils/generate_vf_coils.py

Output is always written to inputs/vf_biotsavart.json relative to the
banana_drivers project root.
"""
import argparse
import os

from hbt_parameters import *

from simsopt._core import save
from simsopt.field import BiotSavart, Coil, Current
from simsopt.field.coil import ScaledCurrent
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurveXYZFourier,
    create_equally_spaced_curves,
    curves_to_vtk,
)

parser = argparse.ArgumentParser(
    description="Generate VF coil BiotSavart for finite-current scenarios.")
parser.add_argument(
    "-p", "--plot", action="store_true",
    help="Generate VTK files for visualization of coil geometry.")
args = parser.parse_args()
plot = args.plot

# ── Output path (relative to project root) ───────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF_OUT_FILE  = os.path.join(PROJECT_ROOT, "inputs", "vf_biotsavart.json")
TF_OUT_FILE  = os.path.join(PROJECT_ROOT, "inputs", "tf_biotsavart.json")

# -- Coil currents ------------------------------------------------------------
VF_CURRENT =  1e3
TF_CURRENT = 80e3

# -- Coil quadpoints ----------------------------------------------------------
NQPTS = 32

vf_coils = []
for R, Z, sign in zip(VF_RS, VF_ZS, VF_SIGN_CURR):
    curve = CurveXYZFourier(NQPTS, order=1)
    curve.set("xc(1)", R)
    curve.set("ys(1)", R)
    curve.set("zc(0)", Z)
    current = ScaledCurrent(Current(1), sign * VF_CURRENT)
    curve.fix_all()
    current.fix_all()
    vf_coils.append(Coil(curve, current))

tf_coils = []
for curve in create_equally_spaced_curves(
    N_TF_COILS,
    TF_NFP,
    stellsym=False,
    R0=TFCOIL_MAJOR_R,
    R1=TFCOIL_MINOR_R,
    order=1,
    numquadpoints=NQPTS,
):
    current = ScaledCurrent(Current(1), TF_CURRENT)
    curve.fix_all()
    current.fix_all()
    tf_coils.append(Coil(curve, current))

vf_biotsavart = BiotSavart(vf_coils)
tf_biotsavart = BiotSavart(tf_coils)

save(vf_biotsavart, VF_OUT_FILE)
save(tf_biotsavart, TF_OUT_FILE)

print(f"Saved VF coil BiotSavart to {VF_OUT_FILE}")
print(f"Saved TF coil BiotSavart to {TF_OUT_FILE}")

if plot:
    vf_curves = [coil.curve for coil in vf_coils]
    tf_curves = [coil.curve for coil in tf_coils]
    curves = vf_curves + tf_curves
    curves_to_vtk(curves, "tf_vf_coils", close=True)
    print("Saved coil geometry to tf_vf_coils.vtk")