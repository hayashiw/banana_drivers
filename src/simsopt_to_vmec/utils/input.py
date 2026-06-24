import numpy as np
import pandas as pd
import re

from simsopt._core import load
from simsopt.mhd import Vmec
from simsopt.geo import BoozerSurface, SurfaceRZFourier

from ..paths import HBT_COEFF_FILE

THIN_NTHETA = 256
PCURR_TYPE = "power_series_i"
PIOTA_TYPE = "power_series"

DEFAULT_NCURR = 1
DEFAULT_CURREDGE = 0.0
DEFAULT_IOTAEDGE = 0.2
DEFAULT_NS_ARRAY = [13]
DEFAULT_NITER_ARRAY = [1000]
DEFAULT_FTOL_ARRAY = [1e-6]
DEFAULT_AXIS_SCALE = 0.5

def write_fixedb_input_from_boozersurface(
    boozersurface: BoozerSurface | str,
    input_filename: str,
    ncurr: int = DEFAULT_NCURR,
    curredge: float = DEFAULT_CURREDGE,
    iotaedge: float = DEFAULT_IOTAEDGE,
    ns_array: int | list[int] = DEFAULT_NS_ARRAY,
    niter_array: int | list[int] = DEFAULT_NITER_ARRAY,
    ftol_array: float | list[float] = DEFAULT_FTOL_ARRAY,
    axis_scale: float = DEFAULT_AXIS_SCALE,
) -> None:
    assert ncurr in [0, 1], "ncurr must be 0 (fixed iota) or 1 (fixed current)"
    
    if isinstance(boozersurface, str):
        boozersurface = load(boozersurface)

    biotsavart = boozersurface.biotsavart
    surface_xyz = boozersurface.surface
    mpol = surface_xyz.mpol
    nfp = surface_xyz.nfp
    ntor = nfp + 1
    if mpol == ntor: mpol += 1

    surface_rz = SurfaceRZFourier(
        mpol=surface_xyz.mpol,
        ntor=surface_xyz.ntor,
        nfp=surface_xyz.nfp,
        stellsym=surface_xyz.stellsym,
        quadpoints_phi=surface_xyz.quadpoints_phi,
        quadpoints_theta=surface_xyz.quadpoints_theta,
    )
    surface_rz.least_squares_fit(surface_xyz.gamma())

    surface = SurfaceRZFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=surface_rz.stellsym,
        quadpoints_phi=surface_rz.quadpoints_phi,
        quadpoints_theta=surface_rz.quadpoints_theta,
    )
    surface.least_squares_fit(surface_rz.gamma())

    thin = SurfaceRZFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_phi=np.array([0.0]),
        quadpoints_theta=np.linspace(0.0, 1.0, THIN_NTHETA, endpoint=False),
    )
    thin.x = surface.x

    g0 = thin.gamma()[0]
    dg0 = thin.gammadash2()[0]
    biotsavart.set_points(g0)
    A = biotsavart.A()
    B = biotsavart.B()
    dq = 1.0 / THIN_NTHETA
    A_dot_tangent = np.einsum("ij,ij->i", A, dg0)
    magnitude = abs(float(np.sum(A_dot_tangent) * dq))
    Bphi_at_phi0 = B[:, 1]
    sign = np.sign(Bphi_at_phi0.mean())
    if sign == 0: sign = 1
    phiedge = sign * magnitude

    vmec = Vmec()
    vmec.lfreeb = False

    coeff_df = pd.read_csv(HBT_COEFF_FILE)
    ac = coeff_df["ac"].to_numpy()
    ai = coeff_df["ai"].to_numpy()
    ac *= curredge / np.polyval(ac[::-1], [1.0])
    ai *= iotaedge / np.polyval(ai[::-1], [1.0])

    vmec.indata.ncurr = ncurr
    if ncurr == 1:
        vmec.indata.pcurr_type = PCURR_TYPE
        vmec.indata.ac[:] = 0.0
        vmec.indata.ac[:len(ac)] = ac
        vmec.indata.curtor = curredge
    else:
        vmec.indata.piota_type = PIOTA_TYPE
        vmec.indata.ai[:] = 0.0
        vmec.indata.ai[:len(ai)] = ai

    vmec.indata.ns_array[:]    = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:]  = 0.0

    vmec.indata.ns_array[:len(ns_array)] = ns_array
    vmec.indata.niter_array[:len(niter_array)] = niter_array
    vmec.indata.ftol_array[:len(ftol_array)] = ftol_array

    vmec.indata.mpol = mpol
    vmec.indata.ntor = ntor

    vmec.indata.ntheta = 3*mpol + 1
    vmec.indata.nzeta  = 3*ntor + 1

    vmec.indata.phiedge = phiedge

    vmec.boundary = surface

    vmec.verbose = True

    vmec.write_input(input_filename)

    r0 = surface.major_radius()
    ntor_seed = min(ntor, 4)
    raxis_cc = [r0]  + [axis_scale * surface.get_rc(0, n) for n in range(1, ntor_seed + 1)]
    zaxis_cs = [0.0] + [axis_scale * surface.get_zs(0, n) for n in range(1, ntor_seed + 1)]
    raxis_str = " ".join(f"{v:.10e}" for v in raxis_cc)
    zaxis_str = " ".join(f"{v:.10e}" for v in zaxis_cs)

    with open(input_filename, "r") as f:
        text = f.read()

    text = re.sub(r"^\s*(RAXIS_CC|ZAXIS_CS)\s*=.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^/\s*$",
                f"\nRAXIS_CC = {raxis_str}\nZAXIS_CS = {zaxis_str}\n/",
                text, count=1, flags=re.MULTILINE)

    with open(input_filename, "w") as f:
        f.write(text)

    return 0
