import os
import numpy as np

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier

surface_classes = (SurfaceRZFourier, SurfaceXYZTensorFourier)

DEFAULT_NPHI = 65
DEFAULT_NTHETA = 64

def convert_rz_to_xyztensor(surface_rz: SurfaceRZFourier | str) -> SurfaceXYZTensorFourier:
    if isinstance(surface_rz, str):
        surface_rz = load(surface_rz)
    assert isinstance(surface_rz, surface_classes), f"Expected SurfaceRZFourier, SurfaceXYZTensorFourier, or filepath, got {type(surface_rz)}"
    if isinstance(surface_rz, SurfaceXYZTensorFourier):
        surface = surface_rz
    else:
        surface = SurfaceXYZTensorFourier(
            mpol=surface_rz.mpol,
            ntor=surface_rz.ntor,
            nfp=surface_rz.nfp,
            stellsym=surface_rz.stellsym,
            quadpoints_phi=surface_rz.quadpoints_phi,
            quadpoints_theta=surface_rz.quadpoints_theta,
        )
        surface.least_squares_fit(surface_rz.gamma())
    return surface

def change_surface_range(
    surface_in: SurfaceRZFourier | SurfaceXYZTensorFourier | str,
    *,
    surf_range: str = "field period",
    nphi: int = DEFAULT_NPHI,
    ntheta: int = DEFAULT_NTHETA,
) -> SurfaceXYZTensorFourier:
    # Also applies to changing surface quadpoints
    surface_in = convert_rz_to_xyztensor(surface_in)

    nfp = surface_in.nfp
    if surf_range == "full torus":
        phimax = 1
    elif surf_range == "field period":
        phimax = 1 / nfp
    elif surf_range == "half period":
        phimax = 1 / nfp / 2
    else:
        raise ValueError(f"Invalid surface range: {surf_range}")

    surface = SurfaceXYZTensorFourier(
        mpol=surface_in.mpol,
        ntor=surface_in.ntor,
        nfp=surface_in.nfp,
        stellsym=surface_in.stellsym,
        quadpoints_phi=np.linspace(0, phimax, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surface.x = surface_in.x.copy()
    return surface

def change_surface_resolution(
    surface_in: SurfaceRZFourier | SurfaceXYZTensorFourier | str,
    *,
    mpol: int | None = None,
    ntor: int | None = None,      
):
    surface_in = convert_rz_to_xyztensor(surface_in)

    if mpol is None: mpol = surface_in.mpol
    if ntor is None: ntor = surface_in.ntor
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=surface_in.nfp,
        stellsym=surface_in.stellsym,
        quadpoints_phi=surface_in.quadpoints_phi,
        quadpoints_theta=surface_in.quadpoints_theta,
    )
    surface.least_squares_fit(surface_in.gamma())
    return surface

def convert_to_boozerexact_surface(
    surface_in: SurfaceRZFourier | SurfaceXYZTensorFourier | str,
    *,
    surf_range: str = "field period",
    mpol: int | None = None,
    ntor: int | None = None,      
):
    surface_in = convert_rz_to_xyztensor(surface_in)

    if mpol is None: mpol = surface_in.mpol
    if ntor is None: ntor = surface_in.ntor
    if (mpol != surface_in.mpol) or (ntor != surface_in.ntor):
        surface_in = change_surface_resolution(surface_in, mpol=mpol, ntor=ntor)

    nphi = 2*ntor + 1
    ntheta = 2*mpol + 1
    surface = change_surface_range(surface_in, surf_range=surf_range, nphi=nphi, ntheta=ntheta)
    return surface

def rebuild_surface(
    surface_in: SurfaceRZFourier | SurfaceXYZTensorFourier | str,
    *,
    surf_range: str | None = None,
    mpol: int | None = None,
    ntor: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
) -> SurfaceXYZTensorFourier:
    if (
        (surf_range is None)
        and (mpol is None)
        and (ntor is None)
        and (nphi is None)
        and (ntheta is None)
    ):
        return surface_in
    surface = convert_rz_to_xyztensor(surface_in)

    nfp = surface.nfp
    phimax = surface.quadpoints_phi.max()
    iphi = np.abs(np.array([1/nfp/2, 1/nfp, 1]) - phimax).argmin()
    surf_range_in = ["half period", "field period", "full torus"]
    surf_range_in = surf_range_in[iphi]

    if (
        ((surf_range is not None) and (surf_range != surf_range_in))
        or ((nphi is not None) and (nphi != surface.quadpoints_phi.size))
        or ((ntheta is not None) and (ntheta != surface.quadpoints_theta.size))
    ):
        surface = change_surface_range(
            surface,
            surf_range=surf_range or surf_range_in,
            nphi=nphi or surface.quadpoints_phi.size,
            ntheta=ntheta or surface.quadpoints_theta.size,
        )
    
    if (
        ((mpol is not None) and (mpol != surface.mpol))
        or ((ntor is not None) and (ntor != surface.ntor))
    ):
        surface = change_surface_resolution(
            surface,
            mpol=mpol or surface.mpol,
            ntor=ntor or surface.ntor,
        )
    return surface
