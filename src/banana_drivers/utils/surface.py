import os
from numpy import linspace

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier

_surface_classes = (SurfaceRZFourier, SurfaceXYZTensorFourier)

DEFAULT_NPHI = 128
DEFAULT_NTHETA = 127

def build_surface(
    surf_file: str,
    s: float = 1.0,
    scale: float | None = None,
    surf_range: str = "field period",
    nphi: int = DEFAULT_NPHI,
    ntheta: int = DEFAULT_NTHETA
) -> SurfaceRZFourier | SurfaceXYZTensorFourier:
    ext = os.path.splitext(surf_file)[1]

    if ext == ".json":
        surface = load(surf_file)
        if not isinstance(surface, _surface_classes):
            raise TypeError(f"Expected surface of type {_surface_classes}, got {type(surface)}")
    elif ext == ".nc":
        surface = SurfaceRZFourier.from_wout(surf_file, s=s)
        if scale is not None:
            surface.set_dofs(surface.get_dofs() * scale / surface.major_radius())
    else:
        raise ValueError(f"Unsupported surface file extension: {ext}")
    
    surface = resize_surface(surface, surf_range, nphi=nphi, ntheta=ntheta)

    return surface

def resize_surface(
    init_surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    surf_range: str,
    mpol: int | None = None,
    ntor: int | None = None,
    nphi: int = DEFAULT_NPHI,
    ntheta: int = DEFAULT_NTHETA
) -> SurfaceRZFourier | SurfaceXYZTensorFourier:
    if not isinstance(init_surface, _surface_classes):
        raise TypeError(f"Expected surface of type {_surface_classes}, got {type(init_surface)}")
    
    nfp = init_surface.nfp
    if surf_range == "field period":
        phimax = 1 / nfp
    elif surf_range == "full torus":
        phimax = 1
    elif surf_range == "half period":
        phimax = 1 / nfp / 2
    else:
        raise ValueError(f"Unsupported range: {surf_range}")
    
    qpts_phi = linspace(0, phimax, nphi, endpoint=False)
    qpts_theta = linspace(0, 1, ntheta, endpoint=False)

    if mpol is None: mpol = init_surface.mpol
    if ntor is None: ntor = init_surface.ntor
    
    surface_func = type(init_surface)
    surface = surface_func(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=init_surface.stellsym,
        quadpoints_phi=qpts_phi,
        quadpoints_theta=qpts_theta,
    )
    if (mpol, ntor) == (init_surface.mpol, init_surface.ntor):
        surface.set_dofs(init_surface.get_dofs())
    else:
        if (mpol >= init_surface.mpol) and (ntor >= init_surface.ntor):
            for key in init_surface.dof_names:
                surface.set(key, init_surface.get(key))
        else:
            surface.least_squares_fit(init_surface.gamma())

    return surface