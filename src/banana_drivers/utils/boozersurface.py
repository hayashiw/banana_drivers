from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface, SurfaceXYZTensorFourier, Volume

from .surface import resize_surface
from ..hardware import BANANA_IDX, PROXY_IDX

DEFAULT_CONSTRAINT_WEIGHT = 1e3

from numpy import pi
MU0 = pi * 4e-7

def build_boozersurface(
    biotsavart: BiotSavart,
    init_surf: Surface,
    constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
    plasma_current: float = 0.0,
) -> BoozerSurface:
    if isinstance(init_surf, SurfaceXYZTensorFourier):
        surface = init_surf
    else:
        surface = SurfaceXYZTensorFourier(
            mpol=init_surf.mpol,
            ntor=init_surf.ntor,
            nfp=init_surf.nfp,
            stellsym=init_surf.stellsym,
            quadpoints_phi=init_surf.quadpoints_phi,
            quadpoints_theta=init_surf.quadpoints_theta,
        )
        surface.least_squares_fit(init_surf.gamma())
    targetlabel = surface.volume()

    label = Volume(surface)
    boozersurface = BoozerSurface(
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=constraint_weight,
        I=plasma_current,
        options=dict(
            verbose=True,
        )
    )
    return boozersurface

def load_boozersurface_from_biotsavart(
    init_biotsavart: str | BiotSavart,
    init_surface: str | Surface,
    mpol: int | None = None,
    ntor: int | None = None,
    banana_order: int | None = None,
    constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
) -> BoozerSurface:
    if isinstance(init_biotsavart, str):
        init_biotsavart = load(init_biotsavart)
    if isinstance(init_surface, str):
        init_surface = load(init_surface)

    rebuild_surface = False
    if mpol is None:
        mpol = init_surface.mpol
        rebuild_surface = True
    if ntor is None:
        ntor = init_surface.ntor
        rebuild_surface = True
    if rebuild_surface:
        surface = resize_surface(init_surface, "field period", mpol=mpol, ntor=ntor)
    else:
        surface = init_surface

    if banana_order is None:
        # banana_order = init_biotsavart.coils[BANANA_IDX].curve.order
        pass

    biotsavart = init_biotsavart
    proxy_coil = biotsavart.coils[PROXY_IDX]
    proxy_current = proxy_coil.current.get_value()
    I = proxy_current * MU0

    return build_boozersurface(
        biotsavart, surface, constraint_weight=constraint_weight, plasma_current=I
    )

def load_boozersurface_from_file(
    boozersurface: str | BoozerSurface,
    mpol: int | None = None,
    ntor: int | None = None,
    banana_order: int | None = None,
    constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
) -> BoozerSurface:
    if isinstance(boozersurface, str):
        boozersurface = load(boozersurface)

    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    return load_boozersurface_from_biotsavart(
        biotsavart, surface, mpol=mpol, ntor=ntor,
        banana_order=banana_order, constraint_weight=constraint_weight
    )