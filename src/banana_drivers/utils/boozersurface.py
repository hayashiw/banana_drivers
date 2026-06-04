from simsopt._core import load
from simsopt.field import BiotSavart, Coil
from simsopt.geo import (
    BoozerSurface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume
)

from ..hardware import TF_IDX, BANANA_IDX, PROXY_IDX, VF_IDX
from .biotsavart import rebuild_biotsavart
from .constants import MU0
from .surface import convert_rz_to_xyztensor, convert_to_boozerexact_surface, rebuild_surface

OPTION = dict(verbose=True)

def build_boozersurface(
    biotsavart: BiotSavart | str,
    surface: SurfaceRZFourier | SurfaceXYZTensorFourier | str,
    constraint_weight: float | None = None,
    targetlabel: float | None = None,
    proxy_coil_from_surface: bool = True,
    proxy_coil: Coil | None = None,
) -> BoozerSurface:
    if isinstance(biotsavart, str):
        biotsavart = load(biotsavart)
    if isinstance(surface, str):
        surface = load(surface)
    if constraint_weight == 0: constraint_weight = None

    if constraint_weight is None:
        surface = convert_to_boozerexact_surface(surface)
    else:
        surface = convert_rz_to_xyztensor(surface)

    if proxy_coil is not None: proxy_coil_from_surface = False
    if proxy_coil_from_surface:
        proxy_rz = (surface.major_radius(), 0.0)
        biotsavart = rebuild_biotsavart(biotsavart, proxy_rz=proxy_rz)

    if proxy_coil is None: proxy_coil = biotsavart.coils[PROXY_IDX]
    current = proxy_coil.current.get_value()
    I = current * MU0

    label = Volume(surface)
    if targetlabel is None: targetlabel = surface.volume()
    boozersurface = BoozerSurface(biotsavart, surface, label, targetlabel, constraint_weight=constraint_weight, options=OPTION, I=I)
    return boozersurface

def rebuild_boozersurface(
    boozersurface: BoozerSurface | str,
    *,
    constraint_weight: float | None = None, # If None, inherits from boozersurface. Use 0 to set BoozerExact.
    targetlabel: float | None = None,
    proxy_coil_from_surface: bool = True,
    tf_current_ka: float | None = None,
    tf_fix_current: bool | None = None,
    banana_current_ka: float | None = None,
    banana_fix_current: bool | None = None,
    banana_dofs: dict[str, float] | None = None,
    banana_order: int | None = None,
    banana_qpts_per_order: int | None = None,
    proxy_current_ka : float | None = None,
    proxy_rz : tuple[float, float] | None = None,
    vf_current_ka: float | None = None,
    vf_fix_current: bool | None = None,
    surf_range: str | None = None,
    mpol: int | None = None,
    ntor: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
) -> BoozerSurface:
    if isinstance(boozersurface, str):
        boozersurface = load(boozersurface)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface
    
    if constraint_weight is None:
        constraint_weight = boozersurface.constraint_weight
        use_boozer_exact = True
    elif constraint_weight == 0:
        constraint_weight = None
        use_boozer_exact = False

    new_mpol = mpol or surface.mpol
    new_ntor = ntor or surface.ntor
    if use_boozer_exact:
        new_ntheta = 2*new_mpol+1
        new_nphi = 2*new_ntor+1
    else:
        new_ntheta = ntheta or surface.ntheta
        new_nphi = nphi or surface.nphi

    new_proxy_rz = (surface.major_radius(), 0.0) if proxy_coil_from_surface else proxy_rz

    surface_kwargs = dict(
        surf_range=surf_range,
        mpol=new_mpol,
        ntor=new_ntor,
        ntheta=new_ntheta,
        nphi=new_nphi,
    )
    new_surface = rebuild_surface(surface, **surface_kwargs)

    biotsavart_kwargs = dict(
        tf_current_ka=tf_current_ka,
        tf_fix_current=tf_fix_current,
        banana_current_ka=banana_current_ka,
        banana_fix_current=banana_fix_current,
        banana_dofs=banana_dofs,
        banana_order=banana_order,
        banana_qpts_per_order=banana_qpts_per_order,
        proxy_current_ka=proxy_current_ka,
        proxy_rz=new_proxy_rz,
        vf_current_ka=vf_current_ka,
        vf_fix_current=vf_fix_current,
    )
    new_biotsavart = rebuild_biotsavart(biotsavart, **biotsavart_kwargs)

    proxy_coil = new_biotsavart.coils[PROXY_IDX]
    current = proxy_coil.current.get_value()
    I = current * MU0

    label = Volume(new_surface)
    if targetlabel is None: targetlabel = new_surface.volume()
    new_boozersurface = BoozerSurface(new_biotsavart, new_surface, label=label, targetlabel=targetlabel, constraint_weight=constraint_weight, options=OPTION, I=I)

    return new_boozersurface
