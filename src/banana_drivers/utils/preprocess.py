from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import Surface

from banana_drivers.utils.surface import build_surface, resize_surface
from banana_drivers.utils.coils import build_biotsavart, extract_banana_dofs, read_banana_dofs
from banana_drivers.hardware import (
    N_COILS,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
    DEFAULT_PROXY_RZ,
)

def _current_ka(coil):
    return coil.current.get_value() / 1e3

def _current_is_fixed(coil):
    return len(coil.current.x) == 0

def load_inputs(args):
    if args.boozersurface_file:
        bsurf = load(args.boozersurface_file)
        return bsurf.biotsavart, resize_surface(bsurf.surface, "field period")
    biotsavart = None if getattr(args, "build", False) else load(args.biotsavart_file)
    surface = build_surface(args.surface_file or args.wout_file,
                            s=args.vmec_s, scale=args.scale)
    return biotsavart, surface

def process_args(args) -> tuple[BiotSavart, Surface]:
    init_biot, surface = load_inputs(args)
    biotsavart = build_biotsavart(**process_coil_args(init_biot, args))
    biotsavart.set_points(surface.gamma().reshape(-1, 3))
    return biotsavart, surface

def process_coil_args(init_biot: BiotSavart, args) -> dict:
    overrides = dict(
        tf_current_ka     = args.tf_current,
        tf_fix            = args.tf_fix_current,
        banana_current_ka = args.banana_current,
        banana_fix        = args.banana_fix_current,
        banana_order      = args.banana_order,
        banana_dofs       = read_banana_dofs(args.banana_init_file)
                            if args.banana_init_file else None,
        proxy_current_ka  = args.proxy_current,
        proxy_rz          = tuple(args.proxy_rz) if args.proxy_rz else None,
        vf_current_ka     = args.vf_current,
        vf_fix            = args.vf_fix_current,
    )

    if args.build:
        missing = [k for k, v in overrides.items() if v is None]
        if missing:
            raise ValueError(f"--build requires all coil args; missing: {missing}")
        return overrides
    
    coils = init_biot.coils
    if len(coils) != N_COILS:
        raise ValueError(f"Expected {N_COILS} coils in BiotSavart, got {len(coils)}")
    banana_curve = coils[BANANA_IDX].curve
    inherited = dict(
        tf_current_ka     = _current_ka(coils[TF_IDX]),
        tf_fix            = _current_is_fixed(coils[TF_IDX]),
        banana_current_ka = _current_ka(coils[BANANA_IDX]),
        banana_fix        = _current_is_fixed(coils[BANANA_IDX]),
        banana_order      = banana_curve.order,
        banana_dofs       = extract_banana_dofs(banana_curve),
        proxy_current_ka  = _current_ka(coils[PROXY_IDX]),
        proxy_rz          = DEFAULT_PROXY_RZ,
        vf_current_ka     = _current_ka(coils[VF_IDX]),
        vf_fix            = _current_is_fixed(coils[VF_IDX]),
    )
    return {k: (overrides[k] if overrides[k] is not None else inherited[k])
            for k in inherited}