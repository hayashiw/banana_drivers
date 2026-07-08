import numpy as np
import yaml

from simsopt.field import (
    Coil,
    Current,
    RegularizedCoil,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
    coils_via_symmetries,
    regularization_circ,
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)

from .finitebuild import create_cws_multifilament_grid
from ..paths import BANANA_DOFS_INIT_FILE
from ..hardware import (
    hbt_tf,
    hbt_banana_fb,
    hbt_banana_ws,
    hbt_vf,
)

DEFAULT_BANANA_ORDER = 4
DEFAULT_QPTS_PER_ORDER = 96
PROXY_NQPTS = 128
VF_NQPTS = 128

def generate_tf_coils(
    tf_current_ka: float,
    fix_current: bool = True
) -> list[Coil]:
    tf_curves = create_equally_spaced_curves(
        hbt_tf.n_coils, hbt_tf.nfp, hbt_tf.stellsym,
        R0=hbt_tf.major_radius, R1=hbt_tf.minor_radius, order=hbt_tf.order
    )
    for tf_curve in tf_curves: tf_curve.fix_all()

    tf_current = ScaledCurrent(Current(1.0), tf_current_ka*1e3)
    if fix_current: tf_current.fix_all()
    tf_currents = [tf_current] * hbt_tf.n_coils

    tf_coils = [
        Coil(curve, current) for curve, current in zip(tf_curves, tf_currents)
    ]
    return tf_coils

def read_banana_dofs(dofs_file: str) -> dict:
    with open(dofs_file, "r") as f:
        dofs = yaml.safe_load(f)
    order = 0
    for key, val in dofs.items():
        n = int(key.split("(")[-1].split(")")[0])
        order = max(order, n)
    return dofs, order

def extract_banana_dofs(curve: CurveCWSFourierCPP) -> dict[str, float]:
    dofs = {}
    for dof_name in curve.local_dof_names:
        dofs[dof_name] = curve.get(dof_name)
    return dofs

def extract_winding_surface(curve: CurveCWSFourierCPP, ndigits=10, Z0=0.0) -> tuple[float, float]:
    x, y, z = curve.gamma().T
    r = np.sqrt(x**2 + y**2)

    A = np.column_stack([r, np.ones_like(r)])
    b = r**2 + (z - Z0)**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    major_radius = c[0] / 2.0
    minor_radius = np.sqrt(c[1] + major_radius**2)

    major_radius = round(major_radius, ndigits)
    minor_radius = round(minor_radius, ndigits)
    
    return major_radius, minor_radius

def generate_banana_coils(
    banana_current_ka: float,
    *,
    fix_current: bool = True,
    order: int = DEFAULT_BANANA_ORDER,
    dofs: dict[str, float] | None = None,
    qpts_per_order: int = DEFAULT_QPTS_PER_ORDER,
    major_radius: float = hbt_banana_ws.major_radius,
    minor_radius: float = hbt_banana_ws.minor_radius,
    finitebuild: bool | None = None,
    regularized: bool | None = None,
) -> list[Coil]:
    winding_surface = SurfaceRZFourier(
        nfp=hbt_banana_ws.nfp, stellsym=hbt_banana_ws.stellsym
    )
    winding_surface.set_rc(0, 0, major_radius)
    winding_surface.set_rc(1, 0, minor_radius)
    winding_surface.set_zs(1, 0, minor_radius)

    nqpts = qpts_per_order * order
    banana_curve = CurveCWSFourierCPP(
        np.linspace(0, 1, nqpts, endpoint=False),
        order=order,
        surf=winding_surface
    )

    # Here `min_order` refers to the minimum necessary Fourier order for direct copying of DOFs
    # In reality `min_order` is the maximum Fourier order present in the DOFs file
    if dofs is None:
        dofs, min_order = read_banana_dofs(BANANA_DOFS_INIT_FILE)
    else:
        min_order = 0
        for key, val in dofs.items():
            n = int(key.split("(")[-1].split(")")[0])
            min_order = max(min_order, n)
    if order < min_order:
        fit_curve = CurveCWSFourierCPP(
            np.linspace(0, 1, nqpts, endpoint=False),
            order=min_order,
            surf=winding_surface
        )
        for key, val in dofs.items():
            fit_curve.set(key, val)
        banana_curve.least_squares_fit(fit_curve.gamma())
    else:
        for key, val in dofs.items():
            banana_curve.set(key, val)

    banana_current = ScaledCurrent(Current(1.0), banana_current_ka*1e3)
    if fix_current: banana_current.fix_all()

    regularizations = [regularization_circ(hbt_banana_fb.wire_radius)] if regularized else None
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [banana_current],
        hbt_banana_ws.nfp,
        hbt_banana_ws.stellsym,
        regularizations=regularizations
    )

    if finitebuild:
        nfil = hbt_banana_fb.numfilaments
        base_coil = banana_coils[0]
        base_curves = [base_coil.curve]
        banana_current = base_coil.current.get_value()
        total_current = ScaledCurrent(Current(1.0), banana_current)
        base_currents = [total_current/nfil]

        base_curves_finite_build = sum([
            create_cws_multifilament_grid(
                c, hbt_banana_ws.major_radius, hbt_banana_fb.numfilaments_n, hbt_banana_fb.numfilaments_b, 
                hbt_banana_fb.offset_from_horizontal, hbt_banana_fb.offset_from_vertical, hbt_banana_fb.horizontal_spacing,
                rotation_order=0)
            for c in base_curves], [])
        base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])

        curves_fb = apply_symmetries_to_curves(
            base_curves_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)
        currents_fb = apply_symmetries_to_currents(
            base_currents_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)

        if regularized:
            reg = regularization_circ(hbt_banana_fb.wire_radius)
            banana_coils_finitebuild = [RegularizedCoil(c, curr, reg) for (c, curr) in zip(curves_fb, currents_fb)]
        else:
            banana_coils_finitebuild = [Coil(c, curr) for c, curr in zip(curves_fb, currents_fb)]

        return banana_coils_finitebuild
    else:
        return banana_coils

def generate_proxy_coils(
    proxy_current_ka: float,
    rz: tuple[float, float]
) -> list[Coil]:
    R, Z = rz
    nqpts = PROXY_NQPTS
    proxy_curve = CurveXYZFourier(nqpts, 1)
    proxy_curve.set('xc(1)', R)
    proxy_curve.set('ys(1)', R)
    proxy_curve.set('zc(0)', Z)
    proxy_curve.fix_all()

    proxy_current = ScaledCurrent(Current(1.0), proxy_current_ka*1e3)
    proxy_current.fix_all()

    proxy_coils = [Coil(proxy_curve, proxy_current)]
    return proxy_coils

def generate_vf_coils(
    vf_current_ka: float,
    fix_current: bool = True
) -> list[Coil]:
    nqpts = VF_NQPTS
    vf_coils = []
    for R, Z, sign in hbt_vf:
        curve = CurveXYZFourier(nqpts, order=1)
        curve.set("xc(1)", R)
        curve.set("ys(1)", R)
        curve.set("zc(0)", Z)
        current = ScaledCurrent(Current(1.0), sign * vf_current_ka*1e3) # independent VF currents
        curve.fix_all()
        if fix_current: current.fix_all()
        vf_coils.append(Coil(curve, current))
    return vf_coils
