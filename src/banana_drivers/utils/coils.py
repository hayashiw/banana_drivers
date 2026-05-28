from numpy import linspace, loadtxt, sign

from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)

from banana_drivers.paths import BANANA_INIT_DOFS
from banana_drivers.hardware import (
    hbt_tf,
    hbt_banana_ws,
    hbt_vf,
    hardware_limits,
    DEFAULT_BANANA_ORDER,
    DEFAULT_PROXY_RZ,
)

tf_max_current_sign = 1
tf_max_current_ka = 0
for val in hardware_limits.tf_current_ka_limits:
    if abs(val) > tf_max_current_ka:
        tf_max_current_sign = sign(val)
    tf_max_current_ka = max(tf_max_current_ka, abs(val))
tf_max_current_ka *= tf_max_current_sign

banana_max_current_sign = 1
banana_max_current_ka = 0
for val in hardware_limits.banana_current_ka_limits:
    if abs(val) > banana_max_current_ka:
        banana_max_current_sign = sign(val)
    banana_max_current_ka = max(banana_max_current_ka, abs(val))
banana_max_current_ka *= banana_max_current_sign

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
    dofs = {}
    for key, val in loadtxt(dofs_file, dtype=str, ndmin=2, comments="#"):
        dofs[key] = float(val)
    return dofs

def extract_banana_dofs(curve: CurveCWSFourierCPP) -> dict:
    keys = curve.local_full_dof_names
    vals = curve.local_full_x
    dofs = dict(zip(keys, vals))
    return dofs

def generate_banana_coils(
    banana_current_ka: float,
    banana_order: int,
    dofs: dict[str, float],
    fix_current: bool = True
) -> list[Coil]:
    winding_surface = SurfaceRZFourier(
        nfp=hbt_banana_ws.nfp, stellsym=hbt_banana_ws.stellsym
    )
    winding_surface.set_rc(0, 0, hbt_banana_ws.major_radius)
    winding_surface.set_rc(1, 0, hbt_banana_ws.minor_radius)
    winding_surface.set_zs(1, 0, hbt_banana_ws.minor_radius)

    nqpts = 64 * banana_order
    banana_curve = CurveCWSFourierCPP(
        linspace(0, 1, nqpts, endpoint=False),
        order=banana_order, surf=winding_surface
    )
    for key, val in dofs.items():
        banana_curve.set(key, val)

    banana_current = ScaledCurrent(Current(1.0), banana_current_ka*1e3)
    if fix_current: banana_current.fix_all()

    banana_coils = coils_via_symmetries(
        [banana_curve],
        [banana_current],
        hbt_banana_ws.nfp,
        hbt_banana_ws.stellsym,
    )
    return banana_coils

def generate_proxy_coils(
    proxy_current_ka: float,
    rz: tuple[float, float]
) -> list[Coil]:
    R, Z = rz
    nqpts = 128
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
    nqpts = 128
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

def build_biotsavart(*,
    tf_current_ka: float = tf_max_current_ka,
    tf_fix: bool = True,
    banana_current_ka: float = banana_max_current_ka,
    banana_order: int = DEFAULT_BANANA_ORDER,
    banana_dofs: dict[str, float] | None = None,
    banana_fix: bool = True,
    proxy_current_ka: float = 0.0,
    proxy_rz: tuple[float, float] = DEFAULT_PROXY_RZ,
    vf_current_ka: float = 0.0,
    vf_fix: bool = True
) -> BiotSavart:
    if banana_dofs is None:
        banana_dofs = read_banana_dofs(BANANA_INIT_DOFS)

    coils = []
    coils += generate_tf_coils(tf_current_ka, fix_current=tf_fix)
    coils += generate_banana_coils(banana_current_ka, banana_order, banana_dofs, fix_current=banana_fix)
    coils += generate_proxy_coils(proxy_current_ka, proxy_rz)
    coils += generate_vf_coils(vf_current_ka, fix_current=vf_fix)
    biotsavart = BiotSavart(coils)
    return biotsavart
