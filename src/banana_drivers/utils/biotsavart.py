import numpy as np

from simsopt._core import load
from simsopt.field import BiotSavart

from .coils import (
    generate_tf_coils,
    generate_banana_coils,
    generate_proxy_coils,
    generate_vf_coils,
    extract_banana_dofs,
    extract_winding_surface,
)
from ..hardware import (
    hbt_shell, hbt_banana_ws, VF_TO_PROXY_CURRENT_RATIO,
    TF_IDX, BANANA_IDX, PROXY_IDX, VF_IDX,
    N_TF, N_BANANA, N_PROXY, N_VF, N_COILS
)

DEFAULT_FROM_WS = True
DEFAULT_PROXY_R = hbt_banana_ws.major_radius if DEFAULT_FROM_WS else hbt_shell.major_radius
DEFAULT_PROXY_RZ = (DEFAULT_PROXY_R, 0.0)

def build_biotsavart(
    tf_current_ka: float,
    banana_current_ka: float,
    proxy_current_ka : float,
    proxy_rz : tuple[float, float],
    vf_current_ka: float,
    *,
    tf_fix_current: bool = True,
    banana_fix_current: bool = True,
    banana_dofs: dict[str, float] | None = None,
    banana_order: int | None = None,
    banana_qpts_per_order: int | None = None,
    banana_ws_major_radius: float | None = None,
    banana_ws_minor_radius: float | None = None,
    vf_fix_current: bool = True,
    finitebuild: bool | None = None,
    regularized: bool | None = None,
) -> BiotSavart:
    tf_coils = generate_tf_coils(tf_current_ka, tf_fix_current)

    banana_kwargs = dict(fix_current=banana_fix_current)
    if banana_order is not None: banana_kwargs["order"] = banana_order
    if banana_dofs is not None: banana_kwargs["dofs"] = banana_dofs
    if banana_qpts_per_order is not None: banana_kwargs["qpts_per_order"] = banana_qpts_per_order
    if banana_ws_major_radius is not None: banana_kwargs["major_radius"] = banana_ws_major_radius
    if banana_ws_minor_radius is not None: banana_kwargs["minor_radius"] = banana_ws_minor_radius
    if finitebuild is not None: banana_kwargs["finitebuild"] = finitebuild
    if regularized is not None: banana_kwargs["regularized"] = regularized
    banana_coils = generate_banana_coils(banana_current_ka, **banana_kwargs)

    proxy_coils = generate_proxy_coils(proxy_current_ka, proxy_rz)

    vf_coils = generate_vf_coils(vf_current_ka, vf_fix_current)

    coils = tf_coils + banana_coils + proxy_coils + vf_coils
    return BiotSavart(coils)

def rebuild_biotsavart(
    biotsavart: str | BiotSavart,
    *,
    tf_current_ka: float | None = None,
    tf_fix_current: bool | None = None,
    banana_current_ka: float | None = None,
    banana_fix_current: bool | None = None,
    banana_dofs: dict[str, float] | None = None,
    banana_order: int | None = None,
    banana_qpts_per_order: int | None = None,
    banana_ws_major_radius: float | None = None,
    banana_ws_minor_radius: float | None = None,
    proxy_current_ka : float | None = None,
    proxy_rz : tuple[float, float] | None = None,
    vf_current_ka: float | None = None,
    vf_fix_current: bool | None = None,
    finitebuild: bool | None = None,
    regularized: bool | None = None,
) -> BiotSavart:
    if isinstance(biotsavart, str):
        biotsavart = load(biotsavart)

    coils = biotsavart.coils
    ncoils = len(coils)
    err_msg = \
        f"Number of coils mismatch: " \
        f"expected TF ({N_TF}) + BANANA ({N_BANANA}) " \
        f"+ PROXY ({N_PROXY}) + VF ({N_VF}) = {N_COILS} or " \
        f"expected TF ({N_TF}) + BANANA ({N_BANANA}) = {N_TF + N_BANANA}, " \
        f"got {ncoils}"    
    assert (ncoils == N_COILS) or (ncoils == (N_TF + N_BANANA)), err_msg

    add_proxy_vf = ncoils == (N_TF + N_BANANA)

    tf_coil = coils[TF_IDX]
    if tf_current_ka is None: tf_current_ka = tf_coil.current.get_value()/1e3
    if tf_fix_current is None: tf_fix_current = (len(tf_coil.current.x) == 0)

    banana_coil = coils[BANANA_IDX]
    if banana_current_ka is None: banana_current_ka = banana_coil.current.get_value()/1e3
    if banana_fix_current is None: banana_fix_current = (len(banana_coil.current.x) == 0)
    if banana_order is None: banana_order = banana_coil.curve.order
    if banana_dofs is None: banana_dofs = extract_banana_dofs(banana_coil.curve)
    if banana_qpts_per_order is None: banana_qpts_per_order = banana_coil.curve.quadpoints.size // banana_order

    if banana_ws_major_radius is None or banana_ws_minor_radius is None:
        major_radius, minor_radius = extract_winding_surface(banana_coil.curve)
        if banana_ws_major_radius is None: banana_ws_major_radius = major_radius
        if banana_ws_minor_radius is None: banana_ws_minor_radius = minor_radius

    # If proxy current is passed in to rebuild, adjust VF coils to be unfixed with initial current 1/6.5*proxy_current
    if add_proxy_vf:
        if vf_current_ka is None:
            if proxy_current_ka is not None:
                vf_current_ka = proxy_current_ka*VF_TO_PROXY_CURRENT_RATIO
            else:
                vf_current_ka = 0.0
        if vf_fix_current is None:
            if (proxy_current_ka is not None) and (proxy_current_ka != 0.0):
                vf_fix_current = False
            else:
                vf_fix_current = True

        if proxy_current_ka is None: proxy_current_ka = 0.0
        if proxy_rz is None: proxy_rz = DEFAULT_PROXY_RZ
    else:
        vf_coil = coils[VF_IDX]
        if vf_current_ka is None:
            if proxy_current_ka is not None:
                vf_current_ka = proxy_current_ka*VF_TO_PROXY_CURRENT_RATIO
            else:
                vf_current_ka = vf_coil.current.get_value()/1e3
        if vf_fix_current is None:
            if (proxy_current_ka is not None) and (proxy_current_ka != 0.0):
                vf_fix_current = False
            else:
                vf_fix_current = (len(vf_coil.current.x) == 0)

        proxy_coil = coils[PROXY_IDX]
        if proxy_current_ka is None: proxy_current_ka = proxy_coil.current.get_value()/1e3
        if proxy_rz is None:
            x, y, z = proxy_coil.curve.gamma().T
            r = np.sqrt(x**2 + y**2)
            proxy_R = r.mean()
            proxy_Z = z.mean()
            proxy_rz = (proxy_R, proxy_Z)

    return build_biotsavart(
        tf_current_ka=tf_current_ka,
        banana_current_ka=banana_current_ka,
        proxy_current_ka=proxy_current_ka,
        proxy_rz=proxy_rz,
        vf_current_ka=vf_current_ka,
        tf_fix_current=tf_fix_current,
        banana_fix_current=banana_fix_current,
        banana_order=banana_order,
        banana_dofs=banana_dofs,
        banana_qpts_per_order=banana_qpts_per_order,
        banana_ws_major_radius=banana_ws_major_radius,
        banana_ws_minor_radius=banana_ws_minor_radius,
        vf_fix_current=vf_fix_current,
        finitebuild=finitebuild,
        regularized=regularized,
    )

