import numpy as np

from scipy.interpolate import RegularGridInterpolator

def interpolate_target_to_surface(surface, virtualcasing):
    if virtualcasing is None:
        return None

    target_in = virtualcasing.B_external_normal_extended
    
    qpts_phi = surface.quadpoints_phi
    qpts_theta = surface.quadpoints_theta

    nphi_in, ntheta_in = target_in.shape[:2]
    qpts_phi_in = np.linspace(0, 1, nphi_in, endpoint=False)
    qpts_theta_in = np.linspace(0, 1, ntheta_in, endpoint=False)

    dbl_qpts_phi_in = np.concatenate([qpts_phi_in, qpts_phi_in + 1])
    dbl_qpts_theta_in = np.concatenate([qpts_theta_in, qpts_theta_in + 1])
    quad_target_in = np.tile(target_in, (2, 2) + (1,) * (target_in.ndim - 2))

    interpolator = RegularGridInterpolator(
        (dbl_qpts_phi_in, dbl_qpts_theta_in),
        quad_target_in,
        bounds_error=False,
        fill_value=None
    )
    target = interpolator(np.stack(np.meshgrid(qpts_phi, qpts_theta, indexing='ij'), axis=-1))

    return target
