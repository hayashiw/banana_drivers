import numpy as np

def calculate_qs_error(boozersurface):
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface

    nfp  = surface.nfp
    mpol = surface.mpol
    ntor = surface.ntor

    xn = (np.arange(ntor+1, dtype=float)*nfp).tolist()
    xm = np.zeros(ntor+1).tolist()
    for m in range(1, mpol+1):
        for n in range(-ntor, ntor+1):
            xn.append(float(n*nfp))
            xm.append(float(m))
    xn = np.array(xn)
    xm = np.array(xm)

    gamma = surface.gamma().copy()
    biotsavart.set_points(gamma.reshape(-1, 3))
    B = biotsavart.B().reshape(gamma.shape)
    modB = np.linalg.norm(B, axis=-1)

    theta = surface.quadpoints_theta * 2*np.pi
    phi = surface.quadpoints_phi * 2*np.pi
    pol, tor = np.meshgrid(theta, phi)
    pol_m = xm[:, None] @ pol.flatten()[None, :]
    tor_n = xn[:, None] @ tor.flatten()[None, :]
    angle = pol_m - tor_n

    bmnc = np.sum(modB.flatten()[None, :] * np.cos(angle), axis=-1) / np.sum(np.cos(angle)**2, axis=-1)

    # For QA, the non-QA coefficients are n != 0
    non_qs_bmnc = bmnc[xn != 0]
    qs_error = np.sum(non_qs_bmnc**2) / np.sum(bmnc**2)

    return qs_error


