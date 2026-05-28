import numpy as np

from banana_drivers.hardware import (
    hbt_vv,
    hbt_banana_ws,
    hbt_shell,
    N_BANANA,
    BANANA_IDX,
)

FONTSIZE = 12

def plot_banana_coil_projections(fig, ax, biotsavart):
    for icoil in range(N_BANANA, N_BANANA + BANANA_IDX):
        gamma = biotsavart.coils[icoil].curve.gamma()
        x, y, z = np.append(gamma, gamma[:1], axis=0).T
        reff = np.sqrt(x**2 + y**2) - hbt_banana_ws.major_radius
        zeff = z - 0.0
        phi_proj = np.arctan2(y, x)
        theta_proj = np.arctan2(zeff, -reff)
        theta_plot = np.pi - theta_proj
        if phi_proj.ptp() > np.pi:
            phi_mid = (phi_proj.max() + phi_proj.min()) / 2
            phi_proj[phi_proj < phi_mid] += 2*np.pi
        ax.plot(phi_proj, theta_plot, c="k")

def plot_modB(fig, ax, biotsavart, surface):
    biotsavart.set_points(surface.gamma().reshape(-1, 3))
    B = biotsavart.B().reshape(surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    phi = surface.quadpoints_phi*2*np.pi
    theta = surface.quadpoints_theta*2*np.pi
    im = ax.contourf(phi, theta, modB.T, levels=21, cmap="viridis")
    _ = fig.colorbar(im, ax=ax)
    ax.set_title(r"$|B|$ [T]", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\phi$ [rad]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\theta$ [rad]", fontsize=FONTSIZE)
    ax.set_box_aspect(1)
    xmin, xmax = phi.min(), phi.max()
    ymin, ymax = theta.min(), theta.max()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plot_banana_coil_projections(fig, ax, biotsavart)

def plot_Bdotn(fig, ax, biotsavart, surface):
    biotsavart.set_points(surface.gamma().reshape(-1, 3))
    B = biotsavart.B().reshape(surface.gamma().shape)
    modB = np.linalg.norm(B, axis=-1)
    nhat = surface.unitnormal()
    Bdotn = np.sum(B * nhat, axis=-1)
    Bdotn_norm = Bdotn / modB
    phi = surface.quadpoints_phi*2*np.pi
    theta = surface.quadpoints_theta*2*np.pi
    im = ax.contourf(phi, theta, Bdotn_norm.T, levels=21, cmap="bwr")
    _ = fig.colorbar(im, ax=ax)
    ax.set_title(r"$B \cdot \hat{n} / |B|$", fontsize=FONTSIZE)
    ax.set_xlabel(r"$\phi$ [rad]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\theta$ [rad]", fontsize=FONTSIZE)
    ax.set_box_aspect(1)
    xmin, xmax = phi.min(), phi.max()
    ymin, ymax = theta.min(), theta.max()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plot_banana_coil_projections(fig, ax, biotsavart)

def plot_cross_sections(fig, ax, surface, nphis=4):
    nfp = surface.nfp
    for iphi in range(nphis):
        phi = iphi / nphis / nfp
        try:
            cs = surface.cross_section(phi)
        except ValueError as e:
            continue
        cs = np.append(cs, cs[:1], axis=0)
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        ax.plot(r, z)

    _theta = np.linspace(0, 2*np.pi, 100)
    for c, ls, lw in [
        (hbt_vv, "-", 2),
        (hbt_banana_ws, "--", 1),
        (hbt_shell, ":", 1),
    ]:
        R = c.major_radius
        a = c.minor_radius
        x = R + a * np.cos(_theta)
        z = a * np.sin(_theta)
        ax.plot(x, z, c="k", ls=ls, lw=lw)

    ax.set_xlabel("R [m]", fontsize=FONTSIZE)
    ax.set_ylabel("Z [m]", fontsize=FONTSIZE)
    ax.set_title("Cross sections", fontsize=FONTSIZE)
    ax.set_box_aspect(1)
    ax.set_aspect("equal")
