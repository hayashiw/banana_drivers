import numpy as np
import plotly.graph_objects as go

from banana_drivers.hardware import (
    hbt_banana_fb,
    hbt_banana_ws,
    hbt_shell,
    hbt_vv,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
    N_TF,
    N_BANANA,
    N_PROXY,
    N_VF,
)

FONTSIZE = 12

def plot_banana_coil_projections(fig, ax, biotsavart, finitebuild=False):
    nfil = hbt_banana_fb.numfilaments if finitebuild else 1
    lw = 1.5/nfil
    for icoil in range(BANANA_IDX, BANANA_IDX + N_BANANA*nfil):
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
        ax.plot(phi_proj, theta_plot, c="k", lw=lw)

def plot_modB(fig, ax, biotsavart, surface, finitebuild=False):
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
    plot_banana_coil_projections(fig, ax, biotsavart, finitebuild=finitebuild)

def plot_Bdotn(fig, ax, biotsavart, surface, finitebuild=False):
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
    plot_banana_coil_projections(fig, ax, biotsavart, finitebuild=finitebuild)

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

def plotly_coils(biotsavart, fig=None, width=6):
    if fig is None:
        fig = go.Figure()

    coils = biotsavart.coils
    tf_coils = coils[TF_IDX:TF_IDX+N_TF]
    banana_coils = coils[BANANA_IDX:BANANA_IDX+N_BANANA]
    proxy_coils = coils[PROXY_IDX:PROXY_IDX+N_PROXY]
    vf_coils = coils[VF_IDX:VF_IDX+N_VF]

    for coilset, color in [
        (tf_coils, "black"),
        (banana_coils, "red"),
        (proxy_coils, "blue"),
        (vf_coils, "gray"),
    ]:
        for coil in coilset:
            gamma = coil.curve.gamma()
            x, y, z = np.append(gamma, gamma[:1], axis=0).T
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                showlegend=False,
                line=dict(width=width, color=color)
            ))
    return fig

def plotly_surface(surface, fig=None, biotsavart=None, surfacecolors="Bdotn_norm", colorscale="Viridis"):
    if fig is None:
        fig = go.Figure()

    if biotsavart is not None:
        assert surfacecolors in ["modB", "Bdotn_norm"], f"Expected surfacecolors in ['modB', 'Bdotn_norm'], got {surfacecolors}"
        biotsavart.set_points(surface.gamma().reshape(-1, 3))
        B = biotsavart.B().reshape(surface.gamma().shape)
        modB = np.linalg.norm(B, axis=-1)
        if surfacecolors == "modB":
            surfacecolors = modB
        elif surfacecolors == "Bdotn_norm":
            surfacecolors = np.sum(B * surface.unitnormal(), axis=-1) / modB

    x, y, z = surface.gamma().T
    if surfacecolors is None:
        surfacecolors = np.sqrt(x**2 + y**2)
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolors,
        colorscale=colorscale
    ))
    return fig
