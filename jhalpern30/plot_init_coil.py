"""Plot the initial banana coil DoFs used by stage2.py in (phi, theta)
of the winding surface. Standalone — run directly with the sims_banana_env
environment active; no SLURM, no dependence on scan outputs.

Optionally overlay the banana coils from an existing biotsavart_opt.json or
bsurf_{init,opt}.json (passed via --overlay) so the initial shape can be
compared side-by-side against a converged run.

See README.md "Banana coil initialization" for the unit convention and
parametric derivation. Edit the DOFS dict below to try new shapes before
committing a change to stage2.py.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from simsopt._core import load
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, CurveCWSFourierCPP, CurveLength, CurveCurveDistance

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'new_objectives'))
from ellipse_width import ProjectedEllipseWidth

# Winding surface — must match stage2.py
SURF_COILS_NFP = 5
SURF_COILS_R0 = 0.976
SURF_COILS_A  = 0.210

# Banana curve params — must match stage2.py
NUM_QUADPOINTS = 128 * 2
ORDER = 2

# Banana coil indices in the combined (tf + banana + proxy + vf) coil list;
# 20 is the pre-symmetry representative, 21-29 are its nfp/stellsym copies.
BANANA_COIL_SLICE = slice(20, 30)

# Target LCFS center for the xyz -> (phi, theta) projection of overlay coils.
# Matches post_process_plots.py.
TARGET_LCFS_MAJOR_R = 0.92

# DoFs (fractions of 2*pi). Mirror the values set in stage2.py.
#
# The curve is parametrized in t in [0, 1) as
#   phi(t)   = phic(0) + phic(1)   cos(2 pi t) + phis(1)   sin(2 pi t) + ...
#   theta(t) = thetac(0) + thetac(1) cos(2 pi t) + thetas(1) sin(2 pi t) + ...
# with every coefficient multiplied by 2*pi to become radians on the winding surface.
# Each DoF moves the curve in (phi, theta) space as follows:
#
#   phic(0)   : toroidal center of the coil. Shifts the whole curve left/right in phi.
#   thetac(0) : poloidal center. Shifts up/down in theta. 0.5 -> theta = pi (inboard).
#   phic(1)   : x-component of the semi-axis vector at t = 0
#               (i.e. the direction from center to the "3 o'clock" point of the ellipse).
#               Negative values tilt that axis toward -phi.
#   thetac(1) : y-component of the same t = 0 semi-axis. Combined with phic(1) it
#               sets one of the two principal axes of the ellipse.
#   phis(1)   : x-component of the semi-axis vector at t = 1/4
#               (the "12 o'clock" point). With phic(1)/thetac(1) this fixes the
#               second principal axis, hence the overall orientation + aspect ratio.
#   thetas(1) : y-component of the t = 1/4 semi-axis. Dominates the poloidal sweep
#               for a near-vertical coil.
#
# For a tilted ellipse with long semi-axis a_l at angle alpha from the phi-axis and
# short semi-axis a_s perpendicular to it:
#   (phic(1), thetac(1))  = (-a_s sin alpha, +a_s cos alpha) / (2 pi)     # short axis
#   (phis(1), thetas(1))  = (+a_l cos alpha, +a_l sin alpha) / (2 pi)     # long axis
# Larger alpha -> more vertical (less tilt). Larger a_s -> thicker. Larger a_l ->
# longer poloidal sweep.
#
# Order-2 DoFs (phic(2), thetac(2)) round out the tips: adding w*cos(4*pi*t) in the
# +v_2 direction (long-axis direction) counter-bends the peak at t=1/4, 3/4. The tip
# curvature in parametric space goes from a_l/a_s^2 to (a_l - 4 w)/a_s^2, so w = a_l/8
# halves it. Side effect: short-axis ends shift outward by w in v_2, widening the ends
# slightly (lentiform shape). See README.md "Banana coil initialization".
DOFS = {
    'phic(0)':   0.06,
    'thetac(0)': 0.5,
    'phic(1)':  -0.0221,   # a_s = 0.025, alpha = 62 deg
    'thetac(1)': 0.0117,
    'phis(1)':   0.0423,   # a_l = 0.090
    'thetas(1)': 0.0795,
    'phic(2)':   0.0053,   # (a_l/8) cos(alpha)   with a_l=0.090, alpha=62 deg
    'thetac(2)': 0.0099,   # (a_l/8) sin(alpha)
}
# Tuned (a_s=0.025, a_l=0.090) so the init satisfies all four hardware caps
# at iteration 0 — kappa_max=96.5, cc=0.056, W=0.134, max|theta_in|=32.4.
# History: the original tilted init (a_s=0.022, a_l=0.140) hit kappa_init ~ 185
# (nearly 2x the 100 m^-1 cap), so stage 2 L-BFGS spent all its gradient
# on tip-rounding and landed in pinched local minima under TF<0 (51908233-
# 51908242, 2026-04-22). A pure a_s bump pushed cc < 0.05; trimming a_l
# reclaimed cc margin without reopening the kappa gap. Pair this init with
# BANANA_CURRENT_MODE_S2=fixed, BANANA_I_FIXED_S2=16000 in stage 2 — free
# current collapses to small |I_b| for low |I_p| and re-induces pinching
# (see 51934327 I=-1.0 where I_b settled at +5.3 kA and the coils pinched).

OUT_PNG = 'plot_init_coil.png'
OUT_DOFS = 'banana_dofs.txt'   # consumed by stage2.py


def max_inboard_theta_deg(curve, R_winding=SURF_COILS_R0, Z_winding=0.0):
    """Actual max |theta_in| in degrees measured from the winding-surface
    inboard midplane — same definition PoloidalExtent uses internally."""
    g = curve.gamma()
    R = np.linalg.norm(g[:, :2], axis=-1)
    Z = g[:, 2]
    theta_in = np.arctan2(Z - Z_winding, -(R - R_winding))
    return float(np.abs(theta_in).max() * 180.0 / np.pi)


def retrieve_winding_surface(curve):
    """xyz -> (phi_proj, theta_proj) projection used in post_process_plots.py.

    Returns arrays with toroidal wrap-around handled so the projection lives
    in [0, 2*pi) for theta and a contiguous range for phi.
    """
    gamma = curve.gamma()
    x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]
    r = np.sqrt(x**2 + y**2)

    reff = r - TARGET_LCFS_MAJOR_R
    zeff = z - 0.0
    theta_proj = np.arctan2(zeff, reff) % (2 * np.pi)
    phi_proj = np.arctan2(y, x) % (2 * np.pi)

    if phi_proj.ptp() > 1.5 * np.pi:
        if phi_proj[phi_proj > np.pi].size > phi_proj[phi_proj < np.pi].size:
            phi_proj[phi_proj < np.pi] += 2 * np.pi
        else:
            phi_proj[phi_proj > np.pi] -= 2 * np.pi

    return phi_proj, theta_proj


def load_biotsavart(path):
    """Accept biotsavart_opt.json (returns BiotSavart) or a BoozerSurface
    JSON (bsurf_{init,opt}.json, whose .biotsavart attribute carries the
    field). Returns a BiotSavart object."""
    obj = load(path)
    if hasattr(obj, 'biotsavart'):
        return obj.biotsavart
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--overlay', type=str, default=None,
        help='Path to a biotsavart_opt.json or bsurf_{init,opt}.json whose '
             'banana coils should be overlaid as a reference.',
    )
    parser.add_argument('--out', type=str, default=OUT_PNG)
    args = parser.parse_args()

    surf = SurfaceRZFourier(nfp=SURF_COILS_NFP, stellsym=True)
    surf.set_rc(0, 0, SURF_COILS_R0)
    surf.set_rc(1, 0, SURF_COILS_A)
    surf.set_zs(1, 0, SURF_COILS_A)

    curve = CurveCWSFourierCPP(np.linspace(0, 1, NUM_QUADPOINTS), order=ORDER, surf=surf)
    for name, value in DOFS.items():
        curve.set(name, value)

    # Symmetrize to the full set of 10 banana coils (nfp * 2 stellsym copies) so the
    # plot mirrors what stage2.py actually optimizes over.
    init_coils = coils_via_symmetries([curve], [Current(1e4)], surf.nfp, surf.stellsym)

    period = 2 * np.pi / SURF_COILS_NFP

    def fold_and_plot(p, t, color, lw, alpha, label=None):
        """Fold phi so mean lands in [0, period), then draw at phi - period,
        phi, phi + period. xlim [0, period] clips everything but the central
        copy and the tails of the two neighbors spilling across the boundary."""
        p_folded = p - np.floor(np.mean(p) / period) * period
        for i, offset in enumerate((-period, 0.0, period)):
            ax.plot(p_folded + offset, t, c=color, lw=lw, alpha=alpha,
                    label=label if i == 1 else None)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, constrained_layout=True)

    # Coil length + max curvature. All 10 stellsym+nfp copies are geometrically
    # identical so one representative per set suffices.
    init_length = CurveLength(curve).J()
    init_kappa_max = float(np.max(curve.kappa()))
    init_ccdist = float(CurveCurveDistance([c.curve for c in init_coils], 0.0).shortest_distance())
    init_theta_in_max = max_inboard_theta_deg(curve)
    init_width = ProjectedEllipseWidth(curve, SURF_COILS_R0, SURF_COILS_A).J()
    th_min, th_max = np.inf, 0
    for coil in init_coils:
        p, t = retrieve_winding_surface(coil.curve)
        th_min = min(th_min, t.min())
        th_max = max(th_max, t.max())
    info_lines = [
        f'init L={init_length:.3f} m, max κ={init_kappa_max:.1f} m^-1, cc={init_ccdist:.3f} m, '
        f'Δθ={(th_max-th_min)*180/np.pi:.1f}°, max|θ_in|={init_theta_in_max:.1f}°, width={init_width:.3f} m'
    ]

    if args.overlay is not None:
        bs = load_biotsavart(os.path.abspath(args.overlay))
        rep_label = os.path.basename(os.path.dirname(os.path.abspath(args.overlay))) or os.path.basename(args.overlay)
        opt_rep_curve = bs.coils[BANANA_COIL_SLICE][0].curve
        opt_length = CurveLength(opt_rep_curve).J()
        opt_kappa_max = float(np.max(opt_rep_curve.kappa()))
        opt_ccdist = float(CurveCurveDistance([c.curve for c in bs.coils[BANANA_COIL_SLICE]], 0.0).shortest_distance())
        opt_theta_in_max = max_inboard_theta_deg(opt_rep_curve)
        opt_width = ProjectedEllipseWidth(opt_rep_curve, SURF_COILS_R0, SURF_COILS_A).J()
        th_min, th_max = np.inf, 0
        for icoil, coil in enumerate(bs.coils[BANANA_COIL_SLICE]):
            p, t = retrieve_winding_surface(coil.curve)
            color = 'k' if icoil == 0 else 'gray'
            lw = 1.5 if icoil == 0 else 0.8
            alpha = 1.0 if icoil == 0 else 0.4
            label = f'banana coils ({rep_label})' if icoil == 0 else None
            fold_and_plot(p, t, color, lw, alpha, label)
            th_min = min(th_min, t.min())
            th_max = max(th_max, t.max())
        info_lines.append(
            f'opt ({rep_label}) L={opt_length:.3f} m, max κ={opt_kappa_max:.1f} m^-1, cc={opt_ccdist:.3f} m, '
            f'Δθ={(th_max-th_min)*180/np.pi:.1f}°, max|θ_in|={opt_theta_in_max:.1f}°, width={opt_width:.3f} m'
        )
        

    for icoil, coil in enumerate(init_coils):
        p, t = retrieve_winding_surface(coil.curve)
        label = 'banana coils (initial)' if icoil == 0 else None
        fold_and_plot(p, t, 'tab:blue', 1.5, 0.7, label)

    ax.axhline(np.pi, color='tab:red', ls='--', alpha=0.4, label=r'$\theta=\pi$ (inboard midplane)')
    # Poloidal-extent threshold bands. PoloidalExtent in stage2.py measures |theta_in|
    # from the winding-surface inboard midplane (R_w=0.976), while the plot's theta uses
    # the LCFS center (R=0.92), so these lines are approximate — the geometric extent
    # reported in the info box is the definitive number.
    ax.text(0.01, 0.99, '\n'.join(info_lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))
    ax.set_xlabel(r'$\phi$ [rad]', fontsize=12)
    ax.set_ylabel(r'$\theta$ [rad]', fontsize=12)
    ax.set_xlim(0, period)
    ax.set_ylim(0, 2*np.pi)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Banana coil initial shape')

    fig.savefig(args.out, bbox_inches='tight')
    print(f"Saved {args.out}")

    dofs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_DOFS)
    with open(dofs_path, 'w') as f:
        f.write('# Banana coil CurveCWSFourierCPP DoFs (fractions of 2*pi)\n')
        f.write('# Written by plot_init_coil.py; consumed by stage2.py.\n')
        f.write('# Format: <name> <value>    (whitespace-separated, # for comments)\n')
        for name, value in DOFS.items():
            f.write(f'{name:<10s} {value!r}\n')
    print(f"Saved {dofs_path}")


if __name__ == '__main__':
    main()
