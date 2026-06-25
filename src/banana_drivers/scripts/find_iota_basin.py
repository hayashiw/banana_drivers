import argparse
import numpy as np

from scipy.optimize import minimize

from simsopt._core import load
from simsopt.geo import boozer_surface_residual

from ..hardware import TF_IDX, PROXY_IDX, VF_IDX, N_TF, N_VF
from ..utils.constants import MU0

DEFAULT_INIT_IOTA = 1.0
DEFAULT_MAXITER = 1500
DEFAULT_TOL = 1e-10

def build_parser():
    parser = argparse.ArgumentParser(description="Characterize the iota basin for a given BoozerSurface JSON file by minimizing the Boozer residual, keeping the surface DOFs constant.")
    parser.add_argument("boozersurface_file", help="Path to the BoozerSurface JSON file")
    parser.add_argument("--init-iota", type=float, default=DEFAULT_INIT_IOTA, help=f"Initial guess for iota. Default: {DEFAULT_INIT_IOTA}.")
    parser.add_argument("--sign-g", type=int, choices=[-1, 1], default=-1, help="Sign of G. Choices are -1 or 1. Default: -1.")
    parser.add_argument("--dof-g", action="store_true", help="If set, also treats G as a DOF and finds the TF current that minimizes the Boozer residual. Default: False.")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER, help=f"Maximum number of iterations for the minimization. Default: {DEFAULT_MAXITER}.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help=f"Tolerance for the minimization. Default: {DEFAULT_TOL}.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = args.boozersurface_file

    print(f"# Input parameters:", flush=True)
    for key, val in vars(args).items():
        print(f"#     {key}: {val}", flush=True)
    print("#", flush=True)

    boozersurface = load(boozersurface_file)
    biotsavart = boozersurface.biotsavart
    surface = boozersurface.surface

    n_sdofs = surface.x.size

    coils = biotsavart.coils
    tf_coils = coils[TF_IDX:TF_IDX+N_TF]
    proxy_coil = coils[PROXY_IDX]
    vf_coils = coils[VF_IDX:VF_IDX+N_VF]
    tf_curr_tot = sum(abs(coil.current.get_value()) for coil in tf_coils)
    proxy_curr_tot = proxy_coil.current.get_value()
    sign_g = args.sign_g
    G = sign_g * tf_curr_tot * MU0
    I = proxy_curr_tot * MU0
    print(f"# TF coil current:  {tf_coils[0].current.get_value()/1e3:.3f} kA", flush=True)
    print(f"# Ip proxy current: {proxy_curr_tot/1e3:.3f} kA", flush=True)
    print(f"# VF coil current:  {vf_coils[0].current.get_value()/1e3:.3f} kA", flush=True)

    iota = args.init_iota
    r_vec, Jac_full = boozer_surface_residual(surface, iota, G, biotsavart, derivatives=1, weight_inv_modB=True, I=I)
    r_max = np.linalg.norm(r_vec.reshape(surface.gamma().shape), axis=-1).max()
    r_obj = 0.5 * np.sum(r_vec**2)
    grad_iota = r_vec @ Jac_full[:, n_sdofs]
    grad = [grad_iota]
    if args.dof_g:
        grad_g = r_vec @ Jac_full[:, n_sdofs+1]
        grad.append(grad_g)
    print(f"# Initial Boozer residual: {r_max:.5e}", flush=True)
    print(f"# Initial penalty value: {r_obj:.5e}", flush=True)
    print(f"# Initial iota gradient: {grad_iota:.5e}", flush=True)
    if args.dof_g:
        print(f"# Initial G gradient: {grad_g:.5e}", flush=True)
    line = f"iters,evals,iota,G,max(|r|),J_r,grad_iota"
    if args.dof_g: line += ",grad_G"
    print(line, flush=True)

    x0 = [iota]
    if args.dof_g: x0.append(G)

    tracker = dict(
        iters=0,
        evals=0,
        iota=iota,
        G=G,
        r_max=r_max,
        r_obj=r_obj,
        grad_iota=grad_iota,
    )
    if args.dof_g: tracker["grad_G"] = grad_g

    def log_row():
        line = f"{tracker['iters']},{tracker['evals']},{tracker['iota']},{tracker['G']},{tracker['r_max']:.5e},{tracker['r_obj']:.5e},{tracker['grad_iota']:.5e}"
        if args.dof_g: line += f",{tracker['grad_G']:.5e}"
        print(line, flush=True)
    log_row()

    def func(x):
        tracker["evals"] += 1
        iota = x[0]
        G_step = x[1] if args.dof_g else G
        r_vec, Jac_full = boozer_surface_residual(surface, iota, G_step, biotsavart, derivatives=1, weight_inv_modB=True, I=I)
        r_max = np.linalg.norm(r_vec.reshape(surface.gamma().shape), axis=-1).max()
        r_obj = 0.5 * np.sum(r_vec**2)
        grad_iota = r_vec @ Jac_full[:, n_sdofs]
        grad = [grad_iota]
        if args.dof_g:
            grad_g = r_vec @ Jac_full[:, n_sdofs+1]
            grad.append(grad_g)
        tracker["iota"] = iota
        tracker["G"] = G_step
        tracker["r_max"] = r_max
        tracker["r_obj"] = r_obj
        tracker["grad_iota"] = grad_iota
        if args.dof_g: tracker["grad_G"] = grad_g
        log_row()
        return r_obj, np.array(grad)
    
    def callback(x):
        tracker["iters"] += 1
        tracker["evals"] = 0
        log_row()

    res = minimize(
        func,
        x0,
        jac=True,
        tol=args.tol,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": args.maxiter}
    )
    print(f"# {res.message}", flush=True)
    print(f"# Final iota: {tracker['iota']}", flush=True)
    print(f"# Final G: {tracker['G']}", flush=True)
    print(f"# Final Boozer residual: {tracker['r_max']:.5e}", flush=True)
    print(f"# Final penalty value: {tracker['r_obj']:.5e}", flush=True)
    print(f"# Final iota gradient: {tracker['grad_iota']:.5e}", flush=True)
    if args.dof_g:
        print(f"# Final G gradient: {tracker['grad_G']:.5e}", flush=True)
        G = tracker['G']
        tf_current_ka = G / (MU0 * len(tf_coils)) / 1e3
        print(f"# Final TF current: {tf_current_ka:.3f} kA", flush=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
