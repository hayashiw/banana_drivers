import numpy as np
import os

from simsopt._core import load
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    InterpolatedField,
    MaxRStoppingCriterion, MinRStoppingCriterion,
    MaxZStoppingCriterion, MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
)
from simsopt.geo import (
    BoozerSurface,
    boozer_surface_residual,
    CurveXYZFourier,
    SurfaceXYZTensorFourier,
    Volume
)
from simsopt.objectives import SquaredFlux

from simsoptpp import fieldline_tracing

VACVES_MAJOR_R = 0.976
VACVES_MINOR_R = 0.222

def find_rax(bs, surf):
    nfp = surf.nfp
    cs = surf.cross_section(0)
    cs = np.append(cs, cs[:1], axis=0)
    r = np.linalg.norm(cs[:, :2], axis=-1)
    z = cs[:, 2]
    R0s = np.array([r.mean(), np.median(r), (r.min() + r.max())/2])
    Z0s = np.array([z.mean(), np.median(z), (z.min() + z.max())/2])

    degree = 3
    rmin = VACVES_MAJOR_R - VACVES_MINOR_R
    rmax = VACVES_MAJOR_R + VACVES_MINOR_R
    zmin = -VACVES_MINOR_R
    zmax =  VACVES_MINOR_R
    rrange   = (rmin, rmax, 36)
    phirange = (0, 2*np.pi/nfp, 36)
    zrange   = (0, zmax, 18)
    field = InterpolatedField(
        bs,
        degree,
        rrange,
        phirange,
        zrange,
        extrapolate=True,
        nfp=nfp,
        stellsym=True
    )

    max_transits = 3000
    stopping_criteria = [
        MinRStoppingCriterion(rmin),
        MaxRStoppingCriterion(rmax),
        MinZStoppingCriterion(zmin),
        MaxZStoppingCriterion(zmax),
        ToroidalTransitStoppingCriterion(max_transits, False),
    ]

    tmax = 3000
    tol = 1e-6
    nphis = 4
    phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
    rax_guesses = [[] for iphi in range(nphis)]
    zax_guesses = [[] for iphi in range(nphis)]
    nlines = R0s.size
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.asarray(R0s)
    xyz_inits[:, 2] = np.asarray(Z0s)
    print(f"Finding Rax with tracing")
    for i in range(nlines):
        print(f"{i + 1}/{nlines}", flush=True)
        res_ty, res_phi_hit = fieldline_tracing(
            field,
            xyz_inits[i, :],
            tmax,
            tol,
            phis=phis,
            stopping_criteria=stopping_criteria
        )
        res_phi_hit = np.asarray(res_phi_hit)
        for iphi in range(nphis):
            t, _, x, y, z = res_phi_hit[res_phi_hit[:, 1] == iphi].T
            r = np.sqrt(x**2 + y**2)
            rax_guesses[iphi].append([(r.min()+r.max())/2, r.ptp()])
            zax_guesses[iphi].append([(z.min()+z.max())/2, z.ptp()])
    raxs, zaxs = [], []
    for iphi in range(nphis):
        phi = iphi / nphis / nfp
        cs = surf.cross_section(phi)
        cs = np.append(cs, cs[:1], axis=0)
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        rax_guess = np.array(rax_guesses[iphi])
        zax_guess = np.array(zax_guesses[iphi])
        rax = rax_guess[:, 0][rax_guess[:, 1].argmin()]
        zax = zax_guess[:, 0][zax_guess[:, 1].argmin()]
        raxs.append(rax)
        zaxs.append(zax)
    rax_avg = np.mean(raxs)
    zax_avg = np.mean(zaxs)
    print(f"(Rax, Zax) = ({rax_avg:.4f}, {zax_avg:.4f})")
    return rax_avg, zax_avg

mu0 = np.pi * 4e-7

text_rows = []

phi = 0
options = dict(verbose=True)

with open(f"plasma_curr_vs_boozer_init_wc.csv", "w") as f:
    f.write("target_iota,mpol,ntor,constraint_weight,plasma_current_ka,target_volume,norm_squared_flux,bres_inf,bres_avg,iota,vol,success,self_intersecting\n")

for iota in [15, 20]:
    print(f"iota={iota/1e2:.2f}", flush=True)
    out_dir = f"iota{iota}/"

    surf_init = load(out_dir + "surf_opt.json")
    nfp = surf_init.nfp
    surface = SurfaceXYZTensorFourier(
        mpol=surf_init.mpol, ntor=surf_init.ntor,
        nfp=nfp, stellsym=surf_init.stellsym,
        quadpoints_phi=surf_init.quadpoints_phi,
        quadpoints_theta=surf_init.quadpoints_theta)
    surface.least_squares_fit(surf_init.gamma())
    sdofs = surface.get_dofs().copy()
    surface = SurfaceXYZTensorFourier(
        mpol=surf_init.mpol, ntor=surf_init.ntor,
        nfp=nfp, stellsym=surf_init.stellsym,
        quadpoints_phi=np.linspace(0, 1/nfp, 64),
        quadpoints_theta=np.linspace(0, 1, 63))
    surface.set_dofs(sdofs)

    biotsavart = load(out_dir + "biotsavart_opt.json").set_points(
        surface.gamma().reshape((-1, 3)))
    R_proxy, Z_proxy = find_rax(biotsavart, surface)

    for _mpol in [6, 12, 18]:
        print(f"--mpol=ntor={_mpol}", flush=True)
        mpol = _mpol
        ntor = _mpol
        for constraint_weight_pow in [0, 1, 2, 3]:
            constraint_weight = 10**constraint_weight_pow
            print(f"----constraint_weight={constraint_weight:.1e}", flush=True)
            for plasma_current_ka in [-5.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 5.0]:
                print(f"------plasma_current_ka={plasma_current_ka:.3f}kA", flush=True)
                for target_volume in [0.08, 0.10, 0.12, 0.2, 0.4]:
                    print(f"--------target_volume={target_volume:.2f}", flush=True)
                    surf_init = load(out_dir + "surf_opt.json")
                    nfp = surf_init.nfp
                    surface = SurfaceXYZTensorFourier(
                        mpol=mpol, ntor=ntor,
                        nfp=nfp, stellsym=surf_init.stellsym,
                        quadpoints_phi=surf_init.quadpoints_phi,
                        quadpoints_theta=surf_init.quadpoints_theta)
                    surface.least_squares_fit(surf_init.gamma())
                    sdofs = surface.get_dofs().copy()
                    surface = SurfaceXYZTensorFourier(
                        mpol=mpol, ntor=ntor,
                        nfp=nfp, stellsym=surf_init.stellsym,
                        quadpoints_phi=np.linspace(0, 1/nfp, 64),
                        quadpoints_theta=np.linspace(0, 1, 63))
                    surface.set_dofs(sdofs)
        
                    biotsavart = load(out_dir + "biotsavart_opt.json").set_points(
                        surface.gamma().reshape((-1, 3)))
                    current_sum = sum(
                        abs(c.current.get_value()) for c in biotsavart.coils[:20])
                    G0 = current_sum * mu0
            
                    tf_coils = biotsavart.coils[:20]
                    banana_coils = biotsavart.coils[20:30]
            
                    proxy_curve = CurveXYZFourier(128, 1)
                    proxy_curve.set('xc(1)', R_proxy)
                    proxy_curve.set('ys(1)', R_proxy)
                    proxy_curve.set('zc(0)', Z_proxy)
                    proxy_curve.fix_all()
                    proxy_current = Current(plasma_current_ka*1e3)
                    proxy_current.fix_all()
                    proxy_coils = [Coil(proxy_curve, proxy_current)]
                    coils = tf_coils + banana_coils + proxy_coils
                    biotsavart = BiotSavart(coils)

                    I0 = plasma_current_ka*mu0
                    iota0 = iota/1e2
                    label = Volume(surface)
                    targetlabel = target_volume
                    bsurf = BoozerSurface(
                        biotsavart,
                        surface,
                        label,
                        targetlabel,
                        constraint_weight=constraint_weight,
                        options=options,
                        I=I0,
                    )
                    res = bsurf.run_code(iota0, G0)
                    success = res["success"]

                    try:
                        cs = bsurf.surface.cross_section(phi)
                        self_intersecting = False
                    except:
                        print(" == self-intersecting == ")
                        self_intersecting = True
        
                    sqf = SquaredFlux(
                        bsurf.surface,
                        bsurf.biotsavart,
                        definition="normalized",
                    ).J()
                    bres = boozer_surface_residual(
                        surface,
                        res["iota"],
                        res["G"],
                        biotsavart,
                        derivatives=0,
                        I=I0,
                    )[0]
                    bres_inf = np.linalg.norm(bres, ord=np.inf)
                    bres_avg = np.mean(bres)
                    _iota = res["iota"]
                    vol0 = target_volume
                    _vol = bsurf.surface.volume()
                    print(f"||r||_inf = {bres_inf:.5e}")
                    print(f" <r>      = {bres_avg:.5e}")
                    print(f"iota out = {_iota:.5f}")
                    print(f" vol out = {_vol:.5f}")
                    print(f"iota % error = {np.abs(iota0 - _iota)/np.abs(iota0)*100:7.3f} %")
                    print(f" vol % error = {np.abs( vol0 -  _vol)/np.abs( vol0)*100:7.3f} %")
                    row = (
                        f"{iota/1e2:>4.2f},"
                        f"{mpol:>2d},"
                        f"{ntor:>2d},"
                        f"{constraint_weight:>7.1e},"
                        f"{plasma_current_ka:>9.2e},"
                        f"{target_volume:>4.2f},"
                        f"{sqf:>12.5e},"
                        f"{bres_inf:>12.5e},"
                        f"{bres_avg:>12.5e},"
                        f"{_iota:>12.5e},"
                        f"{_vol:>12.5e},"
                        f"{success},"
                        f"{self_intersecting}"
                    )
                    print(row, flush=True)
                    with open(f"plasma_curr_vs_boozer_init_wc.csv", "a") as f:
                        f.write(row + "\n")

                    print("")