import numpy as np
import time

from scipy.linalg import lu
from scipy.optimize import minimize

import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt.geo.surfaceobjectives import (
    boozer_surface_residual,
    boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    boozer_surface_dlsqgrad_dcoils_vjp   
)
from functools import partial

class DiagnosticBoozerSurface(Optimizable):
    def __init__(self, biotsavart, surface, label, targetlabel, constraint_weight=None, options=None, I=0., print_func=None):
        super().__init__(depends_on=[biotsavart])

        from simsopt.geo import SurfaceXYZFourier, SurfaceXYZTensorFourier
        if not isinstance(surface, SurfaceXYZTensorFourier) and not isinstance(surface, SurfaceXYZFourier):
            raise Exception("The input surface must be a SurfaceXYZTensorFourier or SurfaceXYZFourier.")

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.boozer_type = 'ls' if constraint_weight else 'exact'
        self.need_to_run_code = True
        self.I = I

        if options is None: options = {}

        if self.boozer_type == 'exact':
            if 'newton_tol' not in options: options['newton_tol'] = 1e-13
            if 'newton_maxiter' not in options: options['newton_maxiter'] = 40
        elif self.boozer_type == 'ls':
            if 'bfgs_tol' not in options: options['bfgs_tol'] = 1e-10
            if 'newton_tol' not in options: options['newton_tol'] = 1e-11
            if 'newton_maxiter' not in options: options['newton_maxiter'] = 40
            if 'bfgs_maxiter' not in options: options['bfgs_maxiter'] = 1500
            if 'limited_memory' not in options: options['limited_memory'] = False
            if 'weight_inv_modB' not in options: options['weight_inv_modB'] = True
        self.options = options

        if print_func is None:
            def print_func(*args, data=False):
                if data:
                    print(*args, flush=True)
                else:
                    print("# ", *args, flush=True)
        self.print_func = print_func

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, iota, G=None):
        if not self.need_to_run_code:
            return

        if G is None:
            assert np.all([c.current.dofs.all_fixed() for c in self.biotsavart.coils])

        if self.boozer_type == 'exact':
            res = self.solve_residual_equation_exactly_newton(iota=iota, G=G, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'])
            return res

        elif self.boozer_type == 'ls':
            assert self.constraint_weight is not None

            res = self.minimize_boozer_penalty_constraints_LBFGS(constraint_weight=self.constraint_weight, iota=iota, G=G,
                                                                 tol=self.options['bfgs_tol'], maxiter=self.options['bfgs_maxiter'], limited_memory=self.options['limited_memory'],
                                                                 weight_inv_modB=self.options['weight_inv_modB'])
            iota, G = res['iota'], res['G']

            self.need_to_run_code = True
            res = self.minimize_boozer_penalty_constraints_newton(constraint_weight=self.constraint_weight, iota=iota, G=G,
                                                                  tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'],
                                                                  weight_inv_modB=self.options['weight_inv_modB'])
            return res

    def boozer_penalty_constraints_vectorized(self, dofs, derivatives=0, constraint_weight=1., optimize_G=False, weight_inv_modB=True):
        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = dofs[:-2]
            iota = dofs[-2]
            G = dofs[-1]
        else:
            sdofs = dofs[:-1]
            iota = dofs[-1]
            G = 2. * np.pi * np.sum(np.abs([coil.current.get_value() for coil in self.biotsavart._coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))

        s = self.surface
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        nsurfdofs = sdofs.size

        s.set_dofs(sdofs)

        surface = self.surface
        biotsavart = self.biotsavart
        x = surface.gamma()
        xphi = surface.gammadash1()
        xtheta = surface.gammadash2()
        nphi = x.shape[0]
        ntheta = x.shape[1]

        xsemiflat = x.reshape((x.size//3, 3)).copy()
        biotsavart.set_points(xsemiflat)
        biotsavart.compute(derivatives)
        B = biotsavart.B().reshape((nphi, ntheta, 3))

        if derivatives >= 1:
            dx_dc = surface.dgamma_by_dcoeff()
            dxphi_dc = surface.dgammadash1_by_dcoeff()
            dxtheta_dc = surface.dgammadash2_by_dcoeff()
            dB_dx = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))

        if derivatives == 2:
            d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))

        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size
        if derivatives == 0:
            val = sopp.boozer_residual(G, iota, xphi, xtheta, B, weight_inv_modB, self.I)
            boozer = val,
        elif derivatives == 1:
            val, dval = sopp.boozer_residual_ds(G, iota, B, dB_dx, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB, self.I)
            boozer = val, dval
        elif derivatives == 2:
            val, dval, d2val = sopp.boozer_residual_ds2(G, iota, B, dB_dx, d2B_by_dXdX, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB, self.I)
            boozer = val, dval, d2val

        boozer = tuple([b/num_res for b in boozer])

        lab = self.label.J()

        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab-self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.)
        r = rnl + 0.5*rl**2 + 0.5*rz**2

        if derivatives == 0:
            return r

        dl = np.zeros(dofs.shape)
        drz = np.zeros(dofs.shape)
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        Jnl = boozer[1]
        if not optimize_G:
            Jnl = Jnl[:-1]

        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl + rl * drl + rz * drz

        if derivatives == 1:
            return r, J

        Hnl = boozer[2]
        if not optimize_G:
            Hnl = Hnl[:-1, :-1]

        d2rl = np.zeros((dofs.shape[0], dofs.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = np.sqrt(constraint_weight)*self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        H = Hnl + drl[:, None] @ drl[None, :] + drz[:, None] @ drz[None, :] + rl * d2rl

        return r, J, H

    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None, limited_memory=True, weight_inv_modB=True):
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))

        tracker = dict(iters=0, evals=0)

        t0 = time.monotonic()
        fun_name = self.boozer_penalty_constraints_vectorized
        def fun(x):
            tracker["evals"] += 1
            r, J = fun_name(x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)
            line = f"{x[-1]}" if G is None else f"{x[-2]},{x[-1]}"
            self.print_func(f"0,{time.monotonic() - t0},{tracker['iters']},{tracker['evals']},{r},{np.linalg.norm(J)},{line},{self.label.J()}", data=True)
            return r, J
        
        def callback(x):
            tracker["iters"] += 1
            tracker["evals"] = 0
            r, J = fun_name(x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)
            line = f"{x[-1]}" if G is None else f"{x[-2]},{x[-1]}"
            self.print_func(f"0,{time.monotonic() - t0},{tracker['iters']},{tracker['evals']},{r},{np.linalg.norm(J)},{line},{self.label.J()}", data=True)

        method = 'L-BFGS-B' if limited_memory else 'BFGS'
        options = {'maxiter': maxiter, 'gtol': tol}
        if limited_memory:
            options['maxcor'] = 200
            options['ftol'] = tol

        line = "iota" if G is None else "iota,G"
        self.print_func(f"I (plasma_current * mu0) = {self.I}")
        self.print_func(f"targetlabel = {self.targetlabel}")
        self.print_func(f"{method} (BoozerLS) solve")
        self.print_func(f"is_newton,time,iters,evals,r,|J|,{line},label", data=True)
        r, J = fun_name(x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)
        line = f"{x[-1]}" if G is None else f"{x[-2]},{x[-1]}"
        self.print_func(f"0,{time.monotonic() - t0},{tracker['iters']},{tracker['evals']},{r},{np.linalg.norm(J)},{line},{self.label.J()}", data=True)
        res = minimize(fun, x, jac=True, method=method, options=options, callback=callback)
        self.print_func("done")

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success, "G": None, 'weight_inv_modB': weight_inv_modB, 'type': 'ls'
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict['G'] = G
        resdict['s'] = s
        resdict['iota'] = iota

        self.res = resdict
        self.need_to_run_code = False

        self.print_func(f"{method} solve - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}")

        return resdict

    def minimize_boozer_penalty_constraints_newton(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, stab=0., weight_inv_modB=True):
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0

        val, dval, d2val = self.boozer_penalty_constraints_vectorized(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)

        t0 = time.monotonic()
        norm = np.linalg.norm(dval)
        tracker = dict(iters=0)
        line = "iota" if G is None else "iota,G"
        self.print_func(f"I (plasma_current * mu0) = {self.I}")
        self.print_func(f"targetlabel = {self.targetlabel}")
        self.print_func(f"NEWTON (BoozerLS) solve")
        self.print_func(f"is_newton,time,iters,evals,r,|J|,{line},label")
        line = f"{x[-1]}" if G is None else f"{x[-2]},{x[-1]}"
        self.print_func(f"1,{time.monotonic() - t0},{tracker['iters']},0,{val},{norm},{line},{self.label.J()}", data=True)
        while i < maxiter and norm > tol:
            d2val += stab*np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            if norm < 1e-9:
                dx += np.linalg.solve(d2val, dval - d2val@dx)
            x = x - dx
            val, dval, d2val = self.boozer_penalty_constraints_vectorized(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)
            norm = np.linalg.norm(dval)
            i = i+1
            tracker["iters"] = i
            line = f"{x[-1]}" if G is None else f"{x[-2]},{x[-1]}"
            self.print_func(f"1,{time.monotonic() - t0},{i},0,{val},{norm},{line},{self.label.J()}", data=True)
        self.print_func("done")

        r = dval

        P, L, U = lu(d2val)
        res = {
            "residual": r, "jacobian": dval, "hessian": d2val, "iter": i, "success": norm <= tol, "G": None,
            "PLU": (P, L, U), "vjp": partial(boozer_surface_dlsqgrad_dcoils_vjp, weight_inv_modB=weight_inv_modB, I=self.I),
            "type": "ls", "weight_inv_modB": weight_inv_modB, "I": self.I
        }
        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            res['G'] = G
        res['iota'] = iota

        self.res = res
        self.need_to_run_code = False

        self.print_func(f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}")

        return res

    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, iota=0., G=None):
        if not self.need_to_run_code:
            return self.res

        from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
        s = self.surface
        if not isinstance(s, SurfaceXYZTensorFourier):
            raise RuntimeError('Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier')

        m = s.get_stellsym_mask()
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()

        label = self.label
        if G is None:
            G = 2. * np.pi * np.sum(np.abs([c.current.get_value() for c in self.biotsavart.coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1, I=self.I)
        norm = 1e6
        tracker = dict(iters=0)
        self.print_func(f"I (plasma_current * mu0) = {self.I}")
        self.print_func(f"targetlabel = {self.targetlabel}")
        self.print_func(f"NEWTON (BoozerExact) solve")
        self.print_func(f"time,iters,r,|J|,iota,G,label", data=True)
        t0 = time.monotonic()
        self.print_func(f"{time.monotonic() - t0},{tracker['iters']},{np.linalg.norm(r)},{np.linalg.norm(J)},{iota},{G},{self.label.J()}")
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel)]))
            else:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel), s.gamma()[0, 0, 2]]))
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if s.stellsym:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                ))
            else:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                    np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0., 0.]))
                ))
            dx = np.linalg.solve(J, b)
            dx += np.linalg.solve(J, b-J@dx)
            x -= dx
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            i += 1
            r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1, I=self.I)
            tracker['iters'] = i
            self.print_func(f"{time.monotonic() - t0},{tracker['iters']},{np.linalg.norm(r)},{np.linalg.norm(J)},{iota},{G},{self.label.J()}", data=True)
        self.print_func("done")

        if s.stellsym:
            J = np.vstack((
                J[mask, :],
                np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
            ))
        else:
            J = np.vstack((
                J[mask, :],
                np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0., 0.]))
            ))

        P, L, U = lu(J)
        res = {
            "residual": r, "jacobian": J, "iter": i, "success": norm <= tol, "G": G, "s": s, "iota": iota, "PLU": (P, L, U),
            "mask": mask, 'type': 'exact', "vjp": partial(boozer_surface_dexactresidual_dcoils_dcurrents_vjp, I=self.I),
            "I": self.I
        }

        self.print_func(f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}")

        self.res = res
        self.need_to_run_code = False
        return res
