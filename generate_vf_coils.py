import numpy as np

from simsopt._core import save
from simsopt.field import BiotSavart, Current, Coil
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveXYZFourier

# Coil parameters for Ohmic heating (OH) and vertical field (VF) coils.
vf_current = 1e3 # 1 kA to 3 kA

phi_to_ccw_dir = 1

vf_R_inside = 0.260350
vf_Zs_inside_topside = np.array([0.3508375, 0.3635375, 0.3762375, 0.3889375])
vf_Zs_inside = np.sort(np.concatenate([-vf_Zs_inside_topside, vf_Zs_inside_topside]))
vf_Rs_inside = np.full_like(vf_Zs_inside, vf_R_inside)
vf_curr_dir_inside = np.full_like(vf_Rs_inside, phi_to_ccw_dir)
vf_curr_inside = vf_curr_dir_inside * vf_current

n_turns_vf_outside = 4
vf_R_outside = 1.572133
vf_Z_outside = 0.6505194
vf_Zs_outside_topside = np.full(n_turns_vf_outside, vf_Z_outside)
vf_Zs_outside = np.sort(np.concatenate([-vf_Zs_outside_topside, vf_Zs_outside_topside]))
vf_Rs_outside = np.full_like(vf_Zs_outside, vf_R_outside)
vf_curr_dir_outside = np.full_like(vf_Rs_outside, -phi_to_ccw_dir)
vf_curr_outside = vf_curr_dir_outside * vf_current

vf_Rs = tuple(np.append(vf_Rs_inside, vf_Rs_outside))
vf_Zs = tuple(np.append(vf_Zs_inside, vf_Zs_outside))
vf_currents = tuple(np.append(vf_curr_inside, vf_curr_outside))

NQPTS = 128
NR    = 16
NZ    = 8
NPHI  = 16

nfp = 5
phi_grid = np.linspace(0, 2*np.pi/nfp, NPHI)

if __name__ == "__main__":
    coils = []
    rmax = 0
    zmax = 0
    for R, Z, I in zip(vf_Rs, vf_Zs, vf_currents):
        curve = CurveXYZFourier(NQPTS, order=1)
        curve.set("xc(1)", R)
        curve.set("ys(1)", R)
        curve.set("zc(0)", Z)
        current = ScaledCurrent(Current(1), I)
        coil = Coil(curve, current)
        coils.append(coil)
        rmax = max(rmax, R)
        zmax = max(zmax, abs(Z))
    biotsavart = BiotSavart(coils)

    rrange = (0, rmax*1.5, NR)
    zrange = (-zmax*1.5, zmax*1.5, NZ)
    r_grid = np.linspace(*rrange)
    z_grid = np.linspace(*zrange)

    r_mesh, z_mesh, phi_mesh = np.meshgrid(r_grid, z_grid, phi_grid, indexing='ij')
    x_mesh = r_mesh * np.cos(phi_mesh)
    y_mesh = r_mesh * np.sin(phi_mesh)

    points = np.stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()], axis=-1)
    biotsavart.set_points(points)
    save(biotsavart, f"inputs/vf_biotsavart.json")