import argparse
import numpy as np
import os

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface

from ..hardware import (
    hbt_banana_ws,
    N_COILS
)

HEADER = (
    "posx,posy,posz,"
    "tanx,tany,tanz,"
    "normx,normy,normz,"
    "binormx,binormy,binormz\n"
)

def build_parser():
    parser = argparse.ArgumentParser(description="Generate coil txt files from a BoozerSurface JSON file or a BiotSavart JSON file.")
    parser.add_argument("input_file", help="Path to the input BoozerSurface or BiotSavart JSON file.")
    parser.add_argument("--out-dir", type=str, default=None, help="Path to output directory. Default is the cwd/{biotsavart_tag}.coils/")
    return parser

def compare_major_radius(curve, tol=1e-6):
    x, y, z = curve.gamma().T
    r = np.sqrt(x**2 + y**2)

    A = np.vstack([r, np.ones_like(r)]).T
    b = r**2 + z**2
    c1, c2 = np.linalg.lstsq(A, b, rcond=None)[0]
    R_fit = round(c1/2, 6)
    r_fit = round(np.sqrt(c2 + R_fit**2), 6)

    R_err = np.abs(R_fit - hbt_banana_ws.major_radius) / hbt_banana_ws.major_radius
    r_err = np.abs(r_fit - hbt_banana_ws.minor_radius) / hbt_banana_ws.minor_radius
    if R_err > tol:
        print(f"Warning: Major radius error {R_err} exceeds tolerance {tol}")
        return_R = R_fit
    else:
        return_R = hbt_banana_ws.major_radius
    if r_err > tol:
        print(f"Warning: Minor radius error {r_err} exceeds tolerance {tol}")
        return_r = r_fit
    else:
        return_r = hbt_banana_ws.minor_radius

    return return_R, return_r

def main(argv=None):
    args = build_parser().parse_args(argv)
    input_file = args.input_file

    obj = load(input_file)
    if isinstance(obj, BoozerSurface):
        biotsavart = obj.biotsavart
    elif isinstance(obj, BiotSavart):
        biotsavart = obj
    else:
        raise ValueError(f"Unsupported input file type: {type(obj)}")
    
    out_dir = args.out_dir or os.path.join(os.getcwd(), input_file.replace(".json", ".coils"))
    os.makedirs(out_dir, exist_ok=True)
    
    coils = biotsavart.coils
    ncoils = len(coils)
    assert ncoils == N_COILS, f"Expected {N_COILS} coils, but found {ncoils}"
    
    print(f"Saving coils from {input_file} to {out_dir}")
    for icoil in range(20, 30):
        curve = biotsavart.coils[icoil].curve
        Rmajor, _ = compare_major_radius(curve)
        
        gamma = curve.gamma()
        tangent_vector = curve.gammadash()
        tangent_vector /= np.linalg.norm(tangent_vector, axis=-1)[:, None]
        
        phi = np.arctan2(gamma[:, 1], gamma[:, 0])
        x = Rmajor*np.cos(phi)
        y = Rmajor*np.sin(phi)
        z = 0*phi
        major_axis = np.column_stack([x, y, z])
        
        normal_vector = gamma - major_axis
        normal_vector /= np.linalg.norm(normal_vector, axis=-1)[:, None]
        
        binormal_vector = np.cross(tangent_vector, normal_vector)
        binormal_vector /= np.linalg.norm(binormal_vector, axis=-1)[:, None]

        line = HEADER
        for row in np.concatenate(
            (gamma, tangent_vector, normal_vector, binormal_vector),
            axis=1
        ):
            line += (",".join(map(str, row)) + "\n")
        savefile = os.path.join(out_dir, f"coil{icoil-19}.csv")
        with open(savefile, "w") as f:
            f.write(line)
        print(f"    Coil {icoil-19} saved to {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
