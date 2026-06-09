import argparse
import os

from simsopt._core import load
from simsopt.geo import SurfaceRZFourier

def build_parser():
    parser = argparse.ArgumentParser(description="Flip a SurfaceRZFourier in the z direction (same as flipping a stellarator-symmetric surface in the phi direction).")
    parser.add_argument("file", type=str, help="Path to SurfaceRZFourier JSON file.")
    return parser

def flip_surface_dofs(surf_original: SurfaceRZFourier) -> SurfaceRZFourier:
    dof_dict = {}
    for key in surf_original.local_dof_names:
        prefix = key[:2]
        m, n = map(int, key.split("(")[1].split(")")[0].split(","))
        if m == 0:
            val = surf_original.get(key)
            if prefix[1] == "s":
                val = -val
        else:
            val = surf_original.get(f"{prefix}({m},{-n})")
        dof_dict[key] = val
    surf_flipped = SurfaceRZFourier(
        mpol=surf_original.mpol,
        ntor=surf_original.ntor,
        nfp=surf_original.nfp,
        stellsym=surf_original.stellsym,
        quadpoints_phi=surf_original.quadpoints_phi,
        quadpoints_theta=surf_original.quadpoints_theta,
    )
    for key, val in dof_dict.items():
        surf_flipped.set(key, val)

    return surf_flipped

def main(argv=None):
    args = build_parser().parse_args(argv)
    file = args.file
    surf_original = load(file)
    if not isinstance(surf_original, SurfaceRZFourier):
        raise ValueError(f"Expected a SurfaceRZFourier, but got {type(surf_original)}.")

    surf_flipped = flip_surface_dofs(surf_original)
    savefile = os.path.splitext(file)[0] + "_flipped.json"
    surf_flipped.save(savefile)
    print(f"Saved flipped surface to {savefile}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())