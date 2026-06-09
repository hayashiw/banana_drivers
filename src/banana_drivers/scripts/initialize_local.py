import os

from simsopt.geo import SurfaceRZFourier

from ..paths import INIT_DIRS, WOUT_ORIGINAL
from ..hardware import DEFAULT_TF_CURRENT_KA, DEFAULT_BANANA_CURRENT_KA
from ..utils.surface import DEFAULT_NPHI, DEFAULT_NTHETA, change_surface_resolution
from ..utils.biotsavart import build_biotsavart
from ..utils.boozersurface import build_boozersurface
from ..utils.constants import MU0
from ..utils.io import save_to_json
from ..utils.tags import (
    load_tags_from_biotsavart,
    load_tags_from_surface,
    load_tags_from_boozersurface,
)

DEFAULT_WOUT_S = 0.24
DEFAULT_WOUT_SCALE = 0.925

def main(argv=None):
    print("Setting up local directories and initial inputs for banana_drivers")
    for directory in INIT_DIRS:
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Directory {directory} already exists.")
    print("")

    _, INPUTS_DIR, BOOZERSURFACES_DIR, SURFACES_DIR, BIOTSAVARTS_DIR, _, _ = INIT_DIRS

    surface = SurfaceRZFourier.from_wout(
        WOUT_ORIGINAL,
        s=DEFAULT_WOUT_S,
        range="field period",
        nphi=DEFAULT_NPHI,
        ntheta=DEFAULT_NTHETA,
    )
    surface.set_dofs(surface.get_dofs() * DEFAULT_WOUT_SCALE / surface.major_radius())
    surface = change_surface_resolution(surface, mpol=12, ntor=12)
    tag_dict = load_tags_from_surface(surface)
    tag_dict["surface"]["tag"] = "original"
    tag_dict["surface"]["stage"] = "init"
    savefile = save_to_json(surface, tag_dict, minimal=False, out_dir=SURFACES_DIR)
    print(f"Initial surface saved to: {savefile}")

    proxy_R = surface.major_radius()
    proxy_Z = 0.0
    biotsavart = build_biotsavart(
        DEFAULT_TF_CURRENT_KA,
        DEFAULT_BANANA_CURRENT_KA,
        0.0,
        (proxy_R, proxy_Z),
        0.0
    )
    tag_dict = load_tags_from_biotsavart(biotsavart)
    tag_dict["biotsavart"]["tag"] = "original"
    tag_dict["biotsavart"]["stage"] = "init"
    savefile = save_to_json(biotsavart, tag_dict, minimal=False, out_dir=BIOTSAVARTS_DIR)
    print(f"Initial biotsavart saved to: {savefile}")

    boozersurface = build_boozersurface(biotsavart, surface, constraint_weight=1e2)
    tag_dict = load_tags_from_boozersurface(boozersurface, volume_target_str="Surface")
    tag_dict["biotsavart"]["tag"] = "original"
    tag_dict["biotsavart"]["stage"] = "init"
    tag_dict["surface"]["tag"] = "original"
    tag_dict["surface"]["stage"] = "init"
    savefile = save_to_json(boozersurface, tag_dict, minimal=False, out_dir=BOOZERSURFACES_DIR)
    print(f"Initial boozersurface saved to: {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
