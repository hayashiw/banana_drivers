import os

from simsopt.geo import SurfaceRZFourier

from ..paths import INIT_DIRS, WOUT_ORIGINAL
from ..hardware import DEFAULT_TF_CURRENT_KA, DEFAULT_BANANA_CURRENT_KA, TF_IDX, N_TF
from ..utils.surface import DEFAULT_NPHI, DEFAULT_NTHETA, change_surface_resolution
from ..utils.biotsavart import build_biotsavart
from ..utils.boozersurface import build_boozersurface
from ..utils.constants import MU0
from ..utils.io import save_to_json

DEFAULT_WOUT_S = 0.24
DEFAULT_WOUT_SCALE = 0.925

def main():
    print("Setting up local directories and initial inputs for banana_drivers")
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
    for directory in INIT_DIRS:
        make_dir(directory)
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
    savefile = save_to_json(surface, "original", init_opt="init", out_dir=SURFACES_DIR)
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
    savefile = save_to_json(biotsavart, "original", init_opt="init", out_dir=BIOTSAVARTS_DIR)
    print(f"Initial biotsavart saved to: {savefile}")

    boozersurface = build_boozersurface(biotsavart, surface, constraint_weight=1.0)
    savefile = save_to_json(boozersurface, "original", "original", init_opt="init", out_dir=BOOZERSURFACES_DIR)
    print(f"Initial boozersurface saved to: {savefile}")

    iota_guess = 0.15
    tf_coils = biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    tf_current = sum(abs(coil.current.get_value()) for coil in tf_coils)
    G_guess = -tf_current * MU0
    state = dict(iota=iota_guess, G=G_guess)
    savefile = save_to_json(state, "original", "original", init_opt="init", out_dir=BOOZERSURFACES_DIR)
    print(f"Initial state saved to: {savefile}")

    inputs_markdown_file = os.path.join(INPUTS_DIR, "inputs.md")
    if not os.path.exists(inputs_markdown_file):
        with open(inputs_markdown_file, "w") as f:
            f.write(
                "| Label | biotsavart | surface |\n"
                "|------|------------|----------|\n"
                "| original | Initial coils: simple ellipse, default values for coil currents | Original wout with s=0.24 and DOFs scaled to 0.925 m major radius |\n"
            )
    print(f"Inputs markdown file saved to: {inputs_markdown_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
