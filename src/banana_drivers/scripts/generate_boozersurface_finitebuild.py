import argparse
import os

from simsopt._core import load
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    apply_symmetries_to_curves,
    apply_symmetries_to_currents,
)
from simsopt.field.coil import ScaledCurrent

from ..hardware import (
    hbt_banana_fb,
    hbt_banana_ws,
    TF_IDX,
    BANANA_IDX,
    PROXY_IDX,
    VF_IDX,
    N_TF,
    N_BANANA,
    N_PROXY,
    N_VF,
)
from ..utils.finitebuild import create_cws_multifilament_grid
from ..utils.boozersurface import build_boozersurface

def build_parser():
    parser = argparse.ArgumentParser(description="Generate a finite build BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", help="Path to a BoozerSurface JSON file.")
    parser.add_argument("--out-dir", type=str, default=None, help="Out directory. Default is same as boozersurface_file.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = os.path.abspath(args.boozersurface_file)
    print(f"Generating finite build for {boozersurface_file}")

    boozersurface = load(boozersurface_file)
    biotsavart = boozersurface.biotsavart

    tf_coils = biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    banana_coils = biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]
    proxy_coils = biotsavart.coils[PROXY_IDX:PROXY_IDX+N_PROXY]
    vf_coils = biotsavart.coils[VF_IDX:VF_IDX+N_VF]

    nfil = hbt_banana_fb.numfilaments_n * hbt_banana_fb.numfilaments_b
    base_coil = banana_coils[0]
    base_curves = [base_coil.curve]
    banana_current = base_coil.current.get_value()
    total_current = ScaledCurrent(Current(1.0), banana_current)
    base_currents = [total_current/nfil]

    base_curves_finite_build = sum([
        create_cws_multifilament_grid(
            c, hbt_banana_ws.major_radius, hbt_banana_fb.numfilaments_n, hbt_banana_fb.numfilaments_b, 
            hbt_banana_fb.offset_from_horizontal, hbt_banana_fb.offset_from_vertical, hbt_banana_fb.horizontal_spacing,
            rotation_order=0)
        for c in base_curves], [])
    base_currents_finite_build = sum([[c]*nfil for c in base_currents], [])

    curves_fb = apply_symmetries_to_curves(
        base_curves_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)
    currents_fb = apply_symmetries_to_currents(
        base_currents_finite_build, hbt_banana_ws.nfp, hbt_banana_ws.stellsym)

    banana_coils_finitebuild = [Coil(c, curr) for c, curr in zip(curves_fb, currents_fb)]

    coils = tf_coils + banana_coils_finitebuild + proxy_coils + vf_coils
    biotsavart_finitebuild = BiotSavart(coils)

    boozersurface_finitebuild = build_boozersurface(
        biotsavart_finitebuild,
        boozersurface.surface,
        constraint_weight=boozersurface.constraint_weight,
        proxy_coil=proxy_coils[0],
    )

    base = os.path.basename(boozersurface_file)
    biotsavart_tag, surface_tag, *remaining = os.path.basename(boozersurface_file).split(".")
    biotsavart_tag += "_finitebuild"
    new_base = f"{biotsavart_tag}.{surface_tag}." + ".".join(remaining)

    out_dir = args.out_dir or os.path.dirname(boozersurface_file)
    savefile = os.path.join(out_dir, new_base)
    boozersurface_finitebuild.save(savefile)
    print(f"Saved BoozerSurface finite build to {savefile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
