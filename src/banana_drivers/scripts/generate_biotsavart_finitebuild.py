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

from banana_drivers.hardware import (
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
from banana_drivers.finitebuild.finitebuild import create_cws_multifilament_grid

def build_parser():
    parser = argparse.ArgumentParser(description="Generate BiotSavart finite build.")
    parser.add_argument("biotsavart_file", help="Path to the BiotSavart file.")
    parser.add_argument("--banana-current-ka", default=None, type=float, help="Override the banana coil current (units kA).")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    biotsavart_file = os.path.abspath(args.biotsavart_file)
    print(f"Generating BiotSavart finite build for {biotsavart_file}")
    if args.banana_current_ka is not None:
        print(f"Overriding banana coil current to {args.banana_current_ka} kA")
        override_curr = True
    else:
        override_curr = False

    biotsavart = load(biotsavart_file)
    tf_coils = biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    banana_coils = biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]
    proxy_coils = biotsavart.coils[PROXY_IDX:PROXY_IDX+N_PROXY]
    vf_coils = biotsavart.coils[VF_IDX:VF_IDX+N_VF]

    nfil = hbt_banana_fb.numfilaments_n * hbt_banana_fb.numfilaments_b
    base_coil = banana_coils[0]
    base_curves = [base_coil.curve]
    total_current = ScaledCurrent(Current(1.0), args.banana_current_ka*1e3) if override_curr else base_coil.current
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
    biotsavart_finite_build = BiotSavart(coils)

    base = os.path.basename(biotsavart_file)
    out_dir = os.path.dirname(biotsavart_file)
    save_file = os.path.join(out_dir, base.replace("biotsavart", "biotsavart_finitebuild"))
    biotsavart_finite_build.save(save_file)
    print(f"Saved BiotSavart finite build to {save_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())