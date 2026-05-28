import argparse

from simsopt._core import load
from simsopt.field import BiotSavart

from banana_drivers.utils.coils import generate_proxy_coils, generate_vf_coils
from banana_drivers.hardware import TF_IDX, BANANA_IDX, N_TF, N_BANANA

def build_parser():
    p = argparse.ArgumentParser(description="Add proxy and VF coils to an existing BiotSavart solution.")
    p.add_argument("biotsavart_file", type=str, help="Path to the input BiotSavart JSON file.")
    p.add_argument("surface_file", type=str, help="Path to the input surface JSON file.")
    p.add_argument("--proxy-R", type=float, default=None, help="Proxy coil R in meters. Overrides major radius from surface.")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    biotsavart = load(args.biotsavart_file)
    tf_coils = biotsavart.coils[TF_IDX:TF_IDX+N_TF]
    banana_coils = biotsavart.coils[BANANA_IDX:BANANA_IDX+N_BANANA]

    proxy_R = args.proxy_R
    if proxy_R is None:
        surface = load(args.surface_file)
        proxy_R = surface.major_radius()
        print(f"Using proxy coil R from surface major radius -> {proxy_R:.3f} m")
    else:
        print(f"Using proxy coil R from command line argument -> {proxy_R:.3f} m")
    proxy_rz = (proxy_R, 0.0)
    proxy_coils = generate_proxy_coils(0.0, proxy_rz)

    vf_coils = generate_vf_coils(0.0)

    coils = tf_coils + banana_coils + proxy_coils + vf_coils
    new_biotsavart = BiotSavart(coils)

    new_file = args.biotsavart_file.replace(".json", "_init.json")
    new_biotsavart.save(new_file)
    print(f"Saved new BiotSavart with proxy and VF coils -> {new_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())