import argparse

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface

from banana_drivers.hardware import TF_IDX, BANANA_IDX, PROXY_IDX, VF_IDX

def build_parser():
    parser = argparse.ArgumentParser(description="Print coil currents from a BiotSavart JSON file or a BoozerSurface JSON file.")
    parser.add_argument("file", type=str, help="Path to BiotSavart JSON file or BoozerSurface JSON file.")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    file = args.file
    print(f"Loading BiotSavart or BoozerSurface from {file}", flush=True)
    simsobj = load(file)
    if isinstance(simsobj, BiotSavart):
        biotsavart = simsobj
    elif isinstance(simsobj, BoozerSurface):
        biotsavart = simsobj.biotsavart
    else:
        raise ValueError(f"Expected a BiotSavart or BoozerSurface JSON file, but got {type(simsobj)}.")

    coils = biotsavart.coils
    n_coils = len(coils)
    
    coil_labels = {TF_IDX: "TF", BANANA_IDX: "Banana", PROXY_IDX: "Proxy", VF_IDX: "VF"}
    print("Coil currents")
    for icoil, coil in enumerate(coils):
        if icoil in coil_labels:
            print(f"\n{coil_labels[icoil]} coils:")
        current_ka = coil.current.get_value() / 1e3
        isfixed = len(coil.current.x) == 0
        label = ("" if isfixed else "un") + "fixed"
        print(f"\t[{icoil:>2}] {current_ka:>9.5f} kA ({label})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())