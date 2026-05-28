import argparse

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface

from banana_drivers.paths import BANANA_IDX, N_BANANA

def build_parser():
    parser = argparse.ArgumentParser(description="Print the coil parameters of the banana coil set.")
    parser.add_argument("file", type=str, help="Path to BiotSavart JSON file or BoozerSurface JSON file.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    file = args.file

    file_obj = load(file)
    if isinstance(file_obj, BiotSavart):
        biotsavart = file_obj
    elif isinstance(file_obj, BoozerSurface):
        biotsavart = file_obj.biotsavart
    else:
        raise ValueError(f"File {file} is not a BiotSavart or BoozerSurface JSON.")
    
    banana_coils = biotsavart.coils[BANANA_IDX:BANANA_IDX + N_BANANA]
    banana_order = banana_coils[0].curve.order
    nqpts = banana_coils[0].curve.quadpoints.size
    print(f"Banana coil parameters")
    print(f"File: {file}")
    print(f"Fourier order: {banana_order}")
    print(f"Number of quadpoints: {nqpts}")
    print(f"Currents (kA):")
    for icoil, coil in enumerate(banana_coils):
        current = coil.current.get_value()
        isfixed = len(coil.current.x) == 0
        label = "fixed" if isfixed else "unfixed"
        print(f"    [{icoil+1}] {current:>9.5f} ({label})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())