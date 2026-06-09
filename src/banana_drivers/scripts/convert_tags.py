import argparse
import os

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from ..utils.io import save_to_json
from ..utils.tags import (
    load_tags_from_biotsavart,
    load_tags_from_surface,
    load_tags_from_boozersurface,
)

def build_parser():
    parser = argparse.ArgumentParser(description="Convert filename of a SIMSOPT JSON file to the new tag format")
    parser.add_argument("file", type=str, help="Path to the SIMSOPT JSON file.")
    parser.add_argument("--biotsavart-tag", type=str, default="biotsavart", help="Tag for the BiotSavart object in the input file. Default: 'biotsavart'")
    parser.add_argument("--surface-tag", type=str, default="surface", help="Tag for the Surface object in the input file. Default: 'surface'")
    parser.add_argument("--biotsavart-stage", type=str, default="init", choices=["init", "stage2opt", "singlestageopt"], help="BiotSavart stage tag. Choose from 'init', 'stage2opt', or 'singlestageopt'. Default: 'init'")
    parser.add_argument("--surface-stage", type=str, default="init", choices=["init", "presolved", "stage2opt", "singlestageopt"], help="Surface stage tag. Choose from 'init', 'presolved', 'stage2opt', or 'singlestageopt'. Default: 'init'")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"Input parameters:")
    for key, val in vars(args).items():
        print(f"    {key}: {val}")
    print()

    file = os.path.abspath(args.file)
    obj = load(file)
    if isinstance(obj, BiotSavart):
        print("Input file is a BiotSavart object.")
        tag_dict = load_tags_from_biotsavart(obj)
        tag_dict["biotsavart"]["tag"] = args.biotsavart_tag
        tag_dict["biotsavart"]["stage"] = args.biotsavart_stage
        savefile = save_to_json(obj, tag_dict)
    elif isinstance(obj, Surface):
        print("Input file is a Surface object.")
        tag_dict = load_tags_from_surface(obj)
        tag_dict["surface"]["tag"] = args.surface_tag
        tag_dict["surface"]["stage"] = args.surface_stage
        savefile = save_to_json(obj, tag_dict)
    elif isinstance(obj, BoozerSurface):
        print("Input file is a BoozerSurface object.")
        tag_dict = load_tags_from_boozersurface(obj)
        tag_dict["biotsavart"]["tag"] = args.biotsavart_tag
        tag_dict["biotsavart"]["stage"] = args.biotsavart_stage
        tag_dict["surface"]["tag"] = args.surface_tag
        tag_dict["surface"]["stage"] = args.surface_stage
        savefile = save_to_json(obj, tag_dict)
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
    print(f"Saved converted file to: {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
