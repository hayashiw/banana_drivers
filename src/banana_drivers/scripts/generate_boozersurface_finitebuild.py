import argparse
import os

from simsopt._core import load

from ..utils.boozersurface import rebuild_boozersurface
from ..utils.tags import resolve_boozersurface_json_filename, generate_boozersurface_filename

def build_parser():
    parser = argparse.ArgumentParser(description="Generate a finite build BoozerSurface JSON file.")
    parser.add_argument("boozersurface_file", help="Path to a BoozerSurface JSON file.")
    parser.add_argument("--out-dir", type=str, default=None, help="Out directory. Default is cwd.")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    boozersurface_file = os.path.abspath(args.boozersurface_file)
    print(f"Generating finite build for {boozersurface_file}")

    boozersurface = load(boozersurface_file)

    boozersurface_finitebuild = rebuild_boozersurface(boozersurface, finitebuild=True)

    tag_dict = resolve_boozersurface_json_filename(boozersurface_file)
    tag_dict["biotsavart"]["finitebuild"] = "finitebuild"
    filename = generate_boozersurface_filename(tag_dict)

    out_dir = args.out_dir or "./"
    savefile = os.path.join(out_dir, filename)
    boozersurface_finitebuild.save(savefile)
    print(f"Saved BoozerSurface finite build to {savefile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
