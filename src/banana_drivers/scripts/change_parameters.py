import argparse
import inspect
import re

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from ..utils.boozersurface import rebuild_boozersurface
from ..utils.biotsavart import rebuild_biotsavart
from ..utils.surface import rebuild_surface
from ..utils.cli import (
    tf_coil_parser,
    banana_coil_parser,
    proxy_coil_parser,
    vf_coil_parser,
    surface_resolution_parser,
    boozersurface_parameters_parser,
    FIX_CURRENT_KW_PATTERN,
    FILENAME_TAG_INPUTS
)
from ..utils.tags import (
    load_tags_from_biotsavart,
    load_tags_from_surface,
    load_tags_from_boozersurface,
    generate_random_tag,
)
from ..utils.io import save_to_json

def build_parser():
    parser = argparse.ArgumentParser(
        description="Change parameters of a SIMSOPT JSON file and save to a new file with updated tags",
        parents=[
            tf_coil_parser(),
            banana_coil_parser(),
            proxy_coil_parser(),
            vf_coil_parser(),
            surface_resolution_parser(),
            boozersurface_parameters_parser(),
        ],
    )
    parser.add_argument("file", type=str, help="Path to the SIMSOPT JSON file.")
    parser.add_argument("--full-name", action="store_true", help="If set, will use `minimal=False` in the filename generation. Default: False (minimal=True)")
    parser.add_argument("--overwrite", action="store_true", help="If the intended save file exists, will raise an error if False. Set to True to overwrite existing files. Default: False")
    return parser

SKIP_ARGS = ["file", "full_name", "overwrite"]

def main(argv=None):
    args = build_parser().parse_args(argv)
    inputs = {}
    for key, val in vars(args).items():
        if val is None: continue
        if hasattr(val, "__len__") and len(val) == 0: continue
        inputs[key] = val
    print(f"Inputs and requested parameter changes:")
    for key, val in inputs.items():
        print(f"    {key}: {val}")
    nargs = len(inputs)
    assert nargs > len(SKIP_ARGS), "No parameters specified to change. Please provide at least one parameter to change."
    
    file = inputs["file"]
    obj = load(file)
    if isinstance(obj, BiotSavart):
        rebuild_func = rebuild_biotsavart
        load_tag_func = load_tags_from_biotsavart
    elif isinstance(obj, Surface):
        rebuild_func = rebuild_surface
        load_tag_func = load_tags_from_surface
    elif isinstance(obj, BoozerSurface):
        rebuild_func = rebuild_boozersurface
        load_tag_func = load_tags_from_boozersurface
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}. Expected BiotSavart, Surface, or BoozerSurface.")
    tag_dict = load_tag_func(file)

    nargs_obj = 0
    function_signature = inspect.signature(rebuild_func)
    keyword_args = [
        name for name, param in function_signature.parameters.items()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY
        )]
    for keyword in keyword_args:
        if keyword in inputs:
            nargs_obj += 1
    assert nargs_obj > 0, f"No valid parameters specified to change for type {type(obj).__name__}. Please provide at least one valid parameter."

    kwarg_inputs = {key: val for key, val in inputs.items() if key not in SKIP_ARGS}
    kwargs = {key: val for key, val in kwarg_inputs.items() if key in keyword_args}
    new_obj = rebuild_func(obj, **kwargs)
    new_tag_dict = load_tag_func(new_obj)
    for key in ["biotsavart", "surface"]:
        if key in tag_dict:
            new_tag_dict[key]["tag"] = tag_dict[key]["tag"]
            new_tag_dict[key]["stage"] = tag_dict[key]["stage"]
    if "boozersurface" in tag_dict:
        if "volume_target_str" in kwarg_inputs:
            new_tag_dict["boozersurface"]["volume_target_str"] = kwarg_inputs["volume_target_str"].replace(".", "d").replace("%", "pct")
        else:
            new_tag_dict["boozersurface"]["volume_target_str"] = tag_dict["boozersurface"]["volume_target_str"]
    if "version_number_str" in tag_dict:
        new_tag_dict["version_number_str"] = tag_dict["version_number_str"]
    if "iter_number" in tag_dict:
        new_tag_dict["iter_number"] = tag_dict["iter_number"]

    tag_kwargs = {
        key: val for key, val in kwarg_inputs.items() if key in FILENAME_TAG_INPUTS}
    non_tag_non_fix_kwargs = {
        key: val for key, val in kwarg_inputs.items() if
        (key not in FILENAME_TAG_INPUTS) and
        not re.match(FIX_CURRENT_KW_PATTERN, key)}
    non_tag_fix_kwargs = {
        key: val for key, val in kwarg_inputs.items() if
        (key not in FILENAME_TAG_INPUTS) and
        re.match(FIX_CURRENT_KW_PATTERN, key)}
    new_random_tag = len(non_tag_non_fix_kwargs) > 0
    
    if new_random_tag:
        new_tag = generate_random_tag()
        print(f"Non-tag, non-fix/unfix parameters:")
        for key, val in non_tag_non_fix_kwargs.items():
            print(f"    {key}: {val}")
        print(f"Changing non-tag, non-fix/unfix keys triggers a new random tag: {new_tag}")
        if "biotsavart" in tag_dict:
            new_tag_dict["biotsavart"]["tag"] = new_tag
        if "surface" in tag_dict:
            new_tag_dict["surface"]["tag"] = new_tag
    else:
        if (len(tag_kwargs) == 0) and (len(non_tag_fix_kwargs) > 0):
            print(f"Only fix/unfix parameters specified:")
            for key, val in non_tag_fix_kwargs.items():
                print(f"    {key}: {val}")
            print(f"If original file is `minimal=True`, then a new file will be saved.")
            print(f"Else, the original file will be overwritten.")
        if (len(tag_kwargs) > 0):
            print(f"Tag parameters specified:")
            for key, val in tag_kwargs.items():
                print(f"    {key}: {val}")
            print(f"Changing tag parameters creates a new file with the same base tags.")

    minimal = not args.full_name
    savefile = save_to_json(new_obj, new_tag_dict, minimal=minimal, overwrite=args.overwrite)
    print(f"Saved updated file to: {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
