import argparse
import inspect

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
)
from ..utils.tags import (
    load_tags_from_biotsavart,
    load_tags_from_surface,
    load_tags_from_boozersurface,
    generate_random_tag,
    compare_tags,
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
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    inputs = {key: val for key, val in vars(args).items() if val is not None}
    nargs = len(inputs)
    assert nargs > 1, "No parameters specified to change. Please provide at least one parameter to change."
    
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
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    for keyword in keyword_args:
        if keyword in inputs:
            nargs_obj += 1
    assert nargs_obj > 0, f"No valid parameters specified to change for type {type(obj).__name__}. Please provide at least one valid parameter."

    kwargs = {key: val for key, val in inputs.items() if key in keyword_args}
    new_obj = rebuild_func(obj, **kwargs)
    new_tag_dict = load_tag_func(new_obj)
    for key in ["biotsavart", "surface"]:
        if key in tag_dict:
            new_tag_dict[key]["tag"] = tag_dict[key]["tag"]
            new_tag_dict[key]["stage"] = tag_dict[key]["stage"]
    if "boozersurface" in tag_dict:
        new_tag_dict["boozersurface"]["volume_target_str"] = tag_dict["boozersurface"]["volume_target_str"]
    if "version_number_str" in tag_dict:
        new_tag_dict["version_number_str"] = tag_dict["version_number_str"]
    if "iter_number" in tag_dict:
        new_tag_dict["iter_number"] = tag_dict["iter_number"]

    new_random_key = compare_tags(tag_dict, new_tag_dict)
    if new_random_key:
        new_tag = generate_random_tag()
        print(f"Generated new random tag: {new_tag}")
        if "biotsavart" in tag_dict:
            new_tag_dict["biotsavart"]["tag"] = new_tag
        if "surface" in tag_dict:
            new_tag_dict["surface"]["tag"] = new_tag

    savefile = save_to_json(new_obj, new_tag_dict, minimal=False)
    print(f"Saved updated file to: {savefile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
