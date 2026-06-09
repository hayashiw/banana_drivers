import os
import random
import re
import string

from collections import OrderedDict

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from ..hardware import hbt_banana_fb, BANANA_IDX, PROXY_IDX, N_BANANA, N_COILS, N_FB_COILS
from ..utils.boozersurface import load_boozersurface

FLOAT_PATTERN = r"\d+d\d+|\d+"

BIOTSAVART_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>[a-zA-Z0-9_]+)$",
    proxy_current_ka_str=rf"proxy(?P<proxy_current_ka_str>{FLOAT_PATTERN})kA",
    order=r"^o(?P<order>\d+)$",
    nqpts=r"^nqpts(?P<nqpts>\d+)$",
    virtualcasing=r"^(?P<virtualcasing>virtualcasing)$",
    stage=r"^(?P<stage>init|(stage2|singlestage)opt)$",
    finitebuild=r"^(?P<finitebuild>finitebuild)$",
)

ENFORCE_BIOTSAVART_TAGS = ["tag", "proxy_current_ka_str", "order", "stage"]

SURFACE_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>[a-zA-Z0-9_]+)$",
    mpol=r"m(?P<mpol>\d+)$",
    ntor=r"n(?P<ntor>\d+)$",
    nphi=r"np(?P<nphi>\d+)$",
    ntheta=r"nt(?P<ntheta>\d+)$",
    stage=r"^(?P<stage>init|presolved|(stage2|singlestage)opt)$"
)

ENFORCE_SURFACE_TAGS = ["tag", "mpol", "ntor", "stage"]

BOOZERSURFACE_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>boozersurface|state)$",
    constraint_weight_str=rf"^cw(?P<constraint_weight_str>Exact|{FLOAT_PATTERN})$",
    volume_target_str=rf"^vol(?P<volume_target_str>Surface|({FLOAT_PATTERN})(pct)?)$"
)

ENFORCE_BOOZERSURFACE_TAGS = ["tag", "constraint_weight_str", "volume_target_str"]

# Version number can be vers0, vers1, etc.
# In the case of branching versions (new run with vers1 even though vers2 already exists), can be vers1_1, vers1_2, etc.
OTHER_PATTERNS = OrderedDict(
    version_number_str=r"^vers(?P<version_number_str>\d+(?:_\d+)*)$",
    iter_number=r"^iter(?P<iter_number>\d+)$"
)

CONVERTERS = OrderedDict(
    tag=str,
    proxy_current_ka_str=str,
    order=int,
    nqpts=int,
    finitebuild=str,
    virtualcasing=str,
    stage=str,
    mpol=int,
    ntor=int,
    nphi=int,
    ntheta=int,
    constraint_weight_str=str,
    volume_target_str=str,
    version_number_str=str,
    iter_number=int,
)

INVERTERS = dict(
    tag=lambda x: x,
    proxy_current_ka_str=lambda x: f"proxy{x}kA",
    proxy_current_ka=lambda x:f"proxy{str(round(x, 6)).replace('.', 'd')}kA",
    order=lambda x: f"o{x}",
    nqpts=lambda x: f"nqpts{x}",
    finitebuild=lambda x: x,
    virtualcasing=lambda x: x,
    stage=lambda x: x,
    mpol=lambda x: f"m{x}",
    ntor=lambda x: f"n{x}",
    nphi=lambda x: f"np{x}",
    ntheta=lambda x: f"nt{x}",
    constraint_weight_str=lambda x: f"cw{x}",
    volume_target_str=lambda x: f"vol{x}",
    version_number_str=lambda x: f"vers{x}",
    iter_number=lambda x: f"iter{x}"
)

def _resolve_tags(tag: str, source: str) -> dict:
    assert source in ["biotsavart", "surface", "boozersurface", "state"], f"Expected `source` to be 'biotsavart', 'surface', 'boozersurface' or 'state', got {source}"
    if source == "biotsavart":
        patterns = BIOTSAVART_PATTERNS
    elif source == "surface":
        patterns = SURFACE_PATTERNS
    else:
        patterns = BOOZERSURFACE_PATTERNS
    split_tag = tag.split("_")
    base_tag = []
    tag_dict = {}
    for _tag in split_tag:
        matched = False
        for key, pattern in patterns.items():
            if key == "tag": continue
            match = re.match(pattern, _tag)
            if match:
                converted_tag = CONVERTERS[key](match.groupdict()[key])
                tag_dict[key] = converted_tag
                matched = True
                break
        if not matched: base_tag.append(_tag)
                
    if not len(base_tag): base_tag = [source]
    tag_dict["tag"] = "_".join(base_tag)
    return {f"{source}": tag_dict}

def _resolve_other_tags(tags: list[str]) -> dict:
    tag_dict = {}
    for _tag in tags:
        matched = False
        for key, pattern in OTHER_PATTERNS.items():
            match = re.match(pattern, _tag)
            if match:
                tag_dict[key] = CONVERTERS[key](match.groupdict()[key])
                matched = True
                break
        if not matched: # only for diagnostic/debugging purposes
            if "other" not in tag_dict: tag_dict["other"] = []
            tag_dict["other"].append(_tag)
    return tag_dict

def check_tags(tag_dict: dict[str, int | str]) -> None:
    err = False
    msg = {}

    enforce_biotsavart_tags = ENFORCE_BIOTSAVART_TAGS.copy()
    for key in ["virtualcasing", "finitebuild"]:
        if ("biotsavart" in tag_dict) and (key in tag_dict["biotsavart"]):
            enforce_biotsavart_tags += [key]

    for enforce_tag_list, pattern_dict, label in [
        (enforce_biotsavart_tags, BIOTSAVART_PATTERNS, "biotsavart"),
        (ENFORCE_SURFACE_TAGS, SURFACE_PATTERNS, "surface"),
        (ENFORCE_BOOZERSURFACE_TAGS, BOOZERSURFACE_PATTERNS, "boozersurface")
    ]:
        if label not in tag_dict: continue
        for tag in enforce_tag_list:
            if tag not in tag_dict[label]:
                if label not in msg: msg[label] = ""
                msg[label] += f"    Missing tag: {tag}\n"
                err = True
                continue
            pattern = pattern_dict[tag]
            fullmatch = re.fullmatch(pattern, INVERTERS[tag](tag_dict[label][tag]))
            if not fullmatch:
                if label not in msg: msg[label] = ""
                msg[label] += f"    Invalid tag format [{tag}]: {INVERTERS[tag](tag_dict[label][tag])}, expected pattern: {pattern}\n"
                err = True

    verbose_err_msg = "\n".join([f"{label} tags:\n{_msg}" for label, _msg in msg.items()])
    assert (not err), f"Missing tags:\n{verbose_err_msg}"

def resolve_biotsavart_json_filename(filename: str, enforce_tags: bool = True) -> dict:
    biotsavart_tag, *other = os.path.basename(filename).removesuffix(".json").split(".")

    biotsavart_tag_dict = _resolve_tags(biotsavart_tag, "biotsavart")
    other_tags_dict = _resolve_other_tags(other)

    full_tag_dict = dict()
    full_tag_dict.update(biotsavart_tag_dict)
    full_tag_dict.update(other_tags_dict)
    if enforce_tags: check_tags(full_tag_dict)
    return full_tag_dict

def resolve_surface_json_filename(filename: str, enforce_tags: bool = True) -> dict:
    surface_tag, *other = os.path.basename(filename).removesuffix(".json").split(".")

    surface_tag_dict = _resolve_tags(surface_tag, "surface")
    other_tags_dict = _resolve_other_tags(other)
    
    full_tag_dict = dict()
    full_tag_dict.update(surface_tag_dict)
    full_tag_dict.update(other_tags_dict)
    if enforce_tags: check_tags(full_tag_dict)
    return full_tag_dict

def resolve_boozersurface_json_filename(filename: str, enforce_tags: bool = True) -> dict:
    biotsavart_tag, surface_tag, boozersurface_tag, *other = \
        os.path.basename(filename).removesuffix(".json").split(".")

    biotsavart_tag_dict = _resolve_tags(biotsavart_tag, "biotsavart")
    surface_tag_dict = _resolve_tags(surface_tag, "surface")
    boozersurface_tag_dict = _resolve_tags(boozersurface_tag, "boozersurface")
    other_tags_dict = _resolve_other_tags(other)

    full_tag_dict = dict()
    full_tag_dict.update(biotsavart_tag_dict)
    full_tag_dict.update(surface_tag_dict)
    full_tag_dict.update(boozersurface_tag_dict)
    full_tag_dict.update(other_tags_dict)
    if enforce_tags: check_tags(full_tag_dict)
    return full_tag_dict

def increment_version_number(version_number_str):
    version_numbers = [int(x) for x in version_number_str.split("_")]
    version_numbers[-1] += 1
    return "_".join(str(x) for x in version_numbers)

def generate_version_number(tag_dict: dict[str, int | str], out_dir: str) -> str:
    version_number_str = tag_dict.get("version_number_str", "0")
    tag_dict["version_number_str"] = version_number_str
    _biotsavart_tag, _surface_tag, _boozersurface_tag, _other_tags = generate_boozersurface_tags(tag_dict)
    filename = ".".join([_biotsavart_tag, _surface_tag, _boozersurface_tag, *_other_tags])
    filename = os.path.join(out_dir, filename+".json")

    if not os.path.exists(filename):
        return version_number_str
    else:
        next_version_number_str = increment_version_number(version_number_str)
        tag_dict["version_number_str"] = next_version_number_str
        _biotsavart_tag, _surface_tag, _boozersurface_tag, _other_tags = generate_boozersurface_tags(tag_dict)
        filename = ".".join([_biotsavart_tag, _surface_tag, _boozersurface_tag, *_other_tags])
        filename = os.path.join(out_dir, filename+".json")
        if not os.path.exists(filename):
            return next_version_number_str
        else:
            next_version_number_str = f"{version_number_str}_0"
            tag_dict["version_number_str"] = next_version_number_str
            return generate_version_number(tag_dict, out_dir)

def generate_biotsavart_tag(
    tag_dict: dict[str, int | str],
    minimal: bool = True,
    enforce_tags: bool = True,
) -> tuple[str, list[str]]:
    if "biotsavart" not in tag_dict: tag_dict = dict(biotsavart=tag_dict)
    if enforce_tags: check_tags(tag_dict)

    enforce_biotsavart_tags = ENFORCE_BIOTSAVART_TAGS.copy()
    for key in ["virtualcasing", "finitebuild"]:
        if key in tag_dict["biotsavart"]:
            enforce_biotsavart_tags += [key]

    tags = []
    for key in BIOTSAVART_PATTERNS:
        if minimal and (key not in enforce_biotsavart_tags): continue
        if key in tag_dict["biotsavart"]:
            tags.append(INVERTERS[key](tag_dict["biotsavart"][key]))
    if (
        ("proxy_current_ka_str" not in tag_dict["biotsavart"]) and
        ("proxy_current_ka" in tag_dict["biotsavart"])
    ):
        key = "proxy_current_ka"
        tags.append(INVERTERS[key](tag_dict["biotsavart"][key]))
    biotsavart_tag = "_".join(tags)

    err = False
    msg = ""
    other_tags = []
    for key in OTHER_PATTERNS:
        if key in tag_dict:
            pattern = OTHER_PATTERNS[key]
            fullmatch = re.fullmatch(pattern, INVERTERS[key](tag_dict[key]))
            if not fullmatch:
                err = True
                msg += f"Invalid tag format for {key}: {INVERTERS[key](tag_dict[key])}, expected pattern: {pattern}\n"
            else:
                other_tags.append(INVERTERS[key](tag_dict[key]))
    
    if err:
        verbose_err_msg = f"Error generating BiotSavart tags:\n{msg}"
        raise ValueError(verbose_err_msg)

    return biotsavart_tag, other_tags

def generate_surface_tag(
    tag_dict: dict[str, int | str],
    minimal: bool = True,
    enforce_tags: bool = True,
) -> tuple[str, list[str]]:
    if "surface" not in tag_dict: tag_dict = dict(surface=tag_dict)
    if enforce_tags: check_tags(tag_dict)

    tags = []
    for key in SURFACE_PATTERNS:
        if minimal and (key not in ENFORCE_SURFACE_TAGS): continue
        if key in tag_dict["surface"]:
            tags.append(INVERTERS[key](tag_dict["surface"][key]))
    surface_tag = "_".join(tags)

    err = False
    msg = ""
    other_tags = []
    for key in OTHER_PATTERNS:
        if key in tag_dict:
            pattern = OTHER_PATTERNS[key]
            fullmatch = re.fullmatch(pattern, INVERTERS[key](tag_dict[key]))
            if not fullmatch:
                err = True
                msg += f"Invalid tag format for {key}: {INVERTERS[key](tag_dict[key])}, expected pattern: {pattern}\n"
            else:
                other_tags.append(INVERTERS[key](tag_dict[key]))
    
    if err:
        verbose_err_msg = f"Error generating Surface tags:\n{msg}"
        raise ValueError(verbose_err_msg)

    return surface_tag, other_tags

def generate_boozersurface_tags(
    tag_dict: dict[str, int | str],
    minimal: bool = True,
    enforce_tags: bool = True,
) -> tuple[str, str, str, list[str]]:
    if enforce_tags: check_tags(tag_dict)

    err = False
    msg = ""
    if "biotsavart" in tag_dict:
        # Ignore the BiotSavart version_number_str and iter_number tags
        biotsavart_tag, _ = generate_biotsavart_tag(tag_dict["biotsavart"], minimal=minimal, enforce_tags=enforce_tags)
    else:
        err = True
        msg += "Missing `biotsavart` tags\n"

    if "surface" in tag_dict:
        # Ignore the Surface version_number_str and iter_number tags
        surface_tag, _ = generate_surface_tag(tag_dict["surface"], minimal=minimal, enforce_tags=enforce_tags)
    else:
        err = True
        msg += "Missing `surface` tags\n"

    if "boozersurface" in tag_dict:
        boozersurface_tags = []
        for key in BOOZERSURFACE_PATTERNS:
            if minimal and (key not in ENFORCE_BOOZERSURFACE_TAGS): continue
            if key in tag_dict["boozersurface"]:
                boozersurface_tags.append(INVERTERS[key](tag_dict["boozersurface"][key]))
        boozersurface_tag = "_".join(boozersurface_tags)
    else:
        err = True
        msg += "Missing `boozersurface` tags\n"
    
    other_tags = []
    for key in OTHER_PATTERNS:
        if key in tag_dict:
            pattern = OTHER_PATTERNS[key]
            fullmatch = re.fullmatch(pattern, INVERTERS[key](tag_dict[key]))
            if not fullmatch:
                err = True
                msg += f"Invalid tag format for {key}: {INVERTERS[key](tag_dict[key])}, expected pattern: {pattern}\n"
            else:
                other_tags.append(INVERTERS[key](tag_dict[key]))
    
    if err:
        verbose_err_msg = f"Error generating BoozerSurface tags:\n{msg}"
        raise ValueError(verbose_err_msg)
    
    return biotsavart_tag, surface_tag, boozersurface_tag, other_tags

def generate_boozersurface_filename(tag_dict: dict[str, int | str], minimal: bool = True, enforce_tags: bool = True) -> str:
    biotsavart_tag, surface_tag, boozersurface_tag, other_tags = generate_boozersurface_tags(tag_dict, minimal=minimal, enforce_tags=enforce_tags)
    filename = ".".join([biotsavart_tag, surface_tag, boozersurface_tag])
    if len(other_tags):
        filename += "." + ".".join(other_tags)
    return filename+".json"

def generate_random_tag(n: int = 8, a_vs_d: float = 2/3) -> str:
    a_vs_d = max(0, min(1, a_vs_d))
    alphas_n = int(n * a_vs_d)
    digits_n = n - alphas_n
    alpha_characters = random.choices(string.ascii_letters, k=alphas_n)
    digit_characters = random.choices(string.digits, k=digits_n)
    characters = alpha_characters + digit_characters
    random.shuffle(characters)
    return "".join(characters)

def load_tags_from_biotsavart(biotsavart: str | BiotSavart, biotsavart_tag: str = "biotsavart", stage: str = "init") -> dict[str, int | str]:
    if isinstance(biotsavart, str):
        filename = biotsavart
        biotsavart = load(biotsavart)
        partial_tag_dict = resolve_biotsavart_json_filename(filename, enforce_tags=False)
        biotsavart_tag = partial_tag_dict["biotsavart"]["tag"]
        stage = partial_tag_dict["biotsavart"]["stage"]
    
    coils = biotsavart.coils
    ncoils = len(coils)
    assert (ncoils in [N_COILS, N_FB_COILS]), f"Expected {N_COILS} or {N_FB_COILS} (finite-build) coils, got {ncoils}"
    is_finitebuild = (ncoils == N_FB_COILS)

    banana_curve = coils[BANANA_IDX].curve
    order = banana_curve.order
    nqpts = banana_curve.quadpoints.size

    proxy_idx = BANANA_IDX + N_BANANA*hbt_banana_fb.numfilaments if is_finitebuild else PROXY_IDX
    proxy_coil = coils[proxy_idx]
    proxy_current_ka = proxy_coil.current.get_value()/1e3
    proxy_current_ka_str = str(round(proxy_current_ka, 6)).replace(".", "d")

    tag_dict = dict(
        tag=biotsavart_tag,
        order=order,
        nqpts=nqpts,
        proxy_current_ka_str=proxy_current_ka_str,
        stage=stage,
    )
    if is_finitebuild: tag_dict["finitebuild"] = "finitebuild"
    return dict(biotsavart=tag_dict)

def load_tags_from_surface(surface: str | Surface, surface_tag: str = "surface", stage: str = "init") -> dict[str, int | str]:
    if isinstance(surface, str):
        filename = surface
        surface = load(surface)
        partial_tag_dict = resolve_surface_json_filename(filename, enforce_tags=False)
        surface_tag = partial_tag_dict["surface"]["tag"]
        stage = partial_tag_dict["surface"]["stage"]

    mpol = surface.mpol
    ntor = surface.ntor
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size

    tag_dict = dict(
        tag=surface_tag,
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
        stage=stage,
    )
    return dict(surface=tag_dict)

def load_tags_from_boozersurface(
    boozersurface: str | BoozerSurface,
    volume_target_str: str = "Surface",
    biotsavart_tag: str = "biotsavart",
    surface_tag: str = "surface",
    biotsavart_stage: str = "init",
    surface_stage: str = "init",
) -> dict[str, int | str]:
    if isinstance(boozersurface, str):
        filename = boozersurface
        partial_tag_dict = resolve_boozersurface_json_filename(filename, enforce_tags=False)
        biotsavart_tag = partial_tag_dict["biotsavart"]["tag"]
        surface_tag = partial_tag_dict["surface"]["tag"]
        boozersurface_tag = partial_tag_dict["boozersurface"]["tag"]
        biotsavart_stage = partial_tag_dict["biotsavart"]["stage"]
        surface_stage = partial_tag_dict["surface"]["stage"]
        volume_target_str = partial_tag_dict["boozersurface"]["volume_target_str"] # overrides input kwarg
        boozersurface = load(boozersurface)
    else:
        boozersurface_tag = "boozersurface"
        partial_tag_dict = dict()

    constraint_weight = boozersurface.constraint_weight
    constraint_weight_str = "Exact" if (constraint_weight is None) else str(round(constraint_weight, 6)).replace(".", "d")
    boozersurface_tag_dict = dict(boozersurface=dict(
        tag=boozersurface_tag,
        constraint_weight_str=constraint_weight_str,
        volume_target_str=volume_target_str
    ))

    biotsavart = boozersurface.biotsavart
    biotsavart_tag_dict = load_tags_from_biotsavart(
        biotsavart, biotsavart_tag=biotsavart_tag, stage=biotsavart_stage)

    surface = boozersurface.surface
    surface_tag_dict = load_tags_from_surface(
        surface, surface_tag=surface_tag, stage=surface_stage)

    tag_dict = dict(
        **boozersurface_tag_dict,
        **biotsavart_tag_dict,
        **surface_tag_dict,
    )
    for key in OTHER_PATTERNS:
        if key in partial_tag_dict:
            tag_dict[key] = partial_tag_dict[key]
    
    return tag_dict

def update_boozersurface_tags_from_args(
    args: dict,
) -> dict[str, int | str]:
    assert "boozersurface_file" in args, "Expected 'boozersurface_file' in args"

    boozersurface_file = args["boozersurface_file"]
    boozersurface = load_boozersurface(boozersurface_file, args=args)
    old_tag_dict = load_tags_from_boozersurface(boozersurface_file)
    if "volume_target_str" not in args:
        volume_target_str = old_tag_dict["boozersurface"]["volume_target_str"]
    else:
        volume_target_str = args["volume_target_str"].replace(".", "d").replace("%", "pct")
    new_tag_dict = load_tags_from_boozersurface(boozersurface, volume_target_str=volume_target_str)
    new_tag_dict["biotsavart"]["tag"] = old_tag_dict["biotsavart"]["tag"]
    new_tag_dict["surface"]["tag"] = old_tag_dict["surface"]["tag"]
    new_tag_dict["boozersurface"]["tag"] = old_tag_dict["boozersurface"]["tag"]
    return new_tag_dict, boozersurface

def compare_tags(tag_dict1: dict[str, int | str], tag_dict2: dict[str, int | str]) -> bool:
    if len(tag_dict1) != len(tag_dict2):
        return False

    all_subdicts = []
    for key, val in tag_dict1.items():
        if key not in tag_dict2:
            return False
        
        if isinstance(val, dict):
            compare_subdict = compare_tags(val, tag_dict2[key])
            all_subdicts.append(compare_subdict)
        else:
            if val != tag_dict2[key]:
                return False
            
    return all(all_subdicts)
        
