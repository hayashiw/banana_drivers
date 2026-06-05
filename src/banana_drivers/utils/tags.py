import os
import random
import re
import string

from collections import OrderedDict

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

from ..hardware import hbt_banana_fb, BANANA_IDX, PROXY_IDX, N_BANANA, N_COILS, N_FB_COILS

FLOAT_PATTERN = r"\d+d\d+|\d+"

BIOTSAVART_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>[a-zA-Z0-9]+)$",
    proxy_current_ka_str=rf"proxy(?P<proxy_current_ka_str>{FLOAT_PATTERN})kA",
    order=r"^o(?P<order>\d+)$",
    nqpts=r"^nqpts(?P<nqpts>\d+)$",
    virtualcasing=r"^(?P<virtualcasing>virtualcasing)$",
    stage=r"^(?P<stage>init|(stage2|singlestage)opt)$",
    finitebuild=r"^(?P<finitebuild>finitebuild)$",
)

ENFORCE_BIOTSAVART_TAGS = ["tag", "proxy_current_ka_str", "order", "stage"]

SURFACE_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>[a-zA-Z0-9]+)$",
    mpol=r"m(?P<mpol>\d+)$",
    ntor=r"n(?P<ntor>\d+)$",
    nphi=r"np(?P<nphi>\d+)$",
    ntheta=r"nt(?P<ntheta>\d+)$",
    stage=r"^(?P<stage>init|presolved|(stage2|singlestage)opt)$"
)

ENFORCE_SURFACE_TAGS = ["tag", "mpol", "ntor", "stage"]

BOOZERSURFACE_PATTERNS = OrderedDict(
    tag=r"^(?P<tag>boozersurface)$",
    constraint_weight_str=rf"^cw(?P<constraint_weight_str>Exact|{FLOAT_PATTERN})$",
    volume_target_str=rf"^vol(?P<volume_target_str>Surface|{FLOAT_PATTERN})$"
)

ENFORCE_BOOZERSURFACE_TAGS = ["tag", "constraint_weight_str", "volume_target_str"]

OTHER_PATTERNS = OrderedDict(
    version_number=r"^vers(?P<version_number>\d+)$",
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
    version_number=int,
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
    version_number=lambda x: f"vers{x}",
    iter_number=lambda x: f"iter{x}"
)

def _resolve_tags(tag: str, source: str) -> dict:
    assert source in ["biotsavart", "surface", "boozersurface"], f"Expected `source` to be 'biotsavart', 'surface' or 'boozersurface', got {source}"
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
    return tag_dict

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

def resolve_boozersurface_json_filename(filename: str, enforce_tags: bool = True) -> dict:
    biotsavart_tag, surface_tag, boozersurface_tag, *other = \
        os.path.basename(filename).removesuffix(".json").split(".")

    biotsavart_tag_dict = _resolve_tags(biotsavart_tag, "biotsavart")
    surface_tag_dict = _resolve_tags(surface_tag, "surface")
    boozersurface_tag_dict = _resolve_tags(boozersurface_tag, "boozersurface")
    other_tags_dict = _resolve_other_tags(other)

    if enforce_tags:
        err = False
        msg = {}
        for enforce_tag_list, tag_dict, label in [
            (ENFORCE_BIOTSAVART_TAGS, biotsavart_tag_dict, "biotsavart"),
            (ENFORCE_SURFACE_TAGS, surface_tag_dict, "surface"),
            (ENFORCE_BOOZERSURFACE_TAGS, boozersurface_tag_dict, "boozersurface")
        ]:
            for tag in enforce_tag_list:
                if tag not in tag_dict:
                    if label not in msg: msg[label] = ""
                    msg[label] += f"    Missing tag: {tag}\n"
                    err = True
        assert (not err), f"Missing tags:\n{msg}"

    full_tag_dict = dict(
        biotsavart=biotsavart_tag_dict,
        surface=surface_tag_dict,
        boozersurface=boozersurface_tag_dict,
    )
    full_tag_dict.update(other_tags_dict)
    return full_tag_dict

def generate_surface_json_filename(tag_dict: dict[str, int | str], minimal=False) -> str:
    tags = []
    for key in SURFACE_PATTERNS:
        if minimal and (key not in ENFORCE_SURFACE_TAGS): continue
        if key in tag_dict:
            tags.append(INVERTERS[key](tag_dict[key]))
    filename = "_".join(tags) + ".json"
    return filename

def generate_biotsavart_json_filename(tag_dict: dict[str, int | str], minimal=False) -> str:
    tags = []
    for key in BIOTSAVART_PATTERNS:
        if minimal and (key not in ENFORCE_BIOTSAVART_TAGS): continue
        if key in tag_dict:
            tags.append(INVERTERS[key](tag_dict[key]))
    if (
        ("proxy_current_ka_str" not in tag_dict) and
        ("proxy_current_ka" in tag_dict)
    ):
        key = "proxy_current_ka"
        tags.append(INVERTERS[key](tag_dict[key]))
    filename = "_".join(tags) + ".json"
    return filename

def generate_boozersurface_json_filename(tag_dict: dict[str, int | str], minimal=False) -> str:
    tags = []
    if "biotsavart" in tag_dict:
        biotsavart_tags = []
        for key in BIOTSAVART_PATTERNS:
            if minimal and (key not in ENFORCE_BIOTSAVART_TAGS): continue
            if key in tag_dict["biotsavart"]:
                biotsavart_tags.append(INVERTERS[key](tag_dict["biotsavart"][key]))
        if (
            ("proxy_current_ka_str" not in tag_dict["biotsavart"]) and
            ("proxy_current_ka" in tag_dict["biotsavart"])
        ):
            key = "proxy_current_ka"
            biotsavart_tags.append(INVERTERS[key](tag_dict["biotsavart"][key]))
        tags.append("_".join(biotsavart_tags))
    if "surface" in tag_dict:
        surface_tags = []
        for key in SURFACE_PATTERNS:
            if minimal and (key not in ENFORCE_SURFACE_TAGS): continue
            if key in tag_dict["surface"]:
                surface_tags.append(INVERTERS[key](tag_dict["surface"][key]))
        tags.append("_".join(surface_tags))
    if "boozersurface" in tag_dict:
        boozersurface_tags = []
        for key in BOOZERSURFACE_PATTERNS:
            if minimal and (key not in ENFORCE_BOOZERSURFACE_TAGS): continue
            if key in tag_dict["boozersurface"]:
                boozersurface_tags.append(INVERTERS[key](tag_dict["boozersurface"][key]))
        tags.append("_".join(boozersurface_tags))
    for key in OTHER_PATTERNS:
        if key in tag_dict:
            other_tag = INVERTERS[key](tag_dict[key])
            tags.append(other_tag)
    filename = ".".join(tags) + ".json"
    return filename

def load_tags_from_biotsavart(biotsavart: str | BiotSavart) -> dict[str, int | str]:
    if isinstance(biotsavart, str):
        biotsavart = load(biotsavart)
    
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
        order=order,
        nqpts=nqpts,
        proxy_current_ka_str=proxy_current_ka_str,
    )
    if is_finitebuild: tag_dict["finitebuild"] = "finitebuild"
    return tag_dict

def load_tags_from_surface(surface: str | Surface) -> dict[str, int | str]:
    if isinstance(surface, str):
        surface = load(surface)

    mpol = surface.mpol
    ntor = surface.ntor
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size

    tag_dict = dict(
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
    )
    return tag_dict

def load_tags_from_boozersurface(boozersurface: str | BoozerSurface, volume: float | None = None) -> dict[str, int | str]:
    if volume == 0: volume = None
    if isinstance(boozersurface, str):
        filename = boozersurface
        partial_tag_dict = resolve_boozersurface_json_filename(filename, enforce_tags=False)
        biotsavart_tag = partial_tag_dict["biotsavart"]["tag"]
        surface_tag = partial_tag_dict["surface"]["tag"]
        boozersurface_tag = partial_tag_dict["boozersurface"]["tag"]
        boozersurface = load(boozersurface)
    else:
        biotsavart_tag = "biotsavart"
        boozersurface_tag = "boozersurface"
        surface_tag = "surface"
        partial_tag_dict = dict()
    biotsavart_tag_dict = dict(tag=biotsavart_tag)
    surface_tag_dict = dict(tag=surface_tag)
    boozersurface_tag_dict = dict(tag=boozersurface_tag)

    constraint_weight = boozersurface.constraint_weight
    constraint_weight_str = "Exact" if (constraint_weight is None) else str(round(constraint_weight, 6)).replace(".", "d")
    volume_str = "Surface" if (volume is None) else str(round(volume, 6)).replace(".", "d")
    boozersurface_tag_dict.update(
        constraint_weight_str=constraint_weight_str,
        volume_target_str=volume_str
    )

    biotsavart = boozersurface.biotsavart
    biotsavart_tag_dict.update(
        load_tags_from_biotsavart(biotsavart)
    )

    surface = boozersurface.surface
    surface_tag_dict.update(
        load_tags_from_surface(surface)
    )

    tag_dict = dict(
        boozersurface=boozersurface_tag_dict,
        biotsavart=biotsavart_tag_dict,
        surface=surface_tag_dict,
    )
    for key in OTHER_PATTERNS:
        if key in partial_tag_dict:
            tag_dict[key] = partial_tag_dict[key]
    
    return tag_dict

def generate_random_tag(n: int = 8, a_vs_d: float = 2/3) -> str:
    a_vs_d = max(0, min(1, a_vs_d))
    alphas_n = int(n * a_vs_d)
    digits_n = n - alphas_n
    alpha_characters = random.choices(string.ascii_letters, k=alphas_n)
    digit_characters = random.choices(string.digits, k=digits_n)
    characters = alpha_characters + digit_characters
    random.shuffle(characters)
    return "".join(characters)

