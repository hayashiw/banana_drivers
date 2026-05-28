import argparse
import os

from banana_drivers.paths import BANANA_INIT_DOFS, BIOTSAVART_INIT, WOUT_ORIGINAL
from banana_drivers.hardware import hardware_limits, DEFAULT_BANANA_ORDER, DEFAULT_PROXY_RZ
from banana_drivers.utils.coils import (
    tf_max_current_ka,
    banana_max_current_ka,
)

DEFAULT_WOUT_S = 0.24
DEFAULT_WOUT_SCALE = 0.925

def coil_current_parser(defaults_none=False):
    if defaults_none:
        tf_def     = None
        banana_def = None
        proxy_def  = None
        vf_def     = None
        fix_def    = None
    else:
        tf_limits = hardware_limits.tf_current_ka_limits
        tf_max = 0
        for lim in tf_limits:
            if abs(lim) > abs(tf_max):
                tf_max = lim
        tf_def     = tf_max
        banana_limits = hardware_limits.banana_current_ka_limits
        banana_max = 0
        for lim in banana_limits:
            if abs(lim) > abs(banana_max):
                banana_max = lim
        banana_def = banana_max
        proxy_def  = 0.0
        vf_def     = 0.0
        fix_def    = True

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tf-current", type=float, default=tf_def,
                   help=f"TF coil current, kA. Default: inherit / hardware value ({tf_max_current_ka} kA).")
    p.add_argument("--tf-fix-current", action=argparse.BooleanOptionalAction, default=fix_def,
                   help="Fix/free the TF current. Omit to inherit.")
    p.add_argument("--banana-current", type=float, default=banana_def,
                   help=f"Banana coil current, kA. Default: inherit / hardware value ({banana_max_current_ka} kA).")
    p.add_argument("--banana-fix-current", action=argparse.BooleanOptionalAction, default=fix_def,
                   help="Fix/free the banana current. Omit to inherit.")
    p.add_argument("--proxy-current", type=float, default=proxy_def,
                   help="Proxy coil current, kA. Default: inherit / 0.")
    p.add_argument("--vf-current", type=float, default=vf_def,
                   help="VF coil current, kA. Default: inherit / 0.")
    p.add_argument("--vf-fix-current", action=argparse.BooleanOptionalAction, default=fix_def,
                   help="Fix/free the VF currents. Omit to inherit.")
    return p

def coil_geometry_parser(defaults_none=False):
    if defaults_none:
        banana_order_def = None
        banana_init_def  = None
        proxy_rz_def     = None
    else:
        banana_order_def = DEFAULT_BANANA_ORDER
        banana_init_def  = BANANA_INIT_DOFS
        proxy_rz_def     = DEFAULT_PROXY_RZ

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--banana-order", type=int, default=banana_order_def,
                   help=f"Banana Fourier order. Default: inherit / {DEFAULT_BANANA_ORDER}.")
    p.add_argument("--banana-init-file", type=str, default=banana_init_def,
                   help=f"Banana init-dofs file. Default: inherit / {BANANA_INIT_DOFS}.")
    p.add_argument("--proxy-rz", type=float, nargs=2, default=proxy_rz_def,
                   help=f"Proxy coil R,Z, meters. Default: inherit / {DEFAULT_PROXY_RZ}.")
    return p

def input_source_parser():
    p = argparse.ArgumentParser(add_help=False)
    coil_src = p.add_mutually_exclusive_group()
    coil_src.add_argument("--biotsavart-file", type=str, default=BIOTSAVART_INIT,
                          help=f"BiotSavart JSON for the background coil set. "
                               f"Default: {BIOTSAVART_INIT}.")
    coil_src.add_argument("--boozersurface-file", type=str, default=None,
                          help="BoozerSurface JSON — supplies both coils and surface.")
    
    surf_src = p.add_mutually_exclusive_group()
    surf_src.add_argument("--wout-file", type=str, default=WOUT_ORIGINAL,
                     help=f"VMEC wout .nc for the target surface. Default: {WOUT_ORIGINAL}.")
    surf_src.add_argument("--surface-file", type=str, default=None,
                     help="Saved surface .json (overrides --wout-file).")
    p.add_argument("--vmec-s", type=float, default=DEFAULT_WOUT_S,
                   help=f"Normalized toroidal flux for the wout surface used as the boundary surface. Default: {DEFAULT_WOUT_S}.")
    p.add_argument("--scale", type=float, default=DEFAULT_WOUT_SCALE,
                   help=f"Target major radius for the wout surface. Scales surface DOFs by <scale>/surface.major_radius(). Default: {DEFAULT_WOUT_SCALE}.")
    return p

def common_parser(output_pre=None, output_post=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output-dir", type=str, default=os.getcwd(),
                   help=f"Output directory. Default: cwd ({os.getcwd()}).")
    if (output_pre is None) and (output_post is None):
        pre_tag_help = ("Filename tag → <pre>_biotsavart_<post>.json. Default: derived "
                    "from --biotsavart-file / --boozersurface-file basename.")
        post_tag_help = pre_tag_help
    elif output_pre is None:
        pre_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: derived from --biotsavart-file / --boozersurface-file basename."
        post_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: {output_post}."
    elif output_post is None:
        pre_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: {output_pre}."
        post_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: derived from --biotsavart-file / --boozersurface-file basename."
    else:
        pre_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: {output_pre}."
        post_tag_help = f"Filename tag → <pre>_biotsavart_<post>.json. Default: {output_post}."
    p.add_argument("--output-pre", type=str, default=output_pre, help=pre_tag_help)
    p.add_argument("--output-post", type=str, default=output_post, help=post_tag_help)
    p.add_argument("--free-limits", action="store_true",
                   help="If True, skip the TF/banana current hardware-limit check. Default: False.")
    p.add_argument("--overwrite", action="store_true",
                   help="If True, overwrite existing output files. Default: False.")
    return p

def resolve_output_tag(args, fallback=("", "opt")):
    prefix = args.output_pre
    suffix = args.output_post
    if (prefix is not None) and (suffix is not None):
        return prefix, suffix
    for attr in ("boozersurface_file", "biotsavart_file"):
        path = getattr(args, attr, None)
        if path:
            base = os.path.splitext(os.path.basename(path))[0]
            for filetype in ("biotsavart", "boozersurface"):
                if filetype in base:
                    pre = base[:base.find(filetype)]
                    if pre[-1] == "_": pre = pre[:-1]
                    post = base[base.find(filetype)+len(filetype):]
                    if post and post[0] == "_": post = post[1:]
                    if prefix is None: prefix = pre
                    if suffix is None: suffix = post
                    break
            if prefix is None: prefix = ""
            if suffix is None: suffix = base
            return prefix, suffix
    return fallback

def check_current_limits(args, parser):
    if getattr(args, "free_limits", False):
        return
    checks = [
        ("--tf-current",     args.tf_current,     hardware_limits.tf_current_ka_limits),
        ("--banana-current", args.banana_current, hardware_limits.banana_current_ka_limits),
    ]
    for name, value, (lo, hi) in checks:
        if value is None:
            continue
        if not (lo <= value <= hi):
            parser.error(
                f"{name} {value} kA is outside hardware limits "
                f"[{lo}, {hi}] kA — use --free-limits to override."
            )