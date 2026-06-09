import argparse
import yaml

from ..objectives.defaults import *
from ..hardware import hardware_limits, DEFAULT_TF_CURRENT_KA, DEFAULT_BANANA_CURRENT_KA
from .coils import DEFAULT_BANANA_ORDER, DEFAULT_QPTS_PER_ORDER
from .surface import DEFAULT_MPOL, DEFAULT_NTOR, DEFAULT_NTHETA, DEFAULT_NPHI
from .stages import STAGE2, STAGES

DEFAULT_SIGN_G = -1

DEFAULT_MAXITER = 1500
DEFAULT_ITER_FREQ = 10
DEFAULT_OUT_DIR = "./"
DEFAULT_TOL = 1e-15

DRIVER_DEFAULTS = dict(
    sign_g=DEFAULT_SIGN_G,
    maxiter=DEFAULT_MAXITER,
    save_iter_dir=None,
    save_iter_freq=DEFAULT_ITER_FREQ,
    out_dir=DEFAULT_OUT_DIR,
    tol=DEFAULT_TOL,
)

FIX_CURRENT_KW_PATTERN = r"^(?P<coil_type>tf|banana|vf)_(?P<fix_status>(no_)?fix)_current$"

# Inputs from parsers that feed into filename tags
FILENAME_TAG_INPUTS = [
    "banana_order",
    "banana_qpts_per_order",
    "proxy_current_ka",
    "mpol",
    "ntor",
    "ntheta",
    "nphi",
    "constraint_weight",
    "volume_target_str",
]

def tf_coil_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tf-current-ka", type=float, default=None,
                   help=f"TF coil current, kA. If None, inherit from existing configuration or {DEFAULT_TF_CURRENT_KA}. Default: None.")
    p.add_argument("--tf-fix-current", action=argparse.BooleanOptionalAction, default=None,
                   help=f"Fix/free the TF current. If None, inherit from existing configuration or True. Default: None.")
    return p

def banana_coil_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--banana-current-ka", type=float, default=None,
                   help=f"Banana coil current, kA. If None, inherit from existing configuration or {DEFAULT_BANANA_CURRENT_KA}. Default: None.")
    p.add_argument("--banana-fix-current", action=argparse.BooleanOptionalAction, default=None,
                   help=f"Fix/free the banana current. If None, inherit from existing configuration or True. Default: None.")
    p.add_argument("--banana-order", type=int, default=None,
                   help=f"Fourier order of the banana coil. If None, inherit from existing configuration or {DEFAULT_BANANA_ORDER}. Default: None.")
    p.add_argument("--banana-qpts-per-order", type=int, default=None,
                   help=f"Quadrature points per Fourier order. If None, inherit from existing configuration or {DEFAULT_QPTS_PER_ORDER}. Default: None.")
    return p

def proxy_coil_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proxy-current-ka", type=float, default=None,
                   help=f"Proxy coil current, kA. If None, inherit from existing configuration or 0.0. Default: None.")
    p.add_argument("--proxy-rz", nargs="+", type=float, default=[],
                   help=f"RZ coordinates of the proxy coil. Default: inherit from existing BiotSavart.")
    return p

def vf_coil_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vf-current-ka", type=float, default=None,
                   help=f"VF coil current, kA. If None, inherit from existing configuration or 0.0. Default: None.")
    p.add_argument("--vf-fix-current", action=argparse.BooleanOptionalAction, default=None,
                   help=f"Fix/free the VF currents. If None, inherit from existing configuration or True. Default: None.")
    return p

def surface_resolution_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mpol", type=int, default=None,
                   help=f"Number of poloidal modes. If None, inherit from existing Surface or {DEFAULT_MPOL}. Default: None.")
    p.add_argument("--ntor", type=int, default=None,
                   help=f"Number of toroidal modes. If None, inherit from existing Surface or {DEFAULT_NTOR}. Default: None.")
    p.add_argument("--nphi", type=int, default=None,
                   help=f"Number of phi points. If None, inherit from existing Surface or {DEFAULT_NPHI}. Default: None.")
    p.add_argument("--ntheta", type=int, default=None,
                   help=f"Number of theta points. If None, inherit from existing Surface or {DEFAULT_NTHETA}. Default: None.")
    return p

def stage2_objectives_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--weight-sqflux", type=float, default=None,
                   help=f"Weight for squared flux objective. Default: {DEFAULT_WEIGHT_SQFLUX}.")
    return p

def singlestage_objectives_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max-coil-surface-distance", type=float, default=None,
                   help="Maximum coil-surface distance. Default: None.")
    p.add_argument("--min-coil-surface-distance", type=float, default=None,
                   help=f"Minimum coil-surface distance. Default: {hardware_limits.min_csdist} m.")
    p.add_argument("--weight-non-quasisymmetric-ratio", type=float, default=None,
                   help=f"Weight for non-quasisymmetry objective. Default: {DEFAULT_WEIGHT_NONQS}.")
    p.add_argument("--weight-boozer-residual", type=float, default=None,
                   help=f"Weight for Boozer residual objective. Default: {DEFAULT_WEIGHT_BRES}.")
    p.add_argument("--weight-iota", type=float, default=None,
                   help=f"Weight for iota objective. Default: {DEFAULT_WEIGHT_IOTA}.")
    p.add_argument("--weight-coil-surface-distance", type=float, default=None,
                   help=f"Weight for curve-surface distance objective. Default: {DEFAULT_WEIGHT_COIL_SURFACE_DISTANCE}.")
    return p

def shared_objectives_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max-coil-length", type=float, default=None,
                   help=f"Maximum coil length. Default: {hardware_limits.max_length} m.")
    p.add_argument("--min-coil-length", type=float, default=None,
                   help="Minimum coil length. Default: None.")
    p.add_argument("--max-coil-curvature", type=float, default=None,
                   help=f"Maximum coil curvature. Default: {hardware_limits.max_curvature} 1/m.")
    p.add_argument("--min-coil-curvature", type=float, default=None,
                   help="Minimum coil curvature. Default: None.")
    p.add_argument("--max-coil-coil-distance", type=float, default=None,
                   help="Maximum coil-coil distance. Default: None.")
    p.add_argument("--min-coil-coil-distance", type=float, default=None,
                   help=f"Minimum coil-coil distance. Default: {hardware_limits.min_ccdist} m.")
    p.add_argument("--weight-coil-length", type=float, default=None,
                   help=f"Weight for coil length objective. Default: {DEFAULT_WEIGHT_COIL_LENGTH}.")
    p.add_argument("--weight-coil-curvature", type=float, default=None,
                   help=f"Weight for coil curvature objective. Default: {DEFAULT_WEIGHT_COIL_CURVATURE}.")
    p.add_argument("--weight-coil-coil-distance", type=float, default=None,
                   help=f"Weight for coil-coil distance objective. Default: {DEFAULT_WEIGHT_COIL_COIL_DISTANCE}.")
    return p

def custom_objectives_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max-poloidal-extent", type=float, default=None,
                   help=f"Maximum poloidal extent. Default: {DEFAULT_MAX_POLOIDAL_EXTENT} degrees.")
    p.add_argument("--min-poloidal-extent", type=float, default=None,
                   help=f"Minimum poloidal extent. Default: None.")
    p.add_argument("--max-ellipse-width", type=float, default=None,
                   help=f"Maximum ellipse width. Default: {DEFAULT_MAX_ELLIPSE_WIDTH} m.")
    p.add_argument("--min-ellipse-width", type=float, default=None,
                   help=f"Minimum ellipse width. Default: {DEFAULT_MIN_ELLIPSE_WIDTH} m.")
    p.add_argument("--max-coil-self-distance", type=float, default=None,
                   help=f"Maximum coil self-distance. Default: None.")
    p.add_argument("--min-coil-self-distance", type=float, default=None,
                   help=f"Minimum coil self-distance. Default: {DEFAULT_MIN_COIL_SELF_DISTANCE} m.")
    p.add_argument("--max-tf-current-ka", type=float, default=None,
                   help=f"Maximum TF coil current, kA. Default: {DEFAULT_TF_CURRENT_KA} kA.")
    p.add_argument("--min-tf-current-ka", type=float, default=None,
                   help=f"Minimum TF coil current, kA. Default: None.")
    p.add_argument("--max-banana-current-ka", type=float, default=None,
                   help=f"Maximum banana coil current, kA. Default: {DEFAULT_BANANA_CURRENT_KA} kA.")
    p.add_argument("--min-banana-current-ka", type=float, default=None,
                   help=f"Minimum banana coil current, kA. Default: None.")
    p.add_argument("--weight-poloidal-extent", type=float, default=None,
                   help=f"Weight for poloidal extent objective. Default: {DEFAULT_WEIGHT_POLOIDAL_EXTENT}.")
    p.add_argument("--weight-ellipse-width", type=float, default=None,
                   help=f"Weight for ellipse width objective. Default: {DEFAULT_WEIGHT_ELLIPSE_WIDTH}.")
    p.add_argument("--weight-coil-self-distance", type=float, default=None,
                   help=f"Weight for coil self-distance objective. Default: {DEFAULT_WEIGHT_COIL_SELF_DISTANCE}.")
    p.add_argument("--weight-currents", type=float, default=None,
                   help=f"Weight for TF and banana coil current objectives. Default: {DEFAULT_WEIGHT_CURRENTS}.")
    return p

def boozersurface_parameters_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--constraint-weight", type=float, default=None, help="Constraint weight. If None, inherits from `boozersurface_file`. Set constraint weight == 0 to use BoozerExact. Default: None.")
    p.add_argument("--volume-target-str", type=str, default=None, help="Volume target. If None, inherits from `boozersurface_file`. Input either a float or a percentage. Default: None.")
    return p

def driver_parser(stage):
    assert stage in STAGES, f"Invalid stage: {stage}. Must be one of {STAGES}"
    is_stage2 = (stage == STAGE2)
    description = (
        "Stage 2 — Optimization of coils." if is_stage2 else
        "Singlestage  -- Optimization of coils and surface via Boozer surface."
    )
    boozersurface_help = "Input BoozerSurface JSON file."
    if is_stage2:
        boozersurface_help += " A BoozerSurface object is used for convenience since it stores a BiotSavart object and Surface object."
        objectives_parser = stage2_objectives_parser
    else:
        objectives_parser = singlestage_objectives_parser

    parents = [
        tf_coil_parser(),
        banana_coil_parser(),
        proxy_coil_parser(),
        vf_coil_parser(),
        surface_resolution_parser(),
        shared_objectives_parser(),
        custom_objectives_parser(),
        objectives_parser(),
    ]
    if not is_stage2:
        parents.append(boozersurface_parameters_parser())
        
    p = argparse.ArgumentParser(
        description=description,
        parents=parents,
    )
    p.add_argument("boozersurface_file", type=str, help=boozersurface_help)
    p.add_argument("--config-file", type=str, default=None,
                   help=f"Config YAML. Default: None.")
    p.add_argument("--maxiter", type=int, default=None,
                   help=f"Optimizer iteration cap. Default: {DEFAULT_MAXITER}.")
    p.add_argument("--save-iter-dir", type=str, default=None,
                   help="Directory to save iteration data. Default: None.")
    p.add_argument("--save-iter-freq", type=int, default=None,
                   help=f"Frequency to save iteration data. Default: {DEFAULT_ITER_FREQ}.")
    p.add_argument("--out-dir", type=str, default=None,
                   help=f"Output directory. Default: {DEFAULT_OUT_DIR}.")
    if is_stage2:
        p.add_argument("--vcasing-file", type=str, default=None, help="Virtual casing netCDF file.")
        p.add_argument("--tol", type=float, default=None, help=f"Tolerance for optimization convergence. Default: {DEFAULT_TOL}.") # Only for stage 2 for now since singlestage has it's own (f|g)tol per mpol
    else:
        p.add_argument("iota", type=float, help="Target iota")
        p.add_argument("--sign-g", type=int, choices=[-1, 1], default=None, help=f"Sign of G for Boozer solve. Default: {DEFAULT_SIGN_G}.")
    return p

def load_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)
    
def write_config_yaml(filename, args):
    inputs, overrides, _ = process_driver_args(args)
    save_args = dict(overrides)
    save_args.update(inputs)
    with open(filename, "w") as f:
        yaml.dump(save_args, f)

# config file contains parameters to pass in as CLI args with explicit CLI args taking priority
# inputs is the actual CLI input options passed by the user
# overrides are from the config file
# args are the collection that is actually passed to the driver
def process_driver_args(args) -> dict:
    inputs = dict()
    for key, val in vars(args).items():
        if val is not None:
            if isinstance(val, list) and len(val) == 0:
                continue # skip empty lists since they may have defaults in the config file
            inputs[key] = val
    if "config_file" in inputs:
        overrides = load_yaml(inputs["config_file"])
        if "config_file" in overrides: # avoid nesting configs
            del overrides["config_file"]
    else:
        overrides = dict()

    driver_args = dict(overrides)
    driver_args.update(inputs) # CLI inputs take priority over config file
    driver_args.update({key:val for key, val in DRIVER_DEFAULTS.items() if key not in driver_args})
    return inputs, overrides, driver_args
