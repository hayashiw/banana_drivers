import argparse
import os
import re

from ..hardware import (
    DEFAULT_TF_CURRENT_KA,
    DEFAULT_BANANA_CURRENT_KA,
)
from .coils import (
    DEFAULT_BANANA_ORDER,
    DEFAULT_QPTS_PER_ORDER,
)
from .surface import (
    DEFAULT_NPHI,
    DEFAULT_NTHETA,
)

def tf_coil_parser(inherit=False):
    tf_current  = None if inherit else DEFAULT_TF_CURRENT_KA
    fix_current = None if inherit else True

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tf-current-ka", type=float, default=tf_current,
                   help=f"TF coil current, kA. Default: {'inherit' if inherit else tf_current}.")
    p.add_argument("--tf-fix-current", action=argparse.BooleanOptionalAction, default=fix_current,
                   help=f"Fix/free the TF current. Default: {'inherit' if inherit else fix_current}.")
    return p

def banana_coil_parser(inherit=False):
    banana_current = None if inherit else DEFAULT_BANANA_CURRENT_KA
    banana_order   = None if inherit else DEFAULT_BANANA_ORDER
    qpts_per_order = None if inherit else DEFAULT_QPTS_PER_ORDER
    fix_current    = None if inherit else True

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--banana-current-ka", type=float, default=banana_current,
                   help=f"Banana coil current, kA. Default: {'inherit' if inherit else banana_current}.")
    p.add_argument("--banana-fix-current", action=argparse.BooleanOptionalAction, default=fix_current,
                   help=f"Fix/free the banana current. Default: {'inherit' if inherit else fix_current}.")
    p.add_argument("--banana-order", type=int, default=banana_order,
                   help=f"Fourier order of the banana coil. Default: {'inherit' if inherit else banana_order}.")
    p.add_argument("--banana-qpts-per-order", type=int, default=qpts_per_order,
                   help=f"Quadrature points per Fourier order. Default: {'inherit' if inherit else qpts_per_order}.")
    return p

def proxy_coil_parser(inherit=False):
    proxy_current = None if inherit else 0.0
    
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--proxy-current-ka", type=float, default=proxy_current,
                   help=f"Proxy coil current, kA. Default: {'inherit' if inherit else proxy_current}.")
    p.add_argument("--proxy-rz", nargs="+", type=float, default=[],
                   help=f"RZ coordinates of the proxy coil. Default: inherit from existing BiotSavart.")
    return p

def vf_coil_parser(inherit=False):
    vf_current  = None if inherit else 0.0
    fix_current = None if inherit else True

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vf-current-ka", type=float, default=vf_current,
                   help=f"VF coil current, kA. Default: {'inherit' if inherit else vf_current}.")
    p.add_argument("--vf-fix-current", action=argparse.BooleanOptionalAction, default=fix_current,
                   help=f"Fix/free the VF currents. Default: {'inherit' if inherit else fix_current}.")
    return p

def surface_resolution_parser(inherit=False):
    nphi = None if inherit else DEFAULT_NPHI
    ntheta = None if inherit else DEFAULT_NTHETA

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mpol", type=int, default=None,
                   help=f"Number of poloidal modes. Default: inherit from existing Surface.")
    p.add_argument("--ntor", type=int, default=None,
                   help=f"Number of toroidal modes. Default: inherit from existing Surface.")
    p.add_argument("--nphi", type=int, default=nphi,
                   help=f"Number of phi points. Default: {'inherit' if inherit else nphi}.")
    p.add_argument("--ntheta", type=int, default=ntheta,
                   help=f"Number of theta points. Default: {'inherit' if inherit else ntheta}.")
    return p

def objectives_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-min-length", action="store_true",
                   help="If True, skip the coil length minimum penalty. Default: False.")
    p.add_argument("--no-width", action="store_true",
                   help="If True, skip the coil width penalties. Default: False.")
    p.add_argument("--no-current", action="store_true",
                   help="If True, skip the coil current penalties. Default: False.")
    p.add_argument("--max-curvature-override", type=float, default=None,
                   help="Override the maximum curvature for the banana coils. Default: None.")
    return p

