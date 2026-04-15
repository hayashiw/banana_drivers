"""utils/near_axis_seed.py

Construct a near-axis-expansion seed boundary for VMEC cold-start.

Given physical parameters (R0, a, iota_target, nfp), this module uses the
Landreman-Sengupta near-axis expansion (via pyQSC) to produce:
  - a VMEC boundary (RBC/ZBS Fourier coefficients),
  - a self-consistent magnetic axis guess (raxis_cc/zaxis_cs),
with nonzero helical content that escapes the zero-beta / zero-iota trap
(see cold_start_stage1_prompt.md failure mode #3).

The `phiedge` derivation from (rbtor, a, R0) lives in
`utils/vmec_helpers.compute_phiedge` (shared with utils/vmec_resize.py).
Callers compute phiedge separately and combine with the output of this
module when populating `vmec.indata`.

Reference
---------
local/example_pyqsc_near_axis.py is the pedagogical walkthrough of the
mathematics and pyQSC API. This module is the reusable version.

Usage
-----
>>> from near_axis_seed import near_axis_seed
>>> seed = near_axis_seed(R0=0.92, a=0.074, iota_target=0.15, nfp=5)
>>> seed['rbc']        # {(m, n): value}
>>> seed['zbs']
>>> seed['raxis_cc']   # list (to be written into vmec.indata.raxis_cc)
>>> seed['zaxis_cs']
>>> seed['etabar']     # diagnostic
>>> seed['r']          # diagnostic (pyQSC internal radial coordinate)
"""

import os
import re
import tempfile

import numpy as np
from scipy.optimize import brentq


# Helical axis excursion as a fraction of R0. A small nonzero torsion is
# required to escape the zero-iota trap. The peak of iota(etabar) scales
# roughly with this delta, so low-iota targets want small delta (gentle
# cross-section) while high-iota targets require larger delta to reach the
# target at all. near_axis_seed walks this list from smallest to largest and
# accepts the first delta whose iota(etabar) curve brackets iota_target —
# that naturally gives the least-elongated seed for each iota target.
_DELTA_CANDIDATES = (0.01, 0.015, 0.02, 0.03, 0.05)

# etabar scan range for bracketing iota(etabar). The mapping is non-monotonic
# (rises, peaks, decays); we take the lower crossing (gentler elongation).
# Log-spaced for better coverage of the steep rise near etabar=0.
_ETABAR_SCAN = np.geomspace(0.01, 8.0, 100)

# Sanity threshold on pyQSC max_elongation. Seeds with elongation above this
# usually fail downstream in VMEC or produce garbage equilibria.
_MAX_ELONGATION_WARN = 10.0

# Tolerances for the root-finds.
_ETABAR_XTOL = 1.0e-10
_R_XTOL = 1.0e-6


def _parse_vmec_boundary(text):
    """Parse RBC/ZBS Fourier coefficients from a pyQSC-written VMEC input file.

    pyQSC's `Qsc.to_vmec()` writes lines like `RBC(n,m) = +1.234e-01`.
    Note the (n, m) ordering — we return a dict keyed by (m, n) to match
    VMEC/SIMSOPT convention.
    """
    rbc = {}
    zbs = {}
    rbc_re = re.compile(r'RBC\((-?\d+),\s*(-?\d+)\)\s*=\s*([+-]?[\d.]+e[+-]?\d+)')
    zbs_re = re.compile(r'ZBS\((-?\d+),\s*(-?\d+)\)\s*=\s*([+-]?[\d.]+e[+-]?\d+)')
    for line in text.splitlines():
        mr = rbc_re.search(line)
        if mr:
            n, m = int(mr.group(1)), int(mr.group(2))
            rbc[(m, n)] = float(mr.group(3))
        mz = zbs_re.search(line)
        if mz:
            n, m = int(mz.group(1)), int(mz.group(2))
            zbs[(m, n)] = float(mz.group(3))
    return rbc, zbs


def _extract_vmec_boundary(stel, r_val):
    """Run `stel.to_vmec(...)` into a tempfile and parse RBC/ZBS."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        tmpf = f.name
    try:
        stel.to_vmec(tmpf, r=r_val)
        with open(tmpf) as f:
            text = f.read()
    finally:
        try:
            os.unlink(tmpf)
        except OSError:
            pass
    return _parse_vmec_boundary(text)


def _bracket_and_solve_etabar(rc, zs, nfp, iota_target):
    """Scan for a sign change in iota(etabar) - iota_target and refine.

    Returns the lower-crossing etabar, or None if no bracket found.
    """
    from qsc import Qsc

    def residual(eb):
        return Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eb).iota - iota_target

    lo = hi = None
    for a, b in zip(_ETABAR_SCAN[:-1], _ETABAR_SCAN[1:]):
        try:
            ra = residual(a)
            rb = residual(b)
        except Exception:
            continue
        if np.isfinite(ra) and np.isfinite(rb) and ra * rb < 0:
            lo, hi = a, b
            break
    if lo is None:
        return None
    return brentq(residual, lo, hi, xtol=_ETABAR_XTOL)


def _find_etabar_adaptive(R0, nfp, iota_target):
    """Walk _DELTA_CANDIDATES and return (etabar, rc, zs, delta_frac) for the
    smallest delta whose iota(etabar) curve brackets iota_target.

    Smaller delta → gentler helical axis excursion → less-elongated seed, so
    the first success is the least-elongated seed that can reach iota_target.
    """
    from qsc import Qsc

    attempts = []
    for delta_frac in _DELTA_CANDIDATES:
        delta = delta_frac * R0
        rc = [R0, delta]
        zs = [0.0, -delta]
        etabar = _bracket_and_solve_etabar(rc, zs, nfp, iota_target)
        if etabar is None:
            attempts.append(f'{delta_frac*100:g}%: no bracket')
            continue
        max_elong = float(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar).max_elongation)
        if max_elong > _MAX_ELONGATION_WARN:
            attempts.append(f'{delta_frac*100:g}%: elong={max_elong:.1f}')
            continue
        return etabar, rc, zs, delta_frac, max_elong

    raise RuntimeError(
        f'near_axis_seed: no delta in {_DELTA_CANDIDATES} produced a seed '
        f'with bracketed etabar and max_elongation <= {_MAX_ELONGATION_WARN} '
        f'for R0={R0}, iota_target={iota_target}, nfp={nfp}. '
        f'Attempts: {"; ".join(attempts)}.'
    )


def _find_r(stel, a_target):
    """Root-find the pyQSC `r` parameter such that a_eff == a_target.

    a_eff is defined as the average of |RBC(1,0)| and |ZBS(1,0)| — the two
    are unequal because the first-order cross-section is an elongated
    rotating ellipse, not a circle.
    """
    def a_eff(r_val):
        rbc, zbs = _extract_vmec_boundary(stel, r_val)
        return (abs(rbc.get((1, 0), 0.0)) + abs(zbs.get((1, 0), 0.0))) / 2.0

    # Frenet→cylindrical conversion in pyQSC can fail at large r where the
    # near-axis approximation breaks down. Back off until we find a safe
    # upper bracket.
    r_upper = 0.5
    while r_upper > 0.05:
        try:
            _ = a_eff(r_upper)
            break
        except Exception:
            r_upper *= 0.8
    else:
        raise RuntimeError('near_axis_seed: could not find a safe upper bound '
                           'for pyQSC r parameter')

    # Walk r_lower down until a_eff(r_lower) < a_target. For thin
    # cross-sections (small a_target) the default lower bound 0.01 can
    # already overshoot — brentq then sees same-sign endpoints and fails.
    r_lower = 0.01
    while r_lower > 1.0e-6:
        try:
            if a_eff(r_lower) < a_target:
                break
        except Exception:
            pass
        r_lower *= 0.5
    else:
        raise RuntimeError(
            f'near_axis_seed: could not find a lower bound for pyQSC r with '
            f'a_eff < a_target={a_target}. The requested minor radius may be '
            f'smaller than the near-axis expansion can represent.'
        )

    return brentq(lambda r: a_eff(r) - a_target, r_lower, r_upper, xtol=_R_XTOL)


def near_axis_seed(R0, a, iota_target, nfp):
    """Generate a near-axis seed boundary + axis guess for VMEC cold-start.

    Parameters
    ----------
    R0 : float
        Major radius of the magnetic axis (m).
    a : float
        Target effective minor radius (m), defined as
        (|RBC(1,0)| + |ZBS(1,0)|) / 2 on the resulting boundary.
    iota_target : float
        Target on-axis rotational transform (positive; sign handled internally).
    nfp : int
        Number of field periods.

    Returns
    -------
    seed : dict
        Keys:
          'rbc'      : dict {(m, n): RBC[m,n]} Fourier coefficients
          'zbs'      : dict {(m, n): ZBS[m,n]} Fourier coefficients
          'raxis_cc' : list of R cosine axis Fourier coefficients
          'zaxis_cs' : list of Z sine   axis Fourier coefficients
          'etabar'   : float, pyQSC expansion parameter (diagnostic)
          'delta_frac' : float, axis excursion as a fraction of R0 (diagnostic)
          'r'        : float, pyQSC internal radial coordinate (diagnostic)
          'iota'     : float, actual on-axis iota from pyQSC (diagnostic)
          'max_elongation' : float, diagnostic

    Notes
    -----
    The returned `rbc[(0,0)]` is close to but NOT exactly `R0` — it is the
    major-radius Fourier coefficient of the BOUNDARY, not the axis. The
    axis R0 is at `raxis_cc[0]`.

    `phiedge` is NOT returned here. Compute it separately via
    `utils.vmec_helpers.compute_phiedge(rbtor, a, R0)`.
    """
    from qsc import Qsc

    # Stage 1: walk delta candidates, find etabar bracket, accept smallest
    # delta whose seed passes the elongation sanity check.
    etabar_opt, rc, zs, delta_frac, _ = _find_etabar_adaptive(R0, nfp, iota_target)
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar_opt)

    # Stage 2: find pyQSC r parameter so that boundary has a_eff == a.
    r_opt = _find_r(stel, a)

    rbc, zbs = _extract_vmec_boundary(stel, r_opt)

    return {
        'rbc': rbc,
        'zbs': zbs,
        'raxis_cc': list(rc),
        'zaxis_cs': [0.0, -zs[1]],
        'etabar': float(etabar_opt),
        'delta_frac': float(delta_frac),
        'r': float(r_opt),
        'iota': float(stel.iota),
        'max_elongation': float(stel.max_elongation),
    }


def _main():
    """Print the baseline seed ($R_0 = 0.92$ m, $V = 0.1$ m^3, iota=0.15)."""
    R0 = 0.92
    V = 0.1
    nfp = 5
    iota_target = 0.15
    a = np.sqrt(V / (2.0 * np.pi**2 * R0))  # ≈ 0.074 m

    print('=' * 70)
    print('near_axis_seed self-test (baseline configuration)')
    print('=' * 70)
    print(f'  R0           = {R0} m')
    print(f'  V (target)   = {V} m^3')
    print(f'  a (derived)  = {a:.6f} m')
    print(f'  iota_target  = {iota_target}')
    print(f'  nfp          = {nfp}')
    print()

    seed = near_axis_seed(R0=R0, a=a, iota_target=iota_target, nfp=nfp)

    print(f'  delta_frac   = {seed["delta_frac"]:.3f}')
    print(f'  etabar       = {seed["etabar"]:.6f}')
    print(f'  pyQSC r      = {seed["r"]:.6f}')
    print(f'  iota (pyQSC) = {seed["iota"]:.6f}')
    print(f'  max elong    = {seed["max_elongation"]:.4f}')
    print(f'  raxis_cc     = {seed["raxis_cc"]}')
    print(f'  zaxis_cs     = {seed["zaxis_cs"]}')
    print()
    print('  Dominant boundary modes (|val| > 1e-4):')
    print(f'  {"(m, n)":>8s}  {"RBC":>14s}  {"ZBS":>14s}')
    print('  ' + '-' * 42)
    keys = sorted(set(list(seed['rbc'].keys()) + list(seed['zbs'].keys())))
    for key in keys:
        m, n = key
        if abs(m) > 5 or abs(n) > 5:
            continue
        rv = seed['rbc'].get(key, 0.0)
        zv = seed['zbs'].get(key, 0.0)
        if abs(rv) > 1e-4 or abs(zv) > 1e-4:
            print(f'  {str(key):>8s}  {rv:14.8f}  {zv:14.8f}')


if __name__ == '__main__':
    _main()
