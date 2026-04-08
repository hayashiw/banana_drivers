"""
01_stage1_driver.py
───────────────────
Stage 1 VMEC fixed-boundary optimization for the banana coil
stellarator-tokamak hybrid targeting quasi-axisymmetry (QA, M=1 N=0).

Optimizes the VMEC boundary shape to improve QA using Boozer coordinates
(booz_xform) with a resolution ramp.  Supports warm start from an existing
wout file and cold start from a programmatic boundary for Pareto scans.

Pipeline:  01_stage1 (this) -> 02_stage2 -> 03_singlestage

At completion, saves the optimized wout and builds the BoozerSurface
(coils + surface) via utils/init_boozersurface for stage 2 warm-start.

Usage (MPI required):
    srun -n 16 python 01_stage1_driver.py
"""
import atexit
import numpy as np
import os
import sys
import time
import yaml

from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir
from init_boozersurface import build_and_save

from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition


# ──────────────────────────────────────────────────────────────────────────────
# MPI setup
# ──────────────────────────────────────────────────────────────────────────────
mpi = MpiPartition()


def proc0_print(*args, **kwargs):
    if mpi.proc0_world:
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
_cfg_path = os.path.join(_base_dir, 'config.yaml')
with open(_cfg_path) as _f:
    cfg = yaml.safe_load(_f)

# Device geometry
NFP      = cfg['device']['nfp']
STELLSYM = cfg['device']['stellsym']

# Stage 1 settings
s1 = cfg['stage1']
COLD_START      = s1['cold_start']
MAX_MODE_STEPS  = s1['max_mode_steps']
VMEC_MPOL       = s1['vmec_mpol']
VMEC_NTOR       = s1['vmec_ntor']
BOOZER_MPOL     = s1['boozer_mpol']
BOOZER_NTOR     = s1['boozer_ntor']
MAX_NFEV        = s1['max_nfev']
QS_SURFACES     = s1['qs_surfaces']

# Physics targets (env var overrides for Pareto scans)
IOTA_TARGET   = float(os.environ.get('BANANA_IOTA', s1['iota_target']))
ASPECT_TARGET = s1['aspect_target']
VOLUME_TARGET = float(os.environ.get('BANANA_VOLUME', s1['volume_target']))

# Objective weights
ASPECT_WEIGHT = s1['aspect_weight']
IOTA_WEIGHT   = s1['iota_weight']
VOLUME_WEIGHT = s1['volume_weight']
QS_WEIGHT     = s1['qs_weight']

# Cold start boundary
COLD_R0      = s1['cold_start_R0']
COLD_A       = s1['cold_start_a']
COLD_PHIEDGE = s1['cold_start_phiedge']

# Warm-start wout path
WOUT_FILE = os.path.join(_base_dir, cfg['warm_start']['wout_filepath'])

# Output filenames
STAGE1_WOUT_FILENAME = cfg['warm_start']['stage1_wout_filename']
INIT_BSURF_FILEPATH  = cfg['warm_start']['init_bsurf_filepath']

# Output directory (env var override for Pareto scans)
OUTPUT_PREFIX = os.environ.get('BANANA_OUTPUT_PREFIX', 'stage1')
OUT_DIR = resolve_output_dir()

DIAGNOSTICS_FILE = os.path.join(OUT_DIR, f'{OUTPUT_PREFIX}_diagnostics.txt')


# ──────────────────────────────────────────────────────────────────────────────
# Output atexit handler
# ──────────────────────────────────────────────────────────────────────────────
def _emit_out_dir_on_exit():
    """Print output directory path so the shell script can move the log file."""
    proc0_print(f"OUT_DIR={OUT_DIR}")


atexit.register(_emit_out_dir_on_exit)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Output prefix:   {OUTPUT_PREFIX}
    MPI ranks:       {mpi.nprocs_world}

    Start mode:      {'COLD start (programmatic boundary)' if COLD_START else 'WARM start from wout'}
    {'Boundary:        R0=' + str(COLD_R0) + ', a=' + str(COLD_A) + ', phiedge=' + str(COLD_PHIEDGE) if COLD_START else 'Wout:            ' + WOUT_FILE}

    Physics targets:
        iota        = {IOTA_TARGET}  {'(env BANANA_IOTA)' if 'BANANA_IOTA' in os.environ else ''}
        aspect      = {ASPECT_TARGET}
        volume      = {VOLUME_TARGET}  {'(env BANANA_VOLUME)' if 'BANANA_VOLUME' in os.environ else ''}

    QA target: M=1, N=0
        surfaces    = {QS_SURFACES}

    Objective weights:
        aspect      = {ASPECT_WEIGHT:.3e}
        iota        = {IOTA_WEIGHT:.3e}
        volume      = {VOLUME_WEIGHT:.3e}
        qs          = {QS_WEIGHT:.3e}

    Resolution ramp ({len(MAX_MODE_STEPS)} steps):
        max_mode    = {MAX_MODE_STEPS}
        vmec mpol   = {VMEC_MPOL}
        vmec ntor   = {VMEC_NTOR}
        boozer mpol = {BOOZER_MPOL}
        boozer ntor = {BOOZER_NTOR}
        max_nfev    = {MAX_NFEV}

    Output directory: {OUT_DIR}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize VMEC
# ──────────────────────────────────────────────────────────────────────────────
if COLD_START:
    proc0_print('Cold start: building VMEC boundary from scratch...')
    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = NFP
    vmec.indata.mpol = VMEC_MPOL[0]
    vmec.indata.ntor = VMEC_NTOR[0]
    vmec.indata.phiedge = COLD_PHIEDGE

    surf = vmec.boundary
    surf.fix_all()
    surf.set_rc(0, 0, COLD_R0)
    surf.set_rc(1, 0, COLD_A)
    surf.set_zs(1, 0, COLD_A)
    proc0_print(f'  R0={COLD_R0}, a={COLD_A}, phiedge={COLD_PHIEDGE}')
else:
    # Warm start: Vmec() requires an input file to be runnable.
    # Load from default input (which gives indata), then transfer the
    # boundary shape and equilibrium parameters from the wout.
    proc0_print(f'Warm start: seeding boundary from {WOUT_FILE}')
    import netCDF4 as nc4
    ds = nc4.Dataset(WOUT_FILE)
    wout_nfp = int(ds.variables['nfp'][:])
    wout_mpol = int(ds.variables['mpol'][:])
    wout_ntor = int(ds.variables['ntor'][:])
    wout_phiedge = float(ds.variables['phi'][:][-1])  # total toroidal flux
    ds.close()

    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = wout_nfp
    vmec.indata.mpol = max(wout_mpol, VMEC_MPOL[0])
    vmec.indata.ntor = max(wout_ntor, VMEC_NTOR[0])
    vmec.indata.phiedge = wout_phiedge

    # Read the LCFS boundary from the wout and assign to vmec
    from simsopt.geo import SurfaceRZFourier
    wout_surf = SurfaceRZFourier.from_wout(WOUT_FILE, range='full torus',
                                           nphi=50, ntheta=50)
    vmec.boundary = wout_surf
    proc0_print(f'  nfp={wout_nfp}, mpol={vmec.indata.mpol}, ntor={vmec.indata.ntor}, '
                f'phiedge={wout_phiedge:.6f}')

vmec.verbose = mpi.proc0_world
surf = vmec.boundary


# ──────────────────────────────────────────────────────────────────────────────
# Configure Boozer + Quasisymmetry objectives
# ──────────────────────────────────────────────────────────────────────────────
proc0_print('Configuring Boozer + Quasisymmetry (M=1, N=0) objectives...')
boozer = Boozer(vmec)
boozer.bx.verbose = mpi.proc0_world

# QA objective on multiple flux surfaces
qs_list = [Quasisymmetry(boozer, s, 1, 0) for s in QS_SURFACES]

# Least-squares problem: aspect, iota (axis + edge), QA on each surface
tuples = [
    (vmec.aspect, ASPECT_TARGET, ASPECT_WEIGHT),
    (vmec.iota_axis, IOTA_TARGET, IOTA_WEIGHT),
    (vmec.iota_edge, IOTA_TARGET, IOTA_WEIGHT),
    (vmec.volume, VOLUME_TARGET, VOLUME_WEIGHT),
]
for qs in qs_list:
    tuples.append((qs.J, 0, QS_WEIGHT))

prob = LeastSquaresProblem.from_tuples(tuples)
proc0_print(f'  {len(tuples)} objective terms (aspect + 2 iota + volume + {len(qs_list)} QS surfaces)')


# ──────────────────────────────────────────────────────────────────────────────
# Per-group working directories (avoids VMEC fort.9 file conflicts across MPI groups)
# ──────────────────────────────────────────────────────────────────────────────
import tempfile, shutil
_orig_dir = os.getcwd()
_group_dir = tempfile.mkdtemp(prefix=f'vmec_g{mpi.group:03d}_', dir=OUT_DIR)
os.chdir(_group_dir)

@atexit.register
def _cleanup_group_dir():
    os.chdir(_orig_dir)
    try:
        shutil.rmtree(_group_dir)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Print initial state
# ──────────────────────────────────────────────────────────────────────────────
# Run VMEC once to get initial equilibrium values
vmec.run()
proc0_print(
    f"""
INITIAL STATE ─────────────────────────────────
    Aspect ratio:        {float(vmec.aspect()):.6f}  (target: {ASPECT_TARGET})
    Iota axis:           {float(vmec.iota_axis()):.6f}  (target: {IOTA_TARGET})
    Iota edge:           {float(vmec.iota_edge()):.6f}  (target: {IOTA_TARGET})
    Volume:              {float(vmec.volume()):.6f} m^3  (target: {VOLUME_TARGET})
    VMEC mpol:           {vmec.indata.mpol}
    VMEC ntor:           {vmec.indata.ntor}
    Boundary ndofs:      {len(surf.get_dofs())}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

if mpi.proc0_world:
    with open(DIAGNOSTICS_FILE, 'w') as f:
        f.write(f'# Stage 1 VMEC QA Optimization Diagnostics\n')
        f.write(f'# Date: {datetime.now()}\n')
        f.write(f'# Start mode: {"COLD" if COLD_START else "WARM"}\n')
        f.write(f'# Targets: iota={IOTA_TARGET}, aspect={ASPECT_TARGET}, volume={VOLUME_TARGET}\n')
        f.write(f'# QA surfaces: {QS_SURFACES}\n')
        f.write(
            'step,max_mode,vmec_mpol,vmec_ntor,boozer_mpol,boozer_ntor,'
            'max_nfev,vmec_iter,'
            'aspect,iota_axis,iota_edge,volume,'
            'qs_total,objective,'
            'runtime\n'
        )


# ──────────────────────────────────────────────────────────────────────────────
# Resolution ramp optimization
# ──────────────────────────────────────────────────────────────────────────────
n_steps = len(MAX_MODE_STEPS)
for step in range(n_steps):
    step_t0 = time.time()

    max_mode  = MAX_MODE_STEPS[step]
    v_mpol    = VMEC_MPOL[step]
    v_ntor    = VMEC_NTOR[step]
    b_mpol    = BOOZER_MPOL[step]
    b_ntor    = BOOZER_NTOR[step]
    max_nfev  = MAX_NFEV[step]

    # Update VMEC resolution
    vmec.indata.mpol = v_mpol
    vmec.indata.ntor = v_ntor

    # Update booz_xform resolution
    boozer.mpol = b_mpol
    boozer.ntor = b_ntor

    # Free boundary modes up to max_mode
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Keep major radius fixed

    n_free = len([d for d, f in zip(surf.dof_names, surf.dofs_free_status) if f])

    proc0_print(
        f"""
STEP {step+1}/{n_steps} ───────────────────────────────────
    max_mode    = {max_mode}
    vmec mpol   = {v_mpol}, ntor = {v_ntor}
    boozer mpol = {b_mpol}, ntor = {b_ntor}
    max_nfev    = {max_nfev}
    free DOFs   = {n_free}
    vmec iter   = {vmec.iter} (cumulative)
"""
    )

    # Optimize
    least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=max_nfev)

    # Preserve wout from this step
    vmec.files_to_delete = []

    # Post-step diagnostics
    step_runtime = time.time() - step_t0
    qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
    obj = prob.objective()

    aspect = float(vmec.aspect())
    iota_ax = float(vmec.iota_axis())
    iota_ed = float(vmec.iota_edge())
    vol = float(vmec.volume())

    proc0_print(
        f"""
    STEP {step+1} RESULTS (runtime: {timedelta(seconds=step_runtime)}):
        Aspect ratio:    {aspect:.6f}  (target: {ASPECT_TARGET})
        Iota axis:       {iota_ax:.6f}  (target: {IOTA_TARGET})
        Iota edge:       {iota_ed:.6f}  (target: {IOTA_TARGET})
        Volume:          {vol:.6f} m^3
        QS metric total: {qs_total:.6e}
        Objective:       {obj:.6e}
        VMEC iter:       {vmec.iter}
"""
    )

    # Write diagnostics row
    if mpi.proc0_world:
        with open(DIAGNOSTICS_FILE, 'a') as f:
            f.write(
                f'{step+1},{max_mode},{v_mpol},{v_ntor},{b_mpol},{b_ntor},'
                f'{max_nfev},{vmec.iter},'
                f'{aspect:.6e},{iota_ax:.6e},{iota_ed:.6e},{vol:.6e},'
                f'{qs_total:.6e},{obj:.6e},'
                f'{step_runtime:.2f}\n'
            )


# ──────────────────────────────────────────────────────────────────────────────
# Termination summary
# ──────────────────────────────────────────────────────────────────────────────
total_runtime = time.time() - t0

qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
obj = prob.objective()
final_aspect = float(vmec.aspect())
final_iota_ax = float(vmec.iota_axis())
final_iota_ed = float(vmec.iota_edge())
final_vol = float(vmec.volume())
aspect_err = abs(final_aspect - ASPECT_TARGET)
iota_axis_err = abs(final_iota_ax - IOTA_TARGET)
iota_edge_err = abs(final_iota_ed - IOTA_TARGET)

# Simple success criteria: iota within 10% of target, QS improved
success = (iota_axis_err / IOTA_TARGET < 0.10) and (iota_edge_err / IOTA_TARGET < 0.50)

proc0_print(
    f"""
[{datetime.now()}] ...optimization complete
Total runtime: {timedelta(seconds=total_runtime)}

{'SUCCESS' if success else 'FAILURE'} ─────────────────────────────────────────
    Aspect ratio:    {final_aspect:.6f}  (target: {ASPECT_TARGET}, err: {aspect_err:.6f})
    Iota axis:       {final_iota_ax:.6f}  (target: {IOTA_TARGET}, err: {iota_axis_err:.6f})
    Iota edge:       {final_iota_ed:.6f}  (target: {IOTA_TARGET}, err: {iota_edge_err:.6f})
    Volume:          {final_vol:.6f} m^3
    QS metric total: {qs_total:.6e}
    Final objective: {obj:.6e}
    VMEC iterations: {vmec.iter}
"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Print final state
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
FINAL STATE ───────────────────────────────────
    Aspect ratio:        {final_aspect:.6f}  (target: {ASPECT_TARGET})
    Iota axis:           {final_iota_ax:.6f}  (target: {IOTA_TARGET})
    Iota edge:           {final_iota_ed:.6f}  (target: {IOTA_TARGET})
    Volume:              {final_vol:.6f} m^3  (target: {VOLUME_TARGET})
    VMEC mpol:           {vmec.indata.mpol}
    VMEC ntor:           {vmec.indata.ntor}
    Boundary ndofs:      {len(surf.get_dofs())}

    Per-surface QS metrics:"""
)
for i, (s, qs) in enumerate(zip(QS_SURFACES, qs_list)):
    proc0_print(f'        s={s:.2f}:  {float(np.sum(qs.J()**2)):.6e}')

proc0_print(f'    QS total:            {qs_total:.6e}')


# ──────────────────────────────────────────────────────────────────────────────
# Save optimized wout
# ──────────────────────────────────────────────────────────────────────────────
wout_opt_path = os.path.join(OUT_DIR, STAGE1_WOUT_FILENAME)

# VMEC saves wout to its working directory; copy to output dir
wout_src = vmec.output_file
if mpi.proc0_world and wout_src and os.path.exists(wout_src):
    shutil.copy2(wout_src, wout_opt_path)
    proc0_print(f'Optimized wout saved to {wout_opt_path}')
else:
    proc0_print(f'WARNING: Could not find VMEC output file to copy')


# ──────────────────────────────────────────────────────────────────────────────
# Return to original directory and clean up per-group temp dirs
# ──────────────────────────────────────────────────────────────────────────────
os.chdir(_orig_dir)
try:
    shutil.rmtree(_group_dir)
except OSError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Build BoozerSurface (coils + surface) for stage 2 warm-start
# ──────────────────────────────────────────────────────────────────────────────
if mpi.proc0_world:
    proc0_print('\nBuilding BoozerSurface from optimized wout for stage 2...')
    bsurf_out_path = os.path.join(_base_dir, INIT_BSURF_FILEPATH)
    build_and_save(cfg, wout_path=wout_opt_path, out_path=bsurf_out_path,
                   save_vtk=True, print_fn=proc0_print)

proc0_print(f'\nDiagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {OUT_DIR}')
