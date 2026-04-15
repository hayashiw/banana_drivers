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
import shutil
import sys
import time
import yaml

from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from output_dir import resolve_output_dir
from init_boozersurface import build_and_save
from hbt_parameters import TF_RBTOR, compute_phiedge
from near_axis_seed import near_axis_seed
from run_registry import RunRegistry, artifact_path, install_atexit_handler, run_dir

from simsopt._core.util import ObjectiveFailure
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

# Seed mode override: BANANA_SEED=warm|cold supersedes stage1.cold_start in
# config.yaml. Lets submit.sh flip warm/cold without touching the config.
_seed_env = os.environ.get('BANANA_SEED')
if _seed_env is None:
    COLD_START = s1['cold_start']
elif _seed_env == 'warm':
    COLD_START = False
elif _seed_env == 'cold':
    COLD_START = True
else:
    raise ValueError(
        f"BANANA_SEED must be 'warm' or 'cold', got {_seed_env!r}"
    )
MAX_MODE_STEPS  = s1['max_mode_steps']
VMEC_MPOL       = s1['vmec_mpol']
VMEC_NTOR       = s1['vmec_ntor']
BOOZER_MPOL     = s1['boozer_mpol']
BOOZER_NTOR     = s1['boozer_ntor']
MAX_NFEV        = s1['max_nfev']
QS_SURFACES     = s1['qs_surfaces']
NS_ARRAY        = s1['ns_array']          # per failure mode #9 — radial resolution ramp

# Physics targets (env var overrides for Pareto scans)
IOTA_TARGET   = float(os.environ.get('BANANA_IOTA', s1['iota_target']))
ASPECT_TARGET = float(os.environ.get('BANANA_ASPECT', s1['aspect_target']))
VOLUME_TARGET = float(os.environ.get('BANANA_VOLUME', s1['volume_target']))

# Objective weights
ASPECT_WEIGHT = s1['aspect_weight']
IOTA_WEIGHT   = s1['iota_weight']
VOLUME_WEIGHT = s1['volume_weight']
QS_WEIGHT     = s1['qs_weight']

# Weight mode: 'user' applies the weights above as-is; 'relative' divides the
# scalar targets' weights by |target| so each residual is a relative error (the
# user weight then acts as a relative-error coefficient). QS target is 0, so QS
# weight is always applied as-is.
WEIGHT_MODE = s1.get('weight_mode', 'user')
if WEIGHT_MODE not in ('user', 'relative'):
    raise ValueError(f"stage1.weight_mode must be 'user' or 'relative', got {WEIGHT_MODE!r}")

# Cold start boundary: user supplies (R0, V, iota); driver derives (a, phiedge,
# helical seed) from device constants. See cold_start_stage1_prompt.md.
COLD_R0     = float(os.environ.get('BANANA_COLD_R0', s1['cold_start_R0']))
COLD_VOLUME = float(os.environ.get('BANANA_VOLUME', s1['cold_start_volume']))
# Cold-start iota target is the same knob as the stage 1 iota objective target,
# so reuse IOTA_TARGET (already set above, with BANANA_IOTA env override).
COLD_A = float(np.sqrt(COLD_VOLUME / (2.0 * np.pi**2 * COLD_R0)))
COLD_PHIEDGE = compute_phiedge(TF_RBTOR, COLD_A, COLD_R0)

# Warm-start wout path
WOUT_FILE = os.path.join(_base_dir, cfg['warm_start']['wout_filepath'])

# Write env-resolved values back into cfg so content-addressed hashing sees
# the effective inputs, not the raw config.yaml values. Pareto sweeps vary
# these via env vars; without this write-back, runs collide on the same run_id.
s1['cold_start']         = bool(COLD_START)
s1['iota_target']        = IOTA_TARGET
s1['aspect_target']      = ASPECT_TARGET
s1['volume_target']      = VOLUME_TARGET
s1['cold_start_R0']      = COLD_R0
s1['cold_start_volume']  = COLD_VOLUME

# Output directory root — stage 1 owns $OUT_DIR/stage1/<run_id>/.
OUT_DIR = resolve_output_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Registry: register this stage 1 run (rank 0 only; broadcast run_id to peers)
# ──────────────────────────────────────────────────────────────────────────────
# Content-addressed ID derived from STAGE1_INPUT_KEYS + git commit. Same
# inputs + same code → same ID. build_input_blob will KeyError if the
# whitelist has drifted away from config.yaml, so typos surface here instead
# of producing stable-but-meaningless hashes.
_slurm_meta = {
    "slurm_qos":           os.environ.get("SLURM_JOB_QOS"),
    "slurm_partition":     os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_ntasks":        int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_NTASKS") else None,
    "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None,
    "slurm_time_limit_s":  None,   # SLURM doesn't expose TIME in a structured way; leave null
}
_slurm_job_id = os.environ.get("SLURM_JOB_ID")

registry: "RunRegistry | None" = None
_reg_payload: "tuple[str, bool] | None" = None
if mpi.proc0_world:
    registry = RunRegistry()
    _reg_payload = registry.register_stage1(cfg, slurm_meta=_slurm_meta)

# Broadcast (run_id, is_new) so every rank can resolve per-run paths and
# print a consistent header. compute_run_id is deterministic, but routing
# through rank 0 means only one process hits git + sqlite.
_reg_payload = mpi.comm_world.bcast(_reg_payload, root=0)
RUN_ID, _is_new = _reg_payload

RUN_DIR = run_dir("stage1", RUN_ID, OUT_DIR)
# All ranks create (exist_ok=True is idempotent) — every rank later needs
# RUN_DIR to exist for its per-group tempfile.mkdtemp and for opening
# diagnostics / wout snapshot files.
os.makedirs(RUN_DIR, exist_ok=True)

DIAGNOSTICS_FILE = artifact_path("stage1", RUN_ID, OUT_DIR, "diagnostics")


# ──────────────────────────────────────────────────────────────────────────────
# Output atexit handler
# ──────────────────────────────────────────────────────────────────────────────
def _emit_out_dir_on_exit():
    """Print the per-run directory so the shell script can move the log file
    into it (run_driver.sh reads this line)."""
    proc0_print(f"OUT_DIR={RUN_DIR}")


atexit.register(_emit_out_dir_on_exit)

# Mark the row 'failed' with error_code=unclean_exit if the interpreter exits
# while still in 'running' (uncaught exception, non-zero sys.exit, etc.).
# Rank-0 only; SIGKILL/OOM is caught by sweep() via sacct.
if mpi.proc0_world and registry is not None:
    install_atexit_handler(registry, "stage1", RUN_ID)


# ──────────────────────────────────────────────────────────────────────────────
# Print input parameters
# ──────────────────────────────────────────────────────────────────────────────
proc0_print(
    f"""
INPUT PARAMETERS ─────────────────────────────
    Config:          {_cfg_path}
    Date:            {datetime.now()}
    Run ID:          {RUN_ID}  ({'new' if _is_new else 'rerun'})
    Run dir:         {RUN_DIR}
    MPI ranks:       {mpi.nprocs_world}

    Start mode:      {'COLD start (near-axis seed via pyQSC)' if COLD_START else 'WARM start from wout'}  {'(env BANANA_SEED)' if 'BANANA_SEED' in os.environ else ''}
    {'Boundary:        R0=' + f'{COLD_R0:.4f}' + ' m, V=' + f'{COLD_VOLUME:.4f}' + ' m^3 (a=' + f'{COLD_A:.4f}' + ' m), phiedge=' + f'{COLD_PHIEDGE:.6e}' + ' Wb (rbtor=' + f'{TF_RBTOR:.4f}' + ' T*m)' if COLD_START else 'Wout:            ' + WOUT_FILE}

    Physics targets:
        iota        = {IOTA_TARGET}  {'(env BANANA_IOTA)' if 'BANANA_IOTA' in os.environ else ''}
        aspect      = {ASPECT_TARGET}  {'(env BANANA_ASPECT)' if 'BANANA_ASPECT' in os.environ else ''}
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
    proc0_print('Cold start: building VMEC boundary from a near-axis seed (pyQSC)...')
    # Derive the seed boundary + axis guess from (R0, a, iota, nfp) via the
    # first-order near-axis expansion. This produces nonzero helical content
    # (modes with n != 0), which is required to escape the zero-beta zero-iota
    # trap (failure mode #3). The axis guess is self-consistent with the
    # boundary by construction, sidestepping ARNORM degeneracy (failure mode #1).
    try:
        seed = near_axis_seed(R0=COLD_R0, a=COLD_A, iota_target=IOTA_TARGET, nfp=NFP)
    except RuntimeError as e:
        # pyQSC near-axis walker could not produce a seed bracketing the
        # iota target under the elongation cap (or could not resolve the
        # minor-radius root). Exit with a distinct status so sweep
        # bookkeeping can separate "rejected seed" (2) from "VMEC/driver
        # crash" (1) and "walltime" (143).
        proc0_print(f'  REJECTED SEED: {e}')
        proc0_print(
            f'  (R0={COLD_R0}, a={COLD_A}, iota_target={IOTA_TARGET}, V={COLD_VOLUME}, nfp={NFP})'
        )
        if mpi.proc0_world and registry is not None:
            registry.mark_failed(
                "stage1", RUN_ID,
                error_code="bad_input",
                error_message=f"near_axis_seed rejected: {e}",
            )
        sys.exit(2)
    proc0_print(
        f'  pyQSC near-axis seed: etabar={seed["etabar"]:.4f}, '
        f'r={seed["r"]:.4f}, iota={seed["iota"]:.6f}, '
        f'max_elongation={seed["max_elongation"]:.4f}'
    )

    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = NFP
    vmec.indata.mpol = VMEC_MPOL[0]
    vmec.indata.ntor = VMEC_NTOR[0]
    vmec.indata.lasym = False
    vmec.indata.phiedge = COLD_PHIEDGE
    # Radial ramp: the driver applies NS_ARRAY[step] per resolution step in the
    # main loop below. Set the initial values here so the first vmec.run() at
    # INITIAL STATE time sees a well-resolved grid (failure mode #9).
    vmec.indata.ns_array[:]    = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:]  = 0.0
    vmec.indata.ns_array[:len(NS_ARRAY)]    = NS_ARRAY
    vmec.indata.niter_array[:len(NS_ARRAY)] = [10000] * len(NS_ARRAY)
    vmec.indata.ftol_array[:len(NS_ARRAY)]  = [1.0e-10] * len(NS_ARRAY)

    # Zero-beta, zero-current, shape-derived-iota profile.
    # ncurr=1 prescribes toroidal current (ac polynomial + curtor) and lets
    # VMEC solve for iota as an output of the boundary shape. With ac[:]=0 and
    # curtor=0, the plasma carries no net current and iota is determined purely
    # by the boundary geometry — this makes iota_edge a meaningful residual
    # with nonzero gradient w.r.t. boundary DOFs. (The ncurr=0 / ai[0]=target
    # formulation pinned iota_edge to ai[0] regardless of shape, giving a
    # constant residual that the optimizer exploited by collapsing to an
    # axisymmetric boundary with trivial QS=0 at I=0.)
    vmec.indata.ncurr = 1
    vmec.indata.ac[:] = 0.0
    vmec.indata.curtor = 0.0
    vmec.indata.ai[:] = 0.0
    vmec.indata.pres_scale = 0.0
    vmec.indata.am[:] = 0.0

    # Self-consistent magnetic axis guess — avoids ARNORM degeneracy.
    vmec.indata.raxis_cc[:] = 0.0
    vmec.indata.zaxis_cs[:] = 0.0
    raxis = seed['raxis_cc']
    zaxis = seed['zaxis_cs']
    vmec.indata.raxis_cc[:len(raxis)] = raxis
    vmec.indata.zaxis_cs[:len(zaxis)] = zaxis

    # Build a fresh SurfaceRZFourier at the starting VMEC resolution and
    # populate it from the near-axis seed Fourier coefficients. Assign it to
    # vmec.boundary — this is the same pattern the warm-start branch uses to
    # replace the input.default boundary with the seed geometry.
    from simsopt.geo import SurfaceRZFourier
    seed_surf = SurfaceRZFourier(
        nfp=NFP, stellsym=True,
        mpol=VMEC_MPOL[0], ntor=VMEC_NTOR[0],
    )
    # Start from zero, then overwrite the modes returned by near_axis_seed.
    # Any mode outside (mpol0, ntor0) is silently truncated here — the seed's
    # first-order boundary concentrates amplitude in low-m/low-n modes anyway,
    # and the ramp loop below frees higher modes step by step.
    seed_surf.x = np.zeros_like(seed_surf.x)
    for (m, n), val in seed['rbc'].items():
        if m <= VMEC_MPOL[0] and abs(n) <= VMEC_NTOR[0]:
            seed_surf.set_rc(m, n, float(val))
    for (m, n), val in seed['zbs'].items():
        if m <= VMEC_MPOL[0] and abs(n) <= VMEC_NTOR[0]:
            seed_surf.set_zs(m, n, float(val))
    vmec.boundary = seed_surf

    # CRITICAL: re-sync the Optimizable DOF cache after writing indata.phiedge
    # and replacing vmec.boundary. Failure mode #6: without this, the next
    # least_squares_mpi_solve broadcast restores the stale input.default
    # phiedge=1.0 and produces |B|~12 T.
    vmec.local_full_x = np.asarray(vmec.get_dofs())

    proc0_print(
        f'  Boundary: R0={COLD_R0:.4f} m, a={COLD_A:.4f} m, '
        f'phiedge={COLD_PHIEDGE:.6e} Wb'
    )
    proc0_print(
        f'  Axis: raxis_cc={raxis}, zaxis_cs={zaxis}'
    )
else:
    # Warm start: the seed wout has been pre-processed by utils/vmec_resize.py
    # to have LCFS (s=1) == target plasma boundary, at the correct major radius
    # and enclosed toroidal flux. Stage 1 just loads it as-is — no rescaling.
    proc0_print(f'Warm start: seeding boundary from {WOUT_FILE}')
    import netCDF4 as nc4
    ds = nc4.Dataset(WOUT_FILE)
    wout_nfp = int(ds.variables['nfp'][:])
    wout_mpol = int(ds.variables['mpol'][:])
    wout_ntor = int(ds.variables['ntor'][:])
    wout_phiedge = float(ds.variables['phi'][:][-1])
    ds.close()

    vmec = Vmec(mpi=mpi)
    vmec.indata.nfp = wout_nfp
    vmec.indata.mpol = max(wout_mpol, VMEC_MPOL[0])
    vmec.indata.ntor = max(wout_ntor, VMEC_NTOR[0])
    vmec.indata.phiedge = wout_phiedge
    # Bump NITER for the high-mpol resolution steps. input.default ships with
    # niter_array[:]=3000, insufficient at mpol=5 for this equilibrium — see
    # job 51257661 where FSQR plateaued at 1.56e-10 above FTOLV=1e-10.
    vmec.indata.niter_array[:] = 10000

    # CRITICAL: Vmec's Optimizable DOF cache was populated from input.default
    # (phiedge=1.0) during __init__. Overriding vmec.indata.phiedge alone does
    # NOT update that cache — so when the least-squares optimizer later calls
    # `prob.x = x`, Vmec.set_dofs is invoked with the stale [1.0, 0.0, 1.0]
    # vector and silently resets indata.phiedge back to 1.0. Re-syncing
    # local_full_x from indata here locks the cache to the seed wout's phiedge.
    vmec.local_full_x = np.asarray(vmec.get_dofs())

    # Load LCFS directly — the resized seed has LCFS == target boundary.
    from simsopt.geo import SurfaceRZFourier
    wout_surf = SurfaceRZFourier.from_wout(WOUT_FILE, range='full torus',
                                           nphi=50, ntheta=50)
    vmec.boundary = wout_surf
    proc0_print(f'  nfp={wout_nfp}, mpol={vmec.indata.mpol}, ntor={vmec.indata.ntor}, '
                f'phiedge={vmec.indata.phiedge:.6f}, '
                f'R0={wout_surf.major_radius():.4f} m')

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


def _build_prob():
    """Build LeastSquaresProblem from current objectives.

    Must be called after updating boozer.mpol/ntor — Quasisymmetry.J() returns
    one residual per Boozer mode, so the residual vector length changes with
    boozer resolution.  LeastSquaresProblem caches nvals on first eval and
    cannot handle a size change, so we rebuild it at each resolution step.
    """
    # Only iota_edge is targeted — iota_axis is left free so the optimizer
    # can preserve whatever magnetic shear the QA solution prefers.
    #
    # Weight mode:
    #   'user'     — weights used as-is (absolute residuals)
    #   'relative' — effective weight = user_weight / |target|, so the scalar
    #                residuals become relative errors. QS target is 0, so the
    #                QS weight is always applied as-is.
    if WEIGHT_MODE == 'relative':
        aw = ASPECT_WEIGHT / abs(ASPECT_TARGET)
        iw = IOTA_WEIGHT   / abs(IOTA_TARGET)
        vw = VOLUME_WEIGHT / abs(VOLUME_TARGET)
    else:
        aw = ASPECT_WEIGHT
        iw = IOTA_WEIGHT
        vw = VOLUME_WEIGHT
    tuples = [
        (vmec.aspect, ASPECT_TARGET, aw),
        (vmec.iota_edge, IOTA_TARGET, iw),
        (vmec.volume, VOLUME_TARGET, vw),
    ]
    for qs in qs_list:
        tuples.append((qs.J, 0, QS_WEIGHT))
    return LeastSquaresProblem.from_tuples(tuples)


prob = _build_prob()
proc0_print(f'  {3 + len(qs_list)} objective terms (aspect + iota_edge + volume + {len(qs_list)} QS surfaces)')


# ──────────────────────────────────────────────────────────────────────────────
# Per-group working directories (avoids VMEC fort.9 file conflicts across MPI groups)
# ──────────────────────────────────────────────────────────────────────────────
import tempfile
_orig_dir = os.getcwd()
_group_dir = tempfile.mkdtemp(prefix=f'vmec_g{mpi.group:03d}_', dir=RUN_DIR)
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
# Save initial wout + boozmn (for comparison plots vs the optimized result)
# ──────────────────────────────────────────────────────────────────────────────
# Snapshot the pre-optimization equilibrium so downstream analysis can diff
# the cold/warm-start seed against the final optimized surface. Uses the
# first ramp step's Boozer resolution; the optimizer will reset these inside
# the ramp loop before the first least_squares_mpi_solve call.
_init_wout_path   = artifact_path("stage1", RUN_ID, OUT_DIR, "wout_init")
_init_boozmn_path = artifact_path("stage1", RUN_ID, OUT_DIR, "boozmn_init")

boozer.mpol = BOOZER_MPOL[0]
boozer.ntor = BOOZER_NTOR[0]
try:
    boozer.run()
    _boozer_init_ok = True
except Exception as e:
    proc0_print(f'WARNING: initial boozer run failed: {e}')
    _boozer_init_ok = False

if mpi.proc0_world:
    if vmec.output_file and os.path.exists(vmec.output_file):
        shutil.copy2(vmec.output_file, _init_wout_path)
        proc0_print(f'Initial wout saved to {_init_wout_path}')
    else:
        proc0_print('WARNING: no initial vmec.output_file to snapshot')
    if _boozer_init_ok:
        try:
            boozer.bx.write_boozmn(_init_boozmn_path)
            proc0_print(f'Initial boozmn saved to {_init_boozmn_path}')
        except Exception as e:
            proc0_print(f'WARNING: could not write initial boozmn: {e}')

# Workaround for upstream SIMSOPT bug: Boozer.run() assigns
# self.bx.compute_surfs AFTER calling self.bx.init_from_vmec(wout.ns, ...),
# so when the ramp loop drops VMEC to a smaller ns than the initial snapshot
# ran at, booz_xform validates the NEW ns against the STALE compute_surfs
# from the previous (larger-ns) run and throws
# "compute_surfs has an entry that is too large for the given ns".
# Clearing compute_surfs here makes init_from_vmec see an empty list, so the
# first ramp-step run() succeeds and writes its own compute_surfs.
# Proper fix belongs upstream in simsopt/mhd/boozer.py (see PLAN.md).
boozer.bx.compute_surfs = np.array([], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize diagnostics file
# ──────────────────────────────────────────────────────────────────────────────
t0 = time.time()

# Running snapshot of the last step whose post-eval VMEC run converged. If the
# optimizer's final accepted point fails to converge (e.g. step 4 at
# mpol=7/ns=51 hitting NITER before FTOLV — job 51429637), we still have a
# valid wout to save from the last good step. Updated inside the ramp loop.
_last_good_wout_src = None
_last_good_step = None

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
if mpi.proc0_world and registry is not None:
    registry.mark_running("stage1", RUN_ID, slurm_job_id=_slurm_job_id)

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

    # Grow the boundary SurfaceRZFourier basis to (v_mpol, v_ntor) if needed.
    # SurfaceRZFourier.change_resolution() returns a *new* surface (it's just a
    # .copy() with new mpol/ntor) that preserves existing coefficients and pads
    # higher modes with zero. Without this, cold-start seed_surf (built at
    # VMEC_MPOL[0]/VMEC_NTOR[0]) and warm-start wout_surf (at wout_mpol/ntor)
    # lack the dof names for high-m/high-n modes, so the fixed_range call below
    # raises "'rc(0,4)' is not in list" when max_mode exceeds the initial basis.
    if surf.mpol < v_mpol or surf.ntor < v_ntor:
        new_surf = surf.change_resolution(max(surf.mpol, v_mpol),
                                          max(surf.ntor, v_ntor))
        vmec.boundary = new_surf
        surf = new_surf
        # Re-sync the Vmec Optimizable DOF cache after replacing boundary
        # (same failure mode #6 fix as the cold-start branch).
        vmec.local_full_x = np.asarray(vmec.get_dofs())

    # Radial resolution ramp. Use all NS stages up to and including the current
    # step's target, so the multi-grid solver climbs to ns = NS_ARRAY[step] on
    # each optimizer eval. Failure mode #9: without this, a fixed ns can't
    # resolve the helical content's gradient and iota drifts from the prescribed
    # profile.
    _ns_up_to_step = NS_ARRAY[: step + 1] if step + 1 <= len(NS_ARRAY) else NS_ARRAY
    vmec.indata.ns_array[:]    = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:]  = 0.0
    vmec.indata.ns_array[:len(_ns_up_to_step)]    = _ns_up_to_step
    vmec.indata.niter_array[:len(_ns_up_to_step)] = [10000] * len(_ns_up_to_step)
    vmec.indata.ftol_array[:len(_ns_up_to_step)]  = [1.0e-10] * len(_ns_up_to_step)

    # Update booz_xform resolution
    boozer.mpol = b_mpol
    boozer.ntor = b_ntor

    # Rebuild LeastSquaresProblem — Quasisymmetry.J() residual length changes
    # with boozer mpol/ntor (one residual per symmetry-breaking Boozer mode).
    prob = _build_prob()

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

    # Post-step diagnostics. Wrapped in try/except because all of these
    # re-trigger vmec.run() on the current DOF state, and scipy can leave
    # the final accepted point at a non-converging VMEC state (scipy uses
    # the fail=1e12 residual fallback to step away from bad evals, so the
    # *last* eval may be the bad one). We still want the loop to proceed
    # to the next resolution step and the final save block.
    step_runtime = time.time() - step_t0
    try:
        qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
        obj = prob.objective()
        aspect = float(vmec.aspect())
        iota_ax = float(vmec.iota_axis())
        iota_ed = float(vmec.iota_edge())
        vol = float(vmec.volume())
        _diag_ok = True
    except ObjectiveFailure as e:
        proc0_print(f'    WARNING: post-step diagnostics failed: {e}')
        proc0_print( '    (scipy left final accepted point at a non-converging VMEC state; '
                     'continuing to next step)')
        qs_total = float('nan')
        obj = float('nan')
        aspect = float('nan')
        iota_ax = float('nan')
        iota_ed = float('nan')
        vol = float('nan')
        _diag_ok = False

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

    # Snapshot the current wout if this step's diagnostics succeeded. We stash
    # the path (not a copy yet) so that at end-of-run we can pick the newest
    # valid one even if the final step's VMEC call fails.
    if _diag_ok and vmec.output_file and os.path.exists(vmec.output_file):
        _last_good_wout_src = vmec.output_file
        _last_good_step = step + 1

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

# Re-evaluate final state. Protected against VMEC failure on the final DOFs
# (same rationale as the per-step diagnostics block above).
try:
    qs_total = float(sum(np.sum(qs.J()**2) for qs in qs_list))
    obj = prob.objective()
    final_aspect = float(vmec.aspect())
    final_iota_ax = float(vmec.iota_axis())
    final_iota_ed = float(vmec.iota_edge())
    final_vol = float(vmec.volume())
    _final_ok = True
except ObjectiveFailure as e:
    proc0_print(f'WARNING: final-state diagnostics failed: {e}')
    proc0_print( '(reporting last successful per-step values instead)')
    # Fall back to whatever the last successful per-step eval captured.
    # These are still in scope from the for-loop above.
    qs_total = qs_total
    obj = obj
    final_aspect = aspect
    final_iota_ax = iota_ax
    final_iota_ed = iota_ed
    final_vol = vol
    _final_ok = False

aspect_err = abs(final_aspect - ASPECT_TARGET) if _final_ok else float('nan')
iota_axis_err = abs(final_iota_ax - IOTA_TARGET) if _final_ok else float('nan')
iota_edge_err = abs(final_iota_ed - IOTA_TARGET) if _final_ok else float('nan')

# Success criterion (cold-start update, 2026-04-12): gate only on VMEC
# convergence (_final_ok). Physics-metric checks (iota, aspect, volume) are
# evaluated externally by a post-processing script over the diagnostics CSV
# so that Pareto-scan runs always complete and produce outputs regardless
# of whether a specific (R0, V, iota) target was achievable — see the
# "Success metrics" section of local/cold_start_stage1_prompt.md. Errors are
# still computed below for the FINAL STATE report.
success = _final_ok

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
    try:
        _qs_val = float(np.sum(qs.J()**2))
    except ObjectiveFailure:
        _qs_val = float('nan')
    proc0_print(f'        s={s:.2f}:  {_qs_val:.6e}')

proc0_print(f'    QS total:            {qs_total:.6e}')


# ──────────────────────────────────────────────────────────────────────────────
# Save optimized wout + boozmn
# ──────────────────────────────────────────────────────────────────────────────
wout_opt_path = artifact_path("stage1", RUN_ID, OUT_DIR, "wout_opt")

# Pick the freshest valid wout: prefer the final-state file if VMEC converged
# on the optimizer's last accepted DOFs, otherwise fall back to the snapshot
# from the last step whose post-eval diagnostics succeeded (tracked above).
_final_src = vmec.output_file if (_final_ok and vmec.output_file
                                  and os.path.exists(vmec.output_file)) else None
wout_src = _final_src or _last_good_wout_src
_wout_saved = False
if mpi.proc0_world and wout_src and os.path.exists(wout_src):
    shutil.copy2(wout_src, wout_opt_path)
    if _final_src:
        proc0_print(f'Optimized wout saved to {wout_opt_path}')
    else:
        proc0_print(
            f'Optimized wout saved to {wout_opt_path} '
            f'(from last converged step {_last_good_step}; final step failed to converge)'
        )
    _wout_saved = True
else:
    proc0_print(
        'WARNING: No valid wout to save — neither the final accepted point '
        'nor any earlier step produced a converging VMEC run. Skipping wout, '
        'boozmn, and BoozerSurface writes.'
    )

# Write classic boozmn_*.nc via booz_xform. boozer.bx is a direct reference to
# the native booz_xform.Booz_xform C++ object (SIMSOPT's Boozer.save() only
# writes the generic Optimizable JSON, not the standard boozmn NetCDF).
# The FINAL STATE block above calls qs.J() which triggers boozer.bx.run() on
# the final equilibrium, so bx results are current. Naming convention mirrors
# VMEC's wout_<ext>.nc → boozmn_<ext>.nc.
boozmn_path = artifact_path("stage1", RUN_ID, OUT_DIR, "boozmn_opt")
# Only write boozmn when the final-state eval succeeded. boozer.bx holds
# state from the final booz_xform run — if VMEC failed on the final DOFs,
# bx has stale/corrupt data and the file would not correspond to the saved
# wout (which may be a fallback from an earlier step).
if mpi.proc0_world and _final_ok:
    try:
        boozer.bx.write_boozmn(boozmn_path)
        proc0_print(f'Boozer transform saved to {boozmn_path}')
    except Exception as e:
        proc0_print(f'WARNING: Could not write boozmn file: {e}')
elif mpi.proc0_world:
    proc0_print(
        'Skipping boozmn write — final VMEC state did not converge, so '
        'boozer.bx contents do not match the saved wout.'
    )


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
_bsurf_saved = False
if mpi.proc0_world and _wout_saved:
    proc0_print('\nBuilding BoozerSurface from optimized wout for stage 2...')
    bsurf_out_path = artifact_path("stage1", RUN_ID, OUT_DIR, "bsurf_opt")
    try:
        build_and_save(cfg, wout_path=wout_opt_path, out_path=bsurf_out_path,
                       save_vtk=True, print_fn=proc0_print)
        proc0_print(f'BoozerSurface saved to {bsurf_out_path}')
        _bsurf_saved = True
    except Exception as e:
        proc0_print(f'WARNING: BoozerSurface build failed: {e}')
elif mpi.proc0_world:
    proc0_print('\nSkipping BoozerSurface build — no valid wout was saved.')

proc0_print(f'\nDiagnostics saved to {DIAGNOSTICS_FILE}')
proc0_print(f'Outputs saved to {RUN_DIR}')


# ──────────────────────────────────────────────────────────────────────────────
# Registry: finalize the row with metrics or an error code
# ──────────────────────────────────────────────────────────────────────────────
# Success criterion mirrors the termination banner above (_final_ok) but also
# requires that we actually saved the wout and built the BoozerSurface — a
# stage 1 row without those artifacts is useless to downstream stages.
if mpi.proc0_world and registry is not None:
    _metrics = {
        "final_iota_axis": None if not _final_ok else final_iota_ax,
        "final_iota_edge": None if not _final_ok else final_iota_ed,
        "final_aspect":    None if not _final_ok else final_aspect,
        "final_volume":    None if not _final_ok else final_vol,
        "final_qs_metric": None if not _final_ok else qs_total,
        "nfev":            int(vmec.iter),
        "runtime_s":       float(total_runtime),
    }
    if _wout_saved and _bsurf_saved and _final_ok:
        registry.mark_success("stage1", RUN_ID, metrics=_metrics,
                              slurm_wall_s=float(total_runtime))
    else:
        if not _wout_saved:
            err_code = "solver_diverged"
            err_msg  = "no converging VMEC run across resolution ramp"
        elif not _bsurf_saved:
            err_code = "file_save_failed"
            err_msg  = "wout saved but BoozerSurface build failed"
        else:
            err_code = "solver_diverged"
            err_msg  = "final VMEC state did not converge"
        registry.mark_failed("stage1", RUN_ID,
                             error_code=err_code, error_message=err_msg,
                             slurm_wall_s=float(total_runtime),
                             metrics=_metrics)
