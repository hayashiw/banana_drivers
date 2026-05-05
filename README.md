# Banana Drivers

**NOTE: All recent work is being done in the `jhalpern30/` directory**
The top level drivers (`01_stage1_driver.py`, `02_stage2_driver.py`, `03_singlestage_driver.py`) are failing to recover baseline behavior of the example scripts in the `accessibility` branch of the `jhalpern30` SIMSOPT fork.
As a intermediate solution, new driver scripts built directly off of the `jhalpern30` example are being used as the primary driver scripts.
These new driver scripts are in `banana_drivers/jhalpern30`.

Driver and utility scripts for optimization of **banana coils** in a stellarator-tokamak hybrid device using SIMSOPT.

See `CLAUDE.md` for Claude-specific instructions and design decisions. See `PLAN.md` for current status and planned work.

---

## Goal

Produce a realizable set of banana coils for a stellarator-tokamak hybrid capable of:
- **(a)** Pure stellarator scenarios (vacuum field)
- **(b)** Finite-current scenarios (proxy coil + VF coils)

These coil designs are intended to be manufactured.

---

## Submitting Jobs

All drivers are submitted via `submit.sh`, which handles SLURM queue selection and per-driver settings:

```bash
./submit.sh 01                       # stage 1 VMEC QA (auto: debug → shared fallback, MPI)
./submit.sh 02                       # stage 2 coil-only (auto mode)
./submit.sh 02 --poincare-gate       # stage 2 + post-run --quick Poincare trace (afterok)
./submit.sh 03 shared                # single-stage (shared queue only)
./submit.sh 03_singlestage debug     # single-stage (debug queue only)
```

`run_driver.sh` is the generic SLURM batch script called by `submit.sh`. It finds `${DRIVER}_driver.py`, runs it with `srun` when MPI tasks > 1, and moves logs to the output directory printed by the driver's `atexit` handler (`OUT_DIR=...`).

## Scripts

### Pipeline (run in order)

| # | Script | Purpose | Inputs | Outputs |
|---|--------|---------|--------|---------|
| 01 | `01_stage1_driver.py` | Stage 1: VMEC fixed-boundary QA optimization ($M=1$, $N=0$) with resolution ramp. MPI required. | seed wout (warm) or config (cold) | `wout_stage1_opt.nc`, `boozmn_stage1_opt.nc`, `stage1_boozersurface_opt.json` (all in `$OUT_DIR`) |
| 02 | `02_stage2_driver.py` | Stage 2: fixed plasma boundary, optimize banana coil shape (SquaredFlux + geometric penalties) | `stage1_boozersurface_opt.json` | `stage2_boozersurface_opt.json` |
| 03 | `03_singlestage_driver.py` | Stage 3: jointly optimize coil shapes and plasma boundary (BoozerLS + penalties) | stage 2 BoozerSurface | `singlestage_boozersurface_opt.json` |

### Unnumbered Drivers

| Script | Purpose | Status |
|--------|---------|--------|
| `boozxform_driver.py` | Run booz_xform to extract Boozer-coordinate surface + equilibrium iota/G | Diagnostic tool |

### Diagnostics (`local/`)

| Script | Purpose |
|--------|---------|
| `local/diag_iota_from_bs.py` | Compute $\iota$ from BiotSavart field on VMEC surfaces via contravariant $B^\theta/B^\phi$ decomposition |
| `local/diag_field_decomposition.py` | Visualize $\|B_\text{banana}\| / \|B_\text{TF}\|$ on toroidal cross sections (2x2 plot) |
| `local/diag_iota_basin.py` | One-off BoozerLS iota basin investigation (iota scan, surface fingerprint) |
| `local/diag_coil_capability.py` | Cold coil capability probe: maximize $\|\iota\|$ over banana DOFs via BoozerLS |
| `local/sweeps/current_poincare/` | Current Poincaré sweep: stage 2 at $I\in\{2,4,8,12,16\}$ kA + Poincaré tracing on each |
| `local/sweeps/order_poincare/` | Order Poincaré sweep: re-init + stage 2 at order $\in\{2,3\}$ + Poincaré tracing on each |
| `local/sweeps/curvmax_stage2/` | Stage 2 curvature-threshold sweep: $\kappa_\text{max}\in\{20,30,40,50,60\}$ m⁻¹ |
| `local/sweeps/coilcap_R0_current/` | Cold coil-capability probe sweep over $(R_0, I_\text{banana})$ |
| `local/sweeps/stage1_pareto/` | Stage 1 Pareto scan over $(R_0, V, \iota)$ targets using `near_axis_seed` cold start |

### Utilities (`utils/`)

| Script | Purpose |
|--------|---------|
| `utils/init_boozersurface.py` | Build TF+banana coils and plasma surface, assemble BoozerSurface. Importable functions + standalone CLI. |
| `utils/near_axis_seed.py` | pyQSC Landreman-Sengupta near-axis expansion seeder for stage 1 cold start. Adaptive delta walker picks the least-elongated helical axis excursion that brackets the iota target. Used by `01_stage1_driver.py` when `stage1.cold_start: true`. |
| `utils/vmec_resize.py` | One-time preprocessing: extract s=0.24 of original seed wout and re-solve VMEC (two-pass, rbtor-matched to hardware TF coils) to produce `inputs/wout_stage1_seed.nc`. |
| `utils/output_dir.py` | Resolve output directory: `$BANANA_OUT_DIR` → `$SCRATCH/banana_drivers_outputs/` → `./outputs/` |
| `utils/post_process.py` | Extract physics metrics from optimized BoozerSurface files, append to CSV |
| `utils/generate_vf_coils.py` | Generate VF coil BiotSavart for finite-current cases → `inputs/vf_biotsavart.json` |
| `utils/hbt_parameters.py` | HBT-EP machine parameters: major radius, winding surface, TF current, target LCFS geometry |

### Top-level tracked experiment dirs

Tracked top-level while the finite-current / I>0 work is the active line of attack. Will move under `local/` once the pipeline integration settles; do not add infra that assumes the top-level arrangement is permanent.

| Directory | Purpose |
|-----------|---------|
| `jhalpern30/` | Current working drivers (`stage2.py`, `singlestage.py`) with the new hardware limits (80 kA TF, κ≤100 m⁻¹, CC≥5 cm, CS≥1.5 cm, length≤2 m) and the `new_objectives/` penalties wired in. Singlestage supports a Fourier continuation ramp (`BANANA_RAMP=1`: surface `mpol=ntor ∈ [6,8,10,12]` paired with banana order `[3,4,5,5]` and `qp ∈ [192,256,320,320]`; `BANANA_RAMP_STAGES=N` truncates to the first N; `BANANA_RAMP_CUSTOM='[[mpol,ntor,order,qp],...]'` fully overrides), ALM (`--alm`), an I>0 sign-flip mechanism (`--flip-banana-current` in stage 2 + `_flip` parent-dir suffix auto-detected by singlestage), and warm-start resume (`BANANA_RESUME_STAGE=N` loads `$OUT_DIR/stage{N:02d}/bsurf_opt.json` + `state.json`, re-seeds BoozerLS with the saved iota/G, and re-enters the ramp loop at stage $N+1$; each successful stage now writes a sidecar `state.json`). `run_singlestage.sh -c/--cpus N` sets `--ntasks=1 --cpus-per-task=N --mem=64G` + matching `OMP=MKL=OPENBLAS=N` for singlestage-only runs (default 32); the 16×1 MPI layout is preserved for `-p/--post-process`. `current_kA` CLI argument on both stage 2 and singlestage is a bounded float in $[-16, 25]$ kA so sweeps can pass log-uniform samples directly. Active scans: `scan_plasma_curr/` (11-point plasma-current sweep at VF=0; I≤0 converges end-to-end, I=+1.0 kA flip stage 2 done → `scan_plasma_curr/I1.0kA_flip/`, singlestage flip pending) and `iota15_rithik/` / `iota20_rithik/` (Rithik-provided warm-start coil sets — the reliable singlestage basin entry points). Standalone sweeps: **`pareto_negative_current.py` + `run_pareto_negative_current.sh`** — 128-point Pareto scan over $(|I_\text{plasma}|<5\text{ kA}, \iota_\text{target}\in[0.08, 0.25])$ with $I_\text{plasma}<0$ (in-basin) and `BANANA_RAMP_STAGES=2` (mpol ∈ [6, 8], qp ∈ [192, 256]); ProcessPoolExecutor fan-out on a single regular-QoS 128-CPU node with single-threaded BLAS per worker (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), per-point chain stage2 → post-process → singlestage → post-process, `fcntl`-locked `scan_index.csv` append; output to `$SCRATCH/.../pareto_negative_current/`. **`poloidal_sweep.py` + `run_poloidal_sweep.sh`** — I=0 demonstration that the `PoloidalExtent` penalty activates at stage 2 (concluded 2026-04-22; singlestage end-to-end validation is now covered by keeping `BANANA_POLOIDAL_WEIGHT` non-zero in subsequent singlestage runs rather than maintaining a dedicated sweep). **`resolution_scan.py` + `run_resolution_scan.sh`** — 2×2 grid infrastructure retained on-disk but deprecated in favor of a single-point minimum-resolution check (o4_qp256 via `BANANA_RAMP_CUSTOM` + rithik warm start) because 4-worker fan-out at mpol=10 doesn't fit one SLURM allocation. ALM experiments under `alm/`. Archived scans (incl. the legacy `scan_vf_plasma_curr/` 4×5 VF×I grid and TF=100 kA runs) under `jhalpern30/old/` with the original `SCAN_STATUS.md` tracker. |
| `new_objectives/` | CWS-frame manufacturability penalties, wired into `jhalpern30/stage2.py` + `jhalpern30/singlestage.py`. Still need to be integrated into the numbered `02_stage2_driver.py` + `03_singlestage_driver.py`. Files: `cwsobjectives.py` (base), `poloidal_extent.py` (real poloidal-footprint bound; replaces length), `ellipse_width.py` (coil-perpendicular cross-section), `self_intersect.py` (`CurveSelfIntersect`, enables order=4). |

### On Hold (`local/`)

Legacy files, temp-hold drivers, and the master prompt live in `local/`.

---

## Key Parameters

### Hardware Constraints
- **TF coils**: 20 coils, 80 kA each, `R0=0.976 m`, `R1=0.4 m`, order=1 (all fixed)
- **Banana coils**: nfp=5, stellsym, wound on winding surface `R0=0.976 m`, `a=0.210 m`, max 16 kA
- **Banana coil order**: 4. Order=4 was previously unusable because the coils self-intersected during optimization; the `CurveSelfIntersect` objective in [new_objectives/self_intersect.py](new_objectives/self_intersect.py) now prevents that failure mode, so order=4 is the preferred default. Singlestage ramps the coil order as $[3,4,5,5]$ alongside the surface Fourier ramp $[6,8,10,12]$.
- **Banana curvature p-norm**: 4 (L4 produces better coils than L2)
- **Target plasma**: `R0=0.92 m` (baseline LCFS), edge iota $\approx 0.15$ (confirmed achievable by Poincaré field-line tracing at 16 kA), nfp=5, stellsym
- **Engineering tolerances** (`config.yaml:thresholds`, HW-team verified 2026-04-20, enforced unmodified by singlestage): `length_max=2.0 m` (soft — the real poloidal-footprint bound is the CWS-frame `poloidal_extent` objective in [new_objectives/poloidal_extent.py](new_objectives/poloidal_extent.py), not length), `coil_coil_min=0.05 m`, `coil_surface_min=0.015 m`, `curvature_max=100 m⁻¹` (selected conductor has ~1 cm minimum bending radius). Stage 2 loosens the first three via per-threshold relaxation factors (`config.yaml:stage2_relaxation`, default $1.05\times$) so L-BFGS-B has room to drive squared flux lower. Env var overrides: `BANANA_STAGE2_LENGTH_RELAX`, `BANANA_STAGE2_CC_RELAX`, `BANANA_STAGE2_CURV_RELAX`.

### Solver (current baseline target)
- **Boozer method**: BoozerLS (BoozerExact deferred due to Newton initialization issues); `constraint_weight=1.0e+3`
- **Target resolution**: mpol = ntor = 12, reached via the Fourier continuation ramp documented above (surface `[6,8,10,12]` paired with banana order `[3,4,5,5]` and `qp ∈ [192,256,320,320]`). Non-ramp single-stage runs default to mpol=8, ntor=6 and can be overridden via `BANANA_MPOL` / `BANANA_NTOR` / `BANANA_TARGET_ORDER` / `BANANA_TARGET_QP`.
- **Singlestage convergence**: ftol AND gtol satisfied, Boozer residual < 1e-4
- **Stage 2 convergence**: truncated iteration budget — gtol and ftol set to unreachable (1e-15), maxiter=600 is the expected exit path. Real health check is a post-run Poincare trace via `./submit.sh 02 --poincare-gate`, not L-BFGS-B's gradient norm.
- **Optimizer**: L-BFGS-B
- **Coil curve class**: `CurveCWSFourierCPP`

### Objective Function
- `NonQuasiSymmetricRatio` — quasi-symmetry quality
- `BoozerResidual` — Boozer coordinate accuracy (with BoozerLS)
- `Iotas` (QuadraticPenalty) — rotational transform target
- `CurveLength` (QuadraticPenalty, max) — banana coil length constraint; soft 2 m cap. The real poloidal-footprint bound is `new_objectives/poloidal_extent.py`, not this.
- `poloidal_extent` (CWS-frame) — caps banana coil poloidal angular footprint on the winding surface. The Pareto-sweep axis.
- `ellipse_width` (CWS-frame) — coil-perpendicular ellipse-width penalty for manufacturability.
- `CurveSelfIntersect` — prevents order=4 self-intersection; enables order=4 banana coils.
- `CurveCurveDistance` — minimum coil-coil separation
- `CurveSurfaceDistance` — minimum coil-surface separation
- `LpCurveCurvature` (p=4) — curvature penalty
- `SurfaceSurfaceDistance` — **excluded** from objective (measured in post-processing only)

The three CWS-frame objectives above are currently wired into `jhalpern30/stage2.py` + `jhalpern30/singlestage.py`. Integration into `02_stage2_driver.py` + `03_singlestage_driver.py` is a standing TODO (see `PLAN.md`).

---

## Three-Stage Workflow

```
01_stage1 (VMEC QA opt) → 02_stage2 (coil opt) → 03_singlestage (joint opt)
```

- **Stage 1**: VMEC fixed-boundary optimization targeting quasi-axisymmetry ($M=1$, $N=0$). Resolution ramp over boundary Fourier modes. Warm start from an existing wout, or cold start via `utils/near_axis_seed.py` (pyQSC Landreman-Sengupta near-axis expansion, adaptive delta walker) for Pareto scans over $(R_0, V, \iota)$ targets (`BANANA_COLD_R0`, `BANANA_VOLUME`, `BANANA_IOTA` env vars). Produces optimized wout + `stage1_boozersurface_opt.json`.
- **Stage 2**: Coil-only optimization (SquaredFlux + geometric penalties). Banana coil shape only by default; TF coils fixed. Two solvers selectable via `stage2_mode` in `config.yaml`:
  - `weighted` (default) — fixed-weight L-BFGS-B on the scalar objective $J = J_\text{sqf} + w_l J_l + w_{cc} J_{cc} + w_\text{curv} J_\text{curv}$. Current working baseline; weights live in `stage2_weights`. Each hardware threshold (length, coil-coil, curvature) is scaled by its own relaxation factor from `stage2_relaxation` (default 1.05 on all three) — stage 2 only needs to be "good enough for singlestage to polish", so loosening gives L-BFGS-B room to drive squared flux lower. Overrides: `BANANA_STAGE2_LENGTH_RELAX`, `BANANA_STAGE2_CC_RELAX`, `BANANA_STAGE2_CURV_RELAX`. Singlestage always enforces the unrelaxed hardware limits.
  - `alm` (experimental) — augmented Lagrangian method. `f=None` with SquaredFlux and all geometric penalties placed in the constraint list; outer loop ramps per-constraint $\mu_i$ and updates multipliers $\lambda_i$; inner loop is L-BFGS-B on a smooth augmented Lagrangian. Not yet reliable on this geometry — see `PLAN.md`. Opt in via `BANANA_STAGE2_MODE=alm`.

  The banana current is handled separately via `current_mode_stage2` (or `BANANA_CURRENT_MODE_S2`): `fixed` (default, pinned at 16 kA and dropped from free DOFs so stage 2 is shape-only), `penalized` (free DOF with soft upper cap), or `free` (free DOF, no constraint). See `CLAUDE.md` for the failure mode that motivated `fixed` as the default.
- **Stage 3 (singlestage)**: Joint coil + surface optimization using BoozerLS. Minimizes NonQuasiSymmetricRatio + BoozerResidual + geometric penalties. Currently L-BFGS-B on a weighted objective; ALM port planned (see `TODO(ALM)` in `03_singlestage_driver.py` and `PLAN.md`).

---

## Output Directories

| Directory | Contents |
|-----------|---------|
| `$SCRATCH/banana_drivers_outputs/` | Primary output location (NERSC scratch, 8-week purge). All pipeline outputs: wout, BoozerSurface JSON, VTK, diagnostics. |
| `outputs/` | Fallback when `$SCRATCH` unavailable. Also the archive target for `archive.sh`. |

---

## Environment

- **HPC**: Perlmutter @ NERSC (128 CPUs per node, SLURM scheduler)
- **SIMSOPT fork**: `hayashiw/simsopt` on `whjh/auglag_banana` branch (local: `hybrid_torus/banana/simsopt/`)
- **Related repo**: `qi_rso/qi_drivers/` (separate project, shared practices only)
