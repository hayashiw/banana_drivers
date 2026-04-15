# Banana Drivers

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
./submit.sh 01                       # stage 1 VMEC QA (auto: debug → regular fallback, MPI)
./submit.sh 02                       # stage 2 coil-only (auto mode)
./submit.sh 02 --poincare-gate       # stage 2 + post-run --quick Poincare trace (afterok)
./submit.sh 03 regular               # single-stage (regular queue only)
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

### Top-level tracked experiment dirs (temporary — 2026-04-14)

These directories were promoted from `local/` to the top level ahead of Perlmutter maintenance on 2026-04-15 so they can be tracked in git. They will move back under `local/` once the working state re-syncs. Do not add new infra that assumes the top-level arrangement is permanent.

| Directory | Purpose |
|-----------|---------|
| `jhalpern30/` | In-flight VF × plasma-current 4×5 scan under `scan_vf_plasma_curr/`; per-cell stage-2/Poincaré/singlestage driver variants; baseline `single_stage_banana_example.py`; TF80kA audit; `bananacoils.png` winding-surface projection; `analyze.ipynb`. See `jhalpern30/scan_vf_plasma_curr/SCAN_STATUS.md` for the per-cell status table (stage 2, Poincaré, singlestage, and winding-surface projection checks). |
| `new_objectives/` | CWS-frame poloidal-extent and ellipse-width objectives (`cwsobjectives.py`) + `new_objectives_plan.md`. Reviewed but **not yet** integrated into the drivers. See `PLAN.md` — current roadmap has this as higher priority than before. |

### On Hold (`local/`)

Legacy files, temp-hold drivers, and the master prompt live in `local/`.

---

## Key Parameters

### Hardware Constraints
- **TF coils**: 20 coils, 80 kA each, `R0=0.976 m`, `R1=0.4 m`, order=1 (all fixed)
- **Banana coils**: nfp=5, stellsym, wound on winding surface `R0=0.976 m`, `a=0.210 m`, max 16 kA
- **Banana coil order**: 2 (order=4 produces bad coils)
- **Banana curvature p-norm**: 4 (L4 produces better coils than L2)
- **Target plasma**: `R0=0.92 m` (baseline LCFS), edge iota $\approx 0.15$ (confirmed achievable by Poincaré field-line tracing at 16 kA), nfp=5, stellsym
- **Engineering tolerances** (`config.yaml:thresholds`, enforced unmodified by singlestage): `length_max=1.75 m`, `coil_coil_min=0.05 m`, `coil_surface_min=0.02 m`, `curvature_max=40 m⁻¹`. Stage 2 loosens the first three via per-threshold relaxation factors (`config.yaml:stage2_relaxation`, default $1.05\times$) so L-BFGS-B has room to drive squared flux lower. Env var overrides: `BANANA_STAGE2_LENGTH_RELAX`, `BANANA_STAGE2_CC_RELAX`, `BANANA_STAGE2_CURV_RELAX`.

### Solver (current baseline target)
- **Boozer method**: BoozerLS (BoozerExact deferred due to Newton initialization issues); `constraint_weight=1.0e+3`
- **Target resolution**: mpol = ntor = 12 (start at 6-8, ramp up via Fourier continuation)
- **Singlestage convergence**: ftol AND gtol satisfied, Boozer residual < 1e-4
- **Stage 2 convergence**: truncated iteration budget — gtol and ftol set to unreachable (1e-15), maxiter=600 is the expected exit path. Real health check is a post-run Poincare trace via `./submit.sh 02 --poincare-gate`, not L-BFGS-B's gradient norm.
- **Optimizer**: L-BFGS-B
- **Coil curve class**: `CurveCWSFourierCPP`

### Objective Function
- `NonQuasiSymmetricRatio` — quasi-symmetry quality
- `BoozerResidual` — Boozer coordinate accuracy (with BoozerLS)
- `Iotas` (QuadraticPenalty) — rotational transform target
- `CurveLength` (QuadraticPenalty, max) — banana coil length constraint
- `CurveCurveDistance` — minimum coil-coil separation
- `CurveSurfaceDistance` — minimum coil-surface separation
- `LpCurveCurvature` (p=4) — curvature penalty
- `SurfaceSurfaceDistance` — **excluded** from objective (measured in post-processing only)

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
