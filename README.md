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
./submit.sh 03 regular               # single-stage (regular queue only)
./submit.sh 03_singlestage debug     # single-stage (debug queue only)
```

`run_driver.sh` is the generic SLURM batch script called by `submit.sh`. It finds `${DRIVER}_driver.py`, runs it with `srun` when MPI tasks > 1, and moves logs to the output directory printed by the driver's `atexit` handler (`OUT_DIR=...`).

## Scripts

### Pipeline (run in order)

| # | Script | Purpose | Inputs | Outputs |
|---|--------|---------|--------|---------|
| 01 | `01_stage1_driver.py` | Stage 1: VMEC fixed-boundary QA optimization ($M=1$, $N=0$) with resolution ramp. MPI required. | seed wout (warm) or config (cold) | `wout_stage1.nc`, `boozmn_stage1.nc`, `inputs/stage1_boozersurface_opt.json` (+ copy in output dir) |
| 02 | `02_stage2_driver.py` | Stage 2: fixed plasma boundary, optimize banana coil shape (SquaredFlux + geometric penalties) | `stage1_boozersurface_opt.json` | `stage2_boozersurface_opt.json` |
| 03 | `03_singlestage_driver.py` | Stage 3: jointly optimize coil shapes and plasma boundary (BoozerLS + penalties) | stage 2 BoozerSurface | `singlestage_boozersurface_opt.json` |

### Unnumbered Drivers

| Script | Purpose | Status |
|--------|---------|--------|
| `boozxform_driver.py` | Run booz_xform to extract Boozer-coordinate surface + equilibrium iota/G | Diagnostic tool |
| `vmec_resize_driver.py` | Resize VMEC LCFS to match target plasma dimensions | Superseded by stage 1 |

### Utilities (`utils/`)

| Script | Purpose |
|--------|---------|
| `utils/init_boozersurface.py` | Build TF+banana coils and plasma surface, assemble BoozerSurface. Importable functions + standalone CLI. |
| `utils/output_dir.py` | Resolve output directory: `$BANANA_OUT_DIR` → `$SCRATCH/banana_drivers_outputs/` → `./outputs/` |
| `utils/post_process.py` | Extract physics metrics from optimized BoozerSurface files, append to CSV |
| `utils/generate_vf_coils.py` | Generate VF coil BiotSavart for finite-current cases → `inputs/vf_biotsavart.json` |

### On Hold (`local/`)

Legacy files, temp-hold drivers, and the master prompt live in `local/`.

---

## Key Parameters

### Hardware Constraints
- **TF coils**: 20 coils, 80 kA each, `R0=0.976 m`, `R1=0.4 m`, order=1
- **Banana coils**: nfp=5, stellsym, wound on winding surface `R0=0.976 m`, `a=0.215 m`, max 16 kA
- **Banana coil order**: 2 (order=4 produces bad coils)
- **Banana curvature p-norm**: 4 (L4 produces better coils than L2)
- **Target plasma**: `R0=0.925 m`, edge iota ~ 0.12, nfp=5, stellsym

### Solver (current baseline target)
- **Boozer method**: BoozerLS (BoozerExact deferred due to Newton initialization issues)
- **Target resolution**: mpol = ntor = 12 (start at 6-8, ramp up via Fourier continuation)
- **Convergence**: ftol AND gtol satisfied, Boozer residual < 1e-4
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

- **Stage 1**: VMEC fixed-boundary optimization targeting quasi-axisymmetry ($M=1$, $N=0$). Resolution ramp over boundary Fourier modes. Warm start from existing wout or cold start for Pareto scans over iota/volume targets (`BANANA_IOTA`, `BANANA_VOLUME` env vars). Produces optimized wout + `stage1_boozersurface_opt.json`.
- **Stage 2**: Coil-only optimization (SquaredFlux + geometric penalties). Banana coil shape only by default; TF coils fixed. Two solvers selectable via `stage2_mode` in `config.yaml`:
  - `alm` (default) — augmented Lagrangian method. `f=None` with SquaredFlux and all geometric penalties (length ≤ 1.75 m, CC ≥ 0.05 m, curvature ≤ 40 m⁻¹) placed in the constraint list. Outer loop ramps per-constraint penalty weights $\mu_i$ and updates Lagrange multipliers $\lambda_i$; inner loop is L-BFGS-B on a smooth augmented Lagrangian — no penalty cliffs. Writes `stage2_alm_summary.json` with per-constraint $(c, \lambda, \mu, w_\text{eff})$.
  - `weighted` (legacy) — fixed-weight L-BFGS-B on the scalar objective $J = J_\text{sqf} + w_l J_l + w_{cc} J_{cc} + w_\text{curv} J_\text{curv}$. Kept available for comparison and as a fallback; weights live in `stage2_weights`.

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
