# Banana Drivers

Banana coil optimization drivers for a stellarator-tokamak hybrid device using SIMSOPT. The overall goal is to produce a realizable set of banana coils capable of (a) pure stellarator scenarios and (b) finite-current scenarios. These coil designs are intended to be manufactured.

**This is a separate project from qi_drivers.** General practices (workflow pattern, formatting, documentation) are shared, but the physics problem, coil geometry, and optimization landscape are different. Do not assume solutions from qi_drivers transfer directly.

## Workflow Pipeline

The optimization runs in stages, each warm-starting from the previous via BoozerSurface JSON:

1. **01_stage1** — VMEC fixed-boundary optimization (QA target) with resolution ramp → `wout_stage1.nc`, `boozmn_stage1.nc`, `inputs/stage1_boozersurface_opt.json`
2. **02_stage2** — Load stage 1 BoozerSurface → coil-only optimization (SquaredFlux + penalties) → `outputs/stage2_boozersurface_opt.json`
3. **03_singlestage** — Load stage 2 BoozerSurface → joint coil + surface optimization (BoozerLS + penalties) → `outputs/singlestage_boozersurface_opt.json`
4. **poincare_tracing** — Field-line tracing with raw surface cross-section overlay
5. **Pareto scan** — Sweep over banana current, volume, iota targets (not yet implemented)
6. **Finite-current** — Proxy coil + VF coils for finite plasma current (not yet implemented)

BoozerSurface JSON is the interchange format between stages (matching qi_drivers pattern). Each stage must converge before proceeding to the next. Always verify stage 2 feasibility before running single-stage.

## File Structure

```
banana_drivers/
  config.yaml                        # Single source of truth for all drivers
  CLAUDE.md                          # This file — Claude instructions
  PLAN.md                            # Current status and planned work
  README.md                          # Technical reference
  submit.sh                          # Unified SLURM launcher (./submit.sh 01|02|03, ./submit.sh poincare ...)
  run_driver.sh                      # Generic SLURM batch script (called by submit.sh)
  run_poincare.sh                    # SLURM batch script for Poincare tracing (MPI)
  archive.sh                         # Move old scratch outputs into a timestamped archive subdir
  01_stage1_driver.py                # Stage 1: VMEC fixed-boundary QA optimization (MPI)
  02_stage2_driver.py                # Stage 2: coil-only optimization (ALM or weighted)
  03_singlestage_driver.py           # Stage 3: joint coil + surface optimization (BoozerLS)
  poincare_tracing.py                # Poincare field-line tracing + raw surface overlay
  patch_surface_quadpoints.py        # One-shot patcher for legacy half-period BoozerSurface JSON files
  boozxform_driver.py                # Legacy booz_xform surface extraction (superseded by stage 1)
  vmec_resize_driver.py              # Legacy VMEC resize utility (superseded by stage 1)
  inputs/                            # wout seed, stage1_boozersurface_opt.json, vf_biotsavart.json
  outputs/                           # Local fallback only — primary outputs go to $SCRATCH/banana_drivers_outputs/
  utils/
    init_boozersurface.py            # Build TF + banana coils + plasma surface → BoozerSurface (used by stage 1 + CLI)
    output_dir.py                    # Resolve output directory ($SCRATCH → ./outputs fallback)
    post_process.py                  # Metrics extraction and CSV comparison
    generate_vf_coils.py             # VF coil generation for finite-current
  local/                             # Legacy files, on-hold drivers, master prompt
    prompt.md                        # Master prompt and requirements
```

Active output directory is **`$SCRATCH/banana_drivers_outputs/`** (resolved by `utils/output_dir.py`). The in-tree `outputs/` directory is only used as a fallback when `$SCRATCH` is unavailable.

## Key Design Decisions

### config.yaml is the single source of truth
All drivers read thresholds, weights, optimizer settings, device geometry, and warm-start paths from `config.yaml`. Never hardcode these values in driver scripts.

### Hardware Constraints (not relaxable)
Unlike qi_drivers, constraint thresholds in this project are **fixed hardware limits** from the device design (coil-coil clearance, max curvature from bending radius, max coil length from available conductor). They are NOT expressed with relaxation factors — the values in config.yaml are non-negotiable. Always cross-check against `config.yaml` (which is the source of truth); the list below is informational.
- TF coil current: 100 kA, 20 coils, R0=0.976 m, R1=0.4 m, order=1 (all fixed, not optimized)
- Maximum banana coil current: 16 kA
- Banana coils: nfp=5, stellsym, wound on winding surface R0=0.976 m, a=0.215 m, order=2 (order=4 produces bad coils — distinct from curvature p-norm)
- Target plasma: R0=0.925 m, iota target 0.15, nfp=5, stellsym
- Stage 2 hardware thresholds: `length_max=1.75 m`, `coil_coil_min=0.05 m`, `curvature_max=40 m⁻¹`
- Singlestage hardware thresholds: `coil_surface_min=0.02 m`, `curvature_max=20 m⁻¹` (tighter)

### Env var overrides
Drivers read select parameters from environment variables to support Pareto scans and ad-hoc tuning without editing config.yaml:
- `BANANA_OUT_DIR` — output directory (default: `$SCRATCH/banana_drivers_outputs/` with `./outputs` fallback)
- `BANANA_OUTPUT_PREFIX` — file prefix for per-run outputs (default: `stage1` / `singlestage`)
- `BANANA_IOTA` — stage 1 iota target override (Pareto axis)
- `BANANA_VOLUME` — stage 1 volume target override (Pareto axis)
- `BANANA_STAGE2_MODE` — `alm` (default) or `weighted` — select stage 2 solver
- `BANANA_CURRENT_MODE_S2` — `fixed` (default), `penalized`, or `free` — how stage 2 treats the banana current DOF (see "Stage 2 current handling" below)
- `BANANA_TAU` — stage 2 ALM penalty growth factor override
- `BANANA_MAXITER_LAG` — stage 2 ALM outer-loop iteration cap override

### Surface Quadpoints Range
All drivers must use `range="field period"` when creating surfaces for the BoozerSurface. This gives quadpoints_phi in `[0, 1/nfp)` (one full field period). Do NOT use `range="half period"` (gives only half a period — caused a bug where stage 2 JSON contained incorrect surface domains) or `range="full torus"` (unnecessary — SquaredFlux averages over whatever points are given, and stellsym makes one period sufficient).

### Boozer Method
- Use BoozerLS for singlestage baseline (`constraint_weight > 0 → boozer_type='ls'`, set in `boozer.constraint_weight` in config.yaml — currently 1.0).
- BoozerExact deferred — revisit for stochastic optimization or tighter residuals.
- Stage 1 and stage 2 do **not** solve a BoozerSurface. Stage 1 builds a surface via `SurfaceXYZTensorFourier.least_squares_fit(gamma)` from the optimized VMEC equilibrium; stage 2 loads that surface as-is and uses it only as a fixed evaluation grid for `SquaredFlux`. Only singlestage (`03_singlestage_driver.py`) calls `boozersurface.run_code(...)` and modifies the surface DOFs.

### Objective Function
- `SurfaceSurfaceDistance` excluded from objective (complicates DOF shifting, minimal benefit) — measure in post-processing only
- `BANANA_CURV_P = 4` (L4 norm) is intentional — produces better banana coils than L2
- Banana coil current is a `ScaledCurrent(Current(1), current_init)`. In singlestage it is always a free DOF; the hardware cap is enforced via `QuadraticPenalty` or a hard L-BFGS-B bound depending on initial feasibility. In stage 2 the treatment is selectable via `current_mode_stage2` — see "Stage 2 current handling" below.

### Stage 2 current handling
Three modes selectable via `current_mode_stage2` in config.yaml or `BANANA_CURRENT_MODE_S2` env var:
- **`fixed` (default)** — `banana_current` is pinned at `current_fixed_stage2` (default 16 kA, the hardware upper bound) and `fix_all()` removes it from the free DOF set. Stage 2 becomes a shape-only optimization; singlestage handles joint current + shape refinement. This is the recommended mode: stage 2 is a warm-start generator, not a final design, and we don't need it to decide the operating current.
- **`penalized`** — current stays free, with `QuadraticPenalty(|I|, current_soft_max_stage2, "max")` added to the ALM constraint list. Known to collapse to $I=0$ from the post-stage-1 warm start because normalized `SquaredFlux` is lower with TF alone than with TF plus an unconverged banana shape; once zeroed, all shape gradients vanish (they scale as $I$). Preserved for experiments but not recommended. See PLAN.md for the 51246996 failure analysis.
- **`free`** — current free, no current constraint. Same collapse mode as `penalized`. Kept for regression tests.

`CurrentPenaltyWrapper` (in `utils/current_penalty.py`) is the adapter that makes `ScaledCurrent` compatible with `QuadraticPenalty`; shared between stage 2 (penalized mode) and singlestage. The gradient uses `sign(I) * scaled_current.vjp([1.0])` — the `vjp` carries the `ScaledCurrent` scale factor via SIMSOPT's chain rule, so a naive `sign(I)` alone would underweight the gradient by ~10^4.

### Stage 2 solver: ALM by default
Stage 2 uses the augmented Lagrangian method (`stage2_mode: alm` in config.yaml). Rationale:
- `f=None` with all four terms (SquaredFlux, length, coil-coil, curvature) placed in the constraint list — no fixed weights, no penalty cliffs.
- `SquaredFlux` is wrapped in `QuadraticPenalty(Jsqf, sqf_target, "max")` so its contribution clips to zero once `Jsqf < sqf_target`. Without this wrapper SquaredFlux is the only non-self-clipping term and dominates the descent indefinitely (see PLAN.md for the 51228710 failure mode that motivated the wrapper).
- `dof_scale` (default 0.1) limits the physical step per inner L-BFGS-B iteration; matches the qi_drivers pattern and prevents inner line-search stall.
- Per-constraint `μᵢ` ramps by `tau` per outer iteration. Lower `tau` is safer for stiff constraint problems (curvature in particular) — see the μ-explosion discussion in PLAN.md.
- Legacy `weighted` mode (single scalar objective with fixed weights) is still available via `BANANA_STAGE2_MODE=weighted` for comparison runs.

### Singlestage solver: BoozerLS L-BFGS-B
- Uses BoozerLS (`constraint_weight > 0 → boozer_type='ls'`) with `constraint_weight=1.0` from `config.yaml`.
- Fixed-weight scalar objective (nonQS + Boozer residual + iota + length + coil-coil + coil-surface + curvature + current). ALM is **not** yet ported to singlestage — see PLAN.md `deferred` section.
- Boozer residual target < 1e-4 minimum acceptable.

## Key Conventions

### Naming
- Use compound prefixes without internal underscores: `singlestage`, `biotsavart`, `boozersurface`
- Prefix hierarchy: `singlestage_unperturbed_auglag` not `single_stage_unperturbed_augmented_lagrangian`
- SIMSOPT objects use full names: `boozersurface`, `biotsavart`, `surface`

### Coil curves
- Use `CurveCWSFourierCPP` (not the Python `CurveCWSFourier`) for performance

### Driver script structure
Every driver follows this pattern (must match qi_drivers formatting):
1. Module docstring (script name, one-line purpose, usage)
2. `proc0_print` wrapper (defaults `flush=True`); ALL output via `proc0_print`
3. Parameter definitions
4. `atexit` handler (prints `OUT_DIR=...` for SLURM log capture)
5. Print `INPUT PARAMETERS` block (date, all params, thresholds, weights, optimizer settings)
6. Load warm-start data / build geometry
7. Build objective and constraints
8. Print `INITIAL STATE` block ("Parameter values" = physics quantities, "Penalty values" = objective terms, `n_dofs`)
9. Initialize diagnostics CSV file with header comments and column names
10. Set up `fun(x)` with `_write_diagnostics_row()` and `callback(x)` with per-iteration state
11. Run optimizer
12. Print `SUCCESS`/`FAILURE` termination banner (scipy message, iterations, grad/ftol conditions)
13. Print `FINAL STATE` block (mirrors INITIAL STATE)
14. Save outputs
15. Use `# ──────` comment blocks between major code sections

### Output and logging
- ALL printing via `proc0_print` (never bare `print()`)
- Drivers print INPUT PARAMETERS, INITIAL STATE, per-iteration STATE+PENALTIES, TERMINATION, and FINAL STATE blocks
- "Parameter values" = actual physics quantities; "Penalty values" = objective function terms
- Diagnostics CSV (`{stage}_diagnostics.txt`) appended per inner-loop eval for crash-resilient data capture
- Shell scripts pipe output via `tee` to log files
- `atexit` handler prints output directory path for shell script to capture

### SLURM
- All SLURM submissions performed manually by the user via `submit.sh`
- `submit.sh` handles debug/regular/auto mode selection and per-driver SLURM settings
- `run_driver.sh` is the generic batch script — driver name passed via `DRIVER` env var
- Drivers print `OUT_DIR=...` via atexit for log capture by `run_driver.sh`
- Stage 1 (VMEC fixed-boundary optimization) is MPI-parallel — uses `least_squares_mpi_solve` with finite-difference gradients, so ndofs+1 ranks is the useful upper bound. `submit.sh` sets `NTASKS=16` by default for stage 1.
- Stages 2 and 3 (single-surface BoozerLS / coil-only optimization) are inherently serial — do not waste CPUs (`NTASKS=1`)
- Poincare tracing parallelizes over fieldlines (one MPI rank per field line)
- Poincare: `./submit.sh poincare <input.json> [debug|regular|auto] [extra args...]`
- `run_poincare.sh` is the MPI batch script — receives `POINCARE_INPUT`, `POINCARE_LABEL`, `POINCARE_ARGS` via env
- Poincare overlay and starting points use the raw loaded surface directly — no BoozerSurface solve (which distorts the surface when volume/iota don't match the actual coil field geometry)

### Documentation
- Document what works AND what doesn't
- Update CLAUDE.md, PLAN.md, and memory as decisions are made
- The more complicated the code, the more documentation it needs
- Comments, docstrings, and typehints for complex sections

## Environment

- **HPC**: Perlmutter @ NERSC (128 CPUs per node, SLURM scheduler)
- **SIMSOPT fork**: `hayashiw/simsopt` on `whjh/auglag_banana` branch, located at `hybrid_torus/banana/simsopt/` (relative to project root)
- **SIMSOPT for banana**: Always reference `hybrid_torus/banana/simsopt/` for banana-specific SIMSOPT issues (e.g., `CurveCWSFourierCPP`, the `auglag_banana` modifications to `augmented_lagrangian.py`). Do NOT check `qi_rso/simsopt/` unless explicitly asked — it is a separate fork for the QI project.
- **Related repo**: `qi_rso/qi_drivers/` (separate project; reference for formatting/workflow patterns and the original ALM implementation that was ported here.)
