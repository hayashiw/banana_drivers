# Banana Drivers

Banana coil optimization drivers for a stellarator-tokamak hybrid device using SIMSOPT. The overall goal is to produce a realizable set of banana coils capable of (a) pure stellarator scenarios and (b) finite-current scenarios. These coil designs are intended to be manufactured.

**This is a separate project from qi_drivers.** General practices (workflow pattern, formatting, documentation) are shared, but the physics problem, coil geometry, and optimization landscape are different. Do not assume solutions from qi_drivers transfer directly.

## Workflow Pipeline

The optimization runs in stages, each warm-starting from the previous via a **content-addressed run id** pinned in `config.yaml:warm_start`. Every run lives in its own per-run directory:

```
$OUT_DIR/
  stage1/<s01_xxxxxx>/    wout_<id>_opt.nc, boozmn_<id>_opt.nc, bsurf_<id>_opt.json, diagnostics_<id>.csv, ...
  stage2/<s02_xxxxxx>/    bsurf_<id>_opt.json, alm_summary_<id>.json, diagnostics_<id>.csv, ...
  singlestage/<s03_xxxxxx>/  bsurf_<id>_opt.json, state_<id>_opt.json, diagnostics_<id>.csv, ...
```

Run ids are SHA256 hashes over a whitelisted subset of `config.yaml` plus the git commit (see `utils/run_registry.py:STAGE{1,2,SINGLESTAGE}_INPUT_KEYS`). Same inputs + same code → same id; any whitelisted change → new id. Registry is a SQLite database at `$OUT_DIR/registry.db` that tracks lifecycle (`pending → running → success | failed | stale`) and parent/child relationships.

1. **01_stage1** — VMEC fixed-boundary optimization (QA target) with resolution ramp → `stage1/<id>/wout_<id>_opt.nc`, `boozmn_<id>_opt.nc`, `bsurf_<id>_opt.json` (plus `_init` snapshots for pre/post comparison)
2. **02_stage2** — Reads `warm_start.stage1_id` from config.yaml, loads parent's `bsurf_opt.json` → coil-only optimization (SquaredFlux + penalties) → `stage2/<id>/bsurf_<id>_opt.json`
3. **03_singlestage** — Reads `warm_start.stage1_id` and `warm_start.stage2_id`, loads stage 2's `bsurf_opt.json` and stage 1's `wout_opt.nc` → joint coil + surface optimization (BoozerLS + penalties) → `singlestage/<id>/bsurf_<id>_opt.json`, `state_<id>_opt.json`

Parent ids are **hard-required**: the drivers raise if `stage1_id` / `stage2_id` are null in config.yaml. No "latest run" fallback — reproducibility over ergonomics.

`inputs/` contains **permanent** inputs only: `wout_stage1_seed.nc` (resized warm-start seed), `vf_biotsavart.json` (VF coil BiotSavart for finite-current cases), `wout_nfp22ginsburg_000_014417_iota15.nc` (original raw seed consumed by `utils/vmec_resize.py`). All per-run artifacts (optimized wouts, boozmns, bsurfs, diagnostics) live in `$OUT_DIR` (`$SCRATCH/banana_drivers_outputs/` by default).
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
  boozxform_driver.py                # Legacy booz_xform surface extraction (superseded by stage 1)
  inputs/                            # PERMANENT inputs only: wout_stage1_seed.nc, vf_biotsavart.json, raw seed wout
  outputs/                           # Local fallback only — primary outputs go to $SCRATCH/banana_drivers_outputs/
  utils/
    init_boozersurface.py            # Build TF + banana coils + plasma surface → BoozerSurface (used by stage 1 + CLI)
    near_axis_seed.py                # pyQSC near-axis expansion seeder for stage 1 cold start (adaptive delta walker)
    vmec_resize.py                   # One-time preprocessing: extract s=0.24 of original seed wout and re-solve VMEC (two-pass, rbtor-matched) to produce wout_stage1_seed.nc
    output_dir.py                    # Resolve output directory ($SCRATCH → ./outputs fallback)
    post_process.py                  # Metrics extraction and CSV comparison
    generate_vf_coils.py             # VF coil generation for finite-current
    hbt_parameters.py                # HBT-EP machine parameters (major radius, winding surface, TF current, target LCFS)
  local/                             # Legacy files, on-hold drivers, diagnostics, master prompt
    prompt.md                        # Master prompt and requirements
    diag_iota_from_bs.py             # Compute iota from BiotSavart field on VMEC surfaces (contravariant B decomposition)
    diag_field_decomposition.py      # Visualize B_banana / B_TF on toroidal cross sections
    diag_iota_basin.py               # One-off BoozerLS iota basin investigation
    diag_coil_capability.py          # Cold coil capability probe (maximize iota over banana DOFs)
    iota_basin_analysis.md           # Analysis doc: iota basin problem evidence, diagnostics, resolution
    new_objectives_plan.md           # Prompt document for new optimization objectives (under review)
    cold_start_stage1_prompt.md      # Prompt document for cold-start stage 1 implementation
    example_pyqsc_near_axis.py       # Pedagogical walkthrough of pyQSC near-axis construction
    sweeps/                          # Sweep infrastructure
      current_poincare/              # Current Poincaré sweep (stage 2 at 2-16 kA + Poincaré tracing)
      order_poincare/                # Order Poincaré sweep (stage 2 at order={2,3} + Poincaré tracing)
      curvmax_stage2/                # Stage 2 curvature-threshold sweep (κ_max ∈ {20,30,40,50,60} m⁻¹)
      coilcap_R0_current/            # Cold coil-capability probe sweep over (R0, I_banana)
      stage1_pareto/                 # Stage 1 Pareto scan over (R0, V, iota) via near_axis_seed cold start
```

Active output directory is **`$SCRATCH/banana_drivers_outputs/`** (resolved by `utils/output_dir.py`). The in-tree `outputs/` directory is only used as a fallback when `$SCRATCH` is unavailable.

## Key Design Decisions

### config.yaml is the single source of truth
All drivers read thresholds, weights, optimizer settings, device geometry, and warm-start paths from `config.yaml`. Never hardcode these values in driver scripts.

### Hardware Constraints and stage 2 relaxation
`config.yaml:thresholds` holds the **true engineering tolerances** from the device design (coil-coil clearance, max curvature from bending radius, max coil length, coil-surface clearance). Singlestage enforces these unmodified. Stage 2 applies per-threshold scaling factors from `config.yaml:stage2_relaxation` on top — stage 2 only needs coils good enough for singlestage to polish, so loosening its constraints gives L-BFGS-B room to drive squared flux lower and shrinks the coil-produced axis drift that singlestage must recover from. Factor=1 reproduces the hardware limit; factor>1 loosens. Concrete hardware values will eventually live in `utils/hbt_parameters.py`; current values in config.yaml are placeholders. Always cross-check against `config.yaml`; the list below is informational.
- TF coil current: 80 kA, 20 coils, R0=0.976 m, R1=0.4 m, order=1 (all fixed, not optimized)
- Maximum banana coil current: 16 kA
- Banana coils: nfp=5, stellsym, wound on winding surface R0=0.976 m, a=0.210 m, order=2 (order=4 produces bad coils — distinct from curvature p-norm)
- Target plasma: R0=0.92 m (baseline LCFS), iota target 0.15, nfp=5, stellsym
- Hardware thresholds: `length_max=1.75 m`, `coil_coil_min=0.05 m`, `coil_surface_min=0.02 m`, `curvature_max=40 m⁻¹`
- Default stage 2 relaxation factors: `length=1.05`, `coil_coil=1.05`, `curvature=1.05` (5% relaxation on all three — 20% was tried in job 51457615 and traded geometry for sqflx without shrinking axis drift)

### Env var overrides
Drivers read select parameters from environment variables to support Pareto scans and ad-hoc tuning without editing config.yaml. **Every driver writes the resolved effective values back into `cfg` in-memory before calling `register_*`**, so content-addressing sees the real inputs and env var variants produce distinct run ids. The mutation is on the in-memory dict only — `config.yaml` on disk is never rewritten, so concurrent sweep jobs don't need a file lock.
- `BANANA_OUT_DIR` — output directory (default: `$SCRATCH/banana_drivers_outputs/` with `./outputs` fallback)
- `BANANA_IOTA` — stage 1 iota target override (Pareto axis)
- `BANANA_VOLUME` — stage 1 volume target override (Pareto axis)
- `BANANA_ASPECT` — stage 1 aspect target override. Combined with `BANANA_VOLUME`, sets $R_{\text{major}} = (V \cdot A^2 / (2\pi^2))^{1/3}$ without adding a new objective term.
- `BANANA_STAGE2_MODE` — `weighted` (default) or `alm` — select stage 2 solver
- `BANANA_ALM_PRESET` — `throttled` (default) or `unthrottled` — ALM inner-loop philosophy (see "Stage 2 ALM presets" below)
- `BANANA_CURRENT_MODE_S2` — `fixed` (default), `penalized`, or `free` — how stage 2 treats the banana current DOF (see "Stage 2 current handling" below)
- `BANANA_TAU` — stage 2 ALM penalty growth factor override
- `BANANA_MAXITER_LAG` — stage 2 ALM outer-loop iteration cap override
- `BANANA_DOF_SCALE` — stage 2 ALM DOF coordinate rescaling (default 0.1 in throttled, None in unthrottled). Rescales DOF space: $y = x / \text{dof\_scale}$. Smaller values make each inner L-BFGS-B step smaller in physical space, but do NOT bound total displacement per outer iteration. Set to `none` to disable.
- `BANANA_STAGE2_LENGTH_RELAX` — stage 2 length relaxation factor override (default from `stage2_relaxation.length`); effective `length_max = length_max_hw * factor`
- `BANANA_STAGE2_CC_RELAX` — stage 2 coil-coil relaxation factor override (default from `stage2_relaxation.coil_coil`); effective `coil_coil_min = coil_coil_min_hw / factor`
- `BANANA_STAGE2_CURV_RELAX` — stage 2 curvature relaxation factor override (default from `stage2_relaxation.curvature`); effective `curvature_max = curvature_max_hw * factor`. Singlestage always enforces the unrelaxed hardware limit.
- `BANANA_I_FIXED_S2` — stage 2 fixed banana current override in Amperes (default: `current_fixed_stage2` from config.yaml, 16000). Only meaningful when `current_mode_stage2='fixed'`. Used by current Poincaré sweep.
- `BANANA_ORDER` — banana coil Fourier order override (default: `banana_coils.order` from config.yaml, 2). Only affects `utils/init_boozersurface.py` CLI mode. Used by order Poincaré sweep.
- `BANANA_INIT_OUT` — output path for `utils/init_boozersurface.py` CLI mode (default: `$OUT_DIR/<warm_start.stage1_bsurf_filename>`). Used by order sweep to write per-order init JSON.

Stage 2's warm-start bsurf is **not** env-overridable — it is resolved from `warm_start.stage1_id` so the hashed parent always matches the file actually loaded.

### Surface Quadpoints Range
All drivers must use `range="field period"` when creating surfaces for the BoozerSurface. This gives quadpoints_phi in `[0, 1/nfp)` (one full field period). Do NOT use `range="half period"` (gives only half a period — caused a bug where stage 2 JSON contained incorrect surface domains) or `range="full torus"` (unnecessary — SquaredFlux averages over whatever points are given, and stellsym makes one period sufficient).

### Boozer Method
- Use BoozerLS for singlestage baseline (`constraint_weight > 0 → boozer_type='ls'`, set in `boozer.constraint_weight` in config.yaml — currently 1.0e+3). A larger constraint weight anchors BoozerLS's surface-label penalty more firmly against the Boozer residual, reducing drift into the spurious iota=0 basin (see PLAN.md helical amplitude hypothesis section).
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

### Stage 1 seed: utils/vmec_resize.py preprocessing
The original `wout_nfp22ginsburg_000_014417_iota15.nc` was sized such that its **s=0.24** flux surface matched the physical plasma boundary at R0=0.925 m; the outer 76% of the volume was auxiliary (see jhalpern30/simsopt STAGE_2/banana_coil_solver.py lines 25-27, 304 for the reference extraction). It was also sized for a stronger TF field than the real hardware (~0.95 T·m rbtor vs. the 80 kA × 20 TF coil set's 0.32 T·m), so the seed |B| was ~3× too strong for the actual coils.

`utils/vmec_resize.py` is a one-time preprocessing step that:
1. Loads s=0.24 of the original seed and rescales coordinates by `vmec_R / major_radius()`
2. Rescales enclosed toroidal flux by `scale²` (phi is linear in s; coordinate rescale adds the length² factor)
3. Remaps the iota profile to the new domain `s_new = s_orig / 0.24` via a constrained polynomial fit (hard BC `iota(s_new=1) = iota_orig(s=0.24)`)
4. Scales the magnetic axis guess by the same factor
5. Re-solves VMEC at a multi-grid ns ramp (first pass)
6. Rescales phiedge so VMEC rbtor matches the actual TF coil rbtor (`mu_0 * N_tf * I_tf / (2*pi)`), then re-solves VMEC (second pass). Zero-beta equilibrium means |B| is linear in phiedge and iota is independent of it, so this corrects field magnitude without touching surface shape or iota profile.
7. Writes `inputs/wout_stage1_seed.nc` whose **LCFS (s=1) IS the target plasma boundary** and whose rbtor matches the hardware TF coils

Stage 1 then warm-starts from this seed with no boundary rescaling — the warm-start branch of `01_stage1_driver.py` loads the LCFS directly. Downstream drivers (`utils/init_boozersurface.py`, `03_singlestage_driver.py`) extract `plasma_surface.vmec_s = 1.0` (i.e., the LCFS of the stage 1 output wout) and never rescale.

Config keys in `stage1_resize`: `seed_wout_filepath`, `output_filepath`, `inner_s`, `poly_deg`, `mpol`, `ntor`, `ns_array`, `niter_array`, `ftol_array`.

### Stage 1 objective
Stage 1 minimizes: `aspect → aspect_target`, `iota_edge → iota_target`, `volume → volume_target`, and quasisymmetry residuals at `qs_surfaces`. **`iota_axis` is NOT in the objective** — it is left free so the optimizer can preserve whatever magnetic shear the QA solution prefers. The success criterion gates only on `iota_edge` error (< 10% of target). Targets in `config.yaml:stage1` are sized for the resized seed, not the original.

### Stage 2 solver: weighted by default
Stage 2 uses the legacy weighted scalar objective (`stage2_mode: weighted` in
config.yaml). This is the **current working default** because ALM has not
yet been made reliable for the banana-coil constraint landscape — the
augmented Lagrangian implementation is in the driver and in the SIMSOPT
fork, but attempts to use it have not converged to a feasible stationary
point for this geometry. Treat the weighted run as the reference stage 2
until that changes.

Consequences of running in weighted mode:
- The objective is a fixed-weight sum (SquaredFlux + length + coil-coil +
  curvature penalties). L-BFGS-B finds a local minimum of that sum, not a
  constraint-satisfying solution. Expect non-zero final `sqflx` (typically
  $10^{-4}$ for this geometry, not the $10^{-15}$ ALM noise floor) and mild
  hardware-threshold violations (seen in practice: `ccdist` a few percent
  under `0.05 m`, `max_kappa` a few percent over `40 m⁻¹`). These are
  accepted trade-offs of the fixed-weight formulation, not bugs.
- Because the target surface is not actually a flux surface of the final
  coil field, the coil-produced magnetic axis is **not** anchored to the
  target plasma centroid and can drift substantially (tens of percent of
  the minor radius). This axis drift is a structural property of
  weighted-mode stage 2 and is what singlestage has to recover from.

ALM mode is still available via `BANANA_STAGE2_MODE=alm` for experiments.
See PLAN.md for the history and current status of the ALM effort.

### Stage 2 ALM presets (only active when stage2_mode=alm)
Two presets control the ALM inner-loop philosophy (`stage2_alm.preset` in
config.yaml, `BANANA_ALM_PRESET` env var). These are only consulted when
`stage2_mode=alm`; they have no effect in the default weighted mode.
- **`throttled` (default)** — inner loop is step-limited via `dof_scale` (0.1) and `maxfun` cap (100), gentle penalty growth (`tau=2`), many outer iterations (`maxiter_lag=80`). Designed for stiff constraint landscapes where banana coils are near hardware limits. The inner L-BFGS-B under-converges at each penalty level; the outer loop compensates via $\mu/\lambda$ updates.
- **`unthrottled`** — inner loop runs to full convergence (no `dof_scale`, no `maxfun` cap), aggressive penalty growth (`tau=10`), fewer outer iterations (`maxiter_lag=50`). Matches PedroGil's simsopt_alm_temp examples. Better when constraints are not near cliffs. This is how SIMSOPT's ALM was designed to be used.

Individual config keys and env vars override preset defaults. Resolution order: env var > config.yaml key > preset default.

### Singlestage solver: BoozerLS L-BFGS-B
- Uses BoozerLS (`constraint_weight > 0 → boozer_type='ls'`) with `constraint_weight=1.0e+3` from `config.yaml`.
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
