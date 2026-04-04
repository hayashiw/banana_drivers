# Banana Drivers

Banana coil optimization drivers for a stellarator-tokamak hybrid device using SIMSOPT. The overall goal is to produce a realizable set of banana coils capable of (a) pure stellarator scenarios and (b) finite-current scenarios. These coil designs are intended to be manufactured.

**This is a separate project from qi_drivers.** General practices (workflow pattern, formatting, documentation) are shared, but the physics problem, coil geometry, and optimization landscape are different. Do not assume solutions from qi_drivers transfer directly.

## Workflow Pipeline

The optimization runs in stages, each warm-starting from the previous:

1. **Stage 1** — VMEC equilibrium solve: perturbed optimization of existing wout for target boundary (not yet implemented)
2. **booz_xform** — Convert VMEC wout to Boozer-coordinate surface for BoozerSurface initialization
3. **Stage 2** — Coil-only optimization (SquaredFlux + geometric penalties via L-BFGS-B)
4. **Single-stage** — Joint coil + surface optimization (BoozerLS + penalties via L-BFGS-B)
5. **Pareto scan** — Sweep over banana current, volume, iota targets (not yet implemented)
6. **Finite-current** — Proxy coil + VF coils for finite plasma current (not yet implemented)

Each stage must converge before proceeding to the next. Always verify stage 2 feasibility before running single-stage.

## File Structure

```
banana_drivers/
  CLAUDE.md                          # This file — Claude instructions
  PLAN.md                            # Current status and planned work
  README.md                          # Technical reference
  banana_drivers_comparison.md       # Source vs new driver comparison
  submit.sh                          # Unified SLURM launcher (./submit.sh 01)
  run_driver.sh                      # Generic SLURM batch script (called by submit.sh)
  01_stage2_driver.py                # Stage 2: coil-only optimization
  02_singlestage_driver.py           # Single-stage: joint coil + surface optimization
  boozxform_driver.py                # Booz_xform surface extraction (candidate pipeline step)
  vmec_resize_driver.py              # VMEC resize utility (candidate pipeline step or superseded)
  inputs/                            # Input files (wout, coil data)
  utils/
    post_process.py                  # Metrics extraction and CSV comparison
    generate_vf_coils.py             # VF coil generation for finite-current
  local/                             # Legacy files, on-hold drivers, master prompt
    prompt.md                        # Master prompt and requirements
```

## Key Design Decisions

### Hardware Constraints
- TF coil current: 80 kA (fixed)
- Maximum banana coil current: 16 kA (upper bound, not a target)
- TF coils: 20 coils, R0=0.976 m, R1=0.4 m, order=1
- Banana coils: nfp=5, stellsym, wound on winding surface R0=0.976 m, a=0.215 m
- Banana coil order: 2 (order=4 produces bad coils — distinct from curvature p-norm)
- Target plasma: R0=0.925 m, edge iota ~ 0.12, nfp=5, stellsym

### Boozer Method
- Use BoozerLS for the baseline (`constraint_weight` > 0 → `boozer_type='ls'`)
- Driver was already BoozerLS (constraint_weight=1.0); increased to 100.0 to match SIMSOPT examples
- BoozerExact deferred — revisit for stochastic optimization or tighter residuals

### Objective Function
- `SurfaceSurfaceDistance` excluded from objective (complicates DOF shifting, minimal benefit) — measure in post-processing only
- `BANANA_CURV_P = 4` (L4 norm) is intentional — produces better banana coils than L2
- Banana coil current fixed per run, swept in Pareto scan (not optimized in objective)

### Convergence
- Target resolution: mpol = ntor = 12 (start at 6-8, ramp via Fourier continuation)
- Require both ftol AND gtol convergence
- Boozer residual < 1e-4 (minimum acceptable)
- Get initial solve working at single resolution before adding Fourier ramp

### ALM (Augmented Lagrangian Method)
- Design drivers to accommodate ALM outer-loop structure, but do not activate
- Wait for validation in qi_drivers, then port
- Use fixed constraint_weight for now

### Three-Stage Workflow
- Stage 1 treated as perturbed optimization of existing wout (no original init data available)
- Stage 1 produces VMEC equilibrium at correct boundary directly — vmec_resize unnecessary
- Full pipeline: Stage 1 → booz_xform → Stage 2 → Single-stage

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
- Single-surface Boozer optimization is inherently serial — do not waste CPUs
- Poincare tracing parallelizes over fieldlines

### Documentation
- Document what works AND what doesn't
- Update CLAUDE.md, PLAN.md, and memory as decisions are made
- The more complicated the code, the more documentation it needs
- Comments, docstrings, and typehints for complex sections

## Environment

- **HPC**: Perlmutter @ NERSC (128 CPUs per node, SLURM scheduler)
- **Conda env**: `sims_banana_env` (NOT `sims_prox_env` — that is for qi_rso)
- **SIMSOPT fork**: `hayashiw/simsopt` on `whjh/auglag_banana` branch, located at `hybrid_torus/banana/simsopt/`
- **SIMSOPT for banana**: Always reference `hybrid_torus/banana/simsopt/` for banana-specific SIMSOPT issues (e.g., `CurveCWSFourierCPP`). Do NOT check `qi_rso/simsopt/` unless explicitly asked — it is a separate fork for the QI project.
- **Related repo**: `qi_rso/qi_drivers/` (separate project; reference for formatting/workflow patterns only. ALM implementation lives there and would need porting.)
