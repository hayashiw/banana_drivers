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
./submit.sh 01                       # stage 2 (auto: debug → regular fallback)
./submit.sh 02 regular               # single-stage (regular queue only)
./submit.sh 02_singlestage debug     # single-stage (debug queue only)
```

`run_driver.sh` is the generic SLURM batch script called by `submit.sh`. It finds `${DRIVER}_driver.py`, runs it, and moves logs to the output directory printed by the driver's `atexit` handler (`OUT_DIR=...`).

## Scripts

### Pipeline (run in order)

| # | Script | Purpose | Inputs | Outputs |
|---|--------|---------|--------|---------|
| 01 | `01_stage2_driver.py` | Stage 2: fixed plasma boundary, optimize banana coil shape (SquaredFlux + geometric penalties) | wout, TF+banana coil params | `outputs/stage2_biotsavart_opt.json` |
| 02 | `02_singlestage_driver.py` | Single-stage: jointly optimize coil shapes and plasma boundary (BoozerLS + penalties) | stage2 BiotSavart, surface init | `outputs/singlestage_*` |

### Candidate Pipeline Steps (unnumbered)

| Script | Purpose | Status |
|--------|---------|--------|
| `boozxform_driver.py` | Run Booz_xform to extract Boozer-coordinate surface + equilibrium iota/G | Will be numbered once pipeline position is confirmed |
| `vmec_resize_driver.py` | Resize VMEC LCFS to match target plasma dimensions | May be superseded by stage 1 driver |

### Utilities (`utils/`)

| Script | Purpose |
|--------|---------|
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

## Three-Stage Workflow (Planned)

```
Stage 1 (VMEC at correct boundary)  -->  booz_xform  -->  Stage 2  -->  Single-stage
```

- Stage 1: perturbed optimization of existing wout for varying volume/iota targets
- vmec_resize becomes unnecessary once stage 1 is operational
- Pareto front scans over banana current, volume, and iota targets

---

## Output Directories

| Directory | Contents |
|-----------|---------|
| `outputs/` | All pipeline outputs: `stage2_*`, `singlestage_*` files (BiotSavart JSON, VTK, diagnostics) |

---

## Environment

- **HPC**: Perlmutter @ NERSC (128 CPUs per node, SLURM scheduler)
- **SIMSOPT fork**: `jhalpern30/simsopt` on `accessibility` branch
- **Related repo**: `qi_rso/qi_drivers/` (separate project, shared practices only)
