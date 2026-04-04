# Banana Drivers

Driver and utility scripts for optimization of **banana coils** in a stellarator–tokamak hybrid device using SIMSOPT.

See `../CLAUDE.md` for full technical background and `../PLAN.md` for current status and planned work.

---

## Scripts

### Pipeline (run in order)

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `vmec_resize_driver.py` | Resize VMEC LCFS to match target plasma dimensions so `s=1` loads the correct surface without ad-hoc rescaling | `wout_nfp22ginsburg_000_014417_iota15.nc` | `outputs_vmec_resize/wout_nfp05iota012_000_000000.nc` |
| `boozxform_driver.py` | Run Booz_xform on the resized wout to extract a surface already in Boozer coordinates, plus the VMEC equilibrium iota and G | resized wout | `outputs_boozxform/booz_gamma_s100.npz` |
| `stage2_driver.py` | Stage-2 coil optimization: fixed plasma boundary, optimize banana coil shape to minimize SquaredFlux | resized wout, TF+banana coil parameters | `outputs_stage2/stage2_biotsavart_opt.json`, VTK files |
| `singlestage_driver.py` | Single-stage optimization: simultaneously optimize coil shapes and plasma boundary for quasi-symmetry (NonQuasiSymmetricRatio) | `stage2_biotsavart_opt.json`, `booz_gamma_s100.npz` | `outputs_singlestage/` |

### Utilities

| Script | Purpose |
|--------|---------|
| `generate_vf_coils.py` | Generate vertical-field coil geometry |
| `post_process.py` | Post-processing and diagnostics on optimization outputs |
| `singlestage_finitecurrent_driver.py` | Single-stage with finite plasma current proxy (experimental) |

---

## Key Parameters

### Physical
- **TF coils**: 20 coils, 80 kA each, `R0=0.976 m`, `R1=0.4 m`, `order=1`
- **Banana coils**: `nfp=5`, stellsym, wound on winding surface `R0=0.976 m`, `a=0.215 m`, 16 kA each
- **Target plasma**: `R0=0.925 m`, edge iota ≈ 0.12, `nfp=5`, stellsym

### Solver
- Surface resolution: `MPOL=NTOR=6` (singlestage), fits up to `MPOL=18` planned
- BoozerSurface: `CONSTRAINT_WEIGHT=None` → BoozerExact Newton
- Optimizer: L-BFGS-B (`maxcor=300`, `FTOL=1e-15`, `GTOL=1e-3`)

---

## Known Issues (see PLAN.md for fixes)

1. **IOTA_TARGET is wrong** (BLOCKING): `singlestage_driver.py` uses the Booz_xform iota
   (0.147, VMEC equilibrium) as `IOTA_TARGET`, but the stage2 coils actually produce iota ≈ 0.024
   in vacuum. The iota penalty dominates the objective and causes BoozerExact to fail on every
   L-BFGS-B step. Fix: add a stage2 iota probe to `stage2_driver.py` and load the result in
   `singlestage_driver.py`.

2. **BoozerExact fragility** (structural): Unconstrained L-BFGS-B + BoozerExact Newton is
   fragile — any non-trivial step can break Newton convergence. Fix: port the Augmented
   Lagrangian Method (ALM) from `proxima_qi` with per-DOF scaling for surface DOFs.

3. **G_init mismatch**: G from Booz_xform ≈ 0.95 T·m reflects VMEC equilibrium field strength;
   our 80kA TF coils give G ≈ 0.32 T·m. Newton adjusts G during the solve but the 3× gap may
   slow convergence. Fix: use G from the stage2 iota probe instead.

---

## Output Directories

| Directory | Contents |
|-----------|---------|
| `outputs_vmec_resize/` | Resized VMEC wout file |
| `outputs_boozxform/` | Booz_xform `.npz` files with Boozer-coordinate surface data |
| `outputs_stage2/` | Stage-2 optimized BiotSavart JSON, VTK coil/surface snapshots |
| `outputs_singlestage/` | Single-stage optimization outputs, VTK snapshots |
