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
  jhalpern30/                        # In-flight scans, scripts, baselines (tracked top-level while the I>0 /
                                     # finite-current work is the active line of attack — will move under local/ once settled)
                                     #   stage2.py, singlestage.py                — current working drivers (80 kA TF,
                                     #                                                CurveSelfIntersect, poloidal_extent,
                                     #                                                ellipse_width active; current_kA CLI
                                     #                                                arg is a bounded float in [-16, 25] kA,
                                     #                                                not a discrete grid, so Pareto scans
                                     #                                                pass log-uniform samples directly)
                                     #   scan_plasma_curr/                         — 11-point plasma-current scan at VF=0
                                     #                                                (I ∈ {0, ±0.2, ±0.9, ±1, ±2, ±5} kA).
                                     #                                                End-to-end success for I≤0;
                                     #                                                I=+1.0 kA flip stage 2 done →
                                     #                                                scan_plasma_curr/I1.0kA_flip/;
                                     #                                                singlestage flip pending.
                                     #   pareto_negative_current.py,
                                     #   run_pareto_negative_current.sh            — 128-point Pareto scan over
                                     #                                                (|I|<5 kA, iota_target∈[0.08, 0.25]),
                                     #                                                I<0 in-basin, truncated Fourier ramp
                                     #                                                (BANANA_RAMP_STAGES=2 → mpol=[6,8]).
                                     #                                                ProcessPoolExecutor fan-out on a
                                     #                                                single 128-CPU node; fcntl-locked
                                     #                                                scan_index.csv; submitted as job
                                     #                                                51859030.
                                     #   poloidal_sweep.py,
                                     #   run_poloidal_sweep.sh                     — I=0 / weighted-only poloidal-extent
                                     #                                                demonstration sweep (order=2, no ramp;
                                     #                                                theta_target × weight × n-points).
                                     #   resolution_scan.py,
                                     #   run_resolution_scan.sh                    — 2×2 grid (order ∈ {4,5} × qp ∈ {256,320},
                                     #                                                mpol=ntor=10) off one shared stage 2
                                     #                                                warm start.
                                     #   iota15_rithik/, iota20_rithik/            — warm-start sources (Rithik-provided
                                     #                                                coil sets + boozersurface JSONs at
                                     #                                                iota=0.15 and 0.20)
                                     #   alm/                                      — ALM-stage2 / ALM-singlestage experiments
                                     #   post_process.py, analyze.ipynb,
                                     #   eval_curr_dirs.ipynb                      — scan analysis + Poincaré gating
                                     #   old/                                      — archived scans (scan_vf_plasma_curr/
                                     #                                                4×5 VF×I grid, legacy TF=100 kA runs,
                                     #                                                SCAN_STATUS.md)
  new_objectives/                    # CWS-frame manufacturability objectives (wired into jhalpern30/{stage2,singlestage}.py)
                                     #   cwsobjectives.py                          — shared CWS base class
                                     #   poloidal_extent.py                        — poloidal footprint cap (replaces
                                     #                                                length as the real poloidal bound)
                                     #   ellipse_width.py                          — ellipse-width (coil-perpendicular)
                                     #                                                manufacturability penalty
                                     #   self_intersect.py                         — CurveSelfIntersect; enables order=4
                                     # Still NOT integrated into the numbered 01/02/03 pipeline drivers — they run only
                                     # under jhalpern30/. Integration into 02_stage2_driver.py + 03_singlestage_driver.py
                                     # is a standing TODO.
  markdowns/                         # Tracked markdown for prompt docs, iota basin analysis, cold-start prompt, pyQSC walkthrough
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
- Banana coils: nfp=5, stellsym, wound on winding surface R0=0.976 m, a=0.210 m, `order=4`. Order=4 was previously unusable because the coils self-intersected during optimization; the CWS-frame `CurveSelfIntersect` objective in `new_objectives/` now prevents that failure mode, so order=4 is the preferred default and order=2 is only kept as a warm-start starting point for the singlestage Fourier-order ramp.
- Target plasma: R0=0.92 m (baseline LCFS), iota target 0.15, nfp=5, stellsym
- Hardware thresholds (HW-team verified, updated 2026-04-23): `length_max=1.9 m`, `poloidal_extent_half_deg_max=45°` (the real poloidal-footprint bound), `coil_coil_min=0.05 m`, `coil_surface_min=0.015 m`, `curvature_max=100 m⁻¹` (selected conductor has ~1 cm minimum bending radius). Earlier thresholds (`length_max=1.75` / `1.8` / `2.0`, `poloidal_extent=50°`, `coil_surface_min=0.02`, `curvature_max=40`) are obsolete — update any outstanding notes that cite them.
- Default stage 2 relaxation factors: `length=1.05`, `coil_coil=1.05`, `curvature=1.05` (5% relaxation on all three — 20% was tried in job 51457615 and traded geometry for sqflx without shrinking axis drift). The factors are still meaningful under the new hardware limits because stage 2 needs a touch of slack to let L-BFGS-B drive `sqflx` lower before singlestage polishes.

### Env var overrides
Drivers read select parameters from environment variables to support Pareto scans and ad-hoc tuning without editing config.yaml. **Every driver writes the resolved effective values back into `cfg` in-memory before calling `register_*`**, so content-addressing sees the real inputs and env var variants produce distinct run ids. The mutation is on the in-memory dict only — `config.yaml` on disk is never rewritten, so concurrent sweep jobs don't need a file lock.
- `BANANA_OUT_DIR` — output directory (default: `$SCRATCH/banana_drivers_outputs/` with `./outputs` fallback)
- `BANANA_IOTA` — stage 1 iota target override (Pareto axis)
- `BANANA_VOLUME` — stage 1 volume target override (Pareto axis)
- `BANANA_ASPECT` — stage 1 aspect target override. Combined with `BANANA_VOLUME`, sets $R_{\text{major}} = (V \cdot A^2 / (2\pi^2))^{1/3}$ without adding a new objective term.
- `BANANA_STAGE2_MODE` — `weighted` (default) or `alm` — select stage 2 solver
- `BANANA_ALM_PRESET` — `throttled` (default) or `unthrottled` — ALM inner-loop philosophy (see "Stage 2 ALM presets" below)
- `BANANA_CURRENT_MODE_S2` — (numbered pipeline `02_stage2_driver.py` only; `jhalpern30/stage2.py` uses the presence of `BANANA_I_FIXED_S2` as the toggle instead) `fixed` (default), `penalized`, or `free` — how stage 2 treats the banana current DOF (see "Stage 2 current handling" below)
- `BANANA_TAU` — stage 2 ALM penalty growth factor override
- `BANANA_MAXITER_LAG` — stage 2 ALM outer-loop iteration cap override
- `BANANA_DOF_SCALE` — stage 2 ALM DOF coordinate rescaling (default 0.1 in throttled, None in unthrottled). Rescales DOF space: $y = x / \text{dof\_scale}$. Smaller values make each inner L-BFGS-B step smaller in physical space, but do NOT bound total displacement per outer iteration. Set to `none` to disable.
- `BANANA_STAGE2_LENGTH_RELAX` — stage 2 length relaxation factor override (default from `stage2_relaxation.length`); effective `length_max = length_max_hw * factor`
- `BANANA_STAGE2_CC_RELAX` — stage 2 coil-coil relaxation factor override (default from `stage2_relaxation.coil_coil`); effective `coil_coil_min = coil_coil_min_hw / factor`
- `BANANA_STAGE2_CURV_RELAX` — stage 2 curvature relaxation factor override (default from `stage2_relaxation.curvature`); effective `curvature_max = curvature_max_hw * factor`. Singlestage always enforces the unrelaxed hardware limit.
- `BANANA_I_FIXED_S2` — stage 2 fixed banana current override. Two different conventions depending on driver:
  - **`jhalpern30/stage2.py`** (active): value in **kA**. **Default behaviour is unset → banana current is a free DOF with ±10 kA initial.** If the env var is set (including `0`), the banana current is pinned at that value (scaled by `BANANA_CURRENT_SIGN`) and the underlying `Current` DOF is `fix_all()`'d so stage 2 becomes shape-only. Presence of the env var is the toggle; there is no companion `BANANA_CURRENT_MODE_S2` in this driver. Required under TF<0 with `BANANA_I_FIXED_S2=16` (kA) to avoid the low-$|I_b|$ pinched-shape failure mode.
  - **`02_stage2_driver.py`** (numbered pipeline, not in active use): value in **Amperes**; only meaningful when `BANANA_CURRENT_MODE_S2='fixed'`; default magnitude from `config.yaml:current_fixed_stage2` (16000 A).
- `BANANA_ORDER` — banana coil Fourier order override (default: `banana_coils.order` from config.yaml, 2). Only affects `utils/init_boozersurface.py` CLI mode. Used by order Poincaré sweep.
- `BANANA_INIT_OUT` — output path for `utils/init_boozersurface.py` CLI mode (default: `$OUT_DIR/<warm_start.stage1_bsurf_filename>`). Used by order sweep to write per-order init JSON.
- `BANANA_IOTA_TARGET` — singlestage iota target override (default: 0.15). Used by `jhalpern30/singlestage.py` only. Multiplied by `IOTA_TARGET_SIGN` (auto-set to −1 when the output parent dir ends in `_flip`) so the flip branch targets iota=−0.15 without a second env var. Consumed by `pareto_negative_current.py` to sweep the iota axis.
- `BANANA_RAMP_STAGES` — truncate the Fourier continuation ramp to the first N stages (default: 4 = full ramp `[(6,3,192), (8,4,256), (10,5,320), (12,5,320)]`). Used by `jhalpern30/singlestage.py` only. `pareto_negative_current.py` sets N=2 to cap per-point wall time at the mpol=6→8 / order=3→4 / qp=192→256 sub-ramp.
- `BANANA_RAMP_CUSTOM` — JSON list of `[mpol, ntor, order, qp]` stages that fully overrides the ramp (takes precedence over `BANANA_RAMP` / `BANANA_RAMP_STAGES`). Used by `jhalpern30/resolution_scan.py` to land the final stage at a scan cell that doesn't match a prefix of the hardcoded ramp while still pre-warming BoozerLS via earlier stages.
- `BANANA_RESUME_STAGE` — resume a ramped singlestage run from `$OUT_DIR/stage{N:02d}/bsurf_opt.json` + `state.json`. Skips stages `0..N` and re-enters the loop at `N+1`, using the loaded biotsavart + surface + iota + G. Requires the same ramp configuration (`BANANA_RAMP` / `BANANA_RAMP_STAGES` / `BANANA_RAMP_CUSTOM`) as the original run. Each successful stage now writes a sidecar `state.json` with `{iota, G, volume, iota_target, stage_idx, stage_mpol, stage_ntor, stage_order, stage_qp}` for resume + per-stage trajectory inspection.

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

### TF sign convention (TF<0 operating setup)
TF coil current in [stage2.py:333](jhalpern30/stage2.py#L333) is **−80 kA** (flipped from the original +80 kA on 2026-04-22). This is the hardware-intended operating polarity; positive TF was a historical choice that has been retired.

**Sign relationships under TF<0 with the canonical tilted `+slope` init** (reference = the first banana coil, `banana_coils[0]` / log line `[Coil 20]`; the rest alternate sign via stellsym):
- **Reference banana coil current must be NEGATIVE** for the $\iota=+0.15$ basin. Empirical: free-current stage 2 at TF<0 + tilted init + I=0.0 (51942805) converged to $I_b(\text{coil 20}) = -25.94$ kA. Under the prior TF>0 convention the reference banana coil was positive for the same iota basin — so the sign flips with TF. Pin via `BANANA_I_FIXED_S2=-16` (NOT `+16`). The `_flip` wrapper workaround exists for legacy TF>0 cases (I>0 basin flip); it is not the mechanism used for the TF<0 pivot.
- **`G0` (BoozerLS enclosed poloidal current argument)** must also flip sign under TF<0. Fixed at [singlestage.py:422-423](jhalpern30/singlestage.py#L422-L423) — current form multiplies the `abs()` by `-1` to track the TF<0 assumption explicitly. If TF is ever set back to +80 kA, remove that minus sign. A wrong-sign `G0` is the most likely root cause of the wrong-basin iota≈0 failures seen in pre-fix TF<0 singlestage runs (e.g. 51941436): BoozerLS can't find a consistent surface with (iota_init, G_wrong_sign) and defaults to the degenerate solution.
- **`iota` (BoozerLS initial guess) stays positive.** $(\text{TF}<0, I_b<0)$ and $(\text{TF}>0, I_b>0)$ are mirror field configurations; both have the same iota sign in Boozer coordinates because both poloidal and toroidal senses flip together. `iota_target = +0.15` and `IOTA_TARGET_SIGN = +1` are correct.

Empirical findings from the TF<0 pivot (2026-04-22 → 2026-04-23):
- **Coil tilt is TF-sign invariant.** A simple (untilted) ellipse init under TF<0 converges to a $+$slope tilted thin-D shape (confirmed by the I=0.0 `I0.0kA_tfneg_simpleellipse` stage 2 run 51899830). The canonical `+slope` tilted init works for both TF polarities. Caveat: the simpleellipse run landed in the OPPOSITE banana-sign basin ($I_b = +21.89$ kA) from the tilted-init run ($I_b = -25.94$ kA) — the tilt biases which helicity basin stage 2 falls into, so don't use simpleellipse if you want the iota=+0.15 basin under TF<0.
- **Init width/curvature tuning.** The original tilted init (`a_s=0.022, a_l=0.140`) had `kappa_init ≈ 185 m⁻¹` (nearly 2× the 100 m⁻¹ cap), which forced stage 2 into pinched local minima. Tuned values in [plot_init_coil.py](jhalpern30/plot_init_coil.py) are now `a_s=0.025, a_l=0.090` with all four hardware constraints (κ, cc, W, poloidal extent) satisfied at iteration 0.
- **Stage 2 current must be pinned.** See "Stage 2 current handling" below.

### Stage 2 current handling

**`jhalpern30/stage2.py` (active driver)** — banana current is a **free DOF by default**. To pin it (required under TF<0 for production sweeps), set `BANANA_I_FIXED_S2=<kA>` when invoking the driver. Presence of the env var is the toggle; there is no mode selector. Implementation: [stage2.py:393-409](jhalpern30/stage2.py#L393-L409) builds a `ScaledCurrent(Current(1.0), sign*I_A)` and calls `fix_all()` on the underlying `Current` so the scalar current DOF is removed from `JF.x`. Stage 2 then becomes a shape-only optimization; singlestage handles joint current+shape refinement. Under TF<0, running with the current unset (free) exposes a low-$|I_p|$ pinched-shape failure mode: the optimizer lets $|I_b|$ collapse and finds a crumpled local minimum (e.g. 51934327 I=−1.0 settled at $I_b \approx 5$ kA in a free run). Pinning $|I_b|$ at 16 kA (hardware cap) forces the optimizer to find a shape that works with a hardware-max helical field. **Sign matters when pinning** (unlike the free case, the optimizer can no longer choose the sign): production 7-point sweeps under TF<0 with the canonical tilted `+slope` init must pass **`BANANA_I_FIXED_S2=-16`** (negative), because the reference banana coil needs to be negative for the $\iota=+0.15$ basin. Under the prior TF>0 convention it was +16. See "TF sign convention" above for the sign relationships.

**`02_stage2_driver.py` (numbered pipeline, not actively used)** — three modes selectable via `current_mode_stage2` in config.yaml or `BANANA_CURRENT_MODE_S2` env var:
- **`fixed` (default)** — banana current pinned at `current_fixed_stage2` (default 16 kA, hardware upper bound), `fix_all()`'d, shape-only optimisation.
- **`penalized`** — current free with `QuadraticPenalty(|I|, current_soft_max_stage2, "max")` in the ALM constraint list. Known to collapse to $I=0$ from the post-stage-1 warm start because normalized `SquaredFlux` is lower with TF alone than with TF plus an unconverged banana shape; once zeroed, all shape gradients vanish (they scale as $I$). Preserved for experiments. See PLAN.md for the 51246996 failure analysis.
- **`free`** — current free, no constraint. Collapses to low $|I_b|$ + pinched shape under TF<0. Kept for regression tests; avoid for production runs.

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
  hardware-threshold violations of a few percent on whichever constraint
  is active at the chosen weights (at `stage2_relaxation=1.05` and the
  HW-verified limits `ccdist ≥ 0.05 m` / `max_kappa ≤ 100 m⁻¹`,
  the usual violation is `ccdist` grazing a few percent under). These are
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
- `submit.sh` handles debug/shared/auto mode selection and per-driver SLURM settings
- `run_driver.sh` is the generic batch script — driver name passed via `DRIVER` env var
- Drivers print `OUT_DIR=...` via atexit for log capture by `run_driver.sh`
- Stage 1 (VMEC fixed-boundary optimization) is MPI-parallel — uses `least_squares_mpi_solve` with finite-difference gradients, so ndofs+1 ranks is the useful upper bound. `submit.sh` sets `NTASKS=16` by default for stage 1.
- Stages 2 and 3 (single-surface BoozerLS / coil-only optimization) are inherently serial — do not waste CPUs (`NTASKS=1`)
- Poincare tracing parallelizes over fieldlines (one MPI rank per field line)
- Poincare: `./submit.sh poincare <input.json> [debug|shared|auto] [extra args...]`
- `run_poincare.sh` is the MPI batch script — receives `POINCARE_INPUT`, `POINCARE_LABEL`, `POINCARE_ARGS` via env
- Poincare overlay and starting points use the raw loaded surface directly — no BoozerSurface solve (which distorts the surface when volume/iota don't match the actual coil field geometry)

### Documentation
- Document what works AND what doesn't
- Update CLAUDE.md, PLAN.md, and memory as decisions are made
- The more complicated the code, the more documentation it needs
- Comments, docstrings, and typehints for complex sections

## Handoff Notes (2026-04-29)

**Stage 2 default banana init sign FLIPPED to negative (sign fix).** [stage2.py:413](jhalpern30/stage2.py#L413) now reads `ScaledCurrent(Current(1), BANANA_CURRENT_SIGN * -1e4)` (was `* 1e4`), so the default unfixed init is `−10 kA` instead of `+10 kA`. This matches the TF<0 operating convention — under TF<0 the reference banana coil needs to be negative for the iota=+0.15 basin. Pre-fix, the optimizer was sometimes managing to cross zero current to reach the correct basin (51942805) and sometimes not (51999527, +27.65 kA wrong basin). Post-fix, with init sign matching the desired basin, the free optimizer converges deterministically negative. Verified by 52217433 (I=0.0, $I_b = -18.06$ kA) and 52218293 (I=0.2, $I_b = -17.98$ kA). `BANANA_I_FIXED_S2` semantics unchanged — user passes signed kA. `--flip-banana` (legacy) now flips back to +10 kA init for diagnostic.

**Mpol=12 singlestages converged for I=1.0 and I=2.0**: 52173491 (21h45m, full ramp completion via stage 2 resume), 52173493 (15h21m). Both have `bsurf_opt.json` at top-level OUT_DIR copied from `stage03/` with iota=0.150 exact. Memory cap had to be bumped to 128 GB (vs initial 64 GB which OOM-killed earlier attempts at ~9h).

**I=2.0 ss3 Poincaré is genuinely slow** — fails 30 min debug AND 1 h regular. Mpol=12 field has structure that takes >1h to fully trace at NLINES=48 / new wide seed range. Workarounds: longer wall (2-4h regular), or fewer NLINES, or narrower seed range. I=1.0 ss3 traces in 1m44s (52217517) so the issue is specific to I=2.0's field topology.

**I=0.0 (pure stellarator) — UNSOLVED.** Three approaches tried, all fail singlestage:
- Pinned $I_b=-16$ kA in stage 2 → singlestage stages 0+1 ABNORMAL (original sweep)
- Free current with old +10 kA init → wrong banana sign (+27.65 kA), singlestage garbage (52139101)
- Free current with new −10 kA init → correct sign ($I_b = -18$ kA, 52217433) but singlestage still went wrong-basin (cancelled 52218210)

Pattern: at $I_p = 0$, banana coils are the only iota source. Stage 2's "minimize sqflx only" objective doesn't pick coil shapes that also support iota=+0.15 at the target surface. With small plasma current (I=0.2 currently converging at iota=0.149) the plasma contribution helps anchor the basin; at I=0.0 we lose that help and BoozerLS lands at iota≈0. **User's call on next direction** (no decision yet); options under discussion include homotopy from a converged neighbor, init shape retune, stage-2 iota-soft objective, iota continuation in singlestage ramp, or accepting I=0.1 as the "near-stellarator" baseline.

**I=0.2 sign-fixed singlestage in flight (52218672)**: 1h in, stage 0 converged at iota=0.1490 with grad=4.5e-11. On track to complete the 4-stage ramp in 24h. This will be our cleanest near-stellarator baseline if I=0.0 stays unsolved.

**Final design analysis is user-led** (memory: `feedback_user_does_design_review.md`) — don't author candidate-design summaries proactively.

---

## Handoff Notes (2026-04-24 afternoon)

**Resume-singlestage push to mpol=12 timed out across the board.** Three resume jobs (51998499 I=0.2 from stage01; 51998508 I=1.0 from stage02; 51998510 I=2.0 from stage02) all hit the 16 h shared-QoS wall before completing their target stage. Best converged state per I_p remains the highest-numbered `stage_NN/bsurf_opt.json` written: I=0.2 → stage01 (mpol=8); I=1.0 and I=2.0 → stage02 (mpol=10). No I_p has a final-stage (mpol=12) bsurf yet.

**24 h resubmit pattern adopted** (replaces 16 h `-p` runs). Direct `sbatch --qos=shared --time=24:00:00 --ntasks=1 --cpus-per-task=32 --mem=64G` with `POST_PROCESS=false` so singlestage runs at 32 OMP threads (3-5× faster than the legacy 16×1 MPI layout that stayed single-threaded at the singlestage step). Poincaré is now run as a separate post job. In-flight: 52139101 (I=0.0 fresh from new free-current stage 2), 52139102 (I=1.0 resume stage 2), 52139104 (I=2.0 resume stage 2).

**I_p = 0 pinning diagnostic — stage 2 done, basin instability flagged.** Free-current TF<0 stage 2 at I=0 (51999527, 16m29s) converged with $|I_b|$ = 27.65 kA. Same-configuration earlier run (51942805) converged to $|I_b|$ = 25.94 kA but with **opposite sign on the reference banana coil** (coil 20: 51942805 negative, 51999527 positive). Same TF<0, same tilted init, same iota-target, same plasma current. The free-current basin selection is sensitive to small numerical state — should be revisited after singlestage 52139101 reports to see whether sign matters for iota=+0.15 reachability.

**Poincaré tracer for singlestage-output bsurfs — interp-error plateau.** Bumping `post_process.py` grid 50³ → 70³ did NOT reduce max interp error on the singlestage mpol=10 outputs (1.7e-4 / 4e-5 essentially unchanged). Suggests a localized field feature (likely banana coil close-approach to the plasma surface at certain phi) that's sharper than the cubic spline can resolve at any reasonable grid density. Three diagnostic jobs in flight (52139124 DEGREE=4, 52139125 grid=120³, 52139127 TOL=1e-6) — each writes a `diag_<tag>_summary.json` next to the input bsurf indicating whether that variant fixes the segfault.

**New temporary diagnostic file: [diag_poincare.py](jhalpern30/diag_poincare.py)**. Loads a bsurf, runs `compute_fieldlines` with knobs for grid / DEGREE / TOL / TMAX / NLINES / seed range / no-interp. **Slated for archive/deletion** once the chosen fix lands in `post_process.py` — header notes this. Don't expand it into production tooling; it's a flag-grid harness, not a long-lived module.

**Final design review explicitly user-driven.** New feedback memory (`feedback_user_does_design_review.md`): I do not author candidate-design summaries / design-review docs / final-write-up materials proactively. Status updates and analysis assistance on request only.

---

## Handoff Notes (2026-04-24 morning)

**7-point TF<0 sweep outcomes — sign convention confirmed, one surprise.** After 14–16 h singlestage walls overnight, the sweep produced a clean split by $I_p$ sign (all runs at TF=−80 kA, tilted init, `BANANA_I_FIXED_S2=-16` kA):

| $I_p$ [kA] | Singlestage outcome | Final iota |
|---|---|---|
| −2.0 | FAILED 14h17m — all 4 ramp stages `ABNORMAL:` | — |
| −1.0 | TIMEOUT 16h — stage 0 converged but BoozerLS in wrong basin | ≈ −0.024 |
| −0.2 | TIMEOUT 16h — stages 0, 1 `ABNORMAL:`, stuck in stage 2 | — |
|  0.0 | TIMEOUT 16h — stages 0, 1 `ABNORMAL:`, stuck in stage 2 | — |
|  0.2 | TIMEOUT 16h — converging iota≈0.149, κ grazing 100, did not finish ramp | ≈ 0.149 |
|  1.0 | TIMEOUT 16h — stages 0, 1, 2 `CONVERGENCE`, stage 3 in progress at wall | 0.150 |
|  2.0 | TIMEOUT 16h — same as I=1.0 | 0.150 |

**Sign-convention answer**: under TF<0 + pinned $I_b=-16$ kA, **$I_p > 0$ is tokamak-assisted** (plasma poloidal flux co-rotates with banana, converges cleanly to iota=+0.15) and **$I_p < 0$ is hybrid-Shafranov** (not reachable from iota_init=+0.15 — either wrong-basin attractor or `ABNORMAL:`). Swapped from the TF>0 convention as expected. I=1.0 and I=2.0 have usable `stage02/bsurf_opt.json` at (mpol=10, order=4, qp=256) for downstream analysis; stage 3 (mpol=12) didn't complete in the 16 h wall.

**New open question — $I_p = 0$ failure under pinning.** Stages 0 and 1 of the singlestage ramp both terminated `ABNORMAL:` at I=0.0. This is the "pure stellarator" baseline that worked under the TF>0 era (I=0 converged with free $I_b \approx +25$ kA). Under TF<0 + pinned $I_b=-16$ kA it doesn't. Strong suspect: the 16 kA hardware cap starves the banana helical field — free-current TF<0 stage 2 at I=0 converged to $I_b=-25.94$ kA, 62% above the cap. If the field produced by $|I_b|=16$ kA can't support iota=+0.15 at the target surface, BoozerLS falls out of basin and L-BFGS sees no descent direction. **Diagnostic proposed**: rerun I=0.0 singlestage with `BANANA_I_FIXED_S2=-24` (still negative for the correct basin, 50% higher magnitude). If it converges, pinning is the bottleneck; if not, something else is in play. Documented as PLAN.md TODO 2.

**Usable operating points from the sweep (for Pareto refresh / downstream work)**: I=1.0 and I=2.0 `stage02/bsurf_opt.json` in `scan_plasma_curr/I{1.0,2.0}kA/stage02/` — both iota=0.150 exactly, constraint margins tight (κ_max grazing 100, length ≤ 1.9 m). I=0.2 `stage01/bsurf_opt.json` may also be usable as a converged (mpol=8, order=4) intermediate. The I=−1.0 stage 0 bsurf exists but at iota=−0.024, wrong basin — discard.

---

## Handoff Notes (2026-04-23)

**TF<0 operating pivot.** TF coil current flipped from +80 kA → −80 kA in [stage2.py:333](jhalpern30/stage2.py#L333) on 2026-04-22 to match the hardware-intended polarity. All prior TF>0 `scan_plasma_curr/I{X}kA*` artifacts archived under `scan_plasma_curr/archive_tfpos_20260422T204355/`; free-current TF<0 runs from earlier in the day archived under `scan_plasma_curr/archive_tfneg_freecurrent_*/`. Fresh TF<0 runs with pinned banana current write to the default paths.

**Sign relationships under TF<0 — everything that needs flipping** (details in "TF sign convention"):
- Reference banana coil current sign **flips**: was positive under TF>0, is negative under TF<0 (for the iota=+0.15 basin with the canonical tilted `+slope` init). Under pinning this means `BANANA_I_FIXED_S2=-16` (NOT `+16`).
- `G0` input to `BoozerSurface.run_code` **flips**: hotfix at [singlestage.py:422-423](jhalpern30/singlestage.py#L422-L423) adds an explicit `-1 * abs(...)` that makes the TF<0 assumption literal (remove the `-1` if TF ever goes back to +80).
- `iota_target` and `IOTA_TARGET_SIGN` **do NOT flip** — the two configurations are mirror images with identical iota sign in Boozer coordinates.

**Loosened tilted init.** The canonical tilted thin-D init in [banana_dofs.txt](jhalpern30/banana_dofs.txt) (regenerated via `python plot_init_coil.py`) now uses `a_s=0.025, a_l=0.090, alpha=62°` — tuned so all four hardware caps (κ, cc, W, poloidal extent) are satisfied at iteration 0. History trail inline in [plot_init_coil.py](jhalpern30/plot_init_coil.py) under the `DOFS` dict.

**Stage 2 current pinning required under TF<0.** Free-current stage 2 at low $|I_p|$ under TF<0 collapses $|I_b|$ and finds a crumpled local minimum (51934327 I=−1.0 settled at $I_b \approx 5$ kA). Pin with `BANANA_I_FIXED_S2=-16` (kA) — presence of the env var is the toggle, sign is literal. See "Stage 2 current handling" for the full semantics and for how it differs from the numbered-driver's `BANANA_CURRENT_MODE_S2` mechanism.

**Stage 2 order default dropped 4 → 3 for zero-DOF-loss handoff.** [stage2.py:371-372](jhalpern30/stage2.py#L371-L372) now defaults to `BANANA_ORDER=3` with `num_quadpoints = 64 * BANANA_ORDER` (=192 at order=3), so stage 2 output matches the first singlestage ramp stage (6, 6, 3, 192) exactly — the rescale at singlestage stage 0 is a no-op instead of a Fourier-mode drop.

**Singlestage ramp flattened to order=4/qp=256.** Default ramp in [singlestage.py:484-489](jhalpern30/singlestage.py#L484-L489) is now `[(6,6,3,192), (8,8,4,256), (10,10,4,256), (12,12,4,256)]` — no order=5/qp=320 stages. Stage 0 matches stage 2's (order=3, qp=192); stages 1–3 ramp mpol at fixed (order=4, qp=256). If order=4/qp=256 proves insufficient for any sweep point, we re-introduce an order=5/qp=320 final stage.

**Singlestage callback robustness.** [singlestage.py:317-325](jhalpern30/singlestage.py#L317-L325) now wraps `boozer_surface.surface.is_self_intersecting()` in try/except with `intersecting = True` on exception. Matches the pattern already used in `fun(x)` at [line 237-239](jhalpern30/singlestage.py#L237). Pre-fix symptom: scipy's callback path raised `cylindrical angle not monotonically increasing` out of the `cross_section` call and killed the whole optimizer (51911090 I=−1.0, 51911093 I=0.0, 51911095 I=0.2 all died this way).

**Stage 2 SLURM allocation.** QoS `debug → shared`, wall time `30 min → 1h`, `--mem=0 → --mem=32G`, MAXITER `300 (?)→ 1500`. `shared` has a higher concurrent-job ceiling than `debug` and similar queue times; the 1h cap + deeper maxiter gives stage 2 headroom from the tighter init. **Poincaré decoupled from stage 2 `-p`**: `run_stage2.sh -p` runs `post_process.py --modb` only (modB / B·n / cross-section plots). Stage 2 Poincaré caused MPI-rank segfaults (51934327, 51934329) and isn't critical for the workflow. Poincaré tracing stays in `run_singlestage.sh -p`.

**Hardware threshold update (late 2026-04-23).** `LENGTH_TARGET = 1.9 m` (unified across stage 2 and singlestage at [stage2.py:442](jhalpern30/stage2.py#L442) + [singlestage.py:665](jhalpern30/singlestage.py#L665); was 1.7 m weighted / 2.0 m ALM in stage 2, 1.8 m in singlestage). `POLOIDAL_THRESHOLD = 45°` half-width (was 50°, default at [stage2.py:448](jhalpern30/stage2.py#L448) + [singlestage.py:666](jhalpern30/singlestage.py#L666)). Overridable via `BANANA_POLOIDAL_TARGET_DEG`. The first half of the 2026-04-23 sweep (jobs 51950930–44 / 51950981–97) was submitted with the OLD thresholds (50° / 1.7–2.0 m / 1.8 m); I=−2.0 and I=−1.0 were re-done as 51955087/88 stage 2 + 51955913/51955986 singlestage after the threshold change to keep the sweep homogeneous. The other 5 points pick up the new thresholds at Python runtime (driver files re-read on job start).

**Poincaré tracer settings upgraded (post_process.py).** `NR=50, NPHI=50, NZ=NR//2=25` (was 30×30×15), `TOL=1e-7` (was 1e-8 — tightening exacerbated adaptive-RK issues near x-points, 1e-7 matches SIMSOPT's reference example), seed R range narrowed to inboard-only and away from the proxy exclusion torus (see [post_process.py:244-247](jhalpern30/post_process.py#L244-L247); effective range ~`[0.82, 0.855]` with 5 cm margins on both sides).

**Known unresolved — Poincaré tracer segfaults at intermediate $|I_p|$.** Under TF<0 + pinned $I_b=-16$ kA + new settings: Poincaré completes cleanly for $|I_p|=2.0$ (51956817 I=−2.0, 51957884 I=2.0) and $|I_p|=1.0$ (51957851 I=1.0). **Segfaults (rc=143) persist for $|I_p| \in \{0.0, 0.2, 1.0\text{ initial}, -0.2, -1.0\}$** even after grid refinement + seed-range tightening + tol loosening. Different MPI ranks segfault each run, suggesting the adaptive-RK integrator is hitting x-points or interp-spline artifacts the boundary-stopping criteria don't catch fast enough. Tagged for later investigation (2026-04-23); sweep physics analysis can still proceed using the 3 good Poincaré traces + the stage2_plots.png modB/B·n diagnostics that ran for every point.

**Current active sweep state (2026-04-23 EOD).** 7-point $I_p \in \{-2.0, -1.0, -0.2, 0.0, 0.2, 1.0, 2.0\}$ kA at TF=−80 kA, tilted init, pinned $I_b = -16$ kA in stage 2, singlestage at ramp-flattened order=4/qp=256. **All 7 stage 2 jobs COMPLETED.** Singlestages: 3 RUNNING (I=−0.2 / 0.0 / 0.2), 4 PENDING (I=−2.0 / −1.0 / 1.0 / 2.0). Singlestage for I=−2.0 and I=−1.0 was resubmitted (51955913/51955986) after the threshold change; the original 51950981/82 were cancelled. The sweep answers: under TF<0, which sign of $I_p$ is "tokamak-assisted" (plasma poloidal flux boosts iota) vs "hybrid-Shafranov" (outward axis drift)? Expected mapping: since $I_b$ now wants to be negative for iota>0, $I_p>0$ co-rotates and is the tokamak-assisted regime (swapped from the TF>0 convention). Required before the negative-current Pareto scan is physically meaningful.

**SBATCH_DEPENDENCY quirk through `run_singlestage.sh`.** Setting `SBATCH_DEPENDENCY=afterok:<id>` in the invoking shell did NOT attach the dependency to the submitted job at NERSC — jobs showed `Dependency=(null)` (seen 2026-04-23). The sbatch env var path doesn't propagate through the wrapper's `exec sbatch --export=ALL,...` chain reliably. **Workaround**: submit first, then `scontrol update JobId=<ss_id> Dependency=afterok:<s2_id>` for each pair while the singlestages are still PENDING. SLURM rejects `afterok:<already-completed-job>`, so skip already-finished stage 2s (their biotsavart is real — no dependency needed). Permanent fix (deferred): add a `-d/--dependency` CLI flag to `run_singlestage.sh` that passes through as an explicit `--dependency=...` on the sbatch call.

---

## Handoff Notes (2026-04-22)

The active line of attack is `jhalpern30/stage2.py` + `jhalpern30/singlestage.py`, running on the hardware-correct setup (80 kA TF, `CurveSelfIntersect` + `poloidal_extent` + `ellipse_width` active, order=4 banana coils, `num_quadpoints = 64 × TARGET_ORDER` so order=4 → qp=256). These drivers are clones-with-diffs of the numbered pipeline (`01_stage1_driver.py` / `02_stage2_driver.py` / `03_singlestage_driver.py`) — the numbered drivers still exist and are the eventual canonical home, but the new objectives are only wired into the `jhalpern30/` variants until the integration TODO lands.

**Baseline (I=0, VF=0) is recovered** — [jhalpern30/scan_plasma_curr/I0.0kA/log.txt](jhalpern30/scan_plasma_curr/I0.0kA/log.txt) shows 33 L-BFGS-B iterations, final $J=5.80\text{e-}3$, iota=0.149, length=2.00 m, $\kappa_\text{max}=67\text{ m}^{-1}$, CC=5.4 cm, CS=5.3 cm — every hardware constraint satisfied under the new limits. This is the reference point everything else is compared against.

**Sign asymmetry in plasma current.** The 11-point `jhalpern30/scan_plasma_curr/` sweep (VF=0, banana current free, plasma current $I \in \{0, \pm 0.2, \pm 0.9, \pm 1, \pm 2, \pm 5\}$ kA) lands in the correct iota=+0.15 basin for **every** $I \leq 0$ case that ran and lands in the wrong iota≈−0.03 to −0.06 basin for **every** $I \geq 0.2$ case. Physics: $I<0$ reinforces banana helicity, $I>0$ opposes it, and BoozerLS (inner BFGS + Newton) locks into whichever helicity basin is nearest — the optimizer can't cross $I=0$ because it's a saddle, not a gradient direction. Two remediation options on the table, simplest first:
1. Reverse the sign of the banana current (`ScaledCurrent` sign flip) for $I>0$ runs. Free — one line of driver code. Should flip iota basin.
2. Mirror the banana coil geometry ($\theta \to -\theta$ on the winding surface). Requires redesign.

Option (1) is now wired ([jhalpern30/stage2.py:389](jhalpern30/stage2.py#L389) applies `BANANA_CURRENT_SIGN` to the banana `ScaledCurrent` when the output dir ends in `_flip`; [jhalpern30/singlestage.py:382](jhalpern30/singlestage.py#L382) applies the matching `IOTA_TARGET_SIGN` to `iota_target`). Stage 2 has run at I=+1.0 kA with the flip active → [jhalpern30/scan_plasma_curr/I1.0kA_flip/](jhalpern30/scan_plasma_curr/I1.0kA_flip/) has `biotsavart_opt.json` + Poincaré outputs. The singlestage flip run on that warm start is the next action to verify whether the basin flip holds all the way through joint optimization.

**Current known-good recipe for new runs.** Start from [jhalpern30/stage2.py](jhalpern30/stage2.py) (order=3 default, qp=192; `BANANA_ORDER` env var override) → [jhalpern30/singlestage.py](jhalpern30/singlestage.py) (Fourier order ramp `[3, 4, 5, 5]` matched to surface `mpol=ntor ∈ [6, 8, 10, 12]`). The `[5, 5]` repeat is deliberate — when the surface goes from 10→12 the coil order already saturates at 5. Stage 2 is pinned at `(order=3, qp=192)` so it exactly matches the first singlestage ramp stage — no order downsample / no DOF loss on handoff. `num_quadpoints = 64 × BANANA_ORDER` in both stage 2 and singlestage for symmetry (so if `BANANA_ORDER=4` is ever set on stage 2, it yields qp=256 matching singlestage stage 1).

**Singlestage basin reliability depends on the warm-start seed.** Two paths behave very differently at ramp stage 0 (mpol=6, order=3, qp=192):
- `iota{15,20}_rithik/biotsavart_opt.json` → BoozerLS converges cleanly at iota≈0.12–0.17 inside the correct basin (see 51854494 iota=0.1592 reference).
- `scan_plasma_curr/I0.0kA/biotsavart_opt.json` → wrong-basin disaster at stage 0 (iota=−14857, surface self-intersects, Newton diverges; observed in 51867128, cancelled 2026-04-22).

Use the rithik seeds for new resolution checks and for any baseline singlestage comparison. The scan_plasma_curr stage-2 seed is not a reliable singlestage entry point even with the matched ramp; what differs relative to the rithik seed is an open question (basin proximity of the stage-2-final coil shape under the current weighted stage-2 configuration).

**Singlestage allocation shape matters.** Default `run_singlestage.sh` was sized for Poincaré post-processing (`--ntasks=16 --cpus-per-task=1` + `OMP_NUM_THREADS=1`), which means singlestage ran single-threaded — 51854494 took 15h40m just to reach mpol=10/order=4. The wrapper now takes `-c/--cpus N` (default 32 when not post-processing): allocates `--ntasks=1 --cpus-per-task=N` + `OMP=MKL=OPENBLAS=N`. Use `-c 32` for any singlestage-only run; reserve the 16×1 default for `-p/--post-process`.

**Pending pieces before the pipeline is canonical.** (i) Integrate `new_objectives/` into `02_stage2_driver.py` + `03_singlestage_driver.py`. (ii) Port ALM from stage 2 into singlestage — **done** (wired 2026-04-21). (iii) Coil-force objectives — review SIMSOPT `LpCurveForce` vs `SquaredMeanForce` + self vs mutual terms — source review **done**, post-processing wiring pending. (iv) I>0 helicity-flip test — mechanism wired 2026-04-21, stage 2 ran at I=+1.0 kA → `scan_plasma_curr/I1.0kA_flip/` (biotsavart + Poincaré outputs present), singlestage flip and I>0 repeats pending. (v) I=0 poloidal-extent **demonstration** sweep — **concluded 2026-04-22** (stage 2 proves the `PoloidalExtent` penalty works structurally; keep `BANANA_POLOIDAL_WEIGHT` non-zero in singlestage for end-to-end validation in subsequent runs, rather than maintaining a dedicated sweep). (vi) Order/qp resolution question — **pivoted to H'** (single-point o4_qp256 check at 32 cores, job 51895375 queued 2026-04-22). The earlier 2×2 scan was deprecated because 4-worker fan-out at mpol=10 doesn't fit one SLURM allocation; a single minimum-resolution check against a higher-resolution rithik reference answers the actionable question. (vii) Negative-current Pareto scan — `jhalpern30/pareto_negative_current.py`, 128-point scan over $(|I|<5\text{ kA}, \iota_\text{target}\in[0.08, 0.25])$ with $I<0$ and the truncated ramp `BANANA_RAMP_STAGES=2`; resubmitted 2026-04-22 as job 51894344 with 4h / 6h / 14h per-point / wall time budgets after 51859030 hit the old 2h per-point ceiling.

**Warm-start / resume.** `BANANA_RESUME_STAGE=N` resumes a ramped singlestage from `$OUT_DIR/stage{N:02d}/bsurf_opt.json` + `state.json`. Each successful stage now writes `state.json` ({iota, G, volume, iota_target, stage_mpol, stage_ntor, stage_order, stage_qp}) alongside `bsurf_opt.json`. Resume requires matching ramp configuration (same `BANANA_RAMP` / `BANANA_RAMP_STAGES` / `BANANA_RAMP_CUSTOM`) as the original run. Useful for partial-progress recovery and for fanning out multiple endpoint variants from a shared early-ramp warmup.

**Open questions still on file.** The old Q1 ("why does the `jhalpern30/` scan variant succeed where `03_singlestage_driver.py` fails") is obsolete — the jhalpern30 drivers are now the canonical path; the numbered drivers will inherit the same objectives once integration lands. The remaining live questions: the sign asymmetry above, and why the `scan_plasma_curr/I0.0kA/biotsavart_opt.json` seed fails the ramp stage-0 basin check while `iota{15,20}_rithik/biotsavart_opt.json` succeeds (both are I=0 stage-2 outputs). See [PLAN.md](PLAN.md) for the current task list.

## Environment

- **HPC**: Perlmutter @ NERSC (128 CPUs per node, SLURM scheduler)
- **SIMSOPT fork**: `hayashiw/simsopt` on `whjh/auglag_banana` branch, located at `hybrid_torus/banana/simsopt/` (relative to project root)
- **SIMSOPT for banana**: Always reference `hybrid_torus/banana/simsopt/` for banana-specific SIMSOPT issues (e.g., `CurveCWSFourierCPP`, the `auglag_banana` modifications to `augmented_lagrangian.py`). Do NOT check `qi_rso/simsopt/` unless explicitly asked — it is a separate fork for the QI project.
- **Related repo**: `qi_rso/qi_drivers/` (separate project; reference for formatting/workflow patterns and the original ALM implementation that was ported here.)
