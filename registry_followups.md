# Run Registry — Follow-up Fixes

Review of `utils/run_registry.py`, `utils/run_registry_schema.sql`, and the
three drivers after wiring the registry end-to-end (2026-04-13). Picking
this up after the break: fix #1 and #2 before trusting any new registry
rows; #3 is a design choice.

## Correctness issues

### 1. Env var overrides bypass content-addressing (P1)

Every stage reads env vars that mutate runtime behavior but hashes `cfg`
unchanged, so runs with different env values collide on the same
`run_id`. This directly breaks the Pareto sweep use case — the whole
point of the sweep is varying these axes.

Affected env vars:

- **Stage 1** (`01_stage1_driver.py`): `BANANA_SEED`, `BANANA_IOTA`,
  `BANANA_ASPECT`, `BANANA_VOLUME`, `BANANA_COLD_R0`
- **Stage 2** (`02_stage2_driver.py`): `BANANA_STAGE2_MODE`,
  `BANANA_ALM_PRESET`, `BANANA_CURRENT_MODE_S2`, `BANANA_TAU`,
  `BANANA_MAXITER_LAG`, `BANANA_DOF_SCALE`,
  `BANANA_STAGE2_LENGTH_RELAX`, `BANANA_STAGE2_CC_RELAX`,
  `BANANA_STAGE2_CURV_RELAX`, `BANANA_I_FIXED_S2`
- **Stage 3**: none significant

**Fix:** before calling `register_*(cfg, ...)`, mutate `cfg` in place
with the resolved effective values. Write back into the same keys the
whitelist already hashes:

- Stage 1 → `cfg['stage1']['cold_start']`, `cfg['stage1']['iota_target']`,
  `cfg['stage1']['aspect_target']`, `cfg['stage1']['volume_target']`,
  `cfg['stage1']['cold_start_R0']`, `cfg['stage1']['cold_start_volume']`
- Stage 2 → `cfg['stage2_mode']`, `cfg['stage2_alm']['preset']`,
  `cfg['banana_coils']['current_mode_stage2']`,
  `cfg['banana_coils']['current_fixed_stage2']`,
  `cfg['stage2_alm']['tau']`, `cfg['stage2_alm']['maxiter_lag']`,
  `cfg['stage2_alm']['dof_scale']`,
  `cfg['stage2_relaxation']['length']`,
  `cfg['stage2_relaxation']['coil_coil']`,
  `cfg['stage2_relaxation']['curvature']`

Do the write-back at the top of each driver, right after env resolution
and before `register_*`. The module-level derived constants
(`IOTA_TARGET`, `STAGE2_MODE`, `LENGTH_RELAX`, …) can then read from the
mutated cfg instead of re-parsing env, or keep the env-parse and add an
explicit `cfg[...] = <resolved>` line alongside.

### 2. Stage 3 whitelist omits plasma-surface init keys (P1)

[03_singlestage_driver.py:133](03_singlestage_driver.py#L133) seeds the
BoozerLS surface via
`SurfaceRZFourier.from_wout(WOUT_FILE, nphi=NPHI, ntheta=NTHETA, s=VMEC_S)`.
`SINGLESTAGE_INPUT_KEYS` in
[utils/run_registry.py:176-202](utils/run_registry.py#L176-L202) does
not include these, so changing `plasma_surface.nphi`/`ntheta`/`vmec_s`
does not bump the singlestage id even though it changes the seed
surface.

**Fix:** add to `SINGLESTAGE_INPUT_KEYS`:

```python
"plasma_surface.nphi",
"plasma_surface.ntheta",
"plasma_surface.vmec_s",
```

Stage 2 does **not** need these (it loads the surface from the bsurf
JSON, which already carries quadpoints internally).

### 3. `BANANA_INIT_BSURF` override removed (2026-04-13)

Resolved via option (a): the override was dropped from
`02_stage2_driver.py`. The stage 2 warm-start bsurf is now determined
exclusively by `stage1_id`, so the hashed parent always matches the
file actually loaded.

## Minor

### 4. Stage 1 whitelist omits `warm_start.wout_filepath`

Warm-start mode depends on the seed wout file content. If someone swaps
the file in place the hash won't notice. Low risk because the seed is
stable and produced by `utils/vmec_resize.py`, but worth a line comment
in `STAGE1_INPUT_KEYS` or an explicit include. Cold-start mode doesn't
consume this field, so including it bumps cold-start hashes unnecessarily
— probably not worth it. Document as a known limitation.

### 5. Dead default in `STAGE_EXTRA_COLS`

[utils/run_registry.py:215](utils/run_registry.py#L215) uses
`blob.get("stage1.cold_start", False)`, but `build_input_blob` already
KeyErrors on missing whitelist keys. Drop the `False` default — it is
misleading and implies a silent fallback that cannot happen.

### 6. `_register` is non-atomic across row-exists / INSERT / UPDATE

Benign today (serial writers per stage). A concurrent Pareto sweep on
stage 2 could race on the existing-row UPDATE branch and silently drop
one writer's `slurm_meta` refresh. `INSERT OR IGNORE` makes the insert
path race-safe; the UPDATE is not. Not worth fixing until a sweep
actually races.

## Acceptable / no action

- **6-hex prefix** ≈ 4k-run birthday collision bound — fine at research
  scale. Revisit only if a stage accumulates thousands of rows.
- **Parent-exists check allows `status='failed'` parents** — deliberate
  flexibility for manual reruns and post-mortem analysis.
- **Atexit ordering** (`OUT_DIR=` emit vs `install_atexit_handler`) does
  not matter in practice — both fire on normal exit and the unclean-exit
  handler is a no-op when status is not `running`.
- **Stage 1 `mark_failed` on rejected near-axis seed fires before
  `_group_dir` is created** — safe; the `_cleanup_group_dir` handler is
  registered later via decorator, so it isn't on the atexit stack yet.
- **Stage 2 `stage2_ok`** correctly treats `BUDGET_EXHAUSTED` as success
  under the truncated-iteration regime (gtol/ftol are intentionally
  unreachable; maxiter is the happy path).

## Order when resuming

1. Apply #1 (cfg mutation before registration) in all three drivers.
2. Apply #2 (three-key whitelist addition) in `utils/run_registry.py`.
3. Decide #3 (a or b) and apply.
4. Update CLAUDE.md with the new output layout + env var documentation.
5. End-to-end smoke test: stage 1 → stage 2 → stage 3 with a fresh
   `registry.db`, confirm run IDs differ across env var variants.
