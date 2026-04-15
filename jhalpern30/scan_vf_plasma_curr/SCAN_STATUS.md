# VF × Plasma Current Scan Status

4×5 = 20 pairs. Stage 2 (banana_coil_solver) → Poincaré → singlestage.

## Resume notes (2026-04-14, for pickup on another machine)

This directory was promoted from `banana_drivers/local/jhalpern30/scan_vf_plasma_curr/` to `banana_drivers/jhalpern30/scan_vf_plasma_curr/` ahead of Perlmutter maintenance 2026-04-15 so it can be tracked in git. Log files are still NOT tracked — they live in:
- Running jobs' per-iteration `./I{I}kA_VF{VF}kA/log.txt` (relative to the scan directory)
- SLURM stdout/stderr under `$SCRATCH/banana_drivers_outputs/` (or, for historical runs, the user's previous machine under `local/jhalpern30/`)
- `sacct -j <jobid>` works regardless of machine and is the canonical way to recover job state

**In flight at time of writeup:**
- Singlestage `51577013` (I=0/VF=0) from before the path move, still running — unaffected by the directory rearrangement.
- Singlestage `51579433`–`51579445` (12 pairs, `51579439` missing in the range for reasons unknown) queued after the path-fix resubmit. Original batch `51578769`–`51578781` all failed at `bsurf_init` save with `FileNotFoundError` because the per-point `I*_VF*/` subdirs were accessed via the old `local/jhalpern30/` path — fixed by running a `sed` rewrite on the driver shell scripts and resubmitting.

**Driver variants used here** (minimal-diff clones of the jhalpern30 baseline, NOT the main `banana_drivers/` pipeline):
- `banana_coil_solver.py` — stage 2 with finite plasma current via proxy coil
- `poincare_simple.py` — field-line tracing with `LevelsetStoppingCriterion` exclusion torus around the proxy coil
- `singlestage_banana.py` — singlestage with finite-current API (`BoozerSurface(..., I=μ₀·I_plasma)`)

**Known degenerate/broken cells** (do not retry without fixing root cause):
- `I=-8.0/VF=-3.0` Poincaré — reproducible segfault in `compute_fieldlines` across 3 attempts (`51577418`, `51578079`, `51578297`), different nodes, `R_EXCL ∈ {0.03, 0.02}`. Abandoned.
- `I=-0.1/VF=0.0` and the `VF=3.0` column stage-2 runs — hidden-degenerate collapse (L-BFGS-B reported converged, but `Len ∈ {0.5, 0.7} m` and `Ibanana/Itf = 0.100` = init value). See the Stage 2 / Winding Surface Projection tables below.

**Note:** the main `banana_drivers/03_singlestage_driver.py` does NOT reliably converge on this geometry; `singlestage_banana.py` (this directory) does. The delta is an open question — see the main [PLAN.md](../../PLAN.md) `Open Questions` section before assuming "singlestage now works" for the main pipeline.

Legend: `·` not started · `Q` queued · `R` running · `✓` done · `✗` failed

## Stage 2 (banana_coil_solver)

| I \ VF  | -3.0 | -1.0 |  0.0 |  1.0 |  3.0 |
|---------|------|------|------|------|------|
|  -8.0   |  ✓   |  ✓   |  ✓   |  ✓   |  ✓   |
|  -1.0   |  ✓   |  ✓   |  ✓   |  ✓   |  ✗   |
|  -0.1   |  ✓   |  ✗   |  ✗   |  ✓   |  ✗   |
|   0.0   |  ✓   |  ✓   |  ✓   |  ✗   |  ✗   |

## Winding Surface Projection (visual coil check)

Cross-check of stage 2 output from `jhalpern30/bananacoils.png` (φ–θ
projection on the winding surface). `✓` = proper banana shape; `S` =
self-intersecting; `s` = stunted (collapsed into a small oval, not a banana).

| I \ VF  | -3.0 | -1.0 |  0.0 |  1.0 |  3.0 |
|---------|------|------|------|------|------|
|  -8.0   |  ✓   |  ✓   |  ✓   |  ✓   |  ✓   |
|  -1.0   |  ✓   |  ✓   |  ✓   |  ✓   |  s   |
|  -0.1   |  ✓   |  S   |  s   |  ✓   |  s   |
|   0.0   |  ✓   |  ✓   |  ✓   |  s   |  s   |

## Poincaré

| I \ VF  | -3.0 | -1.0 |  0.0 |  1.0 |  3.0 |
|---------|------|------|------|------|------|
|  -8.0   |  ✗   |  ✓   |  ✓   |  ✓   |  ✓   |
|  -1.0   |  ✓   |  ✓   |  ✓   |  ✓   |  ·   |
|  -0.1   |  ✓   |  ✓   |  ✓   |  ✓   |  ·   |
|   0.0   |  ✓   |  ✓   |  ✓   |  ·   |  ·   |

## Singlestage

| I \ VF  | -3.0 | -1.0 |  0.0 |  1.0 |  3.0 |
|---------|------|------|------|------|------|
|  -8.0   |  ·   |  Q   |  Q   |  Q   |  Q   |
|  -1.0   |  Q   |  Q   |  Q   |  Q   |  ·   |
|  -0.1   |  Q   |  ·   |  ·   |  Q   |  ·   |
|   0.0   |  Q   |  Q   |  R   |  ·   |  ·   |

## Job IDs

| I     | VF    | Stage   | JobID    | Status | Converged | Notes                  |
|-------|-------|---------|----------|--------|-----------|------------------------|
|  0.0  |  0.0  | stage2  | 51562322 | ✓      | ✓         | Ibanana/Itf = 0.206    |
| -1.0  |  0.0  | stage2  | 51562323 | ✓      | ✓         | Ibanana/Itf = 0.196    |
| -0.1  |  0.0  | stage2  | 51562389 | ✓      | ✗         | DEGENERATE: Len=0.5m, Ibanana/Itf=0.100 (hidden; L-BFGS-B falsely converged) |
| -8.0  |  0.0  | stage2  | 51562391 | ✓      | ✓         | Ibanana/Itf = 0.140    |
|  0.0  | -1.0  | stage2  | 51562626 | ✓      | ✓         | Ibanana/Itf = 0.208    |
| -0.1  | -1.0  | stage2  | 51562628 | ✓      | ✓         | Ibanana/Itf = 0.204; self-intersecting |
| -1.0  | -1.0  | stage2  | 51562629 | ✓      | ✓         | Ibanana/Itf = 0.184    |
| -8.0  | -1.0  | stage2  | 51562630 | ✓      | ✓         | Ibanana/Itf = 0.142    |
|  0.0  | -3.0  | stage2  | 51563202 | ✓      | ✓         | Ibanana/Itf = 0.200    |
| -0.1  | -3.0  | stage2  | 51563204 | ✓      | ✓         | Ibanana/Itf = 0.188    |
| -1.0  | -3.0  | stage2  | 51563208 | ✓      | ✓         | Ibanana/Itf = 0.194    |
| -8.0  | -3.0  | stage2  | 51563214 | ✓      | ✓         | Ibanana/Itf = 0.134    |
|  0.0  |  0.0  | poinc   | 51563243 | ✓      | —         | traced 32 lines, R_EXCL=0.03 |
|  0.0  |  0.0  | poinc   | 51563460 | ✓      | —         | rerun, R_EXCL=0.01 (inboard clamp 0.921) |
|  0.0  |  0.0  | poinc   | 51563760 | ✓      | —         | uniform R_start spacing |
|  0.0  |  3.0  | stage2  | 51574768 | ✓      | ✗         | DEGENERATE: Len=0.7m, CC=0.12m, Ibanana/Itf=0.100 |
| -0.1  |  3.0  | stage2  | 51574769 | ✓      | ✗         | DEGENERATE: Len=0.7m, CC=0.12m, Ibanana/Itf=0.100 |
| -1.0  |  3.0  | stage2  | 51574770 | ✓      | ✗         | DEGENERATE: Len=0.7m, CC=0.12m, Ibanana/Itf=0.100 |
| -8.0  |  3.0  | stage2  | 51574771 | ✓      | ✓         | Ibanana/Itf = 0.145    |
|  0.0  |  1.0  | stage2  | 51563461 | ✓      | ✗         | Ibanana/Itf = 0.099; DEGENERATE: J=7.7e-04, Len=0.7m, CC=0.14m — coils collapsed |
| -0.1  |  1.0  | stage2  | 51563462 | ✓      | ✓         | Ibanana/Itf = 0.206    |
| -1.0  |  1.0  | stage2  | 51563463 | ✓      | ✓         | Ibanana/Itf = 0.193    |
| -8.0  |  1.0  | stage2  | 51563464 | ✓      | ✓         | Ibanana/Itf = 0.128    |
| -8.0  | -1.0  | poinc   | 51576148 | ✓      | —         | wave 1                 |
| -1.0  | -1.0  | poinc   | 51576149 | ✓      | —         | wave 1                 |
| -0.1  | -1.0  | poinc   | 51576150 | ✓      | —         | wave 1                 |
|  0.0  | -1.0  | poinc   | 51576152 | ✓      | —         | wave 1                 |
| -8.0  |  1.0  | poinc   | 51576153 | ✓      | —         | wave 1                 |
|  0.0  |  0.0  | single  | 51574995 | ✗      | —         | srun cpus-per-task env conflict on shared QOS |
| -1.0  |  1.0  | poinc   | 51577000 | ✓      | —         | wave 2                 |
| -0.1  |  1.0  | poinc   | 51577002 | ✓      | —         | wave 2                 |
| -8.0  |  0.0  | poinc   | 51577003 | ✓      | —         | wave 2                 |
| -1.0  |  0.0  | poinc   | 51577005 | ✓      | —         | wave 2                 |
| -0.1  |  0.0  | poinc   | 51577006 | ✓      | —         | wave 2                 |
|  0.0  |  0.0  | single  | 51577013 | R      |           | resubmit w/o srun      |
| -8.0  | -3.0  | poinc   | 51577418 | ✗      | —         | wave 3; segfault in compute_fieldlines |
| -1.0  | -3.0  | poinc   | 51577419 | ✓      | —         | wave 3                 |
| -0.1  | -3.0  | poinc   | 51577420 | ✗      | —         | wave 3; task failure (same node as 51577418) |
|  0.0  | -3.0  | poinc   | 51577421 | ✗      | —         | wave 3; task failure   |
| -8.0  |  3.0  | poinc   | 51577422 | ✓      | —         | wave 3; 165.8s (slow)  |
| -8.0  | -3.0  | poinc   | 51578079 | ✗      | —         | wave 3 resubmit; segfault again |
| -0.1  | -3.0  | poinc   | 51578081 | ✓      | —         | wave 3 resubmit        |
|  0.0  | -3.0  | poinc   | 51578082 | ✗      | —         | wave 3 resubmit; segfault again (different node) |
| -8.0  | -3.0  | poinc   | 51578297 | ✗      | —         | segfault persists w/ R_EXCL=0.02; abandoning |
|  0.0  | -3.0  | poinc   | 51578298 | ✓      | —         | resubmit w/ R_EXCL=0.02 |
| -8.0  | -1.0  | single  | 51578769 | ✗      |           | pre-move; save failed at bsurf_init |
| -8.0  |  0.0  | single  | 51578770 | ✗      |           | pre-move; save failed at bsurf_init |
| -8.0  |  1.0  | single  | 51578771 | ✗      |           | pre-move; save failed at bsurf_init |
| -8.0  |  3.0  | single  | 51578772 | ✗      |           | pre-move; save failed at bsurf_init |
| -1.0  | -3.0  | single  | 51578773 | ✗      |           | pre-move; save failed at bsurf_init |
| -1.0  | -1.0  | single  | 51578774 | ✗      |           | pre-move; save failed at bsurf_init |
| -1.0  |  0.0  | single  | 51578775 | ✗      |           | pre-move; save failed at bsurf_init |
| -1.0  |  1.0  | single  | 51578776 | ✗      |           | pre-move; save failed at bsurf_init |
| -0.1  | -3.0  | single  | 51578778 | ✗      |           | pre-move; save failed at bsurf_init |
| -0.1  |  1.0  | single  | 51578779 | ✗      |           | pre-move; save failed at bsurf_init |
|  0.0  | -3.0  | single  | 51578780 | ✗      |           | pre-move; save failed at bsurf_init |
|  0.0  | -1.0  | single  | 51578781 | ✗      |           | pre-move; save failed at bsurf_init |
| -8.0  | -1.0  | single  | 51579433 | Q      |           | resubmit after path fix |
| -8.0  |  0.0  | single  | 51579434 | Q      |           | resubmit after path fix |
| -8.0  |  1.0  | single  | 51579435 | Q      |           | resubmit after path fix |
| -8.0  |  3.0  | single  | 51579436 | Q      |           | resubmit after path fix |
| -1.0  | -3.0  | single  | 51579437 | Q      |           | resubmit after path fix |
| -1.0  | -1.0  | single  | 51579438 | Q      |           | resubmit after path fix |
| -1.0  |  0.0  | single  | 51579440 | Q      |           | resubmit after path fix |
| -1.0  |  1.0  | single  | 51579441 | Q      |           | resubmit after path fix |
| -0.1  | -3.0  | single  | 51579442 | Q      |           | resubmit after path fix |
| -0.1  |  1.0  | single  | 51579443 | Q      |           | resubmit after path fix |
|  0.0  | -3.0  | single  | 51579444 | Q      |           | resubmit after path fix |
|  0.0  | -1.0  | single  | 51579445 | Q      |           | resubmit after path fix |
