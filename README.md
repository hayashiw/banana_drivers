# Banana Drivers

Coil optimization drivers for the **banana coils** of the HBT-EP stellarator–tokamak hybrid device, built on [SIMSOPT](https://github.com/hiddenSymmetries/simsopt).

The goal is a manufacturable set of banana coils that supports both pure-stellarator (vacuum) and finite plasma-current scenarios.

---

## Install

The package is laid out as a standard `src/`-style Python project:

```bash
git clone <this repo>
cd banana_drivers
pip install -e .
```

Python 3.10+ is required. The deps in `pyproject.toml` (`numpy`, `scipy`, `matplotlib`, `pyyaml`) are the package's own; SIMSOPT (with the banana-specific `CurveCWSFourierCPP` and CWS-frame objectives) must be installed separately — see [SIMSOPT fork](#simsopt-fork) below.

After install, all CLI entry points listed below are available as `banana-*` shell commands.

---

## Pipeline status

| Stage | Status | Driver |
|-------|--------|--------|
| Generate initial coils + surface | working | `banana-generate-biotsavart`, `banana-generate-default-surface` |
| **Stage 2** — coil-only optimization (`SquaredFlux` + geometry penalties) | working | `banana-stage2` |
| **Singlestage** — joint coil + surface optimization (BoozerLS) | **WIP**, raises on import | `src/banana_drivers/drivers/singlestage.py` |

Stage 2 runs on a fixed target surface, optimizing the banana coil DOFs (shape and, optionally, current). The CLI takes either a `--biotsavart-file` + `--wout-file`/`--surface-file` pair, a `--boozersurface-file`, or `--build` to construct everything from coil args. See `banana-stage2 --help`.

Singlestage is the next piece under construction. Its entry point is intentionally **not** registered in `pyproject.toml` and the module raises at import time so collaborators can't accidentally invoke a half-built driver.

---

## CLI entry points

All `banana-*` commands accept `--help`.

| Command | Purpose |
|---------|---------|
| `banana-generate-biotsavart` | Build an HBT-EP coil set (TF + banana + proxy + VF) and save as a `BiotSavart` JSON. `--default` regenerates the canonical `biotsavart_init.json`. |
| `banana-generate-default-surface` | Extract the target plasma surface from `inputs/wout_original.nc` at $s=0.24$, scale to $R_0 = 0.925$ m, and save `inputs/surface_init.json`. |
| `banana-stage2` | Stage 2 coil-only optimization. |
| `banana-initialize-boozersurface` | Assemble an unsolved `BoozerSurface` JSON from a `BiotSavart` + a surface, ready for downstream `run_code(...)`. |
| `banana-flip-surfacerzfourier` | Flip a `SurfaceRZFourier` toroidally ($\phi \to -\phi$) via DOF relabel. |
| `banana-plot-modB-Bdotn` | Plot $\lvert\mathbf{B}\rvert$, $\mathbf{B}\cdot\hat{\mathbf{n}}/\lvert\mathbf{B}\rvert$, and coil cross-sections for a BiotSavart on a target surface. |
| `banana-print-coil-currents` | Pretty-print all coil currents from a BiotSavart or BoozerSurface JSON. |
| `banana-print-banana-coil-parameters` | Pretty-print banana coil Fourier order, quadpoint count, and per-coil currents. |
| `banana-add-proxy-and-vf-coils` | Add proxy + VF coils to an existing BiotSavart (for finite-current scenarios). |
| `banana-run-virtual-casing` | Run a virtual-casing calculation from a VMEC wout for the finite-current pipeline. |

---

## Package layout

```
banana_drivers/
├── pyproject.toml          # build + entry points
├── README.md               # this file
├── src/banana_drivers/
│   ├── hardware.py         # frozen dataclasses: HBT-EP geometry, hardware limits
│   ├── paths.py            # resolved paths to inputs/, configs
│   ├── drivers/
│   │   ├── stage2.py
│   │   ├── stage2_config.yaml
│   │   ├── singlestage.py        # WIP — raises at import
│   │   └── singlestage_config.yaml
│   ├── scripts/            # one-shot utilities (one CLI per file)
│   ├── utils/
│   │   ├── cli.py          # shared argparse parents (coil currents, geometry, IO)
│   │   ├── coils.py        # coil-set construction
│   │   ├── surface.py      # surface loading / building / rescaling
│   │   ├── preprocess.py   # CLI-args → (biotsavart, surface) resolution
│   │   ├── boozersurface.py
│   │   ├── plot.py
│   │   └── io.py           # DriverLog — diagnostics CSV writer
│   ├── objectives/
│   │   ├── cwsobjectives.py        # CWS-frame manufacturability penalties
│   │   │                            # (PoloidalExtent, ProjectedEllipseWidth, CurveSelfIntersect)
│   │   └── currentobjectives.py    # ScaledCurrentWrapper for current penalties
│   └── inputs/             # checked-in inputs (see below)
└── local/                  # gitignored — working files and run outputs
```

`local/` is reserved for the user's working files (modified inputs, scratch outputs) and is gitignored via the `local*/` rule. Drivers default `--output-dir` to the current working directory, so running from `local/outputs/` keeps run artifacts out of the tree.

---

## Configs

Each driver reads weights and targets from a YAML file next to it:

- `src/banana_drivers/drivers/stage2_config.yaml` — `weights` (sqflux, length, ccdist, curvature, poloidal, width, selfint, currents) and `targets` (length, ccdist, curvature, poloidal, width_max, width_min, selfint).
- `src/banana_drivers/drivers/singlestage_config.yaml` — same shape; singlestage-specific physics knobs (iota target, BoozerLS constraint weight, Fourier ramp) still being wired in alongside the driver.

Override with `--config <path>`. Penalty thresholds and engineering limits live in `hardware.py`, not the config files — see [Hardware](#hardware) below.

---

## Inputs

`src/banana_drivers/inputs/` holds the canonical, version-controlled inputs shared across all runs:

| File | Purpose |
|------|---------|
| `wout_original.nc` | VMEC wout the target plasma surface is extracted from (at $s=0.24$, scaled to $R_0=0.925$ m). |
| `wout_hbt_finite_beta.nc` | Finite-$\beta$ VMEC wout for the finite plasma-current branch. |
| `input.hbt_finite_beta` | VMEC input file used to produce the finite-$\beta$ wout. |
| `HBT_101393_01.eqdsk` | EFIT g-file (shot 101393); reference equilibrium for the finite-current scenario. |
| `banana_init_dofs.txt` | Default banana coil CWS Fourier DOFs (simple ellipse). |

Run-derived inputs (`biotsavart_init.json`, `surface_init.json`, etc.) are produced by `banana-generate-biotsavart` and `banana-generate-default-surface`. Custom variants (flipped surfaces, alternate init DOFs) should live in `local/inputs/`.

---

## Hardware

`src/banana_drivers/hardware.py` defines frozen dataclasses for the HBT-EP geometry and the engineering limits the drivers enforce. Quick reference (always cross-check the source for the live values):

- **TF coils**: $R_0 = 0.976$ m, $a = 0.400$ m, 20 coils at $-80$ kA each (fixed).
- **Banana winding surface**: $R_0 = 0.903$ m, $a = 0.142$ m, nfp=5, stellsym (updated 2026-05-26 — prior 0.976 / 0.210 m values are obsolete).
- **Banana current limit**: $\lvert I_b \rvert \le 16$ kA per coil.
- **Geometric limits**: `max_length = 1.9` m, `max_curvature = 100` m$^{-1}$ (≈1 cm bending radius), `min_ccdist = 0.0462` m, `min_csdist = 0.010` m. Curvature uses an $L^4$ norm (`banana_curv_p = 4`).
- **Target plasma**: $R_0 = 0.92$ m, nfp=5, stellsym.

`banana-stage2` enforces these via `QuadraticPenalty` against thresholds drawn from `hardware.py`; weights come from `stage2_config.yaml`.

---

## SIMSOPT fork

Banana coils use `CurveCWSFourierCPP` (the C++-backed CWS curve) and the CWS-frame manufacturability objectives — these live in a project fork of SIMSOPT, not upstream. Branch: `whjh/auglag_banana` on `hayashiw/simsopt`.
