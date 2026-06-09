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

Python 3.10+ is required. The deps in `pyproject.toml` (`numpy`, `scipy`, `matplotlib`, `pyyaml`) are the package's own; SIMSOPT (with the banana-specific CWS curve and CWS-frame objectives) must be installed separately — see [SIMSOPT fork](#simsopt-fork) below.

After install, all CLI entry points listed below are available as `banana-*` shell commands.

---

## Pipeline

Both optimization drivers read and write a single **`BoozerSurface` JSON**, which bundles a `BiotSavart` (the coil set) and a `Surface`. Stage 2 output therefore feeds directly into singlestage.

| Step | Command | What it does |
|------|---------|--------------|
| 1. Initialize | `banana-initialize-local` | Creates the `local/` working tree and writes the original `BiotSavart`, `Surface`, and `BoozerSurface` JSONs (target surface from `inputs/wout_original.nc` at $s=0.24$, scaled to $R_0=0.925$ m; default coil set). |
| 2. Stage 2 | `banana-stage2 <boozersurface.json>` | Coil-only optimization on a fixed surface: `SquaredFlux` + geometry/manufacturability penalties over the banana coil DOFs (and, optionally, currents). |
| 3. Singlestage | `banana-singlestage <boozersurface.json> <iota>` | Joint coil + surface optimization via a `BoozerSurface` (BoozerLS, or BoozerExact when `--constraint-weight 0`): drives rotational transform toward `iota` and minimizes the non-quasisymmetric ratio while holding the manufacturability penalties. |

Run from inside `local/outputs/` so artifacts stay out of the source tree (the drivers default `--out-dir` to the current directory).

### Driver options

Both drivers share a large set of optional flags built from `utils/cli.py`:

- **Coil overrides** — `--tf-current-ka`, `--banana-current-ka`, `--proxy-current-ka`, `--vf-current-ka`, their `--*-fix-current` toggles, `--banana-order`, `--banana-qpts-per-order`, `--proxy-rz`. Unset values inherit from the input `BoozerSurface`.
- **Surface resolution** — `--mpol`, `--ntor`, `--nphi`, `--ntheta` (inherit when unset).
- **Objective targets/weights** — per objective `--max-*` / `--min-*` bounds and `--weight-*` weights (coil length, curvature, coil–coil distance, poloidal extent, ellipse width, coil self-distance, currents; plus `--weight-sqflux` for stage 2 and `--weight-non-quasisymmetric-ratio` / `--weight-boozer-residual` / `--weight-iota` / `--*-coil-surface-distance` for singlestage). Defaults live in `objectives/defaults.py`.
- **Run control** — `--maxiter`, `--save-iter-dir`, `--save-iter-freq`, `--out-dir`, `--config-file`.
- **Singlestage only** — positional `iota`, `--sign-g`, `--constraint-weight`, `--volume-target-str`.
- **Stage 2 only** — `--vcasing-file` (virtual-casing $B\cdot\hat{n}$ target for finite-current runs).

A YAML passed to `--config-file` may set any of the same option names; explicit CLI flags take priority over the config file. Every run also writes its fully-resolved inputs to `<output>.inputs.yaml` for reproducibility. See `banana-stage2 --help` / `banana-singlestage --help` for the complete list.

`banana-singlestage-multithread` and `banana-singlestage-multithread-openblas` are thin wrappers that set OpenMP/BLAS thread environment variables and `exec` the singlestage driver.

---

## Filenames and tags

Inputs and outputs are named by a **dot-separated tag scheme** (`utils/tags.py`) so a filename fully describes the object:

```
<biotsavart_tag>.<surface_tag>.<boozersurface_tag>[.<other_tags>].json
```

For example:

```
original_proxy0d0kA_o6_stage2opt.original_m8_n8_stage2opt.boozersurface_cw100d0_volSurface.vers0.json
```

- **biotsavart** — `<name>_proxy<I>kA_o<order>_<stage>` (`+_finitebuild` for finite-build sets).
- **surface** — `<name>_m<mpol>_n<ntor>_<stage>`.
- **boozersurface** — `boozersurface_cw<constraint_weight>_vol<volume_target>` (`cwExact` for BoozerExact, `volSurface` for the surface's own volume; a percentage `50%` is stored as `50pct`, a decimal `0.5` as `0d5`).
- **other** — optional `vers<N>` / `iter<N>`. Version numbers branch as `<N>_<M>` for re-runs from an earlier lineage.

`resolve_*_json_filename` parses a name back into a tag dict, `generate_*` rebuilds names from one, and `generate_version_number` picks the next free version. `banana-convert-tags` rewrites a legacy-named SIMSOPT JSON into this scheme (current hardware layout only; pass `--biotsavart-stage` / `--surface-stage`).

---

## CLI entry points

All `banana-*` commands accept `--help`.

| Command | Purpose |
|---------|---------|
| `banana-stage2` | Stage 2 coil-only optimization. |
| `banana-singlestage` | Singlestage joint coil + surface optimization. |
| `banana-singlestage-multithread` / `-openblas` | Threaded singlestage launchers (set thread env, exec the driver). |
| `banana-initialize-local` | Set up the `local/` tree and write the original BiotSavart / Surface / BoozerSurface inputs. |
| `banana-initialize-boozersurface` | Assemble and run-code a `BoozerSurface` from a `BiotSavart` + a `Surface`. |
| `banana-convert-tags` | Rewrite a legacy SIMSOPT JSON filename into the tag scheme. |
| `banana-generate-boozersurface-finitebuild` | Build a finite-build (multifilament) `BoozerSurface` from a filament one. |
| `banana-generate-coil-txt-files` | Export coil geometry to text files. |
| `banana-plot-modB-Bdotn` | Plot $\lvert\mathbf{B}\rvert$, $\mathbf{B}\cdot\hat{\mathbf{n}}/\lvert\mathbf{B}\rvert$, and coil cross-sections for a `BoozerSurface`. |
| `banana-print-parameters` | Pretty-print coil / surface parameters. |
| `banana-run-virtual-casing` | Run a virtual-casing calculation from a VMEC wout. |
| `banana-flip-surfacerzfourier` | Flip a `SurfaceRZFourier` toroidally ($\phi \to -\phi$) via DOF relabel. |

---

## Package layout

```
banana_drivers/
├── pyproject.toml          # build + entry points
├── README.md               # this file
├── src/banana_drivers/
│   ├── hardware.py         # frozen dataclasses: HBT-EP geometry, hardware limits
│   ├── paths.py            # resolved paths to inputs/ and local/
│   ├── drivers/
│   │   ├── stage2.py            # Stage 2: coil-only optimization
│   │   └── singlestage.py       # Singlestage: joint coil + surface
│   ├── objectives/
│   │   ├── build.py             # build_objective — shared objective assembly (both stages)
│   │   ├── defaults.py          # default objective weights and targets
│   │   ├── cwsobjectives.py     # CWS-frame penalties (PoloidalExtent, ProjectedEllipseWidth, CurveSelfIntersect)
│   │   └── currentobjectives.py # ScaledCurrentWrapper for current penalties
│   ├── utils/
│   │   ├── cli.py          # driver_parser, process_driver_args, per-objective parsers
│   │   ├── tags.py         # filename tag system (resolve / generate / version)
│   │   ├── stages.py       # stage name constants
│   │   ├── boozersurface.py # build / rebuild / load BoozerSurface
│   │   ├── biotsavart.py   # coil-set construction
│   │   ├── surface.py      # surface load / build / rescale / resolution
│   │   ├── coils.py, framedcurve.py, finitebuild.py  # coil + finite-build helpers
│   │   ├── plot.py         # |B|, B·n, cross-section plotting
│   │   └── io.py           # DriverLog + save_to_json (tag-based filenames)
│   ├── scripts/            # one-shot utilities (one CLI per file)
│   └── inputs/             # checked-in canonical inputs (see below)
└── local/                  # gitignored — working files and run outputs
```

`local/` is reserved for the user's working files (modified inputs, scratch outputs) and is gitignored via the `local*/` rule.

---

## Configuration

There is no mandatory per-driver config file. Objective **weights and targets** have defaults in `objectives/defaults.py` and are overridable per-objective on the CLI (`--weight-*`, `--max-*`, `--min-*`). A `--config-file <yaml>` may supply any of the same option names as a YAML block, with explicit CLI flags taking priority.

Engineering **thresholds and limits** (max length/curvature, min coil–coil and coil–surface distances, current bounds) live in `hardware.py`, not in config files — see [Hardware](#hardware).

---

## Inputs

`src/banana_drivers/inputs/` holds the canonical, version-controlled inputs shared across runs:

| File | Purpose |
|------|---------|
| `wout_original.nc` | VMEC wout the target plasma surface is extracted from (at $s=0.24$, scaled to $R_0=0.925$ m). |
| `wout_hbt_finite_beta.nc` | Finite-$\beta$ VMEC wout for the finite plasma-current branch. |
| `input.hbt_finite_beta`, `input.hbt_finite_beta_fixed` | VMEC input files used to produce the finite-$\beta$ wout. |
| `HBT_101393_01.eqdsk` | EFIT g-file (shot 101393); reference equilibrium for the finite-current scenario. |
| `original.banana_dofs.yaml` | Default banana coil CWS Fourier DOFs (simple ellipse). |

Run-derived inputs (the original `BiotSavart` / `Surface` / `BoozerSurface` JSONs) are produced by `banana-initialize-local` and written into `local/inputs/`.

---

## Hardware

`src/banana_drivers/hardware.py` defines frozen dataclasses for the HBT-EP geometry and the engineering limits the drivers enforce. Quick reference (always cross-check the source for the live values):

- **TF coils**: $R_0 = 0.976$ m, $a = 0.400$ m, 20 coils at $-80$ kA each (fixed).
- **Banana winding surface**: $R_0 = 0.903$ m, $a = 0.142$ m, nfp=5, stellsym. Banana coil count $= 2\,\mathrm{nfp} = 10$.
- **Banana current limit**: $\lvert I_b \rvert \le 16$ kA per coil.
- **Geometric limits**: `max_length = 1.9` m, `max_curvature = 100` m$^{-1}$ (≈1 cm bending radius), `min_ccdist = 0.0462` m, `min_csdist = 0.010` m. Curvature uses an $L^4$ norm (`banana_curv_p = 4`).
- **Target plasma**: $R_0 = 0.92$ m, nfp=5, stellsym.
- **Full coil set**: 51 coils (20 TF + 10 banana + 1 proxy + 20 VF); indices via `TF_IDX`/`BANANA_IDX`/`PROXY_IDX`/`VF_IDX` in `hardware.py`.

The drivers enforce these via `QuadraticPenalty` against thresholds drawn from `hardware.py`; weights come from `objectives/defaults.py` (overridable on the CLI).

---

## SIMSOPT fork

Banana coils use the C++-backed CWS curve and the CWS-frame manufacturability objectives, which live in a project fork of SIMSOPT, not upstream. Branch: `whjh/auglag_banana` on `hayashiw/simsopt`.
