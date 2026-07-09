# Banana Drivers

Coil optimization drivers and shortlisted candidates for the banana coils of the HBT-EP stellarator–tokamak hybrid, built on [SIMSOPT](https://github.com/hiddenSymmetries/simsopt).

For collaborators, the main deliverable is [`candidates/`](#candidates) — no install required, just clone (or download) and browse.

## Candidates

`candidates/` on this branch (`main`) is the filtered shortlist of finalized coil sets and their diagnostics. The full unfiltered pool lives on the [`all-candidates`](../../tree/all-candidates) branch. Each candidate is a BoozerSurface JSON plus a set of files that share its basename and hold the visualizations and coil geometry.

**Per-candidate files** (`<stem>.*`):

| File | What it is |
|------|-----------|
| `<stem>.json` | BoozerSurface bundle (coils, target surface, solve state). |
| `<stem>.modB_Bdotn.png` | \|B\|, B·n, and coil cross-sections on the target surface. |
| `<stem>.poincare.png` | Poincaré field-line plot with the target surface overlaid. |
| `<stem>.coils/coil<i>.csv` | Per-coil geometry: position + tangent + normal + binormal along the curve (one CSV per banana coil). |

**Companion files:**

- `candidates/finitebuild/` — same candidates re-expanded to multifilament finite-build coils. Filenames add `_finitebuild` to the biotsavart tag; matching `.modB_Bdotn.png` plots sit alongside.
- `metrics_table.csv` (repo root) — one row per candidate summarizing geometric constraints, B·n, Boozer residual, and Poincaré confinement scores (survival fraction + confinement score).

### Filename tags

The stem itself decodes the object. Dot-separated groups:

```
<biotsavart_tag>.<surface_tag>.<boozersurface_tag>[.<other_tags>].json
```

- **biotsavart** — `<name>_proxy<I>kA_o<order>_<stage>` (+`_finitebuild` when expanded to filaments).
- **surface** — `<name>_m<mpol>_n<ntor>_<stage>` (+`_presolved` when the boozer solve has been re-run and cached).
- **boozersurface** — `boozersurface_cw<constraint_weight>_vol<volume_target>` (`cwExact` for BoozerExact; `50%` → `50pct`, `0.5` → `0d5`).
- **other** — optional `vers<N>`, `iter<N>`.

Parsers/generators live in `utils/tags.py`.

## Optimization pipeline

For running new optimizations, install the package (below), then:

| Step | Command | Purpose |
|------|---------|---------|
| Stage 2 | `banana-stage2 <boozersurface.json>` | Coil-only optimization on a fixed surface. |
| Singlestage | `banana-singlestage <boozersurface.json> <iota>` | Joint coil + surface optimization via BoozerLS. |

Both drivers accept CLI overrides for coil DOFs, surface resolution, and per-objective weights/thresholds. Same option names also work via `--config-file <yaml>`; CLI wins. Every run writes its resolved inputs to `<output>.inputs.yaml` for reproducibility.

Threaded variants: `banana-singlestage-multithread` and `-openblas`.

## Install

```bash
git clone <this repo>
cd banana_drivers
pip install -e .
```

Requires Python 3.10+ and the banana SIMSOPT fork (branch `whjh/auglag_banana` on `hayashiw/simsopt`), which supplies `CurveCWSFourierCPP` and the CWS-frame manufacturability objectives. All `banana-*` shell commands become available after install.

`hardware.py` holds the canonical engineering limits (coil counts, currents, geometric thresholds, wire radius, finite-build filament layout) — always the source of truth over what's written here.

## CLI commands

All accept `--help`:

- **Optimization** — `banana-stage2`, `banana-singlestage`, `banana-singlestage-multithread`, `banana-singlestage-multithread-openblas`
- **Object construction / manipulation** — `banana-initialize-local`, `banana-initialize-boozersurface`, `banana-run-boozer-solve`, `banana-change-parameters`, `banana-convert-tags`, `banana-flip-surfacerzfourier`, `banana-generate-boozersurface-finitebuild`
- **Post-processing** — `banana-run-poincare-trace`, `banana-run-virtual-casing`, `banana-plot-modb-bdotn`, `banana-print-parameters`, `banana-generate-coil-txt-files`, `banana-find-iota-basin`, `banana-tabulate-metrics`
- **VMEC integration** (via co-installed `simsopt_to_vmec`) — `banana-simsvmec-make-input`, `banana-simsvmec-extract-coefficients`
