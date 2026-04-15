# Notes on cold-starting stage 1 optimization for the HBT-EP banana coil tokamak-stellarator hybrid device

> **Status (2026-04-12):** PROCESSED. Implementation landed in
> `utils/near_axis_seed.py`, `utils/hbt_parameters.py` (`TF_RBTOR`,
> `compute_phiedge`), `utils/vmec_resize.py`, `01_stage1_driver.py`
> (cold-start branch rewritten, ns_array ramp, success gate relaxed),
> and `config.yaml` (`stage1` section: `cold_start: true`, `ns_array`,
> `max_mode_steps`/`vmec_mpol`/`vmec_ntor`/`boozer_mpol`/`boozer_ntor`
> ramped to step 4, `cold_start_volume` replaces `cold_start_a`,
> `cold_start_phiedge` removed, `tf_coils.rbtor` added). Validation
> (item 6 of Implementation order) is outstanding and will be run
> manually via `./submit.sh 01`.

## Motivation
A cold start with stage 1 serves two purposes:
1. **Enable the Pareto scan.** In our eventual Pareto scan, we want to sweep over volume, iota, coil length, and other objectives. Much of this requires going through the full three stage workflow with physics targets that differ from the seed wout. The warm-start branch re-optimizes from a single fixed seed and is poorly suited to re-targeting $(R_0, V, \iota)$ across a scan — the seed's shape, volume, and iota profile bias every run toward the same neighborhood. A cold start built from device constraints lets every Pareto point start from a boundary matched to its own $(R_0, V, \iota)$ targets with no inherited artifacts.
2. **Decouple stage 1 from the seed wout as an independent variable.** PLAN.md (2026-04-11) attributes the singlestage iota basin problem to the BoozerLS solver itself — the current Poincaré sweep and coilcap sweep confirmed the banana coils ARE capable of $\iota\approx 0.15$ at 16 kA, so the wrong-basin convergence is not a coil-physics or seed-geometry failure. Cold starting does not fix BoozerLS, but it removes the seed wout as a suspect by giving us a VMEC-diagnosable, BOOZ_XFORM-inspectable surface built entirely from device constraints. If BoozerLS still fails from a cold-started equilibrium, the solver is unambiguously the bottleneck; if it succeeds, we gain a cleaner handoff to stage 2/stage 3 for free.

## End goal
A robust stage 1 driver that can scan the relevant parameters and provide feasible plasma surfaces for stage 2 and stage 3 optimization. The surfaces should be well diagnosed to identify failure points in subsequent stages. A stage 1 driver already exists, our goal here is to update it to be more robust.

## Current cold-start branch state
The existing cold-start branch in [01_stage1_driver.py:159-174](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L159-L174) is a stub that builds a circular torus and skips several safeguards. Specifically:

- **Boundary**: `rc(0,0)=R_0`, `rc(1,0)=a`, `zs(1,0)=a` — pure circular torus with no helical content. Hits failure mode #3 (zero-iota trap) by construction.
- **Axis guess**: `raxis_cc/zaxis_cs` not set — defaults to whatever input.default provides, not self-consistent with the boundary. Hits failure mode #1 (ARNORM degeneracy) on stiff steps.
- **phiedge cache**: writes `vmec.indata.phiedge = COLD_PHIEDGE` but does NOT resync `local_full_x`. Hits failure mode #6 — phiedge silently reverts to 1.0 on first optimizer broadcast, producing $|B| \sim 12$ T outputs.

Everything downstream of the branch-specific setup (Boozer + Quasisymmetry construction, `_build_prob()`, resolution loop, diagnostics, BoozerSurface build) is already shared between warm and cold paths and does not need cold-start-specific changes. The scope of this work is therefore: replace the stub setup block with a near-axis-seeded boundary, override `raxis_cc/zaxis_cs` from the same near-axis solution, and add the `local_full_x` resync.

## Approach
In order for the cold-start to be robust, we would like to rely on our physical constraints and scan bounds as much as possible while avoiding anything inferred from the seed wout. This is to avoid inheriting bad artifacts and focusing the optimization on well-defined constraints. The ground truth for all device geometry is [utils/hbt_parameters.py](projects/hybrid_torus/banana/banana_drivers/utils/hbt_parameters.py), which [config.yaml](projects/hybrid_torus/banana/banana_drivers/config.yaml) inherits from. Here are the physical constraints and scan bounds for this optimization:

1. 20 fixed toroidal field (TF) coils operate with currents of 80 kA in each coil to produce an on-axis field of approximately 0.35 T at R = 0.92 m. This axis position is only a reference for our case and we don't need to adhere to guaranteeing an axis position at 0.92. The TF current is set in [config.yaml](projects/hybrid_torus/banana/banana_drivers/config.yaml) (`tf_coils.current: 80.0e+3`), derived from the ground truth in [utils/hbt_parameters.py](projects/hybrid_torus/banana/banana_drivers/utils/hbt_parameters.py).

2. Five field-period QA. We are optimizing for a five field-period quasi-axisymmetric configuration. Since this is designed for: (a) stellarator operation, (b) tokamak operation, and (c) hybrid tokamak-stellarator operation, QA is the obvious choice. Another way of thinking about it is we're aiming for a perturbed tokamak configuration. Additionally, HBT has 20 TF coils with a vacuum vessel that has a circular cross section so lower field periods would (a) collide with the vessel and (b) the banana coils would have odd alignment with the TF coils. With Nfp = 5, we get 10 banana coils (stellsym)
which aligns nicely with the 20 TF coils.

3. We are targeting a last-closed flux surface (LCFS) that is within the circular torus defined by a major radius of 0.92 m and minor radius of 0.15 m (Fourier coefficients $R_{1,0}^c = Z_{1,0}^s = 0.15$ m). These values come from the physical shell inside HBT and are defined in [utils/hbt_parameters.py](projects/hybrid_torus/banana/banana_drivers/utils/hbt_parameters.py) (`TARGET_LCFS_MAJOR_R`, `TARGET_LCFS_MINOR_R`). The axisymmetric torus geometry is defined thus:
$R = 0.92 + 0.15 \cos(\theta)$,
$Z = 0.15 \sin(\theta)$,
where $\theta$ is the poloidal angle. To be specific, this LCFS limit is the outermost LCFS that the plasma could have. This does not define a fixed target LCFS. Additionally, we will also want to scan over configurations with major radius closer to the vessel major radius which allows for the LCFS limit to have a larger minor radius. For configurations that align with the vessel major radius, the LCFS limit essentially has a similar minor radius to the banana coil winding surface.

4. Volume and iota targets. We are interested in scanning over volume and iota in order to have a comprehensive understanding of the possible physics targets. Increased volume and edge iota is attractive for a tokamak-stellarator hybrid. A simple circular torus with major radius of 0.92 m and minor radius of 0.15 m has a volume of $V = 2\pi^2 R_0 a^2 = 2\pi^2 \times 0.92 \times 0.15^2 \approx 0.41~\text{m}^3$.
Of course, we're not interested in a circular-cross section tokamak that fills the entire proposed LCFS but this does give us an upper bound. We could also surpass this threshold if we shift the LCFS major radius towards the vessel major radius of 0.976 m. For now, we'll suppose we fill less than a quarter of the volume so around $0.1~\text{m}^3$. For the iota target, we've previously used 0.15 but we should gather more information about what iotas are possible with the available parameter space. We will likely need to perform a scan over volume and iota to get a better grasp of what values work. Additionally, the volume scan is coupled to the major radius: as we fill larger volumes we need to push the plasma major radius closer to the vessel major radius in order to avoid collisions with the winding surface. This means we will need to complete the diag (iii) task for more information (see PLAN.md for the diag task list).

5. Banana coils and winding surface. The banana coils are defined in the optimization using a winding surface which greatly simplifies our treatment of the coils. We're restricting the coils to a winding surface represented by a simple torus with a major radius of 0.976 m (same as the HBT vessel major radius) and a minor radius of 0.210 m (HBT vacuum vessel is 0.222 m minor radius). These values are defined in [utils/hbt_parameters.py](projects/hybrid_torus/banana/banana_drivers/utils/hbt_parameters.py) (`WINDSURF_MAJOR_R`, `WINDSURF_MINOR_R`). This means that this winding surface is the limiting surface ahead of the vacuum vessel (normally, the vacuum vessel would be the limiting surface since it consists of the first wall near the plasma). Additionally, the banana coils have a hardware limit of 16 kA per coil. We could set the 16 kA as an inequality constraint and allow the current to decrease but based on previous results we need all the current we can get so we have fixed the current at 16 kA. While the banana coils don't directly contribute to stage 1, it is still helpful to keep in mind what will happen downstream.

## In the three stage workflow
The existing [01_stage1_driver.py](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py) already has a `cold_start` config flag that selects between a warm-start branch (loads a pre-resized seed wout via `utils/vmec_resize.py`) and a cold-start branch (programmatic boundary from `cold_start_R0`, `cold_start_a`, `cold_start_phiedge` in `config.yaml` — `cold_start_phiedge` is removed in Implementation order step 4). The current cold-start branch is a known-broken stub: it builds a circular torus with no helical content, doesn't override `raxis_cc/zaxis_cs`, and skips the `local_full_x` resync after writing `phiedge` — see failure modes #1, #3, and #6. This work replaces that stub with a near-axis-seeded boundary plus the warm-start branch's safeguards.

After this work, flip `cold_start: true` in the `stage1` section of `config.yaml` so that cold-start is the default path. Warm-start is retained as a fallback (set `cold_start: false` to use it). Both branches share the same downstream code (see "Current cold-start branch state"), so `stage1_boozersurface_opt.json` is feature-identical regardless of which branch ran.

## Relation to other utilities
The `phiedge` derivation used here is also used in the [utils/vmec_resize.py](projects/hybrid_torus/banana/banana_drivers/utils/vmec_resize.py) utility for resizing the seed wout via matching `rbtor`. Factor the one shared helper (`compute_phiedge`) out of `vmec_resize.py` into a shared location so both the warm-start preprocessing and the cold-start branch call the same function — see Implementation order step 2. No broader refactor of `vmec_resize.py` is in scope.

## Known failure modes

**1. ARNORM OR AZNORM EQUAL ZERO — axis degeneracy.** VMEC cannot determine a self-consistent magnetic axis because the boundary + axis guess are inconsistent. Encountered in stage 1 mpol=5 runs after rescaling (PLAN.md: "hit VMEC niter walls and spurious ARNORM OR AZNORM EQUAL ZERO axis degeneracies because the rescaled full-volume boundary was stiff"). Mitigations: provide a better raxis_cc/zaxis_cs initial guess (not just $R_0$, 0); lower initial mpol; soften the boundary shape; ensure the axis guess lies inside the boundary cross-section at every $\phi$. For cold-start the principled fix is to derive raxis_cc/zaxis_cs from the near-axis solution (see VMEC input construction step 6), which guarantees self-consistency with the seed boundary.

**2. FSQR > FTOLV at niter wall — force-residual convergence failure.** VMEC ran out of iterations before reaching the force tolerance. Job 51257661 hit this with FSQR=1.56e-10 just above FTOLV=1e-10 at niter=3000 because the rescaled equilibrium at high mpol needed more steps. Mitigations: bump niter_array[:]=10000 (already the stage 1 default per the fix applied in that job); check FSQR trend — if monotonically decreasing, more iterations help; if oscillating, the problem is stiffness, not budget. Also: VMEC's force residual has a squared-norm interpretation, so FTOLV=1e-10 is quite tight — can relax to 1e-9 at low-mpol steps in the ramp and tighten only at the final step.

**3. $\iota$ collapses to 0 on a near-axisymmetric boundary** — not a VMEC crash, but a silent zero-beta trap. With ncurr=0 and prescribed ai, if the boundary is axisymmetric (no modes with $n \neq 0$), VMEC's zero-beta equilibrium has $\iota\equiv 0$ regardless of what ai asks for — the optimizer cannot climb out because there's no $\partial \iota/\partial(\text{boundary shape})$ at the trivial point. Cold-start implication: the initial boundary MUST have nonzero non-axisymmetric modes (at least one $R_{m,n}^c$ or $Z_{m,n}^s$ with $n \neq 0$). Note that adding $(m=1, n=0)$ elongation alone does NOT escape the trap — an elongated axisymmetric torus still has $\iota\equiv 0$; you need genuine toroidal-angle dependence. Guard: $|\iota_\text{edge}|$ is reported in the diagnostics CSV and flagged by the post-processing metrics if it collapses below $\iota_\text{floor}$; manually inspect the first ramp step of any new scan configuration to catch the trap before committing to a full Pareto sweep.

**4. VMEC `ObjectiveFailure` mid-diagnostics** — simsopt raises `ObjectiveFailure` from inside a post-optimization `qs.J()` or `vmec.aspect()` call (as distinct from `least_squares_mpi_solve` itself). Job 51257661 crashed this way when step 3's diagnostics block re-evaluated VMEC on a state that was near a force-residual cliff. Mitigation already applied in [01_stage1_driver.py](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py): per-step diagnostics ([01_stage1_driver.py:369-387](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L369-L387)) and final-state diagnostics ([01_stage1_driver.py:421-440](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L421-L440)) are already wrapped in try/except ObjectiveFailure. Both warm and cold branches flow through this shared block — no cold-start-specific change needed.

**5. LeastSquaresProblem residual-dimension mismatch between resolution steps** — scipy-level crash, not a VMEC one strictly, but it fires during VMEC-based optimization and looks like a VMEC failure. Job 51256877 hit ValueError: operands could not be broadcast together with shapes (1492,) (3388,) because Quasisymmetry.J() returns one residual per Boozer mode and boozer.mpol/ntor changed between resolution steps. Mitigation already applied: _build_prob() is factored out and rebuilt inside the shared resolution-ramp loop ([01_stage1_driver.py:335](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L335)). Both warm and cold branches use the same resolution loop, so this is already covered for cold-start — no separate change needed.

**6. phiedge cached-DOF clobber** — `Vmec(mpi=mpi)` loads `input.default` at construction which sets `phiedge=1.0`, and the `Optimizable.__init__` caches `[1.0, 0.0, 1.0]` in `local_full_x`. If the driver subsequently overrides via `vmec.indata.phiedge = ...`, that write bypasses the DOF cache, and the next least_squares_mpi_solve broadcast restores the stale cached value — silently reverting `phiedge` to 1.0 and producing a wout with $|B|\sim 12$ T. Mitigation: after `vmec.indata.phiedge = ...`, force-sync `vmec.local_full_x = np.asarray(vmec.get_dofs())`. The warm-start branch applies this fix at [01_stage1_driver.py:204](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L204) (2026-04-08). The current cold-start branch does NOT — it sets `vmec.indata.phiedge = COLD_PHIEDGE` at [01_stage1_driver.py:165](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L165) but never resyncs the DOF cache. The updated cold-start branch must add the same resync after writing the freshly-computed phiedge.

**7. Line-search stall / zero step** — not a VMEC crash but a common downstream symptom. When VMEC is near a stiffness wall (e.g., helical mode trying to amplify beyond what the radial resolution supports), finite-difference gradients become noisy and the optimizer's line search fails with zero step. Symptoms: first-order optimality jumps by orders of magnitude between steps (step 2 of 51257661 reported 2.11e+18 vs. step 1's 5.31e-01); state freezes; residual barely changes. Mitigations: tighter VMEC convergence (FTOLV), lower step's mpol/ntor (to smooth gradients), or finer finite-difference step. Hard to prevent at the cold-start stage; worth logging first-order optimality per-step so the symptom is visible.

**8. input.default shadowing / working directory pollution** — VMEC reads input.<extension> files from the current working directory. Historically caused issues when a stale input.default was sitting around. Mitigation already applied: [01_stage1_driver.py](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py) creates a per-MPI-group tempdir at [01_stage1_driver.py:257-268](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L257-L268) and os.chdirs into it before the resolution loop, so VMEC writes its input.<ext> and fort.9 files into a clean isolated directory. Both warm and cold branches inherit this — no cold-start-specific change needed.

**9. Radial resolution (ns) too low for the targeted helical content** — VMEC converges but the $\iota$ profile is wrong because the radial grid can't resolve the helical perturbation's gradient. Symptom: $\iota_\text{edge}$ from VMEC disagrees with the prescribed ai by more than a few percent. Mitigation: resolution ramp ns_array=[13, 25, 51] (or higher terminal ns), as [utils/vmec_resize.py](projects/hybrid_torus/banana/banana_drivers/utils/vmec_resize.py) uses in the `stage1_resize` section of [config.yaml](projects/hybrid_torus/banana/banana_drivers/config.yaml). The stage 1 driver does NOT currently ramp `ns` — it only sets `niter_array[:] = 10000` at [01_stage1_driver.py:167](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L167). The cold-start implementation should add an `ns_array` ramp to the `stage1` config section (matching vmec_resize.py's `[13, 25, 51]` terminal resolution) and apply it per resolution step in the driver.

**10. Volume drift under QA optimization without a volume constraint** — not a VMEC failure, but a stage 1 pitfall observed in an early run (PLAN.md: "Volume drifted 0.577→0.643 without constraint — volume targeting now added"). VMEC conserves toroidal flux, not volume, so the QA optimizer can quietly reshape the boundary away from the target volume while improving QS. Mitigation: vmec.volume → volume_target is already in _build_prob() ([01_stage1_driver.py:243](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L243)). Both warm and cold branches use the same objective — already covered.

## Objectives
We are optimizing for (matching the existing warm-start stage 1 driver's `_build_prob()` in [01_stage1_driver.py](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py)):
1. **Boozer-QA (M=1, N=0)** — `Quasisymmetry.J()` evaluated on multiple flux surfaces (`qs_surfaces` in config, e.g. `[0.25, 0.5, 0.75]`). One residual block per surface; residual length scales with Boozer mpol/ntor, which is why the LeastSquaresProblem must be rebuilt at each resolution-ramp step (failure mode #5).
2. **Volume** — `vmec.volume → volume_target`. Required to prevent the QA optimizer from silently reshaping the boundary away from the target volume (failure mode #10).
3. **Aspect ratio** — `vmec.aspect → aspect_target`. In combination with the volume constraint, pins the major radius and prevents ballooning.
4. **Edge iota** — `vmec.iota_edge → iota_target`. **Edge only** — `iota_axis` is not targeted in the objective so the optimizer can preserve whatever magnetic shear the QA solution prefers. Note: since we run with `ncurr=0` and a prescribed degree-0 flat `ai` at `iota_target`, VMEC's output `iota_edge` is nearly pinned to the input, so this residual is close to trivially satisfied. It is retained as a sanity check (catches cases where VMEC silently fails to track the prescribed profile, e.g. under insufficient `ns`) and for consistency with the existing warm-start driver.

The existing warm-start driver gates success on `iota_edge` error < 10%. Cold-start removes this gate — VMEC convergence is the only in-driver success criterion; all physics metrics (iota, aspect, volume) are evaluated in post-processing (see Success metrics).

## VMEC input construction
Each VMEC input is derived from the constraints above with no reference to any seed wout. The derivation chain is:

1. **Pick scan-axis values**: $R_0$ (initial: 0.92 m from the baseline LCFS center, scannable up toward 0.976 m which is the vessel center), target volume $V$ (initial: $0.1~\text{m}^3$ for $R_0$ = 0.92 m, the max target volume is coupled to the $R_0$ value i.e. if $R_0$ = 0.92 then we can only scan volume up to what can fit inside the LCFS, if $R_0$ = 0.976 then the volume can scan up to the vessel volume), target iota $\iota_\text{target}$ (initial: 0.15, gated on diag (iii) results — see PLAN.md). These are the only free knobs the user / Pareto driver supplies (V and $R_0$ are coupled — see constraint #4).
2. **Derive minor radius** $a$ from $V$ and $R_0$: $a = \sqrt{V / (2\pi^2 R_0)}$ (circular cross-section approximation; refined once helical content is added). Start conservative, scan outward.
3. **Derive `rbtor`** from constraint #1: $\text{rbtor} = \mu_0 N_\text{TF} I_\text{TF} / (2\pi)$ with $N_\text{TF}=20$, $I_\text{TF}=80$ kA $\rightarrow$ 0.32 T·m.
4. **Derive `phiedge`** from `rbtor`, $R_0$, $a$: $\phi_\text{edge} \approx \pi \cdot \text{rbtor} \cdot a^2 / R_0$. This is the same formula [utils/vmec_resize.py](projects/hybrid_torus/banana/banana_drivers/utils/vmec_resize.py) uses for the warm path; per the "Relation to other utilities" section, you will factor it into a shared utility.
5. **Derive seed boundary Fourier coefficients** $(R_{m,n}^c, Z_{m,n}^s)$ from $(R_0, a, \iota_\text{target})$ via the near-axis expansion — see "Helical amplitude from near-axis expansion" below. The seed MUST have nonzero modes with $n \neq 0$ (non-axisymmetric content) or VMEC traps at $\iota \equiv 0$ (failure mode #3). Axisymmetric elongation alone is not enough.
6. **Derive `raxis_cc[0]`** from the near-axis solution rather than just setting it to $R_0$. This sidesteps failure mode #1 (ARNORM degeneracy) by giving VMEC a self-consistent axis guess from the start.
7. **Set fixed inputs**: `ai[0] = iota_target; ai[1:] = 0.0` (degree-0 flat profile — see Implementation order step 3 for the bug this form guards against), `ncurr = 0`, `lasym = False`, `pres_scale = am = 0`. Resolution ramps `mpol/ntor` and `ns_array/niter_array` match the existing stage 1 driver.

The table below is a quick-reference summary of the same chain.

| VMEC input | Source | Notes |
|---|---|---|
| `phiedge` | derived from `rbtor` = $\mu_0 N_\text{TF} I_\text{TF} / (2\pi)$ and boundary | $\phi \approx \pi \cdot \text{rbtor} \cdot a^2 / R_0$ (vmec_resize uses this) |
| `rbc[m,n]`, `zbs[m,n]` | full set from `stel.to_vmec()`, truncated to `mpol/ntor` | Modes and amplitudes come from the pyQSC near-axis solution — not hand-derived. VMEC uses field-period-indexed `n`, so on an NFP=5 device `n=1` is a 5-periodic helix in the lab frame. |
| `ai[0]` | `iota_target` (degree-0 flat; `ai[1:] = 0`) | Prescribed-iota profile under `ncurr=0`. Do NOT use `ai[:] = iota_target` — that gives a divergent profile. |
| `raxis_cc[0]` | near-axis solution (not just $R_0$) | Self-consistent axis guess; avoids failure mode #1 |
| `zaxis_cs[0]` | 0 | Stellarator-symmetric axis |
| `ncurr` | 0 | Zero-current (prescribed iota) |
| `lasym` | False | Stellarator symmetry |
| `pres_scale`, `am` | 0 | Zero beta |
| `mpol, ntor` | ramp [3, 4, 5, 7] with `max_mode_steps: [1, 2, 3, 4]` | Set these values in `config.yaml` `stage1` section |
| `ns_array, niter_array` | ramp | `niter_array[:] = 10000` per failure mode #2 |

### Helical amplitude from near-axis expansion
The seed boundary needs nonzero Fourier modes with $n \neq 0$ (non-axisymmetric content) to escape the zero-beta zero-iota trap (failure mode #3), and the size of the helical content controls the resulting on-axis rotational transform at first order in the distance from the axis. The principled way to invert "I want $\iota_\text{target}$" into "what helical amplitude should I seed?" is the **near-axis expansion** of Garren-Boozer / Landreman-Sengupta, which writes the on-axis rotational transform analytically as a function of axis shape and the leading boundary harmonic. For a rotating ellipse (axisymmetric major-axis circle plus an $(m=1, n=1)$ helical perturbation), the first-order construction gives a closed-form $\iota_0(\text{helical amplitude}, \text{ellipticity}, \text{axis curvature})$ that could in principle be inverted analytically to pick the seed amplitude. In practice we solve the full first-order sigma equation numerically via pyQSC rather than inverting the closed-form rotating-ellipse expression directly — the sigma-equation solve handles the ellipticity self-consistently and generalizes cleanly to non-ellipse cross-sections.

References:
- Landreman & Sengupta, *Direct construction of optimized stellarator shapes. Part 1. Theory in cylindrical coordinates*, J. Plasma Phys. 84(6) (2018) — [arXiv:1809.10233](https://arxiv.org/abs/1809.10233). Sets up the cylindrical-coordinate near-axis expansion that avoids the Frenet-Serret discontinuities.
- Landreman, Sengupta & Plunk, *Part 2. Numerical quasisymmetric solutions*, J. Plasma Phys. 85(1) (2019) — [arXiv:1809.10246](https://arxiv.org/abs/1809.10246). Practical numerical solutions; basis for [`pyQSC`](https://github.com/landreman/pyQSC). We use `pyQSC` directly rather than reimplementing these formulae.
- Landreman, *Constructing stellarators with quasisymmetry to high order*, J. Plasma Phys. 85(6) (2019) — [DOI:10.1017/S0022377819000783](https://doi.org/10.1017/S0022377819000783). Extends to higher orders if first-order accuracy is insufficient for our seed quality.

**Implementation: use pyQSC.** We use the [`pyQSC`](https://github.com/landreman/pyQSC) package (Landreman's implementation of the near-axis expansion) rather than implementing the first-order formulae directly. The sigma equation that determines $\iota(\bar\eta)$ involves subtle normalization conventions (B0/G0, arc-length factors, sign conventions) that are easy to get wrong — direct implementation was attempted and produced a factor-of-4 error in iota before the normalization issue was identified. pyQSC is a small, pure-Python, pip-installable dependency that handles all of this correctly.

**Practical recipe**: Given $(R_0, a, \iota_\text{target}, N_{fp})$, the cold-start seed is generated as follows:
1. Define a helical axis: `rc = [R0, delta]`, `zs = [0, -delta]` with `delta = 0.01 * R0` (~1% excursion). A non-planar axis gives nonzero torsion, which is the first-order driver of $\iota$ in QA near-axis theory — a planar axis yields $\iota = 0$.
2. Root-find `etabar` such that `Qsc(rc, zs, nfp, etabar).iota == iota_target`. The mapping $\iota(\bar\eta)$ is non-monotonic (rises then falls); take the lower crossing for a gentler (less elongated) seed.
3. Root-find the pyQSC `r` parameter such that the resulting boundary has `a_eff = (|RBC(1,0)| + |ZBS(1,0)|) / 2 = a_target`.
4. Extract `RBC/ZBS` via `stel.to_vmec(filename, r=r_opt)`. Truncate to the desired `mpol/ntor`.
5. Compute `phiedge = pi * rbtor * a^2 / R0` from device constraints (same formula as `vmec_resize.py`).
6. Set `raxis_cc = rc`, `zaxis_cs = [0, delta]` — self-consistent by construction.

This guarantees (a) nonzero helical content from the first VMEC call, (b) an iota that's already in the right ballpark before stage 1 optimization starts, (c) a self-consistent axis position to seed `raxis_cc/zaxis_cs` (sidestepping failure mode #1), and (d) automatic adaptation across the validated Pareto scan range (see "Validated range" below) with no hand-tuning.

**Key pyQSC parameters:**
- `etabar` ($\bar\eta$): controls the cross-section elongation. The elongation at each $\phi$ is approximately $\max(\bar\eta^2/\kappa^2, \kappa^2/\bar\eta^2)$. Values near $\kappa \approx 1/R_0$ give nearly circular cross-sections.
- `sigma` ($\sigma(\phi)$): the cross-section tilt angle, solved from the sigma equation. Small $|\sigma|$ means gentle shaping. The lower-etabar crossing typically has $|\sigma_\text{max}| < 0.2$ for our parameters.
- `r`: near-axis radial coordinate $r = \sqrt{2\psi/B_0}$. NOT the physical minor radius — related to it through the boundary shape.

**Validated range:** `near_axis_seed()` converges for $\iota \in [0.05, 0.25]$, $R_0 \in [0.90, 0.96]$ m, covering the full Pareto scan space. The baseline validation case is $R_0 = 0.92$ m, $V = 0.1~\text{m}^3$ (→ $a \approx 0.074$ m from $a = \sqrt{V/(2\pi^2 R_0)}$), $\iota = 0.15$. See [local/example_pyqsc_near_axis.py](projects/hybrid_torus/banana/banana_drivers/local/example_pyqsc_near_axis.py) for a detailed walkthrough of the mathematics and implementation.

**Fallback** if pyQSC is unavailable: hand-tune a small helical amplitude (e.g. 5-10% of the minor radius) for a single configuration. Since the in-driver hard gates no longer include an iota-floor check (iota is a post-processing metric), a fallback run that falls into the zero-iota trap will complete silently and only get flagged as infeasible by the post-processing script. Not suitable for Pareto scans (the required amplitude depends on all input parameters) — use only to bootstrap a single configuration, and manually inspect the first run's iota before trusting it.

## Success metrics
Since the stage 1 outputs are intended to be unfinished products that will be polished with single stage we don't need amazing convergence. However, we still need to make sure that the surface is a feasible point for stage 2 into stage 3. We should of course also guarantee that VMEC converges in stage 1. VMEC convergence is governed by ftol_array (per multi-grid step). VMEC reports success when the squared force residual FSQR + FSQZ + FSQL < ftol at the terminal ns step; on failure simsopt's Vmec.run() raises ObjectiveFailure. There is no separate "simsopt convergence" beyond "VMEC ran without raising and the wout exists."

**In-driver hard gates** (any one fails → run is marked as failed, but the driver still completes and saves whatever outputs it can):
- Vmec.run() returns without ObjectiveFailure at the terminal resolution step
- Terminal-step force residual satisfies fsqr < ftol_array[-1] (typical: 1e-10 to 1e-9; can relax intermediate steps to 1e-9 and tighten only the final)
- Boundary s=1 is well-defined (no NaN in wout.rmnc/zmns)

IMPORTANT: The driver should NOT abort on physics metric failures (iota, aspect, volume). In a Pareto scan, we want every run to complete and produce outputs regardless of whether the physics targets were met. The driver should log a SUCCESS/FAILURE banner based on the hard gates above, but always save the wout, diagnostics CSV, and BoozerSurface JSON.

**Post-processing quality metrics** (evaluated externally, NOT gated in the driver):
- $|\iota_\text{edge}|$ > $\iota_\text{floor}$ (e.g. $0.5 \times \iota_\text{target}$) — guards against the silent zero-beta trap from failure mode #3
- $|\iota_\text{edge} - \iota_\text{target}| / \iota_\text{target} < 0.01$
- $|\text{aspect} - \text{aspect\_target}| / \text{aspect\_target} < 0.01$
- $|\text{volume} - \text{volume\_target}| / \text{volume\_target} < 0.05$

These metrics should be reported in the per-run diagnostics CSV and the driver's FINAL STATE block. A post-processing script (recommended, not part of this prompt's scope) should read the diagnostics CSV from a Pareto scan, apply these thresholds, and label each run as "feasible" or "infeasible" for downstream stages.

The existing warm-start stage 1 driver's success gate ([01_stage1_driver.py:448](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L448)) uses `success = _final_ok and (iota_edge_err / IOTA_TARGET < 0.10)`. For the cold-start update, keep `_final_ok` (VMEC convergence) as the driver's success flag but remove the iota gating from the driver — move it to post-processing.

## Implementation order

1. **Create `utils/near_axis_seed.py`** — the `near_axis_seed(R0, a, iota_target, nfp)` function from the "Practical recipe" section. This is a self-contained module with no dependencies beyond pyQSC and scipy. Note: the utility takes minor radius `a` directly; the driver converts $V \to a$ via $a = \sqrt{V/(2\pi^2 R_0)}$ before calling, so `cold_start_volume` / `BANANA_VOLUME` is the user-facing knob while the utility keeps a physics-oriented signature. Testable independently: `python utils/near_axis_seed.py` should print the baseline seed ($R_0 = 0.92$ m, $V = 0.1~\text{m}^3 \Rightarrow a \approx 0.074$ m, $\iota = 0.15$). Reference: [local/example_pyqsc_near_axis.py](projects/hybrid_torus/banana/banana_drivers/local/example_pyqsc_near_axis.py) for the full mathematical walkthrough.

2. **Factor `compute_phiedge(rbtor, a, R0)` from `vmec_resize.py`** — extract this one helper into a shared location (e.g. `hbt_parameters.py` or a new `utils/vmec_helpers.py`). Both the cold-start branch and `vmec_resize.py` need it. `rbtor` itself is a device constant (20 × 80 kA TF coils → 0.32 T·m) and should just be stored as a named constant in `hbt_parameters.py` (or as `tf_coils.rbtor` in `config.yaml`) rather than recomputed — it only changes if the TF coil current scan is revisited. Update `vmec_resize.py` to import `compute_phiedge` from the shared module. This is also a good time to clean up `vmec_resize.py` to make it import-friendly (wrap the script body in `if __name__ == '__main__'`).

3. **Replace the cold-start stub in `01_stage1_driver.py`** ([01_stage1_driver.py:159-174](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L159-L174)) — wire up `near_axis_seed()` to produce the boundary, axis, and phiedge. Add the `local_full_x` resync (failure mode #6). Add `ns_array` ramp from config (failure mode #9). The cold-start branch should:
   - Call `near_axis_seed()` with config values and env var overrides (`BANANA_IOTA`, `BANANA_VOLUME`)
   - Set `vmec.indata.rbc`, `vmec.indata.zbs`, `vmec.indata.raxis_cc`, `vmec.indata.zaxis_cs`, `vmec.indata.phiedge`
   - Resync `vmec.local_full_x = np.asarray(vmec.get_dofs())`
   - Set `vmec.indata.niter_array[:] = 10000` (already in the stub)
   - Set `vmec.indata.ai[0] = iota_target; vmec.indata.ai[1:] = 0.0` (degree-0 flat iota profile — do NOT use `ai[:] = iota_target`, which would scalar-broadcast and produce a divergent $\iota(s) = \text{iota\_target}/(1-s)$)
   - Set `vmec.indata.ncurr = 0`

4. **Update `config.yaml`** — flip `cold_start: true`; add `ns_array: [13, 25, 51]` to `stage1` section; update `vmec_mpol/vmec_ntor` to `[3, 4, 5, 7]` with `max_mode_steps: [1, 2, 3, 4]`; update `aspect_target` and `volume_target` comments to note that cold-start derives these from $(R_0, V)$ rather than from the seed wout. Config-key cleanup for the cold-start path:
   - **Remove `cold_start_phiedge`** — phiedge is now computed inside the driver from `rbtor` and the seed boundary via `compute_phiedge()`. The old value is stale.
   - **Replace `cold_start_a` with `cold_start_volume`** (default $V = 0.1~\text{m}^3$) — the driver derives $a = \sqrt{V/(2\pi^2 R_0)}$ internally before calling `near_axis_seed()`. This makes the config-file knobs line up with the `BANANA_VOLUME` env var and the $(R_0, V, \iota)$ scan axes used throughout the document. `cold_start_R0` stays as-is (default 0.92 m) and is overridden by a scan via config edit or env var as needed.
   - **Add `tf_coils.rbtor: 0.32` (T·m)** as a named device constant (or equivalent in `hbt_parameters.py`) for use by both `compute_phiedge()` call sites.

5. **Update success gating** — modify [01_stage1_driver.py:448](projects/hybrid_torus/banana/banana_drivers/01_stage1_driver.py#L448) to gate only on VMEC convergence (`_final_ok`), not on iota error. Keep all physics metrics in the FINAL STATE block and diagnostics CSV for post-processing.

6. **Validation** — submit a cold-start stage 1 run at the baseline ($R_0 = 0.92$ m, $V = 0.1~\text{m}^3 \Rightarrow a \approx 0.074$ m, $\iota = 0.15$). Check: VMEC converges, iota is in the right ballpark, wout + BoozerSurface JSON are saved. Then feed into stage 2 to verify the full pipeline.

## New dependency

This implementation adds `pyQSC` ([github.com/landreman/pyQSC](https://github.com/landreman/pyQSC)) as a new dependency. It is pip-installable (`pip install qsc`), pure Python, and has no compiled extensions. It is only imported during the cold-start branch of stage 1 — the warm-start path and all other drivers are unaffected. A requirements documentation task is pending separately to capture all banana_drivers dependencies.