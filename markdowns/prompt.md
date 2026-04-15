# Instructions for Claude to proceed with revising driver scripts in banana_drivers
For context, we are working primarily in the banana_drivers/ repository.
Make sure to review the work being done in qi_rso/qi_drivers/.
It is important to point out that this is a separate project from qi_drivers we cannot assume the problem is the same.
However, general practices should be duplicated such as workflow pattern, format styles, etc.
While this banana_drivers system is different from qi_drivers, it may still be useful to review the prompt.md file in qi_drivers/.
Review the qi_drivers prompt file and see how Claude behaved in responding to that prompt.
There may be additional context in memory as well as the CLAUDE.md file.

## Definitions
ALM: Augmented Lagrangian method
TF: Toroidal field
VF: Vertical field
DoF: Degree-of-freedom

## Additional context
The work done here is a continuation/parallel to the work done on the `accessibiility` branch on the SIMSOPT fork `jhalpern30/simsopt`.
The stage-2 script and single-stage script in examples/single_stage_optimization/ are able to successfully produce vacuum-field results.
While those scripts are good examples, we want driver scripts that match our preferred format and configurations.

## Hardware constraints
Toroidal field coil current = 80 kA
Maximum banana coil current = 16 kA

## Main prompt
The basic drivers for single-stage Boozer surface optimization of banana coils for a stellarator-tokamak hybrid have become a bit convoluted.
In order to properly debug the script and move forward with developing the full workflow we will need to conduct a few experiments as well as clean up our driver scripts to be more human-readable.
**The overall goal is to produce a realizable set of banana coils for a stellarator-tokamak hybrid that is capable of (a) pure stellarator scenarios and (b) finite-current scenarios.**
Finite-current effects are approximated using a proxy coil in SIMSOPT along with VF coils.
This has been implemented for Boozer surfaces and mostly tested but we haven't validated convergence of a finite-current case.
The coils are constrained according to real engineering and hardware restrictions.
It cannot be stressed enough, these coil designs are intendend to be manufactured.

### Zeroth point: Check my work
After reviewing the full contents of this file, provide an analysis of what is being prompted.
Are there things I haven't considered?
Are there things I should be concerned with?

### First point: Baseline for our baseline
First, we need a baseline for our baseline.
The zero-current optimization is supposed to be the baseline for the finite-current optimization as well as the baseline for the pure stellarator scenario.
However, the Boozer surface single stage optimization have become quite involved even in its most basic form.
Our first goal is to ensure we can achieve a straight forward convergent solution with the simplest inputs.
*Can we get a low-resolution case to converge nicely?*
Ideally we could just run the example scripts and achieve something reasonable.
However, the example scripts use different coil currents than what we are interested in.
The example TF coil current is 100 kA which is 25% higher than our hardware value of 80 kA.
Additionally, the banana coils are 10 kA which is lower than our maximum hardware value of 16 kA.
Attempts at running the optimization with the actual hardware values has proven unsuccessful so far.
We've attempted quite a few "tricks" such as resolving the VMEC surface to fit the target boundary surface (s=0.24 translated to s=1) and using Booz_xform to ensure the initial surface DoFs are close to Boozer coordinates.
We have also tried scaling the initial banana current to be proportional to the TF coil current so that the 100 - 80 kA reduction is reflected in the banana coils as 10 kA - 8 kA.
So far we have not had great success.
We need to essentially rebuild from scratch and have the most simple case that can converge.

### Second point: Stage 2 optimization to initialize single stage
In order to have a good starting point, even for the most simple single stage case, we should ensure good results from stage 2 optimization.
Stage 2 is critical for setting up the single stage case since we need something resembling banana coils to start.
At some point, we also want to do a volume and iota scan as part of a Pareto scan.
Would this require re-doing stage 2 for each step?
What would the ideal workflow for scanning parameters be?
Something like [re-solve VMEC --> re-do stage 2 --> run single stage] for each case?
This also depends on how we decide to handle the VMEC initial surface which will be discussed below.

### Third point: KISS
Keep it simple stupid.
If it does not need to be complicated, don't make it more complicated.
If something is implemented and it doesn't work, document detailed information about the failure then remove it from the workflow to declutter the space.
Of course, refer back to the previous points on comments.
We obviously cannot guarantee that we can keep it simple but it is good to keep in mind.

### Fourth point: Pareto front
A near term next step once we have a baseline is to build out a Pareto front.
We need to know what our optimal targets should be.
It will also be useful to scan over the proxy plasma current and the VF coil current.
It also has a secondary purpose of producing a database of converged solutions that can be part of a database study.

### Fifth point: consistent formatting
In order to ensure the scripts are human-readable and the outputs are also human-readable, driver scripts should have consistent formatting.

#### Output logs
A combination of output methods has been implemented in qi_drivers/.
Review the work done by Claude in qi_drivers regarding this point.
It may help to review the prompt.md file in qi_drivers/.

#### Naming conventions
In order for SIMSOPT objects to be easily readable, we should use full names i.e. `boozersurface`, `biotsavart`, `surface`, etc.
This is already implemented in some places but not updated in other places.

In order to differentiate driver scripts, output directories, and output files, there should be a naming convention regarding prefixes.
There is some implementation of this with the use of `singlestage` and `stage2`.
Note that I find it more helpful to see `singlestage` instead of `single_stage` and this applies to other cases.
The reason is that when it's used as a prefix, additional unneccessary underscores may confuse parts of the prefix with other tags in the file or directory names.
For example, I find `singlestage_unperturbed_auglag` to be more informative than `single_stage_unperturbed_augmented_lagrangian`.
Each part of the prefix acts almost like a hierarchy.

#### Comments
Ideally I would like to keep our workflows as simple as possible.
Considering how complex the problem is, this may be near impossible.
On the other hand, if the code is well documented with comments, docstrings, and typehints then it will be easier to follow.
A general rule of thumb:
*The more complicated a piece of code, the more documentation it will need to be understandable.*

### Sixth point: Poincare tracing
Ensuring successful optimization requires Poincare tracing and validation of the flux surfaces.
Build out a robust Poincare tracing driver script.
It is very important to treat the finite-current Poincare tracing in a special way.
Previous attempts to run Poincare tracing for a finite current case has timed out.
One hypothesis is that there is a singularity near the proxy coil i.e. at the magnetic axis.
There should be a stopping criterion to avoid going near the proxy coil.

### Seventh point: Parallelization on Perlmutter.
We are working on the Perlmutter Cray supercomputer at NERSC.
This is a powerful machine with excellent methods for parallelization.
This is especially helpful for the Poincare tracing and future stochastic optimization.
We should ensure any slurm scripts make full use of these features.
For example, Poincare tracing is parallelized over fieldines.
If we are tracing 30 fieldlines, we could run all 30 traces in parallel.
We also want to be careful to not waste compute hours.
Any and all slurm submission will be performed manually by myself to ensure control over how much compute time we use.
Since we are only solving for one boozer surface there may not be a way to parallelize the single stage runs.
Double check this.
There may be slurm scripts that run the unperturbed case with 8 CPUs.
If my intuition about the inability to parallelize a optimizing one Boozer surface then that means 7 CPUs are wasted which is not good.

### Eight point: Document, document, document
In addition to documenting progress for my sake, it's also important to keep progress for future Claude.
Regularly update CLAUDE.md, PLAN.md, agent files, and memory as needed.

Additionally, I am curious about agentic work.
If you create new, permanent or temporary agents, describe that process and document it.