# Banana Drivers

Driver and utility scripts for optimization of **banana coils** in a **stellarator–tokamak hybrid** device.

## Overview

This repository contains:

- Optimization driver scripts (parameter scans, solver orchestration, restart handling)
- Pre/post-processing utilities (input generation, diagnostics, plotting, export)

The goal is to automate reproducible coil optimization workflows from baseline geometry to analyzed results.

---

## Repository Layout

```text
banana_drivers/
├── inputs/            # Generated run directories and results
└── README.md
```

---

## Prerequisites

- Python 3.10+ (recommended)
- `pip` or `conda`
- Access to required optimization/physics dependencies used by the scripts
- Optional: cluster scheduler tools (e.g., Slurm) for batch runs

---

## Typical Workflow

1. Prepare baseline coil/device inputs.
2. Define optimization targets and constraints in config files.
3. Launch driver script (local or cluster).
4. Monitor objective/constraint convergence.
5. Post-process fields, geometry, and performance metrics.
6. Export selected coil set and summary artifacts.

---

## Inputs and Outputs

### Inputs
- Baseline geometry/coils
- Device and solver parameters
- Objective weights and constraint definitions
- Scan ranges (optional)

### Outputs
- Per-run logs and checkpoints
- Optimized coil parameters/geometry
- Diagnostic metrics (cost, constraints, penalties)
- Plots and summary tables