"""Resolve the output directory for banana drivers.

Priority:
  1. BANANA_OUT_DIR env var (for Pareto scans or manual override)
  2. $SCRATCH/banana_drivers_outputs/ (if $SCRATCH exists and is writable)
  3. ./outputs/ (fallback when scratch is unavailable)
"""

import os

SCRATCH_SUBDIR = "banana_drivers_outputs"
LOCAL_SUBDIR = "outputs"


def resolve_output_dir():
    """Return the absolute path to the output directory, creating it if needed."""
    # 1. Explicit override
    env_dir = os.environ.get("BANANA_OUT_DIR")
    if env_dir:
        out = os.path.abspath(env_dir)
        os.makedirs(out, exist_ok=True)
        return out

    # 2. Scratch
    scratch = os.environ.get("SCRATCH") or os.environ.get("PSCRATCH")
    if scratch:
        candidate = os.path.join(scratch, SCRATCH_SUBDIR)
        try:
            os.makedirs(candidate, exist_ok=True)
            # Verify writable
            test_file = os.path.join(candidate, ".write_test")
            with open(test_file, "w") as f:
                f.write("")
            os.remove(test_file)
            return os.path.abspath(candidate)
        except OSError:
            pass  # fall through to local

    # 3. Local fallback
    out = os.path.abspath(LOCAL_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out
