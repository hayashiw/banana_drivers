import os

from banana_drivers.paths import INPUTS_DIR, WOUT_ORIGINAL
from banana_drivers.utils.cli import DEFAULT_WOUT_S, DEFAULT_WOUT_SCALE
from banana_drivers.utils.surface import build_surface

def main():
    surface = build_surface(WOUT_ORIGINAL, s=DEFAULT_WOUT_S, scale=DEFAULT_WOUT_SCALE)
    savefile = os.path.join(INPUTS_DIR, "surface_init.json")
    surface.save(savefile)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())