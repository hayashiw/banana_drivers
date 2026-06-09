import os

SOURCE_DIR         = os.path.dirname(os.path.abspath(__file__))
TOPLEVEL_DIR       = os.path.abspath(os.path.join(SOURCE_DIR, "../.."))
LOCAL_DIR          = os.path.join(TOPLEVEL_DIR, "local")
INPUTS_DIR         = os.path.join(LOCAL_DIR, "inputs")
BOOZERSURFACES_DIR = os.path.join(INPUTS_DIR, "boozersurfaces")
SURFACES_DIR       = os.path.join(INPUTS_DIR, "surfaces")
BIOTSAVARTS_DIR    = os.path.join(INPUTS_DIR, "biotsavarts")
OUTPUTS_DIR        = os.path.join(LOCAL_DIR, "outputs")
VMEC_DIR           = os.path.join(LOCAL_DIR, "vmec")
INIT_DIRS = (
    LOCAL_DIR,
    INPUTS_DIR,
    BOOZERSURFACES_DIR,
    SURFACES_DIR,
    BIOTSAVARTS_DIR,
    OUTPUTS_DIR,
    VMEC_DIR,
)

SRC_INPUTS_DIR        = os.path.join(SOURCE_DIR, "inputs")
WOUT_ORIGINAL         = os.path.join(SRC_INPUTS_DIR, "wout_original.nc")
BANANA_DOFS_INIT_FILE = os.path.join(SRC_INPUTS_DIR, "original.banana_dofs.yaml")

# DRIVERS_DIR           = os.path.join(SOURCE_DIR, "drivers")
