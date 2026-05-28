import os

PACKAGE_DIR        = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR         = os.path.join(PACKAGE_DIR, "inputs")
# UTILS_DIR          = os.path.join(PACKAGE_DIR, "utils")
DRIVERS_DIR        = os.path.join(PACKAGE_DIR, "drivers")

BIOTSAVART_INIT    = os.path.join(INPUTS_DIR, "biotsavart_init.json")
WOUT_ORIGINAL      = os.path.join(INPUTS_DIR, "wout_original.nc")
BANANA_INIT_DOFS   = os.path.join(INPUTS_DIR, "banana_init_dofs.txt")
STAGE2_CONFIG      = os.path.join(DRIVERS_DIR, "stage2_config.yaml")
SINGLESTAGE_CONFIG = os.path.join(DRIVERS_DIR, "singlestage_config.yaml")