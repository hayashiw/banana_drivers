import os

script_dir = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR = os.path.join(script_dir, "inputs")
HBT_INPUT_FILE = os.path.join(INPUTS_DIR, "input.hbt_finite_beta")
HBT_WOUT_FILE = os.path.join(INPUTS_DIR, "wout_hbt_finite_beta.nc")
HBT_COEFF_FILE = os.path.join(INPUTS_DIR, "hbt_finite_beta.ac_ai.csv")
