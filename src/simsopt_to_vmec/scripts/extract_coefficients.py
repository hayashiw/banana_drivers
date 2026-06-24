import argparse
import numpy as np
import os
import pandas as pd

from netCDF4 import Dataset

DEFAULT_POLY_DEG = 10

def build_parser():
    parser = argparse.ArgumentParser(description="Extract ac and ai coefficients from a VMEC wout netCDF4 file and save as a csv.")
    parser.add_argument("wout_file", type=str, help="Path to the VMEC wout netCDF4 file.")
    parser.add_argument("--poly-deg", type=int, default=DEFAULT_POLY_DEG, help="Degree of the polynomial fit for ac and ai coefficients.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the extracted coefficients.")
    return parser

def get_ac_ai_from_wout(wout_file, poly_deg=DEFAULT_POLY_DEG):
    with Dataset(wout_file, "r") as nc:
        ns = nc.variables["ns"][:]
        iota_full = nc.variables["iotaf"][:]
        buco_half = nc.variables["buco"][:][1:]
    s_full = np.linspace(0, 1, ns)
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)

    itor_half = 2*np.pi*buco_half / (4e-7*np.pi)

    ai = np.polyfit(s_full, iota_full, poly_deg)
    ac = np.polyfit(s_half, itor_half, poly_deg)

    # numpy polyfit returns coefficients in decreasing order
    # we want the dataframe row to correspond to the correct polynomial index
    # so row 0 is the 0th order term
    ac_ai_df = pd.DataFrame({
        "ac": ac[::-1],
        "ai": ai[::-1]
    })
    ac_ai_df.index.names = ["i"]

    return ac_ai_df

def main(argv=None):
    args = build_parser().parse_args(argv)
    wout_file = args.wout_file
    out_dir = args.out_dir
    print(f"Extracting ac and ai from {wout_file}")
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(wout_file).replace("wout_", ""))[0] + ".ac_ai.csv")

    ac_ai_df = get_ac_ai_from_wout(wout_file, poly_deg=args.poly_deg)
    ac_ai_df.to_csv(out_file)
    print(f"Extracted ac and ai coefficients saved to {out_file}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
