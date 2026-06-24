import argparse
import os

NUM_THREADS = 64
TRGT_NPHI = 65
TRGT_NTHETA = 64
SRC_NPHI = int(1.5*TRGT_NPHI)

def build_parser():
    parser = argparse.ArgumentParser(description="Compute virtual casing data from a VMEC equilibrium.")
    parser.add_argument("wout_file", type=str, help="Path to the VMEC netCDF file.")
    parser.add_argument("--num-threads", type=int, default=NUM_THREADS, help=f"Number of threads to use for OpenBLAS. Default: {NUM_THREADS}. Set to 1 to disable multithreading.")
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    wout_file = os.path.abspath(args.wout_file)
    num_threads = args.num_threads
    print(f"Running virtual casing")
    print(f"VMEC file: {wout_file}")
    print(f"Number of threads: {num_threads}")

    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    from simsopt.mhd import VirtualCasing

    vc_filename = wout_file.replace("wout", "vcasing")

    print(f"Making virtual casing")
    vc = VirtualCasing.from_vmec(wout_file, src_nphi=SRC_NPHI, trgt_nphi=TRGT_NPHI, trgt_ntheta=TRGT_NTHETA)
    vc.save(vc_filename)
    print(f"Saved virtual casing to {vc_filename}")

if __name__ == "__main__":
    raise SystemExit(main())