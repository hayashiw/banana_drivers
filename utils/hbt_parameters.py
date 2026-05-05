# Various machine parameters for the High-beta tokamak -- extended pulse (HBT-EP) at Columbia University.

pi = 3.141592653589793
mu0 = 4.0e-7 * pi  # Vacuum permeability (T·m/A)

# This is the major radius that defines most of the toroidal geometry associated with HBT-EP.
HBT_MAJOR_RADIUS = 0.976 # m

# These are the major and minor radius of the vacuum vessel torus.
VACVES_MAJOR_R = HBT_MAJOR_RADIUS
VACVES_MINOR_R = 0.222

# These are the major and minor radius of the banana coil winding surface.
WINDSURF_MAJOR_R = HBT_MAJOR_RADIUS
WINDSURF_MINOR_R = 0.210 # original value = 0.215

# These are the major and minor radius of the toroidal field coils
TFCOIL_MAJOR_R = HBT_MAJOR_RADIUS
TFCOIL_MINOR_R = 0.4
N_TF_COILS = 20
TF_NFP = 1

# Toroidal field (TF) coil current is derived from B = 0.35 T at R = 0.92 m.
# TFCOIL_CURRENT = 80e3 # commented out for now since we can use config.yaml to vary the coil current
# TODO: Find TFCOIL_CURRENT necessary to produce 0.175 T at R = 0.92 m and conduct a scan.

# Vacuum toroidal field integral from the TF coil set:
#   rbtor = mu_0 * N_TF * I_TF / (2*pi)
# with N_TF = 20, I_TF = 80 kA. This is a device constant — the hardware TF
# current is fixed. Used by utils/vmec_resize.py (warm-start rbtor match) and
# the stage 1 cold-start branch (phiedge derivation).
TF_RBTOR = mu0 * 20 * 80.0e3 / (2.0 * pi)  # ≈ 0.32 T*m

# These values come from the shell inside HBT and defines the proposed LCFS limits.
TARGET_LCFS_MAJOR_R = 0.92 # original value = 0.9115, original wout was rescaled for 0.925.
TARGET_LCFS_MINOR_R = 0.15

# VF coil parameters
# 20 coils total
# 12 inside  (6 top + 6 bottom), R = 0.260350 m
#  8 outside (4 top + 4 bottom), R = 1.572133 m, Z = ±0.6505194 m
# VF current   : 1 kA per turn (adjustable up to TODO: ? kA per turn)
VF_RS = (
    0.260350, 0.260350, 0.260350, 0.260350, 0.260350, 0.260350,
    0.260350, 0.260350, 0.260350, 0.260350, 0.260350, 0.260350,
    1.572133, 1.572133, 1.572133, 1.572133,
    1.572133, 1.572133, 1.572133, 1.572133,
)
VF_ZS = (
     0.3508375,  0.3635375,  0.3762375,  0.3889375,  0.4016375,  0.4143375,
    -0.3508375, -0.3635375, -0.3762375, -0.3889375, -0.4016375, -0.4143375,
     0.6505194,  0.6505194,  0.6505194,  0.6505194,
    -0.6505194, -0.6505194, -0.6505194, -0.6505194,
)
VF_SIGN_CURR = ( # Sign of current with respect to the CCW toroidal direction
     1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1,
    -1, -1, -1, -1,
)

def compute_phiedge(rbtor, a, R0):
    """Compute VMEC phiedge from the vacuum toroidal field integral and geometry.

    For a zero-beta equilibrium with on-axis field B0 = rbtor / R0, the
    enclosed toroidal flux at the LCFS (circular cross-section approximation)
    is:

        phiedge = pi * a^2 * B0 = pi * rbtor * a^2 / R0

    Used by:
      - utils/vmec_resize.py : to rescale the warm-start seed's phiedge so
        its rbtor matches the hardware TF coil set.
      - 01_stage1_driver.py cold-start branch : to set phiedge self-consistently
        with the cold-start boundary geometry.

    Parameters
    ----------
    rbtor : float
        Vacuum toroidal field integral (T·m). Typically TF_RBTOR for this device.
    a : float
        Minor radius (m).
    R0 : float
        Major radius (m).

    Returns
    -------
    phiedge : float
        Enclosed toroidal flux at the LCFS (Wb).
    """
    return pi * rbtor * a * a / R0