from dataclasses import dataclass

@dataclass(frozen=True)
class HBTVacuumVessel:
    major_radius: float = 0.976 # meters
    minor_radius: float = 0.222 # meters

@dataclass(frozen=True)
class HBTShell:
    major_radius: float = 0.92 # meters
    minor_radius: float = 0.15 # meters

@dataclass(frozen=True)
class HBTTFCoils:
    major_radius: float = 0.976 # meters
    minor_radius: float = 0.400 # meters
    n_coils: int = 20
    nfp: int = 1
    order: int = 1
    stellsym: bool = False

@dataclass(frozen=True)
class BananaWindingSurface:
    # The original major and minor radii were 0.976 m and 0.210 m.
    # These values were used up until May 26, 2026.
    major_radius: float = 0.903 # meters
    minor_radius: float = 0.142 # meters
    nfp: int = 5
    stellsym: bool = True

@dataclass(frozen=True)
class HardwareLimits:
    tf_current_ka_limits: tuple[float, float] = (-80.0, 0.0) # kiloAmperes
    banana_current_ka_limits: tuple[float, float] = (0.0, 16.0) # kiloAmperes
    banana_curv_p: int   = 4
    max_curvature: float = 100.0 # 1/meters
    max_length: float    = 1.900 # meters
    # The original minimum coil-coil distance was 0.05 m or 5 cm.
    # This value was used up until May 26, 2026.
    min_ccdist: float    = 0.0462 # meters
    # The original minimum coil-plasma distance was 0.015 m or 1.5 cm.
    # This value was used up until May 26, 2026.
    min_csdist: float    = 0.010 # meters

@dataclass(frozen=True)
class HBTVFCoils:
    rs: tuple[float, ...] = ( # meters
        # inner ring (12)
        0.260350, 0.260350, 0.260350, 0.260350, 0.260350, 0.260350,
        0.260350, 0.260350, 0.260350, 0.260350, 0.260350, 0.260350,
        # outer ring (8)
        1.572133, 1.572133, 1.572133, 1.572133,
        1.572133, 1.572133, 1.572133, 1.572133,
    )
    zs: tuple[float, ...] = ( # meters
         0.3508375,  0.3635375,  0.3762375,  0.3889375,  0.4016375,  0.4143375,
        -0.3508375, -0.3635375, -0.3762375, -0.3889375, -0.4016375, -0.4143375,
         0.6505194,  0.6505194,  0.6505194,  0.6505194,
        -0.6505194, -0.6505194, -0.6505194, -0.6505194,
    )
    sign_curr: tuple[int, ...] = ( # +1 CCW phi, -1 CW phi
         1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1,
        -1, -1, -1, -1,
    )

    def __iter__(self):
        return zip(self.rs, self.zs, self.sign_curr)

hbt_vv          = HBTVacuumVessel()
hbt_shell       = HBTShell()
hbt_tf          = HBTTFCoils()
hbt_banana_ws   = BananaWindingSurface()
hbt_vf          = HBTVFCoils()
hardware_limits = HardwareLimits()

N_TF       = hbt_tf.n_coils
N_BANANA   = 2 * hbt_banana_ws.nfp
TF_IDX     = 0
BANANA_IDX = N_TF
PROXY_IDX  = N_TF + N_BANANA
VF_IDX     = N_TF + N_BANANA + 1
N_COILS    = VF_IDX + len(hbt_vf.rs)

# Default values but not hardware constraints
DEFAULT_BANANA_ORDER = 3
DEFAULT_PROXY_RZ = (0.925, 0.0)
