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
class BananaCoilFiniteBuild:
    offset_from_horizontal: float = 0.00231 # meters = 2.31 mm
    offset_from_vertical: float = 0.00135 # meters = 1.35 mm
    horizontal_spacing: float = 0.00531 # meters = 5.31 mm
    numfilaments_n: int = 2
    numfilaments_b: int = 7
    numfilaments: int = numfilaments_n * numfilaments_b

DEFAULT_TF_CURRENT_KA = -80.0 # kiloAmperes
DEFAULT_BANANA_CURRENT_KA = 16.0 # kiloAmperes
tf_current_min = min(0, DEFAULT_TF_CURRENT_KA)
tf_current_max = max(0, DEFAULT_TF_CURRENT_KA)
banana_current_min = min(0, DEFAULT_BANANA_CURRENT_KA)
banana_current_max = max(0, DEFAULT_BANANA_CURRENT_KA)
@dataclass(frozen=True)
class HardwareLimits:
    tf_current_ka_limits: tuple[float, float] = (tf_current_min, tf_current_max) # kiloAmperes
    banana_current_ka_limits: tuple[float, float] = (banana_current_min, banana_current_max) # kiloAmperes
    banana_curv_p: int   = 4
    max_curvature: float = 100.0 # 1/meters
    max_length: float    = 1.900 # meters
    # The original minimum coil-coil distance was 0.05 m or 5 cm.
    # This value was used up until May 26, 2026.
    min_ccdist: float    = 0.0462 # meters
    # The original minimum coil-plasma distance was 0.015 m or 1.5 cm.
    # This value was used up until May 26, 2026.
    min_csdist: float    = 0.010 # meters

    # Short width is the shortest width of the coil in 3-D Cartesian space
    # i.e. the width necessary to fit through the vacuum port
    max_short_width: float = 0.197 # meters

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

# Empirical value from HBT-EP experiments
VF_TO_PROXY_CURRENT_RATIO = 1/6.5

hbt_vv          = HBTVacuumVessel()
hbt_shell       = HBTShell()
hbt_tf          = HBTTFCoils()
hbt_banana_ws   = BananaWindingSurface()
hbt_banana_fb   = BananaCoilFiniteBuild()
hbt_vf          = HBTVFCoils()
hardware_limits = HardwareLimits()

N_TF       = hbt_tf.n_coils
N_BANANA   = 2 * hbt_banana_ws.nfp
N_PROXY    = 1
N_VF       = len(hbt_vf.rs)
TF_IDX     = 0
BANANA_IDX = N_TF
PROXY_IDX  = N_TF + N_BANANA
VF_IDX     = N_TF + N_BANANA + N_PROXY
N_COILS    = VF_IDX + N_VF
N_FB_COILS = N_TF + N_PROXY + N_VF + N_BANANA*hbt_banana_fb.numfilaments
