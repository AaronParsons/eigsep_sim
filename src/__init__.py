__author__ = "Aaron Parsons"
__version__ = "0.0.1"

from .const import DTYPE_R_NPY, DTYPE_R_JAX
from .basis import BeamBasis, SkyBasis
from .beam import Beam, dipole_beam_maps_jax, dipole_axes_from_angles
from .terrain import (
    HORIZON_MODELS_NPZ,
    Terrain,
    NullTerrain,
    HorizonTerrain,
)
from .observer import Observer, EarthSurface, LunarSurface, LunarOrbit
from .ephemeris import (
    body_directions_gal,
    body_angular_radius,
    body_occulted_by_moon,
    moon_surface_intersection_mcmf,
)
from .sources import (
    quiet_sun_temperature_K,
    solar_activity_envelope,
    sun_temperature_K,
    inject_solar_bursts,
    flag_bursts,
    earth_rfi_temperature_K,
    earth_rfi_tone_temperature_K,
)
from .regolith import (
    subsolar_equilibrium_temperature_K,
    surface_equilibrium_temperature_K,
    solar_geometry,
    em_power_penetration_depth_m,
    diurnal_thermal_skin_depth_m,
    regolith_brightness_temperature_K,
    regolith_reflectivity,
    specular_reflection_direction,
    lambertian_hemisphere_weights,
)
from .sky import Sky
from .simulate import ForwardModel, StackedForwardModel
from .calibrator import Calibrator
from .lunar import (
    LunarCampaign,
    LunarCampaignResult,
    angular_momentum_for_spin_period,
    crossed_rod_inertia,
    integrate_torque_free,
    interpolate_body_rotations,
    make_ecliptic_orbit_normals,
    normalize_vector,
)
from .lunar_recovery import LunarRecoveryAdapter
from .param_recovery import (
    DipoleBeamVarPro,
    build_gsm_sky_prior,
    beam_pix_vecs,
    pack_dipole_params,
    unpack_dipole_params,
    t21_matched_filter,
    t21_template,
    t21_filter_spectral,
    t21_filter_forward,
)
from .recovery import (
    AdditiveDegeneracy,
    RecoverySolution,
    ScaleDegeneracy,
    build_surface_design_matrix,
    normal_solve,
    relative_rms,
    sample_beam_weights,
)

__all__ = [
    "DTYPE_R_NPY",
    "DTYPE_R_JAX",
    "BeamBasis",
    "SkyBasis",
    "Beam",
    "dipole_beam_maps_jax",
    "dipole_axes_from_angles",
    "HORIZON_MODELS_NPZ",
    "Terrain",
    "NullTerrain",
    "HorizonTerrain",
    "Observer",
    "EarthSurface",
    "LunarSurface",
    "LunarOrbit",
    "body_directions_gal",
    "body_angular_radius",
    "body_occulted_by_moon",
    "moon_surface_intersection_mcmf",
    "quiet_sun_temperature_K",
    "solar_activity_envelope",
    "sun_temperature_K",
    "inject_solar_bursts",
    "flag_bursts",
    "earth_rfi_temperature_K",
    "earth_rfi_tone_temperature_K",
    "subsolar_equilibrium_temperature_K",
    "surface_equilibrium_temperature_K",
    "solar_geometry",
    "em_power_penetration_depth_m",
    "diurnal_thermal_skin_depth_m",
    "regolith_brightness_temperature_K",
    "regolith_reflectivity",
    "specular_reflection_direction",
    "lambertian_hemisphere_weights",
    "Sky",
    "ForwardModel",
    "StackedForwardModel",
    "Calibrator",
    "LunarCampaign",
    "LunarCampaignResult",
    "LunarRecoveryAdapter",
    "DipoleBeamVarPro",
    "build_gsm_sky_prior",
    "beam_pix_vecs",
    "pack_dipole_params",
    "unpack_dipole_params",
    "t21_matched_filter",
    "t21_template",
    "t21_filter_spectral",
    "t21_filter_forward",
    "angular_momentum_for_spin_period",
    "AdditiveDegeneracy",
    "RecoverySolution",
    "ScaleDegeneracy",
    "build_surface_design_matrix",
    "normal_solve",
    "relative_rms",
    "sample_beam_weights",
    "crossed_rod_inertia",
    "integrate_torque_free",
    "interpolate_body_rotations",
    "make_ecliptic_orbit_normals",
    "normalize_vector",
]
