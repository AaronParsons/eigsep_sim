"""Shared setup reproducing EIGSEP_Recovery_v001_NewFramework.ipynb.

Builds the same nside=8 ground-based recovery problem the notebook uses so
plateau diagnostics and solver benchmarks run against identical data.
"""

import numpy as np
import healpy
import astropy.units as u
from astropy.time import Time

from eigsep_sim import (
    EarthSurface,
    Beam,
    Sky,
    NullTerrain,
    ForwardModel,
    Calibrator,
    DTYPE_R_NPY,
)

NSIDE = 8
N_FREQ = 20
FREQS_HZ = np.linspace(55.0, 150.0, N_FREQ) * 1e6
DELTA_NU_HZ = float(np.diff(FREQS_HZ).mean())
OBS_EPOCH = Time("2025-01-01")
N_TIMES = 36
N_AZ, N_ALT = 12, 10
N_ORIENT = N_AZ * N_ALT
T_RX_K = 100.0


def smooth_map(nside, ell_max, rng):
    """Random low-ell map with unit RMS for smooth beam perturbations."""
    cl = np.zeros(3 * nside)
    hi = ell_max + 1
    cl[1:hi] = 1.0
    m = healpy.synfast(cl, nside, verbose=False)
    return m / np.sqrt(np.mean(m**2))


def build_problem(
    noise_seed=42, perturb_seed=1, perturb_frac=0.1, perturb_mode="rough"
):
    """Return dict with fwd, geom, data, truth/init params, noise model.

    perturb_mode='rough' multiplies every coefficient by an independent
    uniform factor (the notebook's choice; high-ell beam structure).
    perturb_mode='smooth' perturbs the beam with a random ell<=3 spatial
    pattern, consistent with the harmonic beam regularizer.
    """
    obs = EarthSurface(lat=39.2, lon=-113.4)
    beam = Beam.from_dipole(
        nside=NSIDE,
        freqs_hz=FREQS_HZ,
        arm_lengths_m=2.0,
        u_body=np.eye(3, dtype=DTYPE_R_NPY)[:1],
        K=5,
    )
    sky = Sky.from_gsm(NSIDE, FREQS_HZ, n_modes=5, include_flat=True)
    gsm_maps = sky.init_coeffs()
    fwd = ForwardModel(obs, beam, sky, terrain=NullTerrain())

    base_times = (
        OBS_EPOCH + np.linspace(0, 86400, N_TIMES, endpoint=False) * u.s
    )
    base_rots = obs.rot_gal2top_stack(base_times)
    rots = np.repeat(base_rots, N_ORIENT, axis=0)
    az_rad = np.linspace(0.0, 2.0 * np.pi, N_AZ, endpoint=False)
    alt_rad = np.linspace(0.0, 0.5 * np.pi, N_ALT)
    orient_rots = np.stack(
        [Beam.top2body(az, alt) for alt in alt_rad for az in az_rad]
    )
    body_rots = np.tile(orient_rots, (N_TIMES, 1, 1))
    geom = fwd.precompute_geometry(rots=rots, body_rots=body_rots)

    sky_coeffs = gsm_maps
    beam_coeffs = beam.coeffs.copy()
    antenna_temp = np.asarray(fwd.simulate(sky_coeffs, beam_coeffs, geom=geom))

    # Noise model identical to the notebook (per-sample radiometer sigma).
    tau_per_obs = 86400.0 / (N_TIMES * N_ORIENT)
    beam_maps = beam_coeffs @ beam.basis.A.T
    pixels = np.asarray(geom["beam_px_jax"])
    weights = np.asarray(geom["beam_wgts_jax"])
    beam_weight = np.zeros(antenna_temp.shape, dtype=float)
    for fi in range(N_FREQ):
        for di in range(beam_maps.shape[0]):
            bm = beam_maps[di, :, fi]
            beam_weight[:, di, fi] = sum(
                (bm[pixels[:, k, :]] * weights[:, k, :]).sum(axis=1)
                for k in range(4)
            )
    sigma_noise = (
        np.abs(antenna_temp) + T_RX_K * np.abs(beam_weight)
    ) / np.sqrt(DELTA_NU_HZ * tau_per_obs)

    antenna_temp_flat = antenna_temp.reshape(-1, N_FREQ)
    sigma_noise_flat = sigma_noise.reshape(-1, N_FREQ)
    rng = np.random.default_rng(seed=noise_seed)
    noise = rng.normal(scale=sigma_noise_flat, size=antenna_temp_flat.shape)
    data_noisy = antenna_temp_flat + noise
    inv_noise_var = 1.0 / sigma_noise_flat**2

    prms_tru = {"sky_coeffs": gsm_maps, "beam_coeffs": beam_coeffs}
    prng = np.random.default_rng(seed=perturb_seed)
    if perturb_mode == "rough":
        prms_ini = {
            k: v
            * prng.uniform(
                1.0 - perturb_frac, 1.0 + perturb_frac, size=v.shape
            ).astype(v.dtype)
            for k, v in prms_tru.items()
        }
    elif perturb_mode == "smooth":
        sky_pert = prng.uniform(
            1.0 - perturb_frac, 1.0 + perturb_frac, size=gsm_maps.shape
        ).astype(gsm_maps.dtype)
        np.random.seed(perturb_seed)  # healpy.synfast uses global RNG
        pattern = smooth_map(NSIDE, 3, prng).astype(beam_coeffs.dtype)
        beam_pert = 1.0 + perturb_frac * pattern[None, :, None]
        prms_ini = {
            "sky_coeffs": gsm_maps * sky_pert,
            "beam_coeffs": beam_coeffs * beam_pert,
        }
    else:
        raise ValueError(f"unknown perturb_mode: {perturb_mode}")

    return {
        "fwd": fwd,
        "beam": beam,
        "sky": sky,
        "geom": geom,
        "data_noisy": data_noisy,
        "inv_noise_var": inv_noise_var,
        "sigma_noise_flat": sigma_noise_flat,
        "prms_tru": prms_tru,
        "prms_ini": prms_ini,
    }


def make_calibrator(setup, **kwargs):
    defaults = dict(m_anderson=5, lam_beam=0.01, lam_sky=0.0)
    defaults.update(kwargs)
    return Calibrator(
        fwd=setup["fwd"],
        data=setup["data_noisy"],
        inv_noise_var=setup["inv_noise_var"],
        **defaults,
    )


def accuracy_metrics(setup, params):
    """Truth-relative accuracy of recovered sky and beam maps."""
    sky = setup["sky"]
    beam = setup["beam"]
    sky_tru = setup["prms_tru"]["sky_coeffs"] @ sky.basis.A.T
    sky_fit = np.asarray(params["sky_coeffs"]) @ sky.basis.A.T
    beam_tru = setup["prms_tru"]["beam_coeffs"] @ beam.basis.A.T
    beam_fit = np.asarray(params["beam_coeffs"]) @ beam.basis.A.T

    # Project out the global sky*scale, beam/scale gauge before comparing.
    scale = float(
        np.sum(beam_fit * beam_tru) / np.maximum(np.sum(beam_fit**2), 1e-30)
    )
    beam_fit = beam_fit * scale
    sky_fit = sky_fit / scale

    sky_frac_rms = float(
        np.sqrt(np.mean(((sky_fit - sky_tru) / np.abs(sky_tru)) ** 2))
    )
    beam_rms = float(np.sqrt(np.mean((beam_fit - beam_tru) ** 2)))
    beam_norm = float(np.sqrt(np.mean(beam_tru**2)))
    return {
        "sky_frac_rms": sky_frac_rms,
        "beam_frac_rms": beam_rms / beam_norm,
        "gauge_scale": scale,
    }
