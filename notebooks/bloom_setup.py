"""
bloom_setup.py — Shared simulation setup utilities for BLOOM-21CM notebooks.

Provides common functions for loading sky models, generating spacecraft attitudes,
computing Sun positions, and loading/computing occultation masks. Eliminates
duplication across lunar_stm.py, beam_cal_verify.py, tumbling_beam_verify.py,
and test_multifreq.py.

All parameters are derived from the OrbiterMission config object (bloom_config.yaml).

STM-derived constants
---------------------
Two constants derived from the STM pointing-error simulation are defined here so
they are shared between tumbling_beam_verify.py and beam_cal_verify.py without
duplication.  They are outputs of the simulation (not input parameters), so they
live here rather than in bloom_config.yaml:

    STM_BIAS_PER_DEG_MK   mK bias for a 1° systematic offset at all observations
                          after eigenmode filtering (from pointing_error cache).
    STM_SIGMA_MONO_MK     mean per-channel SIGMA_MONO over the science band.

Usage
-----
    from bloom_setup import setup_simulation, STM_BIAS_PER_DEG_MK, STM_SIGMA_MONO_MK
"""

import os
import sys

import numpy as np
import astropy.units as u
import healpy
from astropy.coordinates import get_body
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from eigsep_sim.sky import SkyModel
from eigsep_sim.sim import compute_masks_and_beams
from eigsep_sim.models import T21cmModel
from sim_cache import config_fingerprint, try_load_setup, save_setup


# ── STM-derived constants ──────────────────────────────────────────────────────
# From the STM pointing-error simulation (pointing_error_deg1.0_seed99 cache):
#   1° systematic offset applied to all observations → bias after eigenmode filter.
#   SIGMA_MONO is the noise on the sky monopole from the inversion (mean over band),
#   NOT per-pixel radiometric noise.
STM_BIAS_PER_DEG_MK = 122.29   # mK bias for 1° systematic pointing error
STM_SIGMA_MONO_MK   = 1.64     # mK SIGMA_MONO mean over science band


# ── Attitude generation ────────────────────────────────────────────────────────

def generate_attitudes(cfg):
    """
    Generate spacecraft attitudes for all orbital planes.

    Uses a fixed random seed (42 + orbit index) for reproducibility.

    Parameters
    ----------
    cfg : OrbiterMission

    Returns
    -------
    rots : list of scipy.spatial.transform.Rotation
        One Rotation object per orbital plane, each of length n_obs.
    """
    n_obs    = cfg.observation.n_obs
    n_orbits = cfg.observation.n_orbits
    if cfg.observation.fixed_spin:
        phi = np.linspace(0.0, 2.0 * np.pi, n_obs, endpoint=False)
        rot_fixed = Rotation.from_rotvec(np.outer(phi, cfg.antenna.l_hat))
        return [rot_fixed for _ in range(n_orbits)]
    else:
        return [
            Rotation.random(n_obs, random_state=42 + o)
            for o in range(n_orbits)
        ]


# ── Observation time grid ──────────────────────────────────────────────────────

def make_obs_times(cfg):
    """
    Return orbits, time array in seconds, and astropy Time array.

    Parameters
    ----------
    cfg : OrbiterMission

    Returns
    -------
    orbits_list : list of LunarOrbit
    t_obs_s     : ndarray, shape (n_obs,)   — times in seconds from epoch
    obs_times   : astropy Time array, shape (n_obs,)
    """
    n_days  = cfg.observation.n_days
    n_obs   = cfg.observation.n_obs
    orbits_list = cfg.observation.make_orbits(rot_spin_vec=(0, 0, 1), spin_period=0.0)
    t_obs_s     = np.linspace(0.0, n_days * 86400.0, n_obs, endpoint=False)
    obs_times   = cfg.observation.obs_epoch + t_obs_s * u.s
    return orbits_list, t_obs_s, obs_times


# ── Sky and 21cm signal models ─────────────────────────────────────────────────

def load_sky_and_models(cfg, freqs_mhz, model_index=0):
    """
    Load the GSM sky model and 21cm signal model.

    Parameters
    ----------
    cfg       : OrbiterMission
    freqs_mhz : array_like, (N_FREQ,)   — science frequency grid [MHz]
    model_index : int  — which T21cmModel index to inject (default 0)

    Returns
    -------
    gsm_maps : ndarray, shape (npix, N_FREQ)
    T_21_INJ : ndarray, shape (N_FREQ,)   — 21cm signal at freqs_mhz [K]
    """
    freqs_mhz = np.asarray(freqs_mhz)
    print("Loading GSM …", flush=True)
    sky_mf   = SkyModel(freqs_mhz * 1e6, nside=cfg.observation.nside, srcs=None)
    gsm_maps = np.asarray(sky_mf.map)
    if gsm_maps.ndim == 1:
        gsm_maps = gsm_maps[:, np.newaxis]

    models_21cm = T21cmModel()
    T_21_INJ    = models_21cm(freqs_mhz * 1e6, model_index=model_index)
    return gsm_maps, T_21_INJ


# ── Sun pixel positions ────────────────────────────────────────────────────────

def compute_sun_pixels(cfg, obs_times):
    """
    Query Sun galactic coordinates and return HEALPix pixel index per timestep.

    Parameters
    ----------
    cfg       : OrbiterMission
    obs_times : astropy Time array, shape (n_obs,)

    Returns
    -------
    J_SUN : ndarray, shape (n_obs,), int32
    """
    print("Querying Sun positions …", flush=True)
    sun_coords = get_body("sun", obs_times)
    sun_gal    = sun_coords.galactic
    l_s, b_s   = sun_gal.l.rad, sun_gal.b.rad
    J_SUN = healpy.vec2pix(
        cfg.observation.nside,
        np.cos(b_s) * np.cos(l_s),
        np.cos(b_s) * np.sin(l_s),
        np.sin(b_s),
    )
    return J_SUN


# ── Occultation masks ──────────────────────────────────────────────────────────

def load_or_compute_masks(cfg, rots, orbits_list, obs_times, freqs_mhz, J_SUN, fp):
    """
    Load occultation masks from cache, or compute and save them.

    Uses the midband frequency for the mask computation (masks are nearly
    frequency-independent at NSIDE=8).

    Parameters
    ----------
    cfg         : OrbiterMission
    rots        : list of Rotation  (from generate_attitudes)
    orbits_list : list of LunarOrbit
    obs_times   : astropy Time array
    freqs_mhz   : array_like   — needed only to pick midband kh
    J_SUN       : ndarray, shape (n_obs,)  — for cache save
    fp          : str  — config fingerprint

    Returns
    -------
    masks : ndarray, shape (n_total, npix), float32
    """
    freqs_mhz = np.asarray(freqs_mhz)
    cached = try_load_setup(fp)
    if cached is not None:
        masks, _ = cached
        print(f"  masks shape = {masks.shape}  "
              f"(mean open fraction = {masks.mean():.2f})")
        return masks

    print("Computing occultation masks …", flush=True)
    kh_mid = cfg.antenna.kh(freqs_mhz[len(freqs_mhz) // 2])
    masks, _, _ = compute_masks_and_beams(
        orbits_list, obs_times, rots,
        cfg.antenna.u_body, kh_mid,
        cfg.observation.nside, verbose=False,
    )
    save_setup(masks, J_SUN, fp)
    print(f"  masks shape = {masks.shape}  "
          f"(mean open fraction = {masks.mean():.2f})")
    return masks


# ── Full simulation setup ──────────────────────────────────────────────────────

def setup_simulation(cfg, freqs_mhz, fp=None, model_index=0):
    """
    Complete simulation setup: attitudes, time grid, sky models, Sun pixels, masks.

    This is the single entry point that eliminates setup duplication across
    lunar_stm.py, beam_cal_verify.py, and test_multifreq.py.

    Parameters
    ----------
    cfg       : OrbiterMission
    freqs_mhz : array_like, (N_FREQ,)
    fp        : str or None — config fingerprint; computed if None
    model_index : int — 21cm model to inject (default 0)

    Returns
    -------
    rots        : list of Rotation
    orbits_list : list of LunarOrbit
    obs_times   : astropy Time array
    gsm_maps    : ndarray, shape (npix, N_FREQ)
    T_21_INJ    : ndarray, shape (N_FREQ,)
    J_SUN       : ndarray, shape (n_obs,)
    masks       : ndarray, shape (n_total, npix)
    """
    if fp is None:
        fp = config_fingerprint(cfg)

    rots                            = generate_attitudes(cfg)
    orbits_list, _, obs_times       = make_obs_times(cfg)
    gsm_maps, T_21_INJ              = load_sky_and_models(cfg, freqs_mhz, model_index)
    J_SUN                           = compute_sun_pixels(cfg, obs_times)
    masks                           = load_or_compute_masks(
                                          cfg, rots, orbits_list, obs_times,
                                          freqs_mhz, J_SUN, fp)
    return rots, orbits_list, obs_times, gsm_maps, T_21_INJ, J_SUN, masks
