"""
sim_cache.py — Shared simulation cache for BLOOM-21CM analysis.

Defines the on-disk NPZ format for expensive intermediate computations so that
lunar_stm.py, test_multifreq.py, and the Lunar Orbit Multi Freq notebook can
resume and share results without re-running the full simulation.

Cache directory
---------------
Default: notebooks/.sim_cache/   (add to .gitignore)

File naming
-----------
All cache files are keyed by a short config fingerprint (8-char hex hash of the
simulation-defining parameters) so that a config change automatically
invalidates cached results without manual cleanup.

  setup_{fp}.npz
      Masks and Sun pixel indices (slow step: compute_masks_and_beams).

  multifreq_{tag}_{fp}.npz
      Per-frequency inversion results.  ``tag`` encodes noise parameters:
        "nl"                  → noiseless (noise_scale=0)
        "noisy_s{scale:.3f}_r{seed}" → noisy with given scale and seed offset

  pointing_error_deg{pert:.1f}_seed{seed}_{fp}.npz
      Pointing-error simulation results for a given perturbation magnitude.

NPZ array specifications
------------------------

setup_{fp}.npz
  masks          float32  (n_total, npix)   occultation masks
  J_SUN          int32    (n_obs,)           HEALPix Sun pixel at each time
  config_fp      str/U8   ()                 fingerprint for validation

multifreq_{tag}_{fp}.npz
  freqs_mhz      float64  (N_FREQ,)          frequency grid [validation]
  T_sky_mean     float64  (N_FREQ,)          recovered sky-mean monopole [K]
  SIGMA_MONO     float64  (N_FREQ,)          per-freq noise estimate [K]
  noise_scale    float64  ()                 noise scale factor used
  noise_seed     int64    ()                 noise_seed_offset used

pointing_error_deg{pert:.1f}_seed{seed}_{fp}.npz
  freqs_mhz           float64  (N_FREQ,)    frequency grid [validation]
  T_sky_mean_pert     float64  (N_FREQ,)    recovered monopole with perturbed A [K]
  masks_pert          float32  (n_total, npix)  masks recomputed with perturbed att.
  pert_deg            float64  ()           perturbation magnitude (degrees)
  seed                int64    ()           RNG seed for perturbation
  mask_pixels_flipped int64    ()           #pixels that changed between true/pert

Usage
-----
Import in any script or notebook:

    from sim_cache import (
        config_fingerprint, cache_path,
        load_setup, save_setup,
        load_multifreq, save_multifreq,
        load_pointing_error, save_pointing_error,
    )
"""

import hashlib
import json
import os
import numpy as np

_NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE  = os.path.join(_NOTEBOOKS_DIR, ".sim_cache")


# ─── Fingerprint ──────────────────────────────────────────────────────────────

def config_fingerprint(cfg):
    """
    Return an 8-char hex fingerprint of the simulation-defining parameters.

    Any change to the quantities listed here invalidates all cached results
    for that config.

    Parameters
    ----------
    cfg : OrbiterMission instance (from lunar_orbit.py)
    """
    params = {
        "nside":           int(cfg.observation.nside),
        "n_obs":           int(cfg.observation.n_obs),
        "n_days":          int(cfg.observation.n_days),
        "n_orbits":        int(cfg.observation.n_orbits),
        "freq_min_mhz":    float(cfg.observation.freq_min_mhz),
        "freq_max_mhz":    float(cfg.observation.freq_max_mhz),
        "nchan_science":   int(cfg.observation.nchan_science),
        "altitude_m":      float(cfg.observation.altitude),
        "fixed_spin":      bool(cfg.observation.fixed_spin),
        "obs_epoch":       str(cfg.observation.obs_epoch.isot),
    }
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:8]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def cache_path(filename, cache_dir=None):
    """Return the full path for a cache file, creating the directory if needed."""
    d = cache_dir or _DEFAULT_CACHE
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def _setup_name(fp):
    return f"setup_{fp}.npz"


def _multifreq_name(fp, noise_scale, noise_seed):
    if noise_scale == 0.0:
        tag = "nl"
    else:
        tag = f"noisy_s{noise_scale:.3f}_r{noise_seed:d}"
    return f"multifreq_{tag}_{fp}.npz"


def _pointing_error_name(fp, pert_deg, seed):
    return f"pointing_error_deg{pert_deg:.1f}_seed{seed:d}_{fp}.npz"


# ─── Setup cache  (masks + J_SUN) ─────────────────────────────────────────────

def save_setup(masks, J_SUN, fp, cache_dir=None):
    """
    Save occultation masks and Sun pixel indices to the setup cache.

    Parameters
    ----------
    masks : ndarray, shape (n_total, npix), float32 or bool
    J_SUN : ndarray, shape (n_obs,), int
    fp : str
        Config fingerprint from :func:`config_fingerprint`.
    """
    p = cache_path(_setup_name(fp), cache_dir)
    np.savez_compressed(p,
        masks=masks.astype(np.float32),
        J_SUN=np.asarray(J_SUN, dtype=np.int32),
        config_fp=np.bytes_(fp),
    )
    print(f"  [cache] saved setup → {os.path.basename(p)}")


def load_setup(fp, cache_dir=None):
    """
    Load masks and J_SUN from the setup cache.

    Returns
    -------
    masks : ndarray, float32
    J_SUN : ndarray, int32

    Raises
    ------
    FileNotFoundError if cache file does not exist.
    ValueError if the fingerprint stored in the file does not match ``fp``.
    """
    p = cache_path(_setup_name(fp), cache_dir)
    data = np.load(p)
    stored_fp = data["config_fp"].item().decode() if hasattr(data["config_fp"].item(), "decode") else str(data["config_fp"].item())
    if stored_fp != fp:
        raise ValueError(f"Setup cache fingerprint mismatch: file={stored_fp!r}, expected={fp!r}")
    masks = data["masks"]
    J_SUN = data["J_SUN"]
    print(f"  [cache] loaded setup from {os.path.basename(p)}")
    return masks, J_SUN


# ─── Multi-frequency inversion cache ──────────────────────────────────────────

def save_multifreq(freqs_mhz, T_sky_mean, SIGMA_MONO,
                   noise_scale, noise_seed, fp, cache_dir=None):
    """
    Save per-frequency inversion results.

    Parameters
    ----------
    freqs_mhz : ndarray, (N_FREQ,)
    T_sky_mean : ndarray, (N_FREQ,)  recovered sky monopole [K]
    SIGMA_MONO : ndarray, (N_FREQ,)  per-frequency noise estimate [K]
    noise_scale : float
    noise_seed : int    noise_seed_offset used in the run
    fp : str            config fingerprint
    """
    p = cache_path(_multifreq_name(fp, noise_scale, noise_seed), cache_dir)
    np.savez_compressed(p,
        freqs_mhz=np.asarray(freqs_mhz, dtype=np.float64),
        T_sky_mean=np.asarray(T_sky_mean, dtype=np.float64),
        SIGMA_MONO=np.asarray(SIGMA_MONO, dtype=np.float64),
        noise_scale=np.float64(noise_scale),
        noise_seed=np.int64(noise_seed),
    )
    tag = "noiseless" if noise_scale == 0.0 else f"noisy (scale={noise_scale:.3f}, seed={noise_seed})"
    print(f"  [cache] saved multifreq ({tag}) → {os.path.basename(p)}")


def load_multifreq(freqs_mhz, noise_scale, noise_seed, fp, cache_dir=None):
    """
    Load per-frequency inversion results.

    Parameters
    ----------
    freqs_mhz : ndarray   used to validate the stored frequency grid
    noise_scale, noise_seed : float, int
    fp : str

    Returns
    -------
    T_sky_mean : ndarray, (N_FREQ,)
    SIGMA_MONO : ndarray, (N_FREQ,)

    Raises
    ------
    FileNotFoundError, ValueError (freq grid mismatch)
    """
    p = cache_path(_multifreq_name(fp, noise_scale, noise_seed), cache_dir)
    data = np.load(p)
    if not np.allclose(data["freqs_mhz"], freqs_mhz):
        raise ValueError("Multifreq cache frequency grid mismatch")
    tag = "noiseless" if noise_scale == 0.0 else f"noisy (scale={noise_scale:.3f}, seed={noise_seed})"
    print(f"  [cache] loaded multifreq ({tag}) from {os.path.basename(p)}")
    return data["T_sky_mean"], data["SIGMA_MONO"]


# ─── Pointing-error simulation cache ──────────────────────────────────────────

def save_pointing_error(freqs_mhz, T_sky_mean_pert, masks_pert,
                         pert_deg, seed, mask_pixels_flipped,
                         fp, cache_dir=None):
    """
    Save pointing-error simulation results.

    Parameters
    ----------
    freqs_mhz : ndarray, (N_FREQ,)
    T_sky_mean_pert : ndarray, (N_FREQ,)
    masks_pert : ndarray, (n_total, npix), float32
    pert_deg : float    perturbation magnitude (degrees)
    seed : int          RNG seed used for the perturbation
    mask_pixels_flipped : int
    fp : str
    """
    p = cache_path(_pointing_error_name(fp, pert_deg, seed), cache_dir)
    np.savez_compressed(p,
        freqs_mhz=np.asarray(freqs_mhz, dtype=np.float64),
        T_sky_mean_pert=np.asarray(T_sky_mean_pert, dtype=np.float64),
        masks_pert=np.asarray(masks_pert, dtype=np.float32),
        pert_deg=np.float64(pert_deg),
        seed=np.int64(seed),
        mask_pixels_flipped=np.int64(mask_pixels_flipped),
    )
    print(f"  [cache] saved pointing error (Δθ={pert_deg}°, seed={seed}) → {os.path.basename(p)}")


def load_pointing_error(freqs_mhz, pert_deg, seed, fp, cache_dir=None):
    """
    Load pointing-error simulation results.

    Returns
    -------
    T_sky_mean_pert : ndarray, (N_FREQ,)
    masks_pert      : ndarray, (n_total, npix)
    mask_pixels_flipped : int

    Raises
    ------
    FileNotFoundError, ValueError (freq grid mismatch)
    """
    p = cache_path(_pointing_error_name(fp, pert_deg, seed), cache_dir)
    data = np.load(p)
    if not np.allclose(data["freqs_mhz"], freqs_mhz):
        raise ValueError("Pointing error cache frequency grid mismatch")
    print(f"  [cache] loaded pointing error (Δθ={pert_deg}°, seed={seed}) from {os.path.basename(p)}")
    return (data["T_sky_mean_pert"],
            data["masks_pert"],
            int(data["mask_pixels_flipped"]))


# ─── Convenience: try-load with fallback ──────────────────────────────────────

def try_load_setup(fp, cache_dir=None):
    """Return (masks, J_SUN) from cache, or None if cache is absent/stale."""
    try:
        return load_setup(fp, cache_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [cache] setup cache miss: {e}")
        return None


def try_load_multifreq(freqs_mhz, noise_scale, noise_seed, fp, cache_dir=None):
    """Return (T_sky_mean, SIGMA_MONO) from cache, or None on miss."""
    try:
        return load_multifreq(freqs_mhz, noise_scale, noise_seed, fp, cache_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [cache] multifreq cache miss: {e}")
        return None


def try_load_pointing_error(freqs_mhz, pert_deg, seed, fp, cache_dir=None):
    """Return (T_sky_mean_pert, masks_pert, flipped) from cache, or None on miss."""
    try:
        return load_pointing_error(freqs_mhz, pert_deg, seed, fp, cache_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [cache] pointing error cache miss: {e}")
        return None
