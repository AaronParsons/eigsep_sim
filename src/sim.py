"""
BLOOM-21cm observation simulator.

compute_masks_and_beams — precompute per-observation lunar occultation masks
    and dipole beam patterns for a set of LunarOrbit instances and spacecraft
    attitudes.

simulate_observations — forward model: beam-weighted sky + regolith + sun
    scalars, with optional radiometer noise.
"""

import numpy as np

from .beam import thin_dipole_pattern


def compute_masks_and_beams(orbits, obs_times, rots_per_orbit, u_body, kh,
                             nside, verbose=True):
    """
    Precompute lunar occultation masks and dipole beam patterns.

    Parameters
    ----------
    orbits : list of LunarOrbit
        One LunarOrbit per orbital plane.
    obs_times : astropy.time.Time array, length n_obs
        Shared observation time grid (same for all orbits).
    rots_per_orbit : list of scipy.spatial.transform.Rotation
        rots_per_orbit[o] is a Rotation of shape (n_obs,) giving the
        spacecraft attitude at each observation time for orbit o.
    u_body : ndarray, shape (2, 3)
        Dipole unit vectors in the body frame.
    kh : ndarray, shape (2,)
        Electrical half-lengths k·h = π·f·L/c for each dipole.
    nside : int
        HEALPix resolution for sky maps.
    verbose : bool
        Print per-orbit progress.

    Returns
    -------
    masks : ndarray, shape (n_total, npix), float32
        Lunar occultation mask (1 = open sky, 0 = blocked by Moon).
    beams : ndarray, shape (n_total, 2, npix), float32
        Dipole beam power pattern for each observation and dipole.
    omega_B : ndarray, shape (n_total, 2)
        Beam solid angle (sum of beam over all pixels).
    """
    import healpy

    n_orbits = len(orbits)
    n_obs = len(obs_times)
    n_total = n_orbits * n_obs
    npix = healpy.nside2npix(nside)

    N_GAL = np.array(healpy.pix2vec(nside, np.arange(npix)))  # (3, npix)

    masks = np.empty((n_total, npix), dtype=np.float32)
    beams = np.empty((n_total, 2, npix), dtype=np.float32)

    for o, orbit in enumerate(orbits):
        rots_obs = rots_per_orbit[o]
        for i, t in enumerate(obs_times):
            k = o * n_obs + i
            orbit.set_time(t)
            masks[k] = orbit.above_horizon(nside).astype(np.float32)

            D_gal = rots_obs[i].apply(u_body)          # (2, 3)
            cos_t = D_gal @ N_GAL                       # (2, npix)
            beams[k] = thin_dipole_pattern(kh[:, np.newaxis], cos_t).astype(np.float32)

        if verbose:
            print(f"  orbit {o} done")

    omega_B = beams.sum(axis=2)   # (n_total, 2)
    return masks, beams, omega_B


def compute_beams(rots_per_orbit, u_body, kh, nside):
    """
    Compute dipole beam patterns for new ``kh`` values, reusing existing
    spacecraft attitudes without re-evaluating occultation masks.

    Use this in the multi-frequency loop after calling
    :func:`compute_masks_and_beams` once to obtain the masks.

    Parameters
    ----------
    rots_per_orbit : list of scipy.spatial.transform.Rotation
        rots_per_orbit[o] is a stacked Rotation of shape (n_obs,).
    u_body : ndarray, shape (2, 3)
        Dipole unit vectors in the body frame.
    kh : ndarray, shape (2,)
        Electrical half-lengths k·h = π·f·L/c for each dipole at the
        new frequency.
    nside : int
        HEALPix resolution (must match the existing masks).

    Returns
    -------
    beams : ndarray, shape (n_total, 2, npix), float32
    omega_B : ndarray, shape (n_total, 2)
        Beam solid angle (sum over pixels).
    """
    import healpy

    n_orbits = len(rots_per_orbit)
    n_obs = len(rots_per_orbit[0])
    n_total = n_orbits * n_obs
    npix = healpy.nside2npix(nside)

    N_GAL = np.array(healpy.pix2vec(nside, np.arange(npix)))  # (3, npix)
    beams = np.empty((n_total, 2, npix), dtype=np.float32)

    for o, rots_obs in enumerate(rots_per_orbit):
        for i in range(n_obs):
            k = o * n_obs + i
            D_gal = rots_obs[i].apply(u_body)          # (2, 3)
            cos_t = D_gal @ N_GAL                       # (2, npix)
            beams[k] = thin_dipole_pattern(kh[:, np.newaxis], cos_t).astype(np.float32)

    omega_B = beams.sum(axis=2)
    return beams, omega_B


def simulate_observations(masks, beams, omega_B, gsm_map, t_regolith, t_sun,
                           J_SUN, sigma_noise, rng=None, t_rx=None):
    """
    Forward model: simulate antenna temperature observations.

    Parameters
    ----------
    masks : ndarray, shape (n_total, npix), float32
    beams : ndarray, shape (n_total, 2, npix), float32
    omega_B : ndarray, shape (n_total, 2)
    gsm_map : ndarray, shape (npix,)
        Sky brightness temperature map [K].
    t_regolith : float
        Lunar regolith temperature [K].
    t_sun : float
        Effective Sun temperature [K].
    J_SUN : ndarray, shape (n_obs,), int
        HEALPix pixel index of the Sun at each observation time (shared
        across orbits).
    sigma_noise : array_like, shape (2,)
        Per-dipole radiometer noise standard deviation [K].
    rng : numpy.random.Generator or None
        Random number generator for noise.  If None, uses default_rng(42).
    t_rx : array_like, shape (2,) or None
        Per-dipole receiver temperature [K].  Added as a constant offset to
        every measurement.  If None, defaults to zero (backward-compatible).

    Returns
    -------
    data : ndarray, shape (n_total, 2)
        Simulated antenna temperatures [K] including noise.
    y : ndarray, shape (n_total * 2,)
        Flattened measurement vector (row-major: data.ravel()).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sigma_noise = np.asarray(sigma_noise)
    t_rx = np.zeros(2) if t_rx is None else np.asarray(t_rx)

    n_total, npix = masks.shape
    n_obs = len(J_SUN)
    n_orbits = n_total // n_obs

    data = np.empty((n_total, 2))
    for o in range(n_orbits):
        for i in range(n_obs):
            k = o * n_obs + i
            m = masks[k]
            T_sky = gsm_map * m + t_regolith * (1.0 - m)
            sun_mask_i = m[J_SUN[i]]

            for d in range(2):
                B = beams[k, d]
                OmB = omega_B[k, d]
                T_ant = (
                    np.dot(B, T_sky)
                    + t_sun * B[J_SUN[i]] * sun_mask_i
                ) / OmB
                data[k, d] = T_ant + t_rx[d] + rng.normal(scale=sigma_noise[d])

    y = data.ravel()
    return data, y
