"""
Antenna beam model.

The Beam class wraps a HEALPix beam pattern (HPM) and adds azimuth/altitude
rotation so the beam can be steered in the topocentric frame without
reloading data.

Beam data can come either from:
  - a stored NPZ beam pattern
  - a built-in analytic dipole beam

Supported analytic dipole models:
  - "short": frequency-independent short-dipole power pattern
  - "thin": frequency-dependent finite thin-dipole power pattern in free space
"""

import os
import healpy
import numpy as np
from scipy.interpolate import interp1d

from .healpix import HPM
from .coord import rot_m
from .const import c as C_LIGHT

BEAM_NPZ = os.path.join(os.path.dirname(__file__), "data", "eigsep_bowtie_v000.npz")

_real_dtype = np.float32


def load_beam_file(freqs, filename=BEAM_NPZ):
    """
    Load and interpolate a HEALPix beam pattern from an NPZ file.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz] at which to evaluate the beam.
    filename : str
        Path to NPZ file containing 'freqs' (Hz) and 'bm' (nfreq, npix).

    Returns
    -------
    bm : ndarray, shape (npix, nfreq), float32
    """
    npz = np.load(filename)
    bm = npz["bm"].T  # (npix, nfreq)
    mdl_interp = interp1d(
        npz["freqs"], bm, kind="cubic", fill_value=0, bounds_error=False
    )
    return mdl_interp(freqs).astype(_real_dtype)


def _normalize_vector(vec, dtype=_real_dtype):
    """Normalize a 3-vector."""
    vec = np.asarray(vec, dtype=dtype)
    norm = np.sqrt(np.sum(vec**2))
    if norm == 0:
        raise ValueError("vector must be nonzero")
    return vec / norm


def short_dipole_beam(
    freqs,
    nside,
    dipole_axis=(1.0, 0.0, 0.0),
    horizon_clip=False,
    dtype=_real_dtype,
):
    """
    Generate an ideal short-dipole scalar power beam on a HEALPix grid.

    The power response is:
        B(rhat) = 1 - (rhat . dhat)^2

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz]. Included for API consistency; the short-dipole
        beam is frequency independent, so the same pattern is repeated at
        every frequency.
    nside : int
        HEALPix nside of the output beam.
    dipole_axis : array_like, shape (3,)
        Unit vector giving the dipole axis in the antenna frame.
    horizon_clip : bool
        If True, set response below the horizon (z < 0) to zero.
        Typically False for a free-space orbiter dipole.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    bm : ndarray, shape (npix, nfreq)
        Scalar power beam pattern.
    """
    freqs = np.asarray(freqs, dtype=dtype)
    crd = np.stack(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))), axis=0).astype(dtype)
    z = crd[2]
    d = _normalize_vector(dipole_axis, dtype=dtype)

    proj = d @ crd
    bm0 = 1.0 - proj**2

    if horizon_clip:
        bm0 = np.where(z >= 0, bm0, 0.0)

    bm = np.repeat(bm0[:, None], freqs.size, axis=1)
    return bm.astype(dtype)


def thin_dipole_beam(
    freqs,
    nside,
    dipole_axis=(1.0, 0.0, 0.0),
    dipole_length=2.0,
    horizon_clip=False,
    dtype=_real_dtype,
    eps=1e-12,
):
    """
    Generate a frequency-dependent free-space thin-dipole scalar power beam.

    For a center-fed linear dipole of total physical length L, the far-field
    amplitude pattern is proportional to

        E(theta) ~ [cos((kL/2) cos(theta)) - cos(kL/2)] / sin(theta)

    and the scalar power beam is |E(theta)|^2.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    nside : int
        HEALPix nside of the output beam.
    dipole_axis : array_like, shape (3,)
        Unit vector giving the dipole axis in the antenna frame.
    dipole_length : float
        Total physical dipole length [m].
    horizon_clip : bool
        If True, set response below the horizon (z < 0) to zero.
        Typically False for a free-space orbiter dipole.
    dtype : numpy dtype
        Output dtype.
    eps : float
        Small floor to avoid division by zero near the dipole axis.

    Returns
    -------
    bm : ndarray, shape (npix, nfreq)
        Scalar power beam pattern.
    """
    freqs = np.asarray(freqs, dtype=dtype)
    crd = np.stack(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))), axis=0).astype(dtype)
    z = crd[2]
    d = _normalize_vector(dipole_axis, dtype=dtype)

    # mu = cos(theta_dipole), where theta_dipole is angle from dipole axis
    mu = d @ crd  # (npix,)
    sin_theta = np.sqrt(np.maximum(1.0 - mu**2, eps))

    k = (2.0 * np.pi * freqs / C_LIGHT).astype(dtype)  # (nfreq,)
    u = 0.5 * dipole_length * k  # kL/2

    # Broadcast to (npix, nfreq)
    mu2 = mu[:, None]
    sin_theta2 = sin_theta[:, None]
    u2 = u[None, :]

    amp = (np.cos(u2 * mu2) - np.cos(u2)) / sin_theta2
    bm = amp**2

    # Smoothly enforce the axis limit: response -> 0 on-axis
    bm = np.where((1.0 - mu[:, None] ** 2) < eps, 0.0, bm)

    if horizon_clip:
        bm = np.where(z[:, None] >= 0, bm, 0.0)

    return bm.astype(dtype)


def analytic_dipole_beam(
    freqs,
    nside,
    dipole_axis=(1.0, 0.0, 0.0),
    dipole_model="thin",
    dipole_length=2.0,
    horizon_clip=False,
    dtype=_real_dtype,
    eps=1e-12,
):
    """
    Generate an analytic scalar dipole beam on a HEALPix grid.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    nside : int
        HEALPix nside of the output beam.
    dipole_axis : array_like, shape (3,)
        Unit vector giving the dipole axis in the antenna frame.
    dipole_model : {'short', 'thin'}
        Analytic dipole model to use.
    dipole_length : float
        Total physical dipole length [m]. Used only for dipole_model='thin'.
    horizon_clip : bool
        If True, set response below the horizon (z < 0) to zero.
    dtype : numpy dtype
        Output dtype.
    eps : float
        Small floor used by the thin-dipole model.

    Returns
    -------
    bm : ndarray, shape (npix, nfreq)
        Scalar power beam pattern.
    """
    if dipole_model == "short":
        return short_dipole_beam(
            freqs,
            nside,
            dipole_axis=dipole_axis,
            horizon_clip=horizon_clip,
            dtype=dtype,
        )
    if dipole_model == "thin":
        return thin_dipole_beam(
            freqs,
            nside,
            dipole_axis=dipole_axis,
            dipole_length=dipole_length,
            horizon_clip=horizon_clip,
            dtype=dtype,
            eps=eps,
        )
    raise ValueError(f"Unknown dipole_model {dipole_model!r}")


def thin_dipole_pattern(kh, cos_theta, eps=1e-12):
    """
    Thin-dipole scalar power pattern given precomputed cos(theta) values.

    Evaluates  B = [(cos(kh · cos θ) − cos kh) / sin θ]²,  the same formula
    used by :func:`thin_dipole_beam`, but without fixing the dipole axis to a
    HEALPix grid.  Use this when *cos_theta* has been precomputed externally
    (e.g. after rotating the dipole axis into the inertial frame).

    Parameters
    ----------
    kh : array_like
        Electrical half-length(s) kL/2 = π f L / c.  Broadcast-compatible
        with *cos_theta*.
    cos_theta : array_like
        Cosine of the angle between each direction and the dipole axis.
    eps : float
        Floor on sin²(θ) used both to avoid division by zero and as the
        on-axis threshold below which the pattern is set to zero.

    Returns
    -------
    pattern : ndarray
        Unnormalised power beam pattern, same shape as the broadcast of
        *kh* and *cos_theta*.  Exactly zero on the dipole axis.
    """
    kh = np.asarray(kh, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    sin2 = np.maximum(1.0 - cos_theta ** 2, eps)
    numer = np.cos(kh * cos_theta) - np.cos(kh)
    return np.where(sin2 > eps, numer ** 2 / sin2, 0.0)


# ── Dipole reception physics ───────────────────────────────────────────────

def gsm_like_tsky_K(freq_mhz):
    """Crude average-sky model: ~1.9e4 K at 30 MHz, spectral index −2.55."""
    return 1.9e4 * (np.asarray(freq_mhz, dtype=float) / 30.0) ** (-2.55)


def short_dipole_radiation_resistance_ohm(length_m, freq_mhz):
    """Radiation resistance Rrad ≈ 80 π² (L/λ)² (short-dipole approximation)."""
    lam = C_LIGHT / (np.asarray(freq_mhz, dtype=float) * 1e6)
    return 80.0 * np.pi ** 2 * (length_m / lam) ** 2


def realized_efficiency(
    length_m, freq_mhz,
    r_loss_ohm=5.0, z_rx_ohm=50.0, x_scale=120.0,
):
    """
    Approximate realised efficiency η = mismatch × Rrad / (Rrad + Rloss).

    Models the impedance mismatch between the antenna and the receiver using
    a short-dipole reactance approximation, then applies ohmic-loss
    derating.

    Parameters
    ----------
    length_m : float
        Total dipole length [m].
    freq_mhz : array_like
        Frequency [MHz].
    r_loss_ohm : float
        Lumped antenna / lead resistance [Ω].
    z_rx_ohm : float
        Receiver input resistance [Ω].
    x_scale : float
        Reactance model scale factor.

    Returns
    -------
    eta : ndarray
        Realised efficiency ∈ [0, 1], same shape as *freq_mhz*.
    """
    f = np.asarray(freq_mhz, dtype=float)
    elec = np.maximum(length_m * f * 1e6 / C_LIGHT, 1e-6)
    rrad = short_dipole_radiation_resistance_ohm(length_m, f)
    rtot = rrad + r_loss_ohm
    zant = rtot - 1j * (x_scale / elec)
    gamma = (zant - z_rx_ohm) / (zant + z_rx_ohm)
    return np.clip((1.0 - np.abs(gamma) ** 2) * (rrad / rtot), 0.0, 1.0)


def antenna_temperature_K(
    length_m, freq_mhz,
    r_loss_ohm=5.0, z_rx_ohm=50.0, x_scale=120.0,
):
    """Delivered sky temperature after antenna efficiency losses."""
    return (
        realized_efficiency(length_m, freq_mhz,
                            r_loss_ohm=r_loss_ohm, z_rx_ohm=z_rx_ohm,
                            x_scale=x_scale)
        * gsm_like_tsky_K(freq_mhz)
    )


def receiver_margin_factor(
    length_m, freq_mhz, trx_K=100.0,
    r_loss_ohm=5.0, z_rx_ohm=50.0, x_scale=120.0,
):
    """Returns 2·Tant / Trx; criterion Trx < 2·Tant passes when result > 1."""
    return (
        2.0
        * antenna_temperature_K(length_m, freq_mhz,
                                r_loss_ohm=r_loss_ohm, z_rx_ohm=z_rx_ohm,
                                x_scale=x_scale)
        / trx_K
    )


class Beam(HPM):
    """
    Antenna beam pattern stored as a HEALPix map with rotation support.

    The beam is defined in a fixed antenna frame. Az/alt rotation matrices
    are applied on every pixel access so the beam can be steered without
    reloading the data.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    filename : str
        Path to NPZ file containing the beam pattern.
    peak_normalize : bool
        If True, normalize the beam to its peak value across all pixels
        and frequencies.
    beam_type : {'file', 'dipole'}
        Select whether to load the beam from file or generate an analytic
        dipole beam.
    nside : int or None
        Required for beam_type='dipole'. Ignored for beam_type='file'.
    dipole_axis : array_like, shape (3,)
        Dipole axis for beam_type='dipole'.
    dipole_model : {'short', 'thin'}
        Analytic dipole model for beam_type='dipole'.
    dipole_length : float
        Total physical dipole length [m] for dipole_model='thin'.
    horizon_clip : bool
        If True, zero the analytic response below the horizon.
    """

    def __init__(
        self,
        freqs,
        filename=BEAM_NPZ,
        peak_normalize=True,
        beam_type="file",
        nside=None,
        dipole_axis=(1.0, 0.0, 0.0),
        dipole_model="thin",
        dipole_length=2.0,
        horizon_clip=False,
    ):
        self.freqs = np.asarray(freqs, dtype=_real_dtype)

        if beam_type == "file":
            bm_data = load_beam_file(self.freqs, filename=filename)

        elif beam_type == "dipole":
            if nside is None:
                raise ValueError("nside required for analytic dipole beam")

            bm_data = analytic_dipole_beam(
                self.freqs,
                nside=nside,
                dipole_axis=dipole_axis,
                dipole_model=dipole_model,
                dipole_length=dipole_length,
                horizon_clip=horizon_clip,
                dtype=_real_dtype,
            )

        else:
            raise ValueError(f"Unknown beam_type {beam_type!r}")

        if peak_normalize:
            bmmax = bm_data.max()
            if bmmax > 0:
                bm_data = bm_data / bmmax

        nside_beam = healpy.npix2nside(bm_data.shape[0])

        self.set_az(0)
        self.set_alt(0)

        HPM.__init__(self, nside_beam, interp=True)
        self.set_map(bm_data.astype(_real_dtype))

    def set_az(self, theta, az_vec=None):
        """Set the azimuth rotation angle (radians) about az_vec."""
        if az_vec is None:
            az_vec = np.array([0, 0, 1], dtype=_real_dtype)
        self.az = theta
        self.rot_az = rot_m(theta, az_vec)

    def set_alt(self, theta, alt_vec=None):
        """Set the altitude rotation angle (radians) about alt_vec."""
        if alt_vec is None:
            alt_vec = np.array([1, 0, 0], dtype=_real_dtype)
        self.alt = theta
        self.rot_alt = rot_m(theta, alt_vec)

    def get_rotation_matrices(self, azs, alts):
        """
        Compute combined az/alt rotation matrices for arrays of angles.

        Parameters
        ----------
        azs, alts : ndarray
            Azimuth and altitude angles (radians), same shape.

        Returns
        -------
        rot_ms : ndarray, shape (*azs.shape, 3, 3)
        """
        azs = np.asarray(azs)
        alts = np.asarray(alts)
        n_rots = azs.size
        rot_ms = np.empty((n_rots, 3, 3), dtype=_real_dtype)

        for i in range(n_rots):
            self.set_az(azs.flat[i])
            self.set_alt(alts.flat[i])
            rot_ms[i] = self.rot_az.dot(self.rot_alt)

        return rot_ms.reshape(azs.shape + (3, 3))

    def __getitem__(self, crd_top):
        """
        Interpolate the beam at topocentric directions *crd_top*, applying
        the current az/alt rotation before lookup.

        Parameters
        ----------
        crd_top : array_like, shape (3, N)
            Topocentric unit vectors.
        """
        rot = self.rot_az.dot(self.rot_alt)
        bx, by, bz = rot.dot(crd_top)
        return HPM.__getitem__(self, (bx, by, bz))
