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

from .const import c as C_LIGHT, DTYPE_R_NPY

BEAM_NPZ = os.path.join(os.path.dirname(__file__), "data", "eigsep_bowtie_v000.npz")


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
    return mdl_interp(freqs).astype(DTYPE_R_NPY)


def _normalize_vector(vec, dtype=DTYPE_R_NPY):
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
    dtype=DTYPE_R_NPY,
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
    dtype=DTYPE_R_NPY,
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
    dtype=DTYPE_R_NPY,
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


class Beam:
    """
    Body-frame HEALPix beam model with spectral basis decomposition.

    Stores beam coefficients (n_dipoles, npix_beam, nmodes) per dipole in a
    spectral basis, enabling compact representation of frequency-dependent
    beam patterns. Evaluation reconstructs the full beam pattern at any frequency
    via deprojection: B_d(f) = coeffs_d @ basis.A[f].T.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the body-frame beam map.
    freqs_hz : ndarray, shape (nfreq,)
        Frequencies [Hz] at which the beam is defined.
    basis : BeamBasis
        Spectral basis for beam decomposition.
    coeffs : ndarray, shape (n_dipoles, npix_beam, nmodes)
        Spatial coefficients per dipole in the spectral basis.
    u_body : ndarray, shape (n_dipoles, 3), optional
        Dipole axis unit vectors in body frame. Defaults to a standard
        orthogonal pair if None.
    """

    def __init__(self, nside, freqs_hz, basis, coeffs, u_body=None):
        self.nside = int(nside)
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.basis = basis
        self.coeffs = np.asarray(coeffs, dtype=DTYPE_R_NPY)

        if u_body is None:
            u_body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE_R_NPY)
        self.u_body = np.asarray(u_body, dtype=DTYPE_R_NPY)

        # Validate shapes
        npix = healpy.nside2npix(self.nside)
        n_dipoles, nmodes_coeffs = self.coeffs.shape[0], self.coeffs.shape[2]
        if self.coeffs.shape != (n_dipoles, npix, nmodes_coeffs):
            raise ValueError(f"coeffs shape {self.coeffs.shape} inconsistent with "
                           f"nside={nside} (npix={npix}) and basis nmodes={self.basis.nmodes}")

    @classmethod
    def from_dipole(cls, nside, freqs_hz, arm_lengths_m, u_body=None, K=5):
        """Initialize beam from thin-dipole analytic model.

        Evaluates the thin-dipole power pattern at all frequencies on a HEALPix
        grid, then performs SVD per dipole to extract K dominant spectral modes.
        The coefficients are initialized from the SVD decomposition.

        Parameters
        ----------
        nside : int
            HEALPix resolution for body-frame beam.
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        arm_lengths_m : float or ndarray, shape (n_dipoles,)
            Dipole arm length(s) [m]. A scalar is used for every dipole.
        u_body : ndarray, shape (n_dipoles, 3), optional
            Dipole axes in body frame. Defaults to a standard orthogonal pair.
        K : int
            Number of spectral modes to retain (default 5).

        Returns
        -------
        Beam
            New beam object initialized from thin-dipole model.
        """
        from .basis import BeamBasis

        if u_body is None:
            u_body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE_R_NPY)
        u_body = np.asarray(u_body, dtype=DTYPE_R_NPY)
        if u_body.ndim != 2 or u_body.shape[1] != 3 or u_body.shape[0] == 0:
            raise ValueError(
                f"u_body must have shape (n_dipoles, 3), got {u_body.shape}"
            )

        n_dipoles = u_body.shape[0]
        arm_lengths_m = np.atleast_1d(
            np.asarray(arm_lengths_m, dtype=DTYPE_R_NPY)
        )
        if arm_lengths_m.size == 1:
            arm_lengths_m = np.repeat(arm_lengths_m, n_dipoles)
        elif arm_lengths_m.size != n_dipoles:
            raise ValueError(
                "arm_lengths_m must be scalar or have one value per dipole "
                f"({n_dipoles}), got {arm_lengths_m.size}"
            )

        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

        # Compute cos(θ) for all beam pixels
        npix = healpy.nside2npix(nside)
        N_GAL = np.array(healpy.pix2vec(nside, np.arange(npix)))  # (3, npix)
        cos_theta = u_body @ N_GAL  # (n_dipoles, npix)

        # Evaluate thin-dipole beam at all frequencies
        nominal_beam = np.zeros(
            (n_dipoles, npix, len(freqs_hz)), dtype=DTYPE_R_NPY
        )
        for f_idx, f_hz in enumerate(freqs_hz):
            kh_f = arm_lengths_m * np.pi * f_hz / C_LIGHT
            nominal_beam[:, :, f_idx] = thin_dipole_pattern(kh_f[:, np.newaxis], cos_theta)

        # SVD along frequency axis for each dipole
        # Note: K is limited by min(npix, nfreq)
        max_rank = min(npix, len(freqs_hz))
        K_actual = min(K, max_rank)
        # Build a shared spectral basis from the averaged dipole. Project each
        # physical dipole map into that basis so coefficient signs and scales
        # match the shared basis rather than an independent per-dipole SVD.
        B_avg = np.mean(nominal_beam, axis=0)
        U, s, Vt = np.linalg.svd(B_avg, full_matrices=False)
        basis_A = Vt[:K_actual].T  # (nfreq, K_actual)
        coeffs = nominal_beam @ basis_A  # (n_dipoles, npix, K_actual)
        basis = BeamBasis(basis_A, freqs_hz=freqs_hz)

        return cls(nside, freqs_hz, basis, coeffs, u_body=u_body)

    @classmethod
    def from_file(cls, path, new_freqs=None):
        """Load beam from npz file, optionally resampling to new frequencies.

        Parameters
        ----------
        path : str
            Path to npz file with keys: nside, freqs_hz, coeffs, basis_A, u_body.
        new_freqs : ndarray, shape (nfreq_new,), optional
            If provided, resample basis to these frequencies.

        Returns
        -------
        Beam
            Loaded (and optionally resampled) beam object.
        """
        from .basis import BeamBasis

        npz = np.load(path, allow_pickle=False)
        nside = int(npz['nside'])
        freqs_hz = npz['freqs_hz']
        coeffs = npz['coeffs']
        basis_A = npz['basis_A']
        u_body = npz['u_body'] if 'u_body' in npz else None

        # Create basis
        basis = BeamBasis(basis_A, freqs_hz=freqs_hz)

        # Resample if requested
        if new_freqs is not None:
            from .basis import _resample_basis
            basis_A_new = _resample_basis(freqs_hz, basis_A, new_freqs)
            basis = BeamBasis(basis_A_new, freqs_hz=new_freqs)
            freqs_hz = new_freqs

        return cls(nside, freqs_hz, basis, coeffs, u_body=u_body)

    def save(self, path):
        """Save beam to npz file.

        Parameters
        ----------
        path : str
            Output npz file path.
        """
        np.savez(path, nside=self.nside, freqs_hz=self.freqs_hz,
                coeffs=self.coeffs, basis_A=self.basis.A, u_body=self.u_body)

    def evaluate(self, freq_idx):
        """Reconstruct (n_dipoles, npix) beam at frequency index.

        Parameters
        ----------
        freq_idx : int
            Frequency index (0 <= freq_idx < nfreq).

        Returns
        -------
        ndarray, shape (n_dipoles, npix)
            Beam power pattern for each dipole at the given frequency.
        """
        if not (0 <= freq_idx < self.basis.nfreq):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {self.basis.nfreq})")

        n_dipoles = self.coeffs.shape[0]
        npix = self.coeffs.shape[1]
        beam = np.zeros((n_dipoles, npix), dtype=DTYPE_R_NPY)

        for d in range(n_dipoles):
            # coeffs_d @ basis.A[freq_idx] → (npix,)
            beam[d] = self.coeffs[d] @ self.basis.A[freq_idx]

        return beam

    def solid_angle(self, freq_idx):
        """Compute (n_dipoles,) beam solid angles at frequency index.

        Solid angle for each dipole: Ω = 4π / npix * sum(beam_pattern).

        Parameters
        ----------
        freq_idx : int
            Frequency index.

        Returns
        -------
        ndarray, shape (n_dipoles,)
            Beam solid angle (steradians) for each dipole.
        """
        beam = self.evaluate(freq_idx)
        npix = healpy.nside2npix(self.nside)
        return 4.0 * np.pi / npix * np.sum(beam, axis=1)

    @property
    def npix(self):
        """Number of HEALPix pixels."""
        return healpy.nside2npix(self.nside)

    @property
    def n_dipoles(self):
        """Number of dipoles."""
        return self.coeffs.shape[0]

    @property
    def nmodes(self):
        """Number of spectral modes."""
        return self.coeffs.shape[2]

    # ── Body-frame rotation helpers ────────────────────────────────────────────

    @staticmethod
    def rot_x(a):
        """Right-handed rotation matrix by angle *a* [rad] around x-axis."""
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=DTYPE_R_NPY)

    @staticmethod
    def rot_z(a):
        """Right-handed rotation matrix by angle *a* [rad] around z-axis."""
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=DTYPE_R_NPY)

    @staticmethod
    def top2body(az, alt):
        """Rotation matrix from topocentric to antenna body frame.

        Implements the EIGSEP scanning convention: azimuth rotates around the
        topocentric ẑ axis, altitude tilts around the topocentric x̂ (east) axis.
        At az=alt=0 the body frame coincides with the topocentric frame
        (x = east, y = north, z = up).

            R_body2top = R_z(az) @ R_x(alt)
            R_top2body = R_x(−alt) @ R_z(−az)

        Parameters
        ----------
        az : float
            Azimuth [rad].
        alt : float
            Altitude tilt [rad].

        Returns
        -------
        ndarray, shape (3, 3), float32
        """
        return Beam.rot_x(-alt) @ Beam.rot_z(-az)
