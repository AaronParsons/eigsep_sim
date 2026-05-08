"""
Spectral basis decomposition for beam and sky models.

Provides BeamBasis and SkyBasis classes that decompose frequency-dependent patterns
(beam, sky brightness) into a product of spatial modes and spectral eigenfunctions.

Basis matrices can be saved/loaded as npz files and support resampling to new frequencies.
"""

import numpy as np
import healpy
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod

from .const import DTYPE_R_NPY

try:
    import pygdsm
    HAS_GSM = True
except ImportError:
    HAS_GSM = False


class _SpectralBasis(ABC):
    """Abstract base class for spectral basis decomposition.

    Stores a canonical projection matrix A of shape (nfreq, nmodes) and optional
    SVD metadata for reconstruction. The basis enables projection and deprojection
    of spatial data to/from spectral coefficient space.

    Parameters
    ----------
    A : ndarray, shape (nfreq, nmodes)
        Projection matrix (typically right singular vectors from SVD).
        Required; all other parameters optional.
    freqs_hz : ndarray, shape (nfreq,), optional
        Frequencies [Hz] at which the basis is defined.
    svd_mean : ndarray, shape (nfreq,), optional
        Mean spectrum subtracted before SVD (for reconstruction).
    svd_modes : ndarray, shape (nfreq, nmodes), optional
        Left singular vectors U (spatial information).
    svd_svals : ndarray, shape (nmodes,), optional
        Singular values (magnitude information).
    n_samples : int, optional
        Number of samples used to build the basis (e.g., number of sky pixels).
    """

    def __init__(self, A, freqs_hz=None, svd_mean=None, svd_modes=None,
                 svd_svals=None, n_samples=None):
        self.A = np.asarray(A, dtype=DTYPE_R_NPY)
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {self.A.shape}")
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64) if freqs_hz is not None else None
        self.svd_mean = np.asarray(svd_mean, dtype=DTYPE_R_NPY) if svd_mean is not None else None
        self.svd_modes = np.asarray(svd_modes, dtype=DTYPE_R_NPY) if svd_modes is not None else None
        self.svd_svals = np.asarray(svd_svals, dtype=DTYPE_R_NPY) if svd_svals is not None else None
        self.n_samples = n_samples

    @property
    def nfreq(self):
        """Number of frequencies."""
        return self.A.shape[0]

    @property
    def nmodes(self):
        """Number of basis modes."""
        return self.A.shape[1]

    @property
    def has_svd(self):
        """True if all SVD metadata is present."""
        return (self.svd_mean is not None and self.svd_modes is not None and
                self.svd_svals is not None and self.n_samples is not None)

    def project(self, data):
        """Project spatial data onto spectral basis.

        Parameters
        ----------
        data : ndarray, shape (..., nfreq)
            Spatial data (trailing axis is frequency).

        Returns
        -------
        ndarray, shape (..., nmodes)
            Basis coefficients.
        """
        return np.matmul(data, self.A)

    def deproject(self, coeffs):
        """Reconstruct frequency-dependent data from basis coefficients.

        Parameters
        ----------
        coeffs : ndarray, shape (..., nmodes)
            Basis coefficients.

        Returns
        -------
        ndarray, shape (..., nfreq)
            Reconstructed spatial data.
        """
        return np.matmul(coeffs, self.A.T)

    def save(self, path):
        """Save basis to npz file.

        Parameters
        ----------
        path : str
            Output npz file path.
        """
        kwargs = {'A': self.A}
        if self.freqs_hz is not None:
            kwargs['freqs_hz'] = self.freqs_hz
        if self.svd_mean is not None:
            kwargs['svd_mean'] = self.svd_mean
        if self.svd_modes is not None:
            kwargs['svd_modes'] = self.svd_modes
        if self.svd_svals is not None:
            kwargs['svd_svals'] = self.svd_svals
        if self.n_samples is not None:
            kwargs['n_samples'] = np.array(self.n_samples)
        np.savez(path, **kwargs)

    @classmethod
    def from_file(cls, path, new_freqs=None):
        """Load basis from npz file, optionally resampling to new frequencies.

        Parameters
        ----------
        path : str
            Input npz file path.
        new_freqs : ndarray, shape (nfreq_new,), optional
            If provided, resample basis to these frequencies via linear interpolation.

        Returns
        -------
        _SpectralBasis
            Loaded (and optionally resampled) basis object.
        """
        npz = np.load(path, allow_pickle=False)
        A = npz['A']
        freqs_hz = npz['freqs_hz'] if 'freqs_hz' in npz else None
        svd_mean = npz['svd_mean'] if 'svd_mean' in npz else None
        svd_modes = npz['svd_modes'] if 'svd_modes' in npz else None
        svd_svals = npz['svd_svals'] if 'svd_svals' in npz else None
        n_samples = int(npz['n_samples']) if 'n_samples' in npz else None

        # If new_freqs requested, resample A via linear interpolation
        if new_freqs is not None and freqs_hz is not None:
            A = _resample_basis(freqs_hz, A, new_freqs)
            freqs_hz = new_freqs

        return cls(A, freqs_hz=freqs_hz, svd_mean=svd_mean, svd_modes=svd_modes,
                   svd_svals=svd_svals, n_samples=n_samples)

    @classmethod
    def from_ensemble(cls, freqs, spectra, n_modes):
        """Build basis from an ensemble of spectra via covariance SVD.

        Computes the mean spectrum, forms the centered data matrix, performs SVD,
        and returns the top n_modes right singular vectors as the basis.

        Parameters
        ----------
        freqs : ndarray, shape (nfreq,)
            Frequency grid [Hz].
        spectra : ndarray, shape (n_samples, nfreq)
            Ensemble of spectra.
        n_modes : int
            Number of basis modes to retain.

        Returns
        -------
        _SpectralBasis
            Basis object with A, SVD metadata, and ensemble statistics.
        """
        freqs = np.asarray(freqs, dtype=np.float64)
        spectra = np.asarray(spectra, dtype=DTYPE_R_NPY)
        n_samples = spectra.shape[0]

        # Center the data
        svd_mean = np.mean(spectra, axis=0)
        spectra_centered = spectra - svd_mean[np.newaxis, :]

        # SVD
        U, s, Vt = np.linalg.svd(spectra_centered, full_matrices=False)
        A = Vt[:n_modes].T  # shape (nfreq, n_modes)
        svd_modes = U[:, :n_modes]
        svd_svals = s[:n_modes]

        return cls(A, freqs_hz=freqs, svd_mean=svd_mean, svd_modes=svd_modes,
                   svd_svals=svd_svals, n_samples=n_samples)


class BeamBasis(_SpectralBasis):
    """Spectral basis for antenna beam patterns.

    Typically constructed from thin-dipole or measured beam spectra via SVD.
    """

    @classmethod
    def from_dipole(cls, freqs_hz, arm_length_m, u_body=None, K=5, nside=8):
        """Build beam basis from thin-dipole analytic model.

        Evaluates the thin-dipole power pattern at all frequencies on a HEALPix grid,
        then performs SVD per dipole to extract K dominant spectral modes.

        Parameters
        ----------
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        arm_length_m : float or ndarray, shape (2,)
            Dipole arm length(s) [m]. If float, used for both dipoles;
            if (2,), per-dipole arm lengths.
        u_body : ndarray, shape (2, 3), optional
            Dipole axis unit vectors in body frame. If None, uses default
            orthogonal dipoles [(1,0,0), (0,1,0)].
        K : int
            Number of spectral modes to retain (default 5).
        nside : int
            HEALPix resolution for beam evaluation (default 8; ~7° pixels).

        Returns
        -------
        BeamBasis
            Basis object; note that per-dipole construction is used, so
            the basis is shared across dipoles but optimized for the given
            arm lengths.
        """
        from .beam import thin_dipole_pattern
        from .const import c as C_LIGHT

        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        arm_length_m = np.atleast_1d(np.asarray(arm_length_m, dtype=DTYPE_R_NPY))
        if arm_length_m.size == 1:
            arm_length_m = np.repeat(arm_length_m, 2)

        if u_body is None:
            u_body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE_R_NPY)

        # Compute cos(θ) for all beam pixels
        npix = healpy.nside2npix(nside)
        N_GAL = np.array(healpy.pix2vec(nside, np.arange(npix)))  # (3, npix)
        cos_theta = u_body @ N_GAL  # (2, npix)

        # Evaluate thin-dipole beam at all frequencies
        nominal_beam = np.zeros((2, npix, len(freqs_hz)), dtype=DTYPE_R_NPY)
        for f_idx, f_hz in enumerate(freqs_hz):
            kh_f = arm_length_m * np.pi * f_hz / C_LIGHT
            nominal_beam[:, :, f_idx] = thin_dipole_pattern(kh_f[:, np.newaxis], cos_theta)

        # SVD along frequency axis for each dipole (to extract spectral modes)
        # We build a shared basis from the average or dominant dipole
        B_avg = (nominal_beam[0] + nominal_beam[1]) / 2  # (npix, nfreq)
        U, s, Vt = np.linalg.svd(B_avg, full_matrices=False)
        A = Vt[:K].T  # (nfreq, K)

        return cls(A, freqs_hz=freqs_hz)


class SkyBasis(_SpectralBasis):
    """Spectral basis for sky brightness models.

    Typically GSM eigenmodes or custom foreground filtering bases.
    """

    @classmethod
    def from_gsm(cls, freqs_hz, n_modes=5, nside=8, include_flat=True):
        """Build sky basis from GSM16 eigenmodes.

        Loads the Global Sky Model (GSM16) via pygdsm, resamples to target
        nside, then performs SVD to extract dominant spectral modes. Optionally
        includes a flat (constant) mode orthogonalized against GSM modes.

        Parameters
        ----------
        freqs_hz : ndarray, shape (nfreq,)
            Frequencies [Hz].
        n_modes : int
            Number of GSM eigenmodes to retain (default 5).
        nside : int
            HEALPix resolution (default 8).
        include_flat : bool
            If True, append a normalized flat mode orthogonalized against
            GSM modes to span the common-mode degeneracy (default True).

        Returns
        -------
        SkyBasis
            Basis object with GSM eigenmodes (+ flat mode if requested).
        """
        if not HAS_GSM:
            raise ImportError("pygdsm is required for from_gsm(); install it via "
                            "`pip install pygdsm`")

        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        freqs_mhz = freqs_hz / 1e6

        # Load GSM16 and resample to target nside
        gsm = pygdsm.GlobalSkyModel16(freq_unit='MHz', resolution='lo')
        gsm_maps = []
        for f_mhz in freqs_mhz:
            m = gsm.generate(f_mhz)  # Returns map at nside=1024
            m_hp = healpy.ud_grade(m, nside)
            gsm_maps.append(m_hp)
        gsm_maps = np.array(gsm_maps).T  # (npix, nfreq)

        # SVD to get dominant spectral modes
        U, s, Vt = np.linalg.svd(gsm_maps, full_matrices=False)
        modes = Vt[:n_modes].T  # (nfreq, n_modes)

        # Optionally add flat mode
        if include_flat:
            nfreq = len(freqs_hz)
            flat = np.ones(nfreq, dtype=DTYPE_R_NPY) / np.sqrt(nfreq)
            # Orthogonalize against GSM modes
            flat_orth = flat - modes @ (modes.T @ flat)
            norm = np.linalg.norm(flat_orth)
            if norm > 1e-10:
                flat_orth = flat_orth / norm
                modes = np.column_stack([modes, flat_orth])

        return cls(modes, freqs_hz=freqs_hz)


def _resample_basis(old_freqs, A_old, new_freqs):
    """Resample basis matrix A to new frequencies via linear interpolation.

    Parameters
    ----------
    old_freqs : ndarray, shape (nfreq_old,)
        Original frequency grid.
    A_old : ndarray, shape (nfreq_old, nmodes)
        Original basis matrix.
    new_freqs : ndarray, shape (nfreq_new,)
        Target frequency grid.

    Returns
    -------
    ndarray, shape (nfreq_new, nmodes)
        Resampled basis matrix.
    """
    nmodes = A_old.shape[1]
    A_new = np.zeros((len(new_freqs), nmodes), dtype=DTYPE_R_NPY)
    for m in range(nmodes):
        interp = interp1d(old_freqs, A_old[:, m], kind='linear',
                         bounds_error=False, fill_value=0.0)
        A_new[:, m] = interp(new_freqs)
    return A_new
