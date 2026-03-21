"""
Spectral foreground filtering for global 21cm analysis.

Functions
---------
gsm_eigenmodes   — compute dominant spectral eigenmodes from GSM sky maps
eigenmode_filter — project GSM eigenmodes out of a spectrum or array of spectra

log_poly_basis   — design matrix for log-polynomial foreground fits
fit_foreground   — fit and subtract a log-polynomial from a spectrum
project_signal   — project a signal through the foreground-subtraction operator
"""

import numpy as np


def gsm_eigenmodes(gsm_maps, n_modes, include_flat=True):
    """
    Compute the dominant spectral eigenmodes of the GSM.

    Performs a thin SVD of the pixel-by-frequency GSM map matrix and returns
    the top ``n_modes`` right singular vectors.  When ``include_flat=True``
    (the default), a spectrally flat (constant) mode is appended after
    Gram-Schmidt orthogonalisation against the GSM modes.  This flat mode
    spans the common-mode receiver-temperature degeneracy (T_rx adds a
    frequency-independent offset to every measurement) and ensures it is
    removed alongside the smooth synchrotron foreground.

    Parameters
    ----------
    gsm_maps : ndarray, shape (npix, N_FREQ)
        GSM brightness-temperature maps at each frequency [K].
    n_modes : int
        Number of GSM spectral eigenmodes to retain.
    include_flat : bool
        If True (default), append a normalised flat mode (orthogonalised
        against the GSM modes) to the returned array.

    Returns
    -------
    modes : ndarray, shape (n_modes [+ 1], N_FREQ)
        Unit-norm orthogonal row vectors.  The last row is the flat mode
        when ``include_flat=True`` and it is not already spanned by the
        GSM eigenmodes.
    """
    _, _, Vt = np.linalg.svd(gsm_maps, full_matrices=False)
    modes = Vt[:n_modes]

    if include_flat:
        N_FREQ = gsm_maps.shape[1]
        flat = np.ones(N_FREQ) / np.sqrt(N_FREQ)
        # Remove the component already captured by the GSM eigenmodes
        flat_orth = flat - modes.T @ (modes @ flat)
        norm = np.linalg.norm(flat_orth)
        if norm > 1e-10:
            modes = np.vstack([modes, flat_orth / norm])

    return modes


def eigenmode_filter(spectrum, modes):
    """
    Project GSM spectral eigenmodes out of a spectrum (or array of spectra).

    Removes the component of each spectrum that lies in the subspace spanned
    by ``modes``:

        filtered = spectrum − modes.T (modes · spectrum)

    Parameters
    ----------
    spectrum : ndarray, shape (N_FREQ,) or (n_models, N_FREQ)
        Spectrum or batch of spectra to filter.
    modes : ndarray, shape (n_modes, N_FREQ)
        Spectral eigenmodes to remove (unit-norm row vectors from
        :func:`gsm_eigenmodes`).

    Returns
    -------
    filtered : ndarray, same shape as ``spectrum``
    """
    spectrum = np.asarray(spectrum, dtype=float)
    # Works for both 1-D (N_FREQ,) and 2-D (n_models, N_FREQ):
    #   coeffs = (..., n_modes) = spectrum @ modes.T
    #   projection = (..., N_FREQ) = coeffs @ modes
    return spectrum - (spectrum @ modes.T) @ modes


def log_poly_basis(freqs_mhz, n_terms, f_ref=None):
    """
    Log-polynomial basis matrix.

    Column k is (log(f / f_ref))^k  for  k = 0, 1, …, n_terms − 1.

    Parameters
    ----------
    freqs_mhz : array_like, shape (nf,)
    n_terms   : int
        Number of basis terms.  Typical values: 3–7.
    f_ref     : float or None
        Reference frequency [MHz].  Defaults to the geometric mean of
        ``freqs_mhz`` so the basis is well-conditioned.

    Returns
    -------
    B : ndarray, shape (nf, n_terms)
    """
    freqs_mhz = np.asarray(freqs_mhz, dtype=float)
    if f_ref is None:
        f_ref = np.exp(np.mean(np.log(freqs_mhz)))
    x = np.log(freqs_mhz / f_ref)
    return np.column_stack([x ** k for k in range(n_terms)])


def fit_foreground(freqs_mhz, T_obs, n_terms=5, f_ref=None, weights=None):
    """
    Fit and subtract a log-polynomial foreground from a monopole spectrum.

    Parameters
    ----------
    freqs_mhz : array_like, shape (nf,)
    T_obs     : array_like, shape (nf,)
        Observed monopole spectrum [K].
    n_terms   : int
        Number of log-polynomial terms.
    f_ref     : float or None
        Reference frequency [MHz] for the basis (geometric mean if None).
    weights   : array_like, shape (nf,) or None
        Inverse-variance weights.  If None, ordinary least squares is used.

    Returns
    -------
    residual : ndarray, shape (nf,)
        T_obs minus the best-fit foreground.
    T_fg     : ndarray, shape (nf,)
        Best-fit foreground model.
    coeffs   : ndarray, shape (n_terms,)
        Polynomial coefficients.
    """
    B = log_poly_basis(freqs_mhz, n_terms, f_ref)
    T_obs = np.asarray(T_obs, dtype=float)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        coeffs, *_ = np.linalg.lstsq(B * w[:, None], T_obs * w, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(B, T_obs, rcond=None)

    T_fg = B @ coeffs
    return T_obs - T_fg, T_fg, coeffs


def project_signal(freqs_mhz, T_signal, n_terms=5, f_ref=None):
    """
    Project a signal through the foreground-subtraction operator.

    Returns the component of ``T_signal`` that survives after subtracting
    the best-fit ``n_terms``-term log-polynomial, i.e.

        T_proj = (I − B (BᵀB)⁻¹ Bᵀ) · T_signal

    This is what a 21cm model would look like in the foreground-subtracted
    residual, accounting for the partial absorption of signal power into the
    foreground basis.

    Parameters
    ----------
    freqs_mhz : array_like, shape (nf,)
    T_signal  : array_like, shape (nf,)
    n_terms   : int
    f_ref     : float or None

    Returns
    -------
    T_proj : ndarray, shape (nf,)
    """
    T_signal = np.asarray(T_signal, dtype=float)
    B = log_poly_basis(freqs_mhz, n_terms, f_ref)
    fit, *_ = np.linalg.lstsq(B, T_signal, rcond=None)
    return T_signal - B @ fit
