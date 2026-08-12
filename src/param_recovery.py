"""Variable-projection recovery of a parametric dipole beam + GSM sky.

Companion to :class:`eigsep_sim.calibrator.Calibrator`, which recovers per-pixel
beam SVD coefficients (the "v001" pipeline, left fully intact).  Here the beam is
parametrised by *physics* — per-dipole arm length and orientation (azimuth,
elevation) — so it has only ``n_dipoles * 3`` degrees of freedom and cannot
absorb the isotropic T21 monopole.  The sky is linear given the beam, so it is
eliminated by an exact weighted least-squares solve (variable projection, VarPro)
and only the handful of beam parameters are optimised with Levenberg-Marquardt.

Also provides two T21 matched-filter targets, both usable with *either* beam
parametrisation (pass ``beam_maps = beam_coeffs @ basis.A.T`` for the coefficient
beam, or :func:`eigsep_sim.beam.dipole_beam_maps_jax` output for the parametric
beam):

* :func:`t21_filter_spectral` — fast, approximate: removes the part of the T21
  spectrum in ``span(A_sky)`` (frequency space only).
* :func:`t21_filter_forward` — exact: removes the part of the T21 *data
  signature* the sky absorbs through the weighted forward model (beam-weighted,
  occultation/``f_vis``-modulated, inverse-variance weighted).  Equals the
  recovered T21 in the noiseless / exact-beam limit.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import healpy

from .const import DTYPE_R_JAX, DTYPE_R_NPY
from .beam import (
    dipole_beam_maps_jax,
    dipole_axes_from_angles,
    v_dipole_beam_maps_jax,
)


# ── parametrisation helpers ─────────────────────────────────────────────────

def beam_pix_vecs(nside):
    """Body-frame HEALPix pixel unit vectors ``(npix, 3)`` for the beam grid."""
    npix = healpy.nside2npix(int(nside))
    return np.asarray(
        healpy.pix2vec(int(nside), np.arange(npix)), dtype=DTYPE_R_NPY
    ).T


def pack_dipole_params(arm_lengths_m, u_body):
    """Pack ``(arm_lengths, axes)`` into the flat phys vector.

    Layout: ``[L_0 … L_{D-1}, az_0, el_0, az_1, el_1, …]`` (radians for angles).
    ``el = 0`` reproduces the planar crossed-dipole geometry.
    """
    u = np.asarray(u_body, dtype=DTYPE_R_NPY)
    az = np.arctan2(u[:, 1], u[:, 0])
    el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
    arms = np.asarray(arm_lengths_m, dtype=DTYPE_R_NPY).ravel()
    return np.concatenate([arms, np.stack([az, el], axis=1).ravel()])


def unpack_dipole_params(phys, n_dipoles):
    """Split the flat phys vector into ``(arm_lengths (D,), angles (D, 2))``."""
    n_dipoles = int(n_dipoles)
    return phys[:n_dipoles], phys[n_dipoles:].reshape(n_dipoles, 2)


def pack_v_dipole_params(lengths_m, axis_angles_rad, opening_angles_deg,
                         clock_angles_deg=None):
    """Pack explicit V-dipole parameters into a flat physical vector.

    Layout: ``[L..., axis_az/el..., opening_rad..., clock_rad...]``.  Each row
    of ``axis_angles_rad`` gives the symmetry axis of one V dipole.  The opening
    angles are the physical angle between the two arms.
    """
    lengths = np.asarray(lengths_m, dtype=DTYPE_R_NPY).ravel()
    axis_angles = np.asarray(axis_angles_rad, dtype=DTYPE_R_NPY).reshape(
        len(lengths), 2
    )
    openings = np.deg2rad(np.asarray(opening_angles_deg, dtype=DTYPE_R_NPY).ravel())
    if clock_angles_deg is None:
        clocks = np.zeros_like(openings)
    else:
        clocks = np.deg2rad(np.asarray(clock_angles_deg, dtype=DTYPE_R_NPY).ravel())
    if openings.shape != lengths.shape or clocks.shape != lengths.shape:
        raise ValueError(
            "lengths, opening angles, and clock angles must have matching length"
        )
    return np.concatenate([lengths, axis_angles.ravel(), openings, clocks])


def unpack_v_dipole_params(phys, n_dipoles):
    """Split explicit V-dipole vector into length, axis, opening, and clock."""
    n_dipoles = int(n_dipoles)
    phys = jnp.asarray(phys)
    i0 = n_dipoles
    i1 = i0 + 2 * n_dipoles
    i2 = i1 + n_dipoles
    i3 = i2 + n_dipoles
    if int(np.size(phys)) != i3:
        raise ValueError(
            f"expected {i3} V-dipole parameters for {n_dipoles} dipoles"
        )
    return (
        phys[:i0],
        phys[i0:i1].reshape(n_dipoles, 2),
        phys[i1:i2],
        phys[i2:i3],
    )


def build_gsm_sky_prior(gsm_maps, sky_coeffs_ref, rep_tol=1.0, max_modes=None):
    """Spatial sky prior ``U_gsm = orthonormal[flat monopole, GSM SVD modes]``.

    The forward model's spatial basis is the HEALPix pixels, but for *recovery*
    the ~1e4 free pixels are underdetermined by the data and would absorb
    T21/noise.  Restricting the sky to the smooth GSM low-rank subspace is a
    physical *prior* (the foreground is the GSM).  The pure monopole (flat) mode
    is included EXPLICITLY — ``sky.basis.svd_modes`` omits it, and the leading
    GSM SVD mode is the mean-sky *shape* not a constant, so without an explicit
    flat mode the sky monopole and the T21 monopole cannot be separated (T21
    then blows up).  GSM SVD modes are added until ``sky_coeffs_ref`` is
    represented to ``< rep_tol`` [K], so the subspace faithfully spans the sky.

    Parameters
    ----------
    gsm_maps : ndarray, shape (npix, nfreq)
        Sky-model maps used to derive the spatial modes.
    sky_coeffs_ref : ndarray, shape (npix, n_spectral)
        Reference sky coefficients whose representability sets the mode count.
    rep_tol : float
        Maximum per-pixel representation error [K].
    max_modes : int, optional
        Cap on the number of GSM SVD modes added.

    Returns
    -------
    U_gsm : ndarray, shape (npix, n_spatial)
        Orthonormal spatial basis (flat mode first).
    """
    gsm_maps = np.asarray(gsm_maps, dtype=DTYPE_R_NPY)
    ref = np.asarray(sky_coeffs_ref, dtype=DTYPE_R_NPY)
    npix = gsm_maps.shape[0]
    flat = np.ones((npix, 1), dtype=DTYPE_R_NPY) / np.sqrt(npix)
    Ug, _, _ = np.linalg.svd(gsm_maps, full_matrices=False)
    kmax = Ug.shape[1] if max_modes is None else min(int(max_modes), Ug.shape[1])
    U = np.linalg.qr(flat)[0]
    for k in range(1, kmax + 1):
        U = np.linalg.qr(np.concatenate([flat, Ug[:, :k]], axis=1))[0]
        if np.max(np.abs(U @ (U.T @ ref) - ref)) < rep_tol:
            break
    return np.ascontiguousarray(U, dtype=DTYPE_R_NPY)


# ── T21 matched-filter targets (usable with any beam via beam_maps) ──────────

def t21_matched_filter(residual, template):
    """Per-frequency matched-filter amplitude ``sum(res*tmpl)/sum(tmpl**2)``.

    ``residual`` and ``template`` are ``(ntimes, n_dipoles, nfreq)`` arrays;
    the sum is over time and dipole.  Returns ``(nfreq,)``.
    """
    r = np.asarray(residual, dtype=DTYPE_R_NPY)
    t = np.asarray(template, dtype=DTYPE_R_NPY)
    return (r * t).sum(axis=(0, 1)) / (t ** 2).sum(axis=(0, 1))


def t21_template(fwd, geom, beam_maps):
    """Matched-filter template and terrain model for a given beam.

    Returns ``(template, terrain)`` each ``(ntimes, n_dipoles, nfreq)`` where
    ``template = simulate(0, T_iso=1) - simulate(0, T_iso=None) =
    visible_weight / denom`` (accounts for time-varying occultation).
    """
    fwd._ensure_jax_arrays()
    n_spectral = int(fwd._sky_basis_A_jax.shape[1])
    nfreq = len(fwd.beam.freqs_hz)
    zsky = jnp.zeros((int(fwd.beam.npix), n_spectral), dtype=DTYPE_R_JAX)
    terr = np.asarray(fwd.simulate(zsky, beam_maps=beam_maps, geom=geom, T_iso=None))
    iso1 = np.asarray(
        fwd.simulate(zsky, beam_maps=beam_maps, geom=geom,
                     T_iso=np.ones(nfreq, dtype=DTYPE_R_NPY))
    )
    return iso1 - terr, terr


def t21_filter_spectral(t21, A_sky):
    """Fast, approximate T21 filter: remove the part of T21 in ``span(A_sky)``.

    A quick estimate of what the GSM sky can absorb — the projection of the T21
    spectrum onto the sky spectral basis, in raw (unweighted) frequency space.
    Ignores the beam / occultation / inverse-variance weighting of the actual
    fit, so it differs from the true filtered target by a smooth low-order
    function of frequency; use :func:`t21_filter_forward` for the exact target.

    Returns the surviving T21 spectrum ``(nfreq,)``.
    """
    A = np.asarray(A_sky, dtype=DTYPE_R_NPY)
    Q, _ = np.linalg.qr(A)
    t = np.asarray(t21, dtype=DTYPE_R_NPY)
    return t - Q @ (Q.T @ t)


def _sky_design(fwd, geom, beam_maps, sky_basis, weight):
    """Weighted sky Jacobian ``Jc`` and weighted terrain, for the linear sky.

    ``Jc[n_data, n_sky]`` is ``d(weighted prediction)/d(sky coeffs)`` at the GSM
    spatial basis ``sky_basis`` (npix, n_spatial); constant since the model is
    linear in the sky.  ``weight`` is the ``sqrt(inv_noise_var)`` array.
    """
    fwd._ensure_jax_arrays()
    U = jnp.asarray(np.asarray(sky_basis), dtype=DTYPE_R_JAX)
    n_spatial = int(U.shape[1])
    n_spectral = int(fwd._sky_basis_A_jax.shape[1])
    n_sky = n_spatial * n_spectral
    w = jnp.asarray(np.asarray(weight), dtype=DTYPE_R_JAX)

    def predw(cflat):
        c = cflat.reshape(n_spatial, n_spectral)
        pred = fwd.simulate(U @ c, beam_maps=beam_maps, geom=geom, T_iso=None)
        return (pred * w).ravel()

    c0 = jnp.zeros(n_sky, dtype=DTYPE_R_JAX)
    const_w = predw(c0)
    Jc = jax.jacfwd(predw)(c0)
    return Jc, const_w, U, n_spatial, n_spectral


def _sky_lstsq(Jc, target_weighted, ridge):
    """Solve the ridged weighted normal equations ``(JcᵀJc+r)c = Jcᵀ tgt``."""
    n_sky = Jc.shape[1]
    A = Jc.T @ Jc
    A = A + ridge * (jnp.trace(A) / n_sky) * jnp.eye(n_sky, dtype=A.dtype)
    return jnp.linalg.solve(A, Jc.T @ target_weighted)


def t21_filter_forward(t21, fwd, geom, beam_maps, sky_basis, inv_noise_var,
                       ridge=1e-8):
    """Exact T21 filtered target via the weighted forward-model sky projection.

    Runs the T21 *data signature* ``simulate(0, T_iso=t21) - terrain`` through
    the SAME weighted sky least-squares the fit uses, and matched-filters the
    remainder (the part the sky cannot absorb).  This is the fair recovery
    target: in the noiseless / exact-beam limit it equals the matched-filter
    estimate of the recovered residual.  Works for any beam via ``beam_maps``.

    Returns the surviving T21 spectrum ``(nfreq,)``.
    """
    w = np.sqrt(np.asarray(inv_noise_var, dtype=DTYPE_R_NPY))
    tmpl, terr = t21_template(fwd, geom, beam_maps)
    contrib = np.asarray(
        fwd.simulate(
            jnp.zeros((int(fwd.beam.npix),
                       int(fwd._sky_basis_A_jax.shape[1])), dtype=DTYPE_R_JAX),
            beam_maps=beam_maps, geom=geom,
            T_iso=jnp.asarray(np.asarray(t21), dtype=DTYPE_R_JAX),
        )
    ) - terr
    Jc, _const, U, n_spatial, n_spectral = _sky_design(
        fwd, geom, beam_maps, sky_basis, w
    )
    tgt = (jnp.asarray(contrib, dtype=DTYPE_R_JAX)
           * jnp.asarray(w, dtype=DTYPE_R_JAX)).ravel()
    c_ab = _sky_lstsq(Jc, tgt, ridge).reshape(n_spatial, n_spectral)
    absorbed = np.asarray(
        fwd.simulate(U @ c_ab, beam_maps=beam_maps, geom=geom, T_iso=None)
    ) - terr
    return t21_matched_filter(contrib - absorbed, tmpl)


# ── VarPro parametric-beam calibrator ───────────────────────────────────────

class DipoleBeamVarPro:
    """Joint dipole-beam + GSM-sky recovery by variable projection.

    Parameters
    ----------
    fwd : ForwardModel
        Forward model (its ``beam`` supplies ``nside``/``n_dipoles``/``freqs``).
    data : ndarray, shape (ntimes, n_dipoles, nfreq)
        Observed antenna temperatures.
    inv_noise_var : ndarray, broadcastable to ``data``
        Inverse noise variance (weights).
    sky_basis : ndarray, shape (npix, n_spatial)
        Orthonormal spatial sky prior (see :func:`build_gsm_sky_prior`).
    geom : dict
        Pre-computed geometry from ``fwd.precompute_geometry``.
    pix_vecs : ndarray, optional
        Body-frame pixel vectors; derived from ``fwd.beam.nside`` if omitted.
    ridge : float
        Tiny Tikhonov ridge on the (well-conditioned) sky solve.  Keep small:
        a large ridge biases the recovered T21 (see the module tests).
    """

    def __init__(self, fwd, data, inv_noise_var, sky_basis, geom,
                 pix_vecs=None, ridge=1e-8):
        fwd._ensure_jax_arrays()
        self.fwd = fwd
        self.geom = geom
        self.ridge = float(ridge)
        self._data = jnp.asarray(np.asarray(data), dtype=DTYPE_R_JAX)
        self._invvar = np.broadcast_to(
            np.asarray(inv_noise_var, dtype=DTYPE_R_NPY), np.asarray(data).shape
        )
        self._w = jnp.asarray(np.sqrt(self._invvar), dtype=DTYPE_R_JAX)
        self.U = jnp.asarray(np.asarray(sky_basis), dtype=DTYPE_R_JAX)
        self.n_spatial = int(self.U.shape[1])
        self.n_spectral = int(fwd._sky_basis_A_jax.shape[1])
        self.n_sky = self.n_spatial * self.n_spectral
        self.n_dipoles = int(fwd.beam.n_dipoles)
        self.freqs_hz = jnp.asarray(fwd.beam.freqs_hz, dtype=DTYPE_R_JAX)
        if pix_vecs is None:
            pix_vecs = beam_pix_vecs(fwd.beam.nside)
        self.pix_vecs = jnp.asarray(np.asarray(pix_vecs), dtype=DTYPE_R_JAX)

    # -- forward pieces -----------------------------------------------------
    def beam_maps(self, phys):
        """Dipole beam maps ``(n_dipoles, npix, nfreq)`` for a phys vector."""
        arms, angles = unpack_dipole_params(jnp.asarray(phys), self.n_dipoles)
        return dipole_beam_maps_jax(
            arms, dipole_axes_from_angles(angles), self.freqs_hz, self.pix_vecs
        )

    def _weighted_pred(self, cflat, maps):
        c = cflat.reshape(self.n_spatial, self.n_spectral)
        pred = self.fwd.simulate(self.U @ c, beam_maps=maps, geom=self.geom,
                                 T_iso=None)
        return (pred * self._w).ravel()

    def sky_solve(self, phys):
        """Exact weighted least-squares sky coeffs ``c (n_spatial, n_spectral)``."""
        maps = self.beam_maps(phys)
        c0 = jnp.zeros(self.n_sky, dtype=DTYPE_R_JAX)
        predw = lambda cf: self._weighted_pred(cf, maps)  # noqa: E731
        const_w = predw(c0)
        Jc = jax.jacfwd(predw)(c0)
        tgt = (self._data * self._w).ravel() - const_w
        return _sky_lstsq(Jc, tgt, self.ridge).reshape(
            self.n_spatial, self.n_spectral
        )

    def predict(self, phys):
        """Model antenna temperatures at ``phys`` with the sky solved exactly."""
        c = self.sky_solve(phys)
        return self.fwd.simulate(self.U @ c, beam_maps=self.beam_maps(phys),
                                 geom=self.geom, T_iso=None)

    def residual(self, phys):
        """Weighted, flattened residual (sky held at its optimum: Kaufman VarPro)."""
        c = jax.lax.stop_gradient(self.sky_solve(phys))
        pred = self.fwd.simulate(self.U @ c, beam_maps=self.beam_maps(phys),
                                 geom=self.geom, T_iso=None)
        return ((pred - self._data) * self._w).ravel()

    def loss(self, phys):
        r = self.residual(phys)
        return float(jnp.mean(r ** 2))

    # -- optimisation -------------------------------------------------------
    def fit(self, phys0, max_iter=20, lam0=1e-2, tol=1e-8, verbose=False):
        """Levenberg-Marquardt over the beam params (sky eliminated each eval).

        Returns a dict with ``phys``, ``loss``, ``losses``, and the recovered
        ``sky_coeffs`` (npix, n_spectral).
        """
        jac = jax.jacfwd(self.residual)
        phys = jnp.asarray(np.asarray(phys0), dtype=DTYPE_R_JAX)
        lam, L = float(lam0), self.loss(phys)
        losses = [L]
        for it in range(int(max_iter)):
            J = np.asarray(jac(phys), dtype=np.float64)
            r = np.asarray(self.residual(phys), dtype=np.float64)
            A, b = J.T @ J, J.T @ r
            dg, dL = np.diag(np.diag(A)), 0.0
            for _ls in range(16):
                delta = -np.linalg.solve(A + lam * dg, b)
                phys_new = phys + jnp.asarray(delta)
                L_new = self.loss(phys_new)
                if L_new < L:
                    phys, lam, dL, L = phys_new, max(lam * 0.5, 1e-11), L - L_new, L_new
                    break
                lam *= 4.0
            else:
                if verbose:
                    print(f"  iter {it:2d}: converged (no downhill step)")
                break
            losses.append(L)
            if verbose:
                p = np.asarray(phys)
                print(f"  iter {it:2d}: loss={L:.4e}  lam={lam:.1e}  "
                      f"arms={np.round(p[:self.n_dipoles], 4)}  dL={dL:.2e}")
            if dL < tol * L:
                if verbose:
                    print(f"  converged: relative decrease {dL/L:.1e} < tol {tol:.1e}")
                break
        c = self.sky_solve(phys)
        return {
            "phys": np.asarray(phys),
            "loss": L,
            "losses": np.asarray(losses),
            "sky_coeffs": np.asarray(self.U @ c),
        }

    # -- T21 -----------------------------------------------------------------
    def recover_t21(self, phys):
        """Matched-filter T21 estimate from the data residual at beam ``phys``."""
        maps = self.beam_maps(phys)
        tmpl, _terr = t21_template(self.fwd, self.geom, maps)
        res = np.asarray(self._data) - np.asarray(self.predict(phys))
        return t21_matched_filter(res, tmpl)

    def t21_filter_forward(self, t21, phys):
        """Exact filtered-truth target at beam ``phys`` (see module function)."""
        return t21_filter_forward(
            t21, self.fwd, self.geom, self.beam_maps(phys), self.U,
            self._invvar, ridge=self.ridge,
        )


class VDipoleBeamVarPro(DipoleBeamVarPro):
    """VarPro recovery for coherent two-arm V dipoles.

    The fitted physical vector is explicit in the V geometry:
    ``[lengths, axis az/el pairs, opening angles, clock angles]``.  Opening and
    clock angles are in radians inside the vector; use
    :func:`pack_v_dipole_params` to build it from degree-valued engineering
    inputs.
    """

    def beam_maps(self, phys):
        lengths, axis_angles, opening_angles, clock_angles = unpack_v_dipole_params(
            jnp.asarray(phys), self.n_dipoles
        )
        return v_dipole_beam_maps_jax(
            lengths,
            axis_angles,
            opening_angles,
            clock_angles,
            self.freqs_hz,
            self.pix_vecs,
        )
