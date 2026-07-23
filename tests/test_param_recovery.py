#!/usr/bin/env python
"""Tests for the parametric dipole-beam VarPro recovery (param_recovery.py).

Guards the "v002" pipeline: the beam is parametrised by physical dipole arm
length + orientation (6 DOF for 2 dipoles), the sky is eliminated by an exact
weighted least-squares solve, and only the beam params are optimised.  Because
the beam has so few DOF it cannot absorb the isotropic T21 monopole (unlike the
per-pixel coefficient beam), so recovery tracks the filtered T21 target.

Also checks the two T21 matched-filter targets and that the coefficient beam
(v001) forward path is unchanged (regression against beam_maps refactor).
"""

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u

from eigsep_sim import (
    Beam, Sky, ForwardModel,
    DipoleBeamVarPro, build_gsm_sky_prior, beam_pix_vecs,
    pack_dipole_params, unpack_dipole_params,
    dipole_beam_maps_jax, dipole_axes_from_angles,
    t21_filter_spectral, t21_filter_forward, t21_template,
)
from eigsep_sim.observer import LunarOrbit


def build_param_problem(nside=8, nfreq=20, seed=0, noiseless=False):
    """Small lunar-orbit problem with a parametric dipole beam + T21."""
    freqs = np.linspace(30e6, 120e6, nfreq)
    arms = [6.0, 4.0]
    a = np.deg2rad(45.0)
    u_body = np.array([[np.cos(a), np.sin(a), 0.0],
                       [np.cos(a), -np.sin(a), 0.0]])
    beam = Beam.from_dipole(nside, freqs, arm_lengths_m=arms, u_body=u_body, K=3)
    sky = Sky.from_gsm(nside, freqs, n_modes=3, include_flat=True)
    obs = LunarOrbit(
        altitude=1e5, rot_orbit_vec=[0.2, 0, 1], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"), occultation_temperature_K=250.0,
    )
    fwd = ForwardModel(obs, beam, sky)
    times = Time("2030-01-01") + np.linspace(0, 6 * 3600, 40) * u.s
    from scipy.spatial.transform import Rotation
    brots = np.stack(
        [Rotation.from_rotvec([0, 0, th]).as_matrix()
         for th in np.linspace(0, 3, 40)]
    )
    geom = fwd.precompute_geometry(times=times, body_rots=brots)

    sky_coeffs = sky.init_coeffs()
    gsm_maps = np.asarray(sky_coeffs) @ np.asarray(sky.basis.A).T
    T21 = (-0.1 * np.exp(-0.5 * ((freqs / 1e6 - 75) / 8) ** 2)).astype(np.float64)

    pix_vecs = beam_pix_vecs(nside)
    phys_true = pack_dipole_params(arms, u_body)
    maps_true = dipole_beam_maps_jax(
        phys_true[:2], dipole_axes_from_angles(phys_true[2:].reshape(2, 2)),
        freqs, pix_vecs,
    )
    data_clean = np.asarray(
        fwd.simulate(sky_coeffs, beam_maps=maps_true, geom=geom, T_iso=T21)
    )
    sigma = np.full((2, nfreq), 0.02)
    if noiseless:
        noise = np.zeros_like(data_clean)
    else:
        rng = np.random.default_rng(seed)
        noise = (rng.standard_normal(data_clean.shape) * sigma[None]).astype(np.float64)
    data = data_clean + noise
    inv_var = np.broadcast_to((1.0 / sigma ** 2)[None], data.shape).copy()

    sky_basis = build_gsm_sky_prior(gsm_maps, np.asarray(sky_coeffs), rep_tol=1.0)
    return dict(
        fwd=fwd, geom=geom, data=data, inv_var=inv_var, sky_basis=sky_basis,
        phys_true=phys_true, T21=T21, sky=sky, beam=beam, sky_coeffs=sky_coeffs,
    )


def test_pack_unpack_roundtrip():
    u_body = np.array([[np.cos(0.6), np.sin(0.6), 0.1],
                       [np.cos(-0.4), np.sin(-0.4), -0.05]])
    u_body /= np.linalg.norm(u_body, axis=1, keepdims=True)
    phys = pack_dipole_params([6.0, 4.0], u_body)
    arms, angles = unpack_dipole_params(phys, 2)
    axes = np.asarray(dipole_axes_from_angles(angles))
    np.testing.assert_allclose(arms, [6.0, 4.0])
    np.testing.assert_allclose(axes, u_body, atol=1e-12)


def test_gsm_prior_includes_monopole_and_represents_sky():
    p = build_param_problem(nside=8, nfreq=12)
    U = p["sky_basis"]
    # flat/monopole mode is representable (constant map in the span)
    flat = np.ones(U.shape[0]) / np.sqrt(U.shape[0])
    resid_flat = flat - U @ (U.T @ flat)
    assert np.max(np.abs(resid_flat)) < 1e-8
    # the true sky is represented to < 1 K
    ref = np.asarray(p["sky_coeffs"])
    assert np.max(np.abs(U @ (U.T @ ref) - ref)) < 1.0


def test_varpro_recovers_beam_and_reaches_floor():
    p = build_param_problem(nside=8, nfreq=20, seed=0)
    vp = DipoleBeamVarPro(
        p["fwd"], p["data"], p["inv_var"], p["sky_basis"], p["geom"],
    )
    phys_true = p["phys_true"]
    # perturb: arms +2%, opening +-1 deg, out-of-plane elevation tilt
    phys0 = np.asarray(phys_true, dtype=float).copy()
    phys0[:2] *= 1.02
    phys0[2] += np.deg2rad(1.0)
    phys0[4] -= np.deg2rad(1.0)
    phys0[3] += 0.03
    phys0[5] -= 0.03

    floor = vp.loss(phys_true)
    res = vp.fit(phys0, max_iter=25, tol=1e-9)

    # arm lengths recovered to sub-mm; angles (incl. out-of-plane el) to < 0.1 deg
    np.testing.assert_allclose(res["phys"][:2], phys_true[:2], atol=1e-3)
    np.testing.assert_allclose(res["phys"][2:], phys_true[2:], atol=2e-3)
    # final loss sits at the noise floor (does NOT plunge below it)
    assert res["loss"] <= 1.05 * floor
    assert res["loss"] >= 0.5 * floor


def test_t21_forward_filter_matches_recovered_noiseless():
    # In the noiseless / exact-beam limit the forward-model filtered target
    # equals the matched-filter estimate of the recovered residual.
    p = build_param_problem(nside=8, nfreq=20, noiseless=True)
    vp = DipoleBeamVarPro(
        p["fwd"], p["data"], p["inv_var"], p["sky_basis"], p["geom"], ridge=1e-10,
    )
    phys_true = p["phys_true"]
    T21_rec = vp.recover_t21(phys_true)
    T21_filt = vp.t21_filter_forward(p["T21"], phys_true)
    # agree to well below a mK
    assert np.max(np.abs(T21_rec - T21_filt)) < 1e-3


def test_t21_spectral_and_forward_filters_differ_smoothly():
    # Fast spectral filter is a low-order-smooth approximation of the exact
    # forward-model filter; both remove a similar fraction of the T21 peak.
    p = build_param_problem(nside=8, nfreq=20, noiseless=True)
    vp = DipoleBeamVarPro(p["fwd"], p["data"], p["inv_var"], p["sky_basis"],
                          p["geom"], ridge=1e-10)
    A_sky = np.asarray(p["sky"].basis.A)
    T21_spec = t21_filter_spectral(p["T21"], A_sky)
    T21_fwd = vp.t21_filter_forward(p["T21"], p["phys_true"])
    # both are strictly smaller in magnitude than the raw signal
    assert abs(T21_spec.min()) < abs(p["T21"].min())
    assert abs(T21_fwd.min()) < abs(p["T21"].min())
    # they broadly agree (same sign, comparable depth) but are not identical
    assert T21_spec.min() < 0 and T21_fwd.min() < 0
    assert 0.4 < T21_fwd.min() / T21_spec.min() < 2.5


def test_ridge_bias_on_recovered_t21():
    # A too-large ridge biases the recovered T21; the small default does not.
    p = build_param_problem(nside=8, nfreq=20, noiseless=True)
    phys = p["phys_true"]
    vp_small = DipoleBeamVarPro(p["fwd"], p["data"], p["inv_var"],
                                p["sky_basis"], p["geom"], ridge=1e-10)
    vp_big = DipoleBeamVarPro(p["fwd"], p["data"], p["inv_var"],
                              p["sky_basis"], p["geom"], ridge=1e-3)
    t_small = vp_small.recover_t21(phys)
    t_big = vp_big.recover_t21(phys)
    ref = vp_small.t21_filter_forward(p["T21"], phys)
    # small ridge tracks the exact target; big ridge is visibly biased
    assert np.max(np.abs(t_small - ref)) < 1e-3
    assert np.max(np.abs(t_big - ref)) > 5e-3


def test_beam_maps_matches_coeff_path():
    # Regression: routing reconstructed coefficient maps through beam_maps=
    # reproduces the coefficient path exactly (v001 pipeline untouched).
    import jax.numpy as jnp
    p = build_param_problem(nside=8, nfreq=12, noiseless=True)
    fwd, geom = p["fwd"], p["geom"]
    beam = p["beam"]
    sky_coeffs = p["sky_coeffs"]
    maps = jnp.asarray(beam.coeffs) @ jnp.asarray(beam.basis.A).T
    d_coeff = np.asarray(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom))
    d_maps = np.asarray(fwd.simulate(sky_coeffs, beam_maps=maps, geom=geom))
    assert np.max(np.abs(d_coeff - d_maps)) == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
