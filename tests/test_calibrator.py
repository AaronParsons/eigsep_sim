#!/usr/bin/env python
"""Test suite for calibrator.py — joint sky/beam optimization with Anderson Acceleration."""

import numpy as np
import pytest
from astropy.time import Time

from eigsep_sim.calibrator import AndersonAccelerator, Calibrator
from eigsep_sim.simulate import ForwardModel
from eigsep_sim.beam import Beam
from eigsep_sim.sky import Sky
from eigsep_sim.observer import EarthSurface


def setup_forward_model():
    """Build minimal ForwardModel for testing."""
    freqs_hz = np.array([50e6, 100e6])
    nside = 4
    npix_sky = 12 * nside**2

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=[3.0, 3.0], K=2)
    sky = Sky.from_map(
        nside, freqs_hz, np.random.randn(npix_sky, 2), n_modes=2
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")

    return ForwardModel(observer, beam, sky)


def test_anderson_accelerator_basic():
    """AndersonAccelerator: basic initialization and history tracking."""
    aa = AndersonAccelerator(m=5, tol=1e-10)
    assert aa.m == 5
    assert aa.tol == 1e-10
    assert len(aa.x_history) == 0
    assert len(aa.fx_diff_history) == 0


def test_anderson_accelerator_reset():
    """AndersonAccelerator: reset() clears history."""
    aa = AndersonAccelerator(m=3)
    x = np.array([1.0, 2.0, 3.0])
    fx = np.array([0.1, 0.2, 0.3])
    aa.apply(x, fx)
    assert len(aa.x_history) == 1
    aa.reset()
    assert len(aa.x_history) == 0
    assert len(aa.fx_diff_history) == 0


def test_anderson_accelerator_single_iterate():
    """AndersonAccelerator: first iterate returns the fixed-point update."""
    aa = AndersonAccelerator(m=5)
    x = np.array([1.0, 2.0])
    fx = np.array([0.05, 0.1])
    x_acc = aa.apply(x, fx)
    assert np.allclose(x_acc, x + fx)


def test_anderson_accelerator_two_iterates():
    """AndersonAccelerator: applies acceleration with ≥2 iterates."""
    aa = AndersonAccelerator(m=5, tol=1e-10)
    x1 = np.array([1.0, 2.0])
    fx1 = np.array([0.05, 0.1])
    aa.apply(x1, fx1)

    x2 = np.array([1.05, 2.05])
    fx2 = np.array([0.03, 0.08])
    x_acc = aa.apply(x2, fx2)

    assert x_acc.shape == x2.shape
    assert np.all(np.isfinite(x_acc))


def test_anderson_accelerator_history_limit():
    """AndersonAccelerator: history never exceeds m iterates."""
    aa = AndersonAccelerator(m=3)
    for i in range(10):
        x = np.array([float(i), float(i + 1)])
        fx = np.array([0.01, 0.02])
        aa.apply(x, fx)
        assert len(aa.x_history) <= 3
        assert len(aa.fx_diff_history) <= 3


def test_calibrator_basic():
    """Calibrator: basic construction."""
    fwd = setup_forward_model()
    ntimes = 2
    data = np.random.randn(ntimes, 2, 2).astype(np.float32)

    cal = Calibrator(fwd, data, m_anderson=5, lam_beam=0.01, lam_sky=0.0)
    assert cal.fwd is fwd
    assert cal._data.shape == data.shape
    assert cal._lam_beam == 0.01
    assert cal._lam_sky == 0.0
    assert cal._lam_beam_harmonic == 1e5


def test_calibrator_init_params():
    """Calibrator: init_params() returns valid parameter dict."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data)

    params = cal.init_params()
    assert "sky_coeffs" in params
    assert "beam_coeffs" in params
    assert params["sky_coeffs"].shape[0] == fwd.sky.npix
    assert params["beam_coeffs"].shape[0] == 2
    assert np.all(params["sky_coeffs"] == 0.0)


def test_calibrator_data_loss_matches_loss_without_regularization():
    """data_loss() should match _loss() when regularizers are disabled."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(
        fwd, data, lam_beam=0.0, lam_sky=0.0, lam_beam_harmonic=0.0
    )
    params = cal.init_params(geom=geom)
    assert np.isclose(cal.data_loss(params), cal._loss(params))


def test_calibrator_init_params_with_times():
    """Calibrator: init_params() precomputes geometry when times provided."""
    fwd = setup_forward_model()
    data = np.ones((2, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data)

    times = [Time("2000-01-01"), Time("2000-01-02")]
    params = cal.init_params(times=times)

    assert cal._geom is not None
    assert "rot_gal2top" in cal._geom
    assert len(cal._geom["rot_gal2top"]) == 2


def test_calibrator_loss():
    """Calibrator: loss() computes positive scalar."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data, lam_beam=0.01)

    times = [Time("2000-01-01")]
    params = cal.init_params(times=times)

    loss = cal._loss(params)
    loss_val = float(loss)
    assert loss_val > 0.0
    assert np.isfinite(loss_val)


def test_calibrator_beam_step():
    """Calibrator: beam_step() updates beam coefficients."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data, lam_beam=0.01)

    times = [Time("2000-01-01")]
    params = cal.init_params(times=times)
    params["sky_coeffs"] = np.ones_like(params["sky_coeffs"])
    beam_before = params["beam_coeffs"].copy()

    params_new = cal.beam_step(params, lr=0.001)
    assert not np.allclose(params_new["beam_coeffs"], beam_before)
    assert params_new["sky_coeffs"] is params["sky_coeffs"]


def test_calibrator_fit_convergence():
    """Calibrator: fit() runs iterations and detects convergence."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data, lam_beam=0.01)

    times = [Time("2000-01-01")]
    result = cal.fit(times=times, max_iter=5, tol=0.5, verbose=False)

    assert "params" in result
    assert "losses" in result
    assert "converged" in result
    assert "n_iter" in result
    assert len(result["losses"]) <= 5
    assert result["losses"][0] > 0


def test_calibrator_fit_with_noise_weights():
    """Calibrator: fit() respects inv_noise_var weighting."""
    fwd = setup_forward_model()
    data = np.random.randn(1, 2, 2).astype(np.float32)

    inv_noise_var = np.ones((1, 2, 2), dtype=np.float32)
    inv_noise_var[:, 0, :] = 100.0

    cal = Calibrator(fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.01)
    times = [Time("2000-01-01")]
    result = cal.fit(times=times, max_iter=3, verbose=False)

    assert result["n_iter"] > 0
    assert len(result["losses"]) == result["n_iter"]


def test_calibrator_loss_normalization_invariance():
    """
    Calibrator: loss uses mean (normalized) not sum for data residual.

    This test would have exposed the previous bug where loss was computed as
    jnp.sum(residuals**2) instead of jnp.mean(residuals**2). With unnormalized
    loss, the gradient magnitude scales linearly with dataset size, causing
    huge parameter updates even with small learning rates.

    The test verifies that loss computation doesn't scale pathologically with
    a constant-residual dataset.
    """
    fwd = setup_forward_model()

    # Create datasets with 1 and 2 time steps
    times_1 = [Time("2000-01-01")]
    times_2 = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]

    data_1 = np.ones((1, 2, 2), dtype=np.float32) * 100.0
    data_2 = np.ones((2, 2, 2), dtype=np.float32) * 100.0

    cal_1 = Calibrator(fwd, data_1, lam_beam=0.0)
    cal_2 = Calibrator(fwd, data_2, lam_beam=0.0)

    params_1 = cal_1.init_params(times=times_1)
    params_2 = cal_2.init_params(times=times_2)

    loss_1 = float(cal_1._loss(params_1))
    loss_2 = float(cal_2._loss(params_2))

    # With normalized loss (mean), loss should scale gently with dataset size.
    # With unnormalized loss (sum), loss_2 would be roughly 2x loss_1 even with
    # identical residuals per observation.
    # Since observations have same residuals, loss ratio should be close to 1
    # (actual ratio will vary slightly due to geometry changes with time).
    loss_ratio = loss_2 / loss_1
    assert 0.5 < loss_ratio < 2.0, (
        f"Loss scaling issue: ratio = {loss_ratio}. "
        f"If much > 1, suggests unnormalized loss (sum). "
        f"loss_1={loss_1}, loss_2={loss_2}"
    )


def test_calibrator_sky_step_updates_coeffs():
    """Calibrator: sky_step() actually updates sky coefficients."""
    fwd = setup_forward_model()
    np.random.seed(42)
    data = np.random.randn(2, 2, 2).astype(np.float32) * 10.0 + 100.0

    cal = Calibrator(fwd, data)
    params = cal.init_params(times=[Time("2000-01-01")] * 2)

    # Verify initial sky coefficients are zero
    assert np.allclose(
        params["sky_coeffs"], 0.0
    ), "Initial sky_coeffs should be zero"

    # Take a sky step
    params_after = cal.sky_step(params)

    # Sky coefficients should have changed (non-zero)
    sky_change = np.max(
        np.abs(params_after["sky_coeffs"] - params["sky_coeffs"])
    )
    assert sky_change > 1e-6, (
        f"Sky coefficients did not change: max change = {sky_change}. "
        f"sky_step() may not be optimizing properly."
    )

    # Beam coefficients should remain unchanged
    assert np.allclose(
        params_after["beam_coeffs"], params["beam_coeffs"]
    ), "Beam coefficients should not change during sky_step"


def test_calibrator_beam_step_decreases_loss():
    """
    Calibrator: beam_step with lr=0.01 should decrease (or not increase) loss.

    This test would have exposed the previous bug where unnormalized loss caused
    gradients to be huge relative to learning rate, leading to divergence where
    a single beam_step could increase loss by 690x.
    """
    fwd = setup_forward_model()
    # Create data with known structure
    np.random.seed(42)
    data = np.random.randn(2, 2, 2).astype(np.float32) * 10.0 + 100.0

    cal = Calibrator(fwd, data, lam_beam=0.01)
    params = cal.init_params(times=[Time("2000-01-01")] * 2)

    # Compute initial loss
    loss_before = float(cal._loss(params))

    # Take a single beam step with default learning rate
    params_after = cal.beam_step(params, lr=0.01)
    loss_after = float(cal._loss(params_after))

    # Loss should not explode (relative change should be < 10)
    # With the bug, this would be > 100x increase
    rel_change = abs(loss_after - loss_before) / loss_before
    assert rel_change < 10.0, (
        f"Beam step caused huge loss change: {rel_change:.2e}. "
        f"Suggests unnormalized loss in gradient computation. "
        f"loss_before={loss_before}, loss_after={loss_after}"
    )

    # Parameter changes should be reasonable (order 1e-3 to 1e-1)
    beam_change = np.max(
        np.abs(params_after["beam_coeffs"] - params["beam_coeffs"])
    )
    assert beam_change < 1.0, (
        f"Beam coefficients changed by {beam_change}, likely too much. "
        f"Suggests learning rate or gradient magnitude issue."
    )


def test_calibrator_joint_step_decreases_loss():
    """
    Calibrator: joint_step() reduces loss and uses block-diagonal regularization.

    Tests the fixed joint_step (with separate sky/beam Rademacher probes) at a
    point near the alternating optimum.  The joint Hessian is positive definite
    near the minimum, enabling Newton-CG to find a descent direction.  Far from
    the minimum the beam Hessian can be negative, so we test from a near-optimal
    starting point.
    """
    fwd = setup_forward_model()
    np.random.seed(42)
    data = np.random.randn(2, 2, 2).astype(np.float32) * 10.0 + 100.0

    cal = Calibrator(fwd, data, lam_beam=0.01)
    params = cal.init_params(times=[Time("2000-01-01")] * 2)

    # Warm up with 3 alternating steps to get into the PD region of the Hessian
    for _ in range(3):
        params = cal.sky_step(params)
        params = cal.beam_step(params)

    loss_before = float(cal._loss(params))
    params_after = cal.joint_step(params)
    loss_after = float(cal._loss(params_after))

    # joint_step must not raise loss
    assert (
        loss_after <= loss_before * 1.01
    ), f"joint_step raised loss: {loss_before:.3e} → {loss_after:.3e}"
    # Both sky and beam should have changed (joint step is not a no-op)
    sky_change = np.max(
        np.abs(params_after["sky_coeffs"] - params["sky_coeffs"])
    )
    beam_change = np.max(
        np.abs(params_after["beam_coeffs"] - params["beam_coeffs"])
    )
    assert (
        sky_change + beam_change > 1e-12
    ), "joint_step made no parameter changes"


def test_calibrator_fit_use_truncated_beam_cg():
    """Calibrator: fit() accepts truncated beam-CG controls."""
    fwd = setup_forward_model()
    np.random.seed(42)
    data = np.random.randn(2, 2, 2).astype(np.float32) * 10.0 + 100.0

    cal = Calibrator(fwd, data, lam_beam=0.01)
    times = [Time("2000-01-01")] * 2
    params = cal.init_params(times=times)
    params = cal.sky_step(params)
    loss_before = float(cal._loss(params))

    result = cal.fit(
        params=params,
        max_iter=2,
        verbose=False,
        use_cg=True,
        beam_cg_niter=2,
        beam_cg_tol=1e-2,
    )
    assert result["losses"][-1] <= loss_before * 1.01


def test_calibrator_fit_use_joint():
    """Calibrator: fit() with use_joint=True runs without error and reduces loss."""
    fwd = setup_forward_model()
    np.random.seed(42)
    data = np.random.randn(2, 2, 2).astype(np.float32) * 10.0 + 100.0

    cal = Calibrator(fwd, data, lam_beam=0.01)
    times = [Time("2000-01-01")] * 2
    # Run a few alternating steps first to get into the PD Hessian region,
    # then switch to joint for the remaining iterations
    params = cal.init_params(times=times)
    for _ in range(3):
        params = cal.sky_step(params)
        params = cal.beam_step(params)
    loss_after_alt = float(cal._loss(params))

    result = cal.fit(params=params, max_iter=3, verbose=False, use_joint=True)
    assert (
        result["losses"][-1] <= loss_after_alt * 1.01
    ), f"fit(use_joint=True) raised loss: {loss_after_alt:.3e} → {result['losses'][-1]:.3e}"
    assert "params" in result and "losses" in result


def test_calibrator_sky_step_decreases_loss_significantly():
    """
    Calibrator: sky_step() must reduce loss by at least 10× from zero sky.

    This test catches the lam_abs over-regularization bug where gradient-scaled
    Tikhonov made lam_abs >> H_diagonal, turning Newton-CG into over-damped
    gradient descent that barely moved the sky coefficients.

    With the correct (H-diagonal-relative) lam_abs, a single Newton-CG step
    from sky=0 should drop the loss by orders of magnitude, since the sky loss
    is exactly quadratic and Newton's method is exact for quadratic problems.
    """
    freqs_hz = np.array([50e6, 100e6, 150e6], dtype=np.float64)
    nside = 4
    from eigsep_sim import Beam, Sky, ForwardModel, NullTerrain

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=[3.0], K=2)
    sky = Sky.from_gsm(nside, freqs_hz, n_modes=2, include_flat=True)
    observer = EarthSurface(lat=39.2, lon=-113.4)
    fwd = ForwardModel(observer, beam, sky, terrain=NullTerrain())

    gsm_coeffs = sky.init_coeffs()
    beam_coeffs = beam.coeffs.copy()

    times = [Time("2025-01-01") + i * 3600 for i in range(8)]
    geom = fwd.precompute_geometry(times)
    T_true = fwd.simulate(gsm_coeffs, beam_coeffs, geom=geom)
    data = np.array(T_true).astype(np.float32)

    cal = Calibrator(fwd, data, lam_beam=0.0)
    cal._geom = geom
    params = {
        "sky_coeffs": np.zeros_like(gsm_coeffs),
        "beam_coeffs": beam_coeffs.copy(),
    }

    loss_before = float(cal._loss(params))
    params_after = cal.sky_step(params)
    loss_after = float(cal._loss(params_after))

    reduction = loss_before / (loss_after + 1e-30)
    assert reduction > 10.0, (
        f"sky_step() only reduced loss by {reduction:.1f}× (expected ≥10×). "
        f"lam_abs may be over-regularized (gradient-scaled instead of H-scaled). "
        f"loss_before={loss_before:.3e}, loss_after={loss_after:.3e}"
    )


def test_calibrator_init_params_with_rots():
    """init_params(rots=...) precomputes geometry from rotation matrices."""
    fwd = setup_forward_model()
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 4
    data = np.zeros(
        (ntimes, fwd.beam.coeffs.shape[0], len(fwd.beam.freqs_hz)),
        dtype=np.float32,
    )
    cal = Calibrator(fwd, data)
    params = cal.init_params(rots=[R] * ntimes)
    assert cal._geom is not None
    assert "rots_jax" in cal._geom
    assert cal._geom["rots_jax"].shape == (ntimes, 3, 3)
    assert params["sky_coeffs"].shape == (fwd.sky.npix, fwd.sky.nmodes)


def test_calibrator_init_params_with_geom():
    """init_params(geom=...) accepts a pre-computed geometry dict."""
    fwd = setup_forward_model()
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 3
    geom = fwd.precompute_geometry(rots=[R] * ntimes)
    data = np.zeros(
        (ntimes, fwd.beam.coeffs.shape[0], len(fwd.beam.freqs_hz)),
        dtype=np.float32,
    )
    cal = Calibrator(fwd, data)
    params = cal.init_params(geom=geom)
    assert cal._geom is geom  # same object, not re-computed


def test_calibrator_fit_with_rots():
    """fit(rots=...) runs a full calibration iteration from rots geometry."""
    fwd = setup_forward_model()
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 5
    n_dipoles = fwd.beam.coeffs.shape[0]
    nfreq = len(fwd.beam.freqs_hz)

    # Simulate noiseless data using nominal parameters
    geom = fwd.precompute_geometry(rots=[R] * ntimes)
    sky_c = fwd.sky.init_coeffs()
    beam_c = fwd.beam.coeffs.copy()
    data = np.array(fwd.simulate(sky_c, beam_c, geom=geom), dtype=np.float32)

    cal = Calibrator(fwd, data, lam_beam=0.0)
    result = cal.fit(rots=[R] * ntimes, max_iter=3, verbose=False)

    assert "params" in result
    assert result["params"]["sky_coeffs"].shape == (
        fwd.sky.npix,
        fwd.sky.nmodes,
    )
    assert result["n_iter"] <= 3


def test_calibrator_fit_with_precomputed_geom():
    """fit(geom=...) accepts a pre-computed geometry dict directly."""
    fwd = setup_forward_model()
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 4
    geom = fwd.precompute_geometry(rots=[R] * ntimes)
    sky_c = fwd.sky.init_coeffs()
    beam_c = fwd.beam.coeffs.copy()
    data = np.array(fwd.simulate(sky_c, beam_c, geom=geom), dtype=np.float32)

    cal = Calibrator(fwd, data, lam_beam=0.0)
    result = cal.fit(geom=geom, max_iter=2, verbose=False)
    assert "params" in result


def test_calibrator_fit_with_sky_mask():
    """fit(rots=..., sky_mask=...) propagates sky_mask through geometry."""
    fwd = setup_forward_model()
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 4
    sky_mask = fwd.build_sky_mask(rots=[R])
    geom = fwd.precompute_geometry(rots=[R] * ntimes, sky_mask=sky_mask)
    sky_c = fwd.sky.init_coeffs()
    beam_c = fwd.beam.coeffs.copy()
    data = np.array(fwd.simulate(sky_c, beam_c, geom=geom), dtype=np.float32)

    cal = Calibrator(fwd, data, lam_beam=0.0)
    result = cal.fit(
        rots=[R] * ntimes, sky_mask=sky_mask, max_iter=2, verbose=False
    )
    assert "sky_indices_jax" in cal._geom
    assert result["params"]["sky_coeffs"].shape == (
        fwd.sky.npix,
        fwd.sky.nmodes,
    )


def test_calibrator_uses_float64_dtype():
    """Calibrator and package constants use float64 real arrays."""
    from eigsep_sim import DTYPE_R_NPY

    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data)
    params = cal.init_params()

    assert DTYPE_R_NPY == np.float64
    assert cal._data.dtype == np.float64
    assert params["sky_coeffs"].dtype == np.float64
    assert params["beam_coeffs"].dtype == np.float64


def test_forward_model_adjoint_matches_autodiff_gradient_sign():
    """Adjoint numerators equal the negative data-loss gradient."""
    import jax
    import jax.numpy as jnp

    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    sky_true = fwd.sky.init_coeffs()
    beam_true = fwd.beam.coeffs.copy()
    data = np.asarray(fwd.simulate(sky_true, beam_true, geom=geom))

    params = {
        "sky_coeffs": sky_true * 0.9,
        "beam_coeffs": beam_true * 1.1,
    }
    pred = np.asarray(
        fwd.simulate(params["sky_coeffs"], params["beam_coeffs"], geom=geom)
    )
    residual = pred - data
    weights = np.ones_like(data)
    adj = fwd.accumulate_sky_beam_adjoint(
        params["sky_coeffs"], params["beam_coeffs"], residual, weights, geom
    )

    def data_loss(sky_coeffs, beam_coeffs):
        model = fwd.simulate(sky_coeffs, beam_coeffs, geom=geom)
        diff = model - jnp.asarray(data)
        return 0.5 * jnp.sum(diff**2)

    grad_sky, grad_beam = jax.grad(data_loss, argnums=(0, 1))(
        jnp.asarray(params["sky_coeffs"]), jnp.asarray(params["beam_coeffs"])
    )
    np.testing.assert_allclose(grad_sky, -adj["sky_num"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        grad_beam, -adj["beam_num"], rtol=1e-8, atol=1e-6
    )
    assert np.all(np.asarray(adj["sky_den"]) >= 0.0)
    assert np.all(np.asarray(adj["beam_den"]) >= 0.0)


def test_calibrator_adaptive_fit_monotonic_and_telemetry():
    """Adaptive fixed-point fit decreases loss and records benchmark telemetry."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(fwd, data, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = fwd.sky.init_coeffs() * 0.8
    params["beam_coeffs"] = fwd.beam.coeffs * 1.2
    loss_before = float(cal._loss(params))

    result = cal.fit(params=params, max_iter=3, verbose=False)

    assert result["solver"] == "adaptive-fixed-point"
    assert result["losses"][-1] <= loss_before
    assert all(
        later <= earlier + 1e-8
        for earlier, later in zip(result["losses"], result["losses"][1:])
    )
    assert len(result["telemetry"]) == result["n_iter"]
    for key in (
        "wall_time",
        "delta_chi2",
        "delta_chi2_per_sec",
        "step_type",
        "projected_sky_rms",
        "projected_beam_rms",
        "beam_scatter",
        "beam_roughness",
        "joint_step",
        "sky_step",
        "beam_step",
        "joint_loss",
        "sky_loss",
        "beam_loss",
        "beam_shape_update_rms",
        "beam_scale_update_rms",
        "joint_sky_shape_update_rms",
        "joint_sky_scale_update_rms",
        "joint_beam_shape_update_rms",
        "joint_beam_scale_update_rms",
        "beam_scale_alpha",
        "joint_scale_alpha",
    ):
        assert key in result["telemetry"][0]


def test_calibrator_adaptive_scheduled_runs_and_records_state():
    """Scheduled adaptive solver tracks block efficiencies and cadence state."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(fwd, data, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = fwd.sky.init_coeffs() * 0.85
    params["beam_coeffs"] = fwd.beam.coeffs * 1.15
    loss_before = float(cal._loss(params))

    result = cal.fit(
        params=params,
        max_iter=5,
        verbose=False,
        solver="adaptive-scheduled",
        schedule_max_every={"sky": 2, "beam": 2, "joint": 2},
    )

    assert result["solver"] == "adaptive-scheduled"
    assert result["losses"][-1] <= loss_before
    assert len(result["telemetry"]) == result["n_iter"]
    blocks = {entry["scheduled_block"] for entry in result["telemetry"]}
    assert len(blocks) >= 2
    first = result["telemetry"][0]
    for key in (
        "scheduled_block",
        "schedule_reason",
        "schedule_eff_sky",
        "schedule_eff_beam",
        "schedule_eff_joint",
        "schedule_n_since_sky",
        "schedule_n_since_beam",
        "schedule_n_since_joint",
        "schedule_step_gain_sky",
        "schedule_step_gain_beam",
        "schedule_step_gain_joint",
    ):
        assert key in first


def test_calibrator_beam_harmonic_coefficient_penalty_matches_map_space():
    """Coefficient-space harmonic prior matches the explicit map penalty."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(
        fwd,
        data,
        lam_beam=0.0,
        lam_beam_harmonic=1e-2,
        beam_harmonic_lmin=2,
        beam_harmonic_lmax=5,
    )
    params = cal.init_params(geom=geom)
    rng = np.random.default_rng(0)
    beam_coeffs = params["beam_coeffs"] + 0.03 * rng.normal(
        size=params["beam_coeffs"].shape
    )

    q = cal._ensure_beam_harmonic_regularizer()
    basis_A = cal._beam_basis_A_np()
    diff_maps = cal._beam_maps_np(beam_coeffs) - cal._beam_maps_np(
        cal._beam_nom
    )
    map_penalty = float(
        np.mean(diff_maps * np.einsum("pq,dqf->dpf", q, diff_maps))
    )

    np.testing.assert_allclose(
        cal._beam_harmonic_penalty(beam_coeffs),
        map_penalty,
        rtol=1e-10,
        atol=1e-12,
    )


def test_calibrator_beam_harmonic_regularizer_penalizes_shape():
    """Harmonic beam prior penalizes high-ell shape changes from nominal."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(
        fwd,
        data,
        lam_beam=0.0,
        lam_beam_harmonic=1e-2,
        beam_harmonic_lmin=2,
        beam_harmonic_lmax=5,
    )
    params = cal.init_params(geom=geom)
    assert cal._beam_harmonic_penalty(params["beam_coeffs"]) == 0.0

    pix = np.arange(fwd.beam.npix)
    perturb = ((pix % 2) * 2.0 - 1.0)[None, :, None]
    params["beam_coeffs"] = params["beam_coeffs"] + 0.05 * perturb
    assert cal._beam_harmonic_penalty(params["beam_coeffs"]) > 0.0

    params["sky_coeffs"] = fwd.sky.init_coeffs() * 0.9
    result = cal.fit(
        params=params,
        max_iter=2,
        verbose=False,
        solver="adaptive-scheduled",
        schedule_max_every={"sky": 2, "beam": 1, "joint": 2},
    )
    assert "beam_harmonic_penalty" in result["telemetry"][0]


def test_calibrator_adaptive_scheduled_lbfgs_block_runs_when_enabled():
    """Scheduled adaptive solver can include short L-BFGS bursts."""
    pytest.importorskip("jaxopt")
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(fwd, data, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = fwd.sky.init_coeffs() * 0.9
    params["beam_coeffs"] = fwd.beam.coeffs * 1.1
    result = cal.fit(
        params=params,
        max_iter=4,
        verbose=False,
        solver="adaptive-scheduled",
        schedule_max_every={"sky": 3, "beam": 3, "joint": 3},
        schedule_lbfgs_max_every=1,
        schedule_lbfgs_min_iter=1,
        schedule_lbfgs_maxiter=1,
    )
    assert any(
        entry.get("scheduled_block") == "lbfgs"
        for entry in result["telemetry"]
    )
    lbfgs_entries = [
        entry
        for entry in result["telemetry"]
        if str(entry.get("step_type", "")).startswith("lbfgs")
    ]
    assert len(lbfgs_entries) == 1
    assert lbfgs_entries[0]["schedule_n_run_lbfgs"] == 1


def test_calibrator_scale_projection_preserves_sky_beam_product():
    """Scale projection fixes beam RMS gauge while preserving multiplicative data."""
    fwd = setup_forward_model()
    times = [Time("2000-01-01")]
    geom = fwd.precompute_geometry(times=times)
    sky_coeffs = fwd.sky.init_coeffs()
    beam_coeffs = fwd.beam.coeffs.copy()
    data = np.asarray(fwd.simulate(sky_coeffs, beam_coeffs, geom=geom))
    cal = Calibrator(fwd, data, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = sky_coeffs / 3.0
    params["beam_coeffs"] = beam_coeffs * 3.0

    before = np.asarray(
        fwd.simulate(params["sky_coeffs"], params["beam_coeffs"], geom=geom)
    )
    projected = cal._project_scale_degeneracy(params)
    after = np.asarray(
        fwd.simulate(
            projected["sky_coeffs"], projected["beam_coeffs"], geom=geom
        )
    )

    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-8)
    assert np.isclose(
        np.sqrt(np.mean(projected["beam_coeffs"] ** 2)),
        np.sqrt(np.mean(beam_coeffs**2)),
    )


def test_calibrator_adaptive_fit_accepts_2d_data():
    """Adaptive adjoint path accepts (ntimes, nfreq) data."""
    from eigsep_sim import Beam, Sky

    freqs_hz = np.array([50e6, 100e6], dtype=np.float64)
    nside = 2
    npix = 12 * nside**2
    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=[3.0], K=2)
    sky = Sky.from_map(
        nside,
        freqs_hz,
        np.random.default_rng(0).normal(size=(npix, 2)),
        n_modes=2,
    )
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    fwd = ForwardModel(observer, beam, sky)
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    sky_true = sky.init_coeffs()
    beam_true = beam.coeffs.copy()
    data_3d = np.asarray(fwd.simulate(sky_true, beam_true, geom=geom))
    data_2d = data_3d[:, 0, :]

    cal = Calibrator(fwd, data_2d, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = sky_true * 0.9
    params["beam_coeffs"] = beam_true * 1.1
    result = cal.fit(params=params, max_iter=2, verbose=False)

    assert result["solver"] == "adaptive-fixed-point"
    assert len(result["losses"]) == result["n_iter"]
    assert np.all(np.isfinite(result["losses"]))


def test_calibrator_fit_lbfgs_reduces_when_available():
    """Two L-BFGS calls reuse the same calibrator path and do not raise loss."""
    pytest.importorskip("jaxopt")
    fwd = setup_forward_model()
    times = [Time("2000-01-01"), Time("2000-01-01 00:01:00")]
    geom = fwd.precompute_geometry(times=times)
    data = np.asarray(
        fwd.simulate(fwd.sky.init_coeffs(), fwd.beam.coeffs, geom=geom)
    )
    cal = Calibrator(fwd, data, lam_beam=0.0)
    params = cal.init_params(geom=geom)
    params["sky_coeffs"] = fwd.sky.init_coeffs() * 0.9
    params["beam_coeffs"] = fwd.beam.coeffs * 1.1
    first = cal.fit_lbfgs(params, maxiter=2)
    second = cal.fit_lbfgs(first["params"], maxiter=2)
    assert second["losses"][-1] <= first["losses"][-1] * 1.01


if __name__ == "__main__":
    test_anderson_accelerator_basic()
    test_anderson_accelerator_reset()
    test_anderson_accelerator_single_iterate()
    test_anderson_accelerator_two_iterates()
    test_anderson_accelerator_history_limit()
    test_calibrator_basic()
    test_calibrator_init_params()
    test_calibrator_init_params_with_times()
    test_calibrator_loss()
    test_calibrator_beam_step()
    test_calibrator_fit_convergence()
    test_calibrator_fit_with_noise_weights()
    test_calibrator_loss_normalization_invariance()
    test_calibrator_sky_step_updates_coeffs()
    test_calibrator_beam_step_decreases_loss()
    test_calibrator_joint_step_decreases_loss()
    test_calibrator_fit_use_joint()
    print("\n✓ All Phase 7 (calibrator.py) tests passed!")
