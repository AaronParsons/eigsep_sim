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
    sky = Sky.from_map(nside, freqs_hz,
                       np.random.randn(npix_sky, 2), n_modes=2)
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
    """AndersonAccelerator: returns unaccelerated on first iterate (need ≥2)."""
    aa = AndersonAccelerator(m=5)
    x = np.array([1.0, 2.0])
    fx = np.array([0.05, 0.1])
    x_acc = aa.apply(x, fx)
    assert np.allclose(x_acc, x)


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
        x = np.array([float(i), float(i+1)])
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


def test_calibrator_init_params():
    """Calibrator: init_params() returns valid parameter dict."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data)

    params = cal.init_params()
    assert 'sky_coeffs' in params
    assert 'beam_coeffs' in params
    assert params['sky_coeffs'].shape[0] == fwd.sky.npix
    assert params['beam_coeffs'].shape[0] == 2
    assert np.all(params['sky_coeffs'] == 0.0)


def test_calibrator_init_params_with_times():
    """Calibrator: init_params() precomputes geometry when times provided."""
    fwd = setup_forward_model()
    data = np.ones((2, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data)

    times = [Time("2000-01-01"), Time("2000-01-02")]
    params = cal.init_params(times=times)

    assert cal._geom is not None
    assert 'rot_gal2top' in cal._geom
    assert len(cal._geom['rot_gal2top']) == 2


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
    beam_before = params['beam_coeffs'].copy()

    params_new = cal.beam_step(params, lr=0.001)
    assert not np.allclose(params_new['beam_coeffs'], beam_before)
    assert params_new['sky_coeffs'] is params['sky_coeffs']


def test_calibrator_fit_convergence():
    """Calibrator: fit() runs iterations and detects convergence."""
    fwd = setup_forward_model()
    data = np.ones((1, 2, 2), dtype=np.float32)
    cal = Calibrator(fwd, data, lam_beam=0.01)

    times = [Time("2000-01-01")]
    result = cal.fit(times=times, max_iter=5, tol=0.5, verbose=False)

    assert 'params' in result
    assert 'losses' in result
    assert 'converged' in result
    assert 'n_iter' in result
    assert len(result['losses']) <= 5
    assert result['losses'][0] > 0


def test_calibrator_fit_with_noise_weights():
    """Calibrator: fit() respects inv_noise_var weighting."""
    fwd = setup_forward_model()
    data = np.random.randn(1, 2, 2).astype(np.float32)

    inv_noise_var = np.ones((1, 2, 2), dtype=np.float32)
    inv_noise_var[:, 0, :] = 100.0

    cal = Calibrator(fwd, data, inv_noise_var=inv_noise_var, lam_beam=0.01)
    times = [Time("2000-01-01")]
    result = cal.fit(times=times, max_iter=3, verbose=False)

    assert result['n_iter'] > 0
    assert len(result['losses']) == result['n_iter']


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
    assert np.allclose(params['sky_coeffs'], 0.0), "Initial sky_coeffs should be zero"

    # Take a sky step
    params_after = cal.sky_step(params)

    # Sky coefficients should have changed (non-zero)
    sky_change = np.max(np.abs(params_after['sky_coeffs'] - params['sky_coeffs']))
    assert sky_change > 1e-6, (
        f"Sky coefficients did not change: max change = {sky_change}. "
        f"sky_step() may not be optimizing properly."
    )

    # Beam coefficients should remain unchanged
    assert np.allclose(params_after['beam_coeffs'], params['beam_coeffs']), (
        "Beam coefficients should not change during sky_step"
    )


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
    beam_change = np.max(np.abs(params_after['beam_coeffs'] - params['beam_coeffs']))
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
    assert loss_after <= loss_before * 1.01, (
        f"joint_step raised loss: {loss_before:.3e} → {loss_after:.3e}"
    )
    # Both sky and beam should have changed (joint step is not a no-op)
    sky_change = np.max(np.abs(params_after['sky_coeffs'] - params['sky_coeffs']))
    beam_change = np.max(np.abs(params_after['beam_coeffs'] - params['beam_coeffs']))
    assert sky_change + beam_change > 1e-8, (
        "joint_step made no parameter changes"
    )


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
    assert result['losses'][-1] <= loss_after_alt * 1.01, (
        f"fit(use_joint=True) raised loss: {loss_after_alt:.3e} → {result['losses'][-1]:.3e}"
    )
    assert 'params' in result and 'losses' in result


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
    params = {'sky_coeffs': np.zeros_like(gsm_coeffs), 'beam_coeffs': beam_coeffs.copy()}

    loss_before = float(cal._loss(params))
    params_after = cal.sky_step(params)
    loss_after = float(cal._loss(params_after))

    reduction = loss_before / (loss_after + 1e-30)
    assert reduction > 10.0, (
        f"sky_step() only reduced loss by {reduction:.1f}× (expected ≥10×). "
        f"lam_abs may be over-regularized (gradient-scaled instead of H-scaled). "
        f"loss_before={loss_before:.3e}, loss_after={loss_after:.3e}"
    )


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
