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
    print("\n✓ All Phase 7 (calibrator.py) tests passed!")
