"""
Tests for eigsep_sim.models — T21cmModel.
"""

import numpy as np
import pytest
from eigsep_sim.models import T21cmModel


@pytest.fixture
def model_file(tmp_path):
    """NPZ with two flat 21cm models: 20 mK and 50 mK over 50–250 MHz."""
    freqs_GHz = np.linspace(0.05, 0.25, 100)
    models_mK = np.stack([
        20.0 * np.ones(100),
        50.0 * np.ones(100),
    ], axis=0)  # (2, 100), mK
    path = str(tmp_path / "models_21cm.npz")
    np.savez(path, freqs=freqs_GHz, models=models_mK)
    return path


class TestT21cmModel:
    def test_len(self, model_file):
        m = T21cmModel(filename=model_file)
        assert len(m) == 2

    def test_call_all_models_shape(self, model_file):
        m = T21cmModel(filename=model_file)
        freqs = np.array([100e6, 150e6])
        result = m(freqs)
        assert result.shape == (2, 2)

    def test_call_all_models_values(self, model_file):
        """Flat models should interpolate to their constant values (mK → K)."""
        m = T21cmModel(filename=model_file)
        freqs = np.array([100e6, 150e6])
        result = m(freqs)
        np.testing.assert_allclose(result[0], 20e-3, rtol=1e-4)
        np.testing.assert_allclose(result[1], 50e-3, rtol=1e-4)

    def test_call_single_model(self, model_file):
        m = T21cmModel(filename=model_file)
        freqs = np.array([120e6])
        result = m(freqs, model_index=0)
        np.testing.assert_allclose(result, [20e-3], rtol=1e-4)

    def test_out_of_range_returns_zero(self, model_file):
        """Frequencies outside the model range should return 0 (fill_value=0)."""
        m = T21cmModel(filename=model_file)
        freqs = np.array([1e9])  # 1 GHz — far outside 50–250 MHz
        result = m(freqs, model_index=0)
        assert result[0] == pytest.approx(0.0)

    def test_getitem_returns_interp(self, model_file):
        """m[i] should return an interp1d callable that matches m(f, i)."""
        m = T21cmModel(filename=model_file)
        freqs = np.array([100e6, 200e6])
        interp = m[1]
        via_interp = interp(freqs)
        via_call = m(freqs, model_index=1)
        np.testing.assert_allclose(via_interp, via_call, rtol=1e-6)

    def test_internal_dtype_float32(self, model_file):
        """Internal arrays are stored in the requested dtype."""
        m = T21cmModel(filename=model_file, dtype='float32')
        assert m._freqs.dtype == np.float32
        assert m._mdls.dtype == np.float32
