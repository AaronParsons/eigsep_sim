"""Tests for shared ground and lunar recovery infrastructure."""

import numpy as np

from eigsep_sim.linear_solver import build_design_matrix
from eigsep_sim.recovery import build_surface_design_matrix


def _legacy_matrix(masks, beams, omega_B, sun_pixels, include_t_rx):
    n_total, ndipole, npix = beams.shape
    sun_pixels = np.tile(sun_pixels, n_total // len(sun_pixels))
    matrix = np.zeros(
        (n_total, ndipole, npix + 2 + (ndipole if include_t_rx else 0))
    )
    weights = beams / omega_B[:, :, None]
    matrix[:, :, :npix] = weights * masks[:, None, :]
    matrix[:, :, npix] = np.sum(weights * (1.0 - masks[:, None, :]), axis=2)
    matrix[:, :, npix + 1] = (
        weights[np.arange(n_total), :, sun_pixels]
        * masks[np.arange(n_total), sun_pixels, None]
    )
    if include_t_rx:
        for dipole in range(ndipole):
            matrix[:, dipole, npix + 2 + dipole] = 1.0
    return matrix.reshape(n_total * ndipole, -1)


def test_shared_surface_design_matrix_columns():
    weights = np.array([[[0.2, 0.3], [0.4, 0.1]]])
    masks = np.array([[1.0, 0.0]])
    matrix = build_surface_design_matrix(
        weights,
        masks,
        unresolved_surface_weight=np.array([[0.5, 0.5]]),
        source_columns={"source": np.array([[2.0, 3.0]])},
        include_receiver_offsets=True,
    )
    expected = np.array(
        [
            [0.2, 0.0, 0.8, 2.0, 1.0, 0.0],
            [0.4, 0.0, 0.6, 3.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected)


def test_legacy_design_matrix_delegation_preserves_values():
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(4, 5)).astype(float)
    beams = rng.uniform(0.1, 1.0, size=(4, 2, 5))
    omega_B = beams.sum(axis=2)
    sun_pixels = np.array([1, 3])
    for include_t_rx in [False, True]:
        actual = build_design_matrix(
            masks, beams, omega_B, sun_pixels, 5, include_t_rx=include_t_rx
        )
        expected = _legacy_matrix(
            masks, beams, omega_B, sun_pixels, include_t_rx
        )
        np.testing.assert_allclose(actual, expected)
