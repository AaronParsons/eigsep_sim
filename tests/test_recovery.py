"""Tests for shared ground and lunar recovery infrastructure."""

import numpy as np

from eigsep_sim.linear_solver import build_design_matrix
from eigsep_sim.recovery import (
    AdditiveDegeneracy,
    RecoverySolution,
    ScaleDegeneracy,
    build_surface_design_matrix,
)


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


def test_recovery_solution_scale_degeneracy_matches_reference_gauge():
    reference = {"beam": np.arange(1.0, 13.0).reshape(2, 3, 2)}
    estimate = {
        "beam": reference["beam"] * np.array([2.0, 0.5])[:, None, None]
    }
    solution = RecoverySolution(
        estimate, [ScaleDegeneracy(["beam"], group_axes=(0,))]
    )
    projected = solution.remove_degen(reference, inplace=False)
    np.testing.assert_allclose(projected.params["beam"], reference["beam"])
    np.testing.assert_allclose(solution.params["beam"], estimate["beam"])


def test_recovery_solution_scale_degeneracy_couples_sky_and_beam():
    reference = {
        "sky": np.array([[10.0, 20.0], [12.0, 23.0]]),
        "beam": np.arange(1.0, 13.0).reshape(2, 3, 2),
    }
    scale = np.array([2.0, 0.5])
    estimate = {
        "sky": reference["sky"] * scale,
        "beam": reference["beam"] / scale,
    }
    degeneracy = ScaleDegeneracy({"sky": 1.0, "beam": -1.0}, group_axes=(-1,))
    projected = RecoverySolution(estimate, [degeneracy]).remove_degen(
        reference, inplace=False
    )
    for key in reference:
        np.testing.assert_allclose(projected.params[key], reference[key])
    np.testing.assert_allclose(
        estimate["sky"][None, :, :] * estimate["beam"][:, :, None, :],
        reference["sky"][None, :, :] * reference["beam"][:, :, None, :],
    )


def test_recovery_solution_additive_degeneracy_moves_common_temperature():
    reference = {
        "sky": np.array([[10.0, 20.0], [12.0, 23.0]]),
        "ground": np.array([300.0, 310.0]),
        "receiver": np.array([[100.0, 110.0], [101.0, 112.0]]),
    }
    offset = np.array([7.0, -3.0])
    estimate = {
        "sky": reference["sky"] + offset,
        "ground": reference["ground"] + offset,
        "receiver": reference["receiver"] - offset,
    }
    degeneracy = AdditiveDegeneracy(
        {"sky": 1.0, "ground": 1.0, "receiver": -1.0},
        group_axes=(-1,),
    )
    projected = RecoverySolution(estimate, [degeneracy]).remove_degen(
        reference, inplace=False
    )
    for key in reference:
        np.testing.assert_allclose(projected.params[key], reference[key])
    sky_weight = np.array([[0.25], [0.75]])
    estimate_temp = (
        sky_weight * estimate["sky"]
        + (1.0 - sky_weight) * estimate["ground"]
        + estimate["receiver"]
    )
    reference_temp = (
        sky_weight * reference["sky"]
        + (1.0 - sky_weight) * reference["ground"]
        + reference["receiver"]
    )
    np.testing.assert_allclose(estimate_temp, reference_temp)


def test_recovery_solution_canonical_additive_gauge_is_joint():
    params = {
        "sky": np.array([[4.0], [6.0]]),
        "ground": np.array([10.0]),
        "receiver": np.array([[2.0], [8.0]]),
    }
    degeneracy = AdditiveDegeneracy(
        {"sky": 1.0, "ground": 1.0, "receiver": -1.0},
        group_axes=(-1,),
    )
    projected = RecoverySolution(params, [degeneracy]).remove_degen(
        inplace=False
    )
    before = (
        params["sky"].sum() + params["ground"].sum() - params["receiver"].sum()
    )
    after = (
        projected.params["sky"].sum()
        + projected.params["ground"].sum()
        - projected.params["receiver"].sum()
    )
    assert abs(after) < 1e-12
    assert before != after
