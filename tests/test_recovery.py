"""Tests for shared ground and lunar recovery infrastructure."""

import numpy as np

from eigsep_sim.recovery import (
    AdditiveDegeneracy,
    RecoverySolution,
    ScaleDegeneracy,
    build_surface_design_matrix,
    normal_solve,
)


def test_normal_solve_recovers_observed_sky_and_surface_columns():
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ]
    )
    truth = np.array([10.0, 20.0, 300.0])
    result = normal_solve(A, A @ truth, npix=2, rcond=1e-12)
    np.testing.assert_allclose(result["sky_map"], truth[:2])
    np.testing.assert_allclose(result["surface"], truth[2])
    np.testing.assert_allclose(result["t_regolith"], truth[2])
    assert result["rank"] == 3
    assert not result["unobserved"].any()


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
