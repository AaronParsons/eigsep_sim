"""
Tests for eigsep_sim.coord — rot_m and coordinate helpers.
"""

import numpy as np
import pytest
from eigsep_sim.coord import rot_m, xyz2thphi, thphi2xyz, azalt2top, top2azalt


# ---------------------------------------------------------------------------
# rot_m
# ---------------------------------------------------------------------------

class TestRotM:
    def test_identity_zero_angle(self):
        for vec in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            R = rot_m(0.0, np.array(vec, dtype=float))
            np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_full_rotation_returns_to_identity(self):
        for vec in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            R = rot_m(2 * np.pi, np.array(vec, dtype=float))
            np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_quarter_turn_z_maps_x_to_y(self):
        R = rot_m(np.pi / 2, np.array([0, 0, 1.0]))
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-14)

    def test_quarter_turn_z_maps_y_to_minus_x(self):
        R = rot_m(np.pi / 2, np.array([0, 0, 1.0]))
        result = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, [-1, 0, 0], atol=1e-14)

    def test_half_turn_x_flips_yz(self):
        R = rot_m(np.pi, np.array([1, 0, 0.0]))
        result = R @ np.array([0, 1, 0.0])
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-14)

    def test_orthogonality(self):
        axis = np.array([1.0, 2.0, 3.0])
        axis = axis / np.linalg.norm(axis)
        for angle in (0.1, 1.0, 2.5, np.pi):
            R = rot_m(angle, axis)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)

    def test_determinant_one(self):
        axis = np.array([0.6, 0.8, 0.0])
        for angle in (0.5, 1.5, 3.0):
            R = rot_m(angle, axis)
            assert abs(np.linalg.det(R) - 1.0) < 1e-13

    def test_axis_vector_is_fixed(self):
        """The rotation axis should be a fixed point of the rotation."""
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        R = rot_m(1.23, axis)
        np.testing.assert_allclose(R @ axis, axis, atol=1e-13)


# ---------------------------------------------------------------------------
# Spherical coordinate roundtrips
# ---------------------------------------------------------------------------

class TestCoordRoundtrips:
    def test_thphi_xyz_roundtrip(self):
        th = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
        phi = np.array([0.1, 1.0, 2.0])
        xyz = thphi2xyz((th, phi))
        th2, phi2 = xyz2thphi(xyz)
        np.testing.assert_allclose(th2, th, atol=1e-12)
        np.testing.assert_allclose(phi2, phi, atol=1e-12)

    def test_azalt_top_roundtrip(self):
        az = np.array([0.0, np.pi / 4, np.pi])
        alt = np.array([0.0, np.pi / 6, np.pi / 3])
        xyz = azalt2top((az, alt))
        az2, alt2 = top2azalt(xyz)
        np.testing.assert_allclose(az2, az, atol=1e-12)
        np.testing.assert_allclose(alt2, alt, atol=1e-12)
