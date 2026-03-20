"""
Tests for eigsep_sim.sim (compute_masks_and_beams, simulate_observations)
and eigsep_sim.linear_solver (build_design_matrix, svd_solve).
"""

import numpy as np
import pytest
import healpy
from astropy.time import Time
from scipy.spatial.transform import Rotation

from eigsep_sim.sim import compute_masks_and_beams, simulate_observations
from eigsep_sim.linear_solver import build_design_matrix, svd_solve


# ──────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ──────────────────────────────────────────────────────────────────────────────

NSIDE   = 4
NPIX    = healpy.nside2npix(NSIDE)   # 192
N_OBS   = 4
N_ORBIT = 2
N_TOTAL = N_ORBIT * N_OBS


class MockOrbit:
    """Minimal LunarOrbit stub: uniform above_horizon mask."""
    def __init__(self, open_sky=True):
        self._val = open_sky

    def set_time(self, t):
        pass

    def above_horizon(self, nside):
        return np.full(healpy.nside2npix(nside), self._val, dtype=bool)


class MockOrbitPartial:
    """LunarOrbit stub: first half of sky blocked, second half open."""
    def set_time(self, t):
        pass

    def above_horizon(self, nside):
        npix = healpy.nside2npix(nside)
        mask = np.ones(npix, dtype=bool)
        mask[:npix // 2] = False
        return mask


def _identity_rots(n):
    return Rotation.from_matrix(np.tile(np.eye(3), (n, 1, 1)))


def _obs_times():
    return [Time('2025-01-01')] * N_OBS


def _u_body():
    """Two orthogonal dipole axes in the body frame."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _kh():
    # kh = π * f * L / c;  f=50 MHz, L=1.0 m  →  kh ≈ 0.52 (well away from π/2 resonance)
    return np.array([np.pi * 50e6 * 1.0 / 3e8] * 2)


def _run_cmb(orbits=None, open_sky=True):
    """Run compute_masks_and_beams with default test parameters."""
    if orbits is None:
        orbits = [MockOrbit(open_sky=open_sky) for _ in range(N_ORBIT)]
    rots = [_identity_rots(N_OBS) for _ in range(len(orbits))]
    return compute_masks_and_beams(
        orbits, _obs_times(), rots, _u_body(), _kh(), NSIDE, verbose=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# compute_masks_and_beams
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMasksAndBeams:

    def test_output_shapes(self):
        masks, beams, omega_B = _run_cmb()
        assert masks.shape   == (N_TOTAL, NPIX)
        assert beams.shape   == (N_TOTAL, 2, NPIX)
        assert omega_B.shape == (N_TOTAL, 2)

    def test_output_dtypes(self):
        masks, beams, omega_B = _run_cmb()
        assert masks.dtype == np.float32
        assert beams.dtype == np.float32

    def test_open_sky_masks_all_ones(self):
        masks, _, _ = _run_cmb(open_sky=True)
        np.testing.assert_array_equal(masks, 1.0)

    def test_blocked_sky_masks_all_zeros(self):
        masks, _, _ = _run_cmb(open_sky=False)
        np.testing.assert_array_equal(masks, 0.0)

    def test_beams_nonnegative(self):
        _, beams, _ = _run_cmb()
        assert np.all(beams >= 0.0)

    def test_omega_B_equals_beam_sum(self):
        _, beams, omega_B = _run_cmb()
        np.testing.assert_allclose(omega_B, beams.sum(axis=2), rtol=1e-5)

    def test_orbit_masks_are_independent(self):
        """Different orbits with different masks produce distinct mask rows."""
        masks, _, _ = _run_cmb(
            orbits=[MockOrbit(open_sky=True), MockOrbit(open_sky=False)]
        )
        np.testing.assert_array_equal(masks[:N_OBS], 1.0)
        np.testing.assert_array_equal(masks[N_OBS:], 0.0)

    def test_rotations_affect_beam_not_mask(self):
        """Rotating the antenna changes beam values but not the occultation mask."""
        rots_0 = [_identity_rots(N_OBS)] * N_ORBIT
        # 45-degree rotation around z
        phi = np.full(N_OBS, np.pi / 4)
        rots_45 = [Rotation.from_rotvec(np.column_stack([np.zeros(N_OBS), np.zeros(N_OBS), phi]))] * N_ORBIT
        orbits = [MockOrbit(open_sky=True)] * N_ORBIT

        masks0, beams0, _ = compute_masks_and_beams(
            orbits, _obs_times(), rots_0, _u_body(), _kh(), NSIDE, verbose=False
        )
        masks45, beams45, _ = compute_masks_and_beams(
            orbits, _obs_times(), rots_45, _u_body(), _kh(), NSIDE, verbose=False
        )

        np.testing.assert_array_equal(masks0, masks45)
        assert not np.allclose(beams0, beams45)


# ──────────────────────────────────────────────────────────────────────────────
# simulate_observations
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateObservations:

    def _inputs(self, open_sky=True):
        masks, beams, omega_B = _run_cmb(open_sky=open_sky)
        gsm_map = np.ones(NPIX) * 1000.0
        J_SUN = np.zeros(N_OBS, dtype=int)
        return masks, beams, omega_B, gsm_map, J_SUN

    def test_output_shapes(self):
        m, b, w, g, J = self._inputs()
        data, y = simulate_observations(m, b, w, g, 200.0, 1e5, J, [0.0, 0.0])
        assert data.shape == (N_TOTAL, 2)
        assert y.shape    == (N_TOTAL * 2,)

    def test_y_is_data_ravel(self):
        m, b, w, g, J = self._inputs()
        data, y = simulate_observations(m, b, w, g, 200.0, 1e5, J, [0.0, 0.0])
        np.testing.assert_array_equal(y, data.ravel())

    def test_all_blocked_gives_t_regolith(self):
        """
        Fully blocked sky (m=0): sun is also occulted, so T_ant = t_regolith
        regardless of beam shape.
        """
        m, b, w, _, J = self._inputs(open_sky=False)
        T_REG = 275.0
        data, _ = simulate_observations(
            m, b, w,
            gsm_map=np.zeros(NPIX), t_regolith=T_REG, t_sun=1e6,
            J_SUN=J, sigma_noise=[0.0, 0.0],
        )
        np.testing.assert_allclose(data, T_REG, rtol=1e-5)

    def test_uniform_open_sky_gives_sky_temperature(self):
        """
        Uniform sky T with all-open mask and t_sun=0: T_ant = T for any beam
        (beam normalisation cancels in numerator / denominator).
        """
        m, b, w, _, J = self._inputs(open_sky=True)
        T_SKY = 3000.0
        data, _ = simulate_observations(
            m, b, w,
            gsm_map=np.full(NPIX, T_SKY), t_regolith=0.0, t_sun=0.0,
            J_SUN=J, sigma_noise=[0.0, 0.0],
        )
        np.testing.assert_allclose(data, T_SKY, rtol=1e-5)

    def test_noise_reproducible(self):
        m, b, w, g, J = self._inputs()
        _, y1 = simulate_observations(m, b, w, g, 200.0, 1e5, J, [10.0, 10.0],
                                       rng=np.random.default_rng(7))
        _, y2 = simulate_observations(m, b, w, g, 200.0, 1e5, J, [10.0, 10.0],
                                       rng=np.random.default_rng(7))
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_give_different_noise(self):
        m, b, w, g, J = self._inputs()
        _, y1 = simulate_observations(m, b, w, g, 200.0, 1e5, J, [10.0, 10.0],
                                       rng=np.random.default_rng(1))
        _, y2 = simulate_observations(m, b, w, g, 200.0, 1e5, J, [10.0, 10.0],
                                       rng=np.random.default_rng(2))
        assert not np.allclose(y1, y2)

    def test_zero_noise_noiseless(self):
        """With sigma_noise=0, noiseless and noisy calls return identical results."""
        m, b, w, g, J = self._inputs()
        _, y_noiseless = simulate_observations(m, b, w, g, 200.0, 1e5, J,
                                               sigma_noise=[0.0, 0.0])
        _, y_noisy = simulate_observations(m, b, w, g, 200.0, 1e5, J,
                                           sigma_noise=[10.0, 10.0],
                                           rng=np.random.default_rng(0))
        # Residual should be ~noise; noiseless call should have smaller residual
        assert np.max(np.abs(y_noiseless - y_noisy)) > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# build_design_matrix
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildDesignMatrix:

    def _inputs(self, open_sky=True):
        masks, beams, omega_B = _run_cmb(open_sky=open_sky)
        J_SUN = np.zeros(N_OBS, dtype=int)
        return masks, beams, omega_B, J_SUN

    def test_output_shape(self):
        m, b, w, J = self._inputs()
        A = build_design_matrix(m, b, w, J, NPIX)
        assert A.shape == (N_TOTAL * 2, NPIX + 2)

    def test_dtype_float64(self):
        m, b, w, J = self._inputs()
        A = build_design_matrix(m, b, w, J, NPIX)
        assert A.dtype == np.float64

    def test_row_normalization(self):
        """
        For every row: Σ_j A[r,j] + A[r,npix] = 1.

        Expanding: Σ_j B[j]*m[j]/OmB + Σ_j B[j]*(1-m[j])/OmB
                 = Σ_j B[j]/OmB = OmB/OmB = 1.

        This holds for any mask and any beam pattern.
        """
        for open_sky in (True, False):
            m, b, w, J = self._inputs(open_sky=open_sky)
            A = build_design_matrix(m, b, w, J, NPIX)
            row_sums = A[:, :NPIX].sum(axis=1) + A[:, NPIX]
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                       err_msg=f"open_sky={open_sky}")

    def test_open_sky_regolith_column_zero(self):
        """When m=1 everywhere, A[:,npix] (regolith) should be ~0."""
        m, b, w, J = self._inputs(open_sky=True)
        A = build_design_matrix(m, b, w, J, NPIX)
        np.testing.assert_allclose(A[:, NPIX], 0.0, atol=1e-6)

    def test_blocked_sky_only_regolith_nonzero(self):
        """
        When m=0 everywhere, sky and sun columns are zero; regolith column = 1.
        """
        m, b, w, J = self._inputs(open_sky=False)
        A = build_design_matrix(m, b, w, J, NPIX)
        np.testing.assert_allclose(A[:, :NPIX],   0.0, atol=1e-6)
        np.testing.assert_allclose(A[:, NPIX + 1], 0.0, atol=1e-6)  # sun blocked
        np.testing.assert_allclose(A[:, NPIX],     1.0, atol=1e-5)

    def test_nonnegative_entries(self):
        """All entries of A are non-negative (beam weights and masks are ≥ 0)."""
        for open_sky in (True, False):
            m, b, w, J = self._inputs(open_sky=open_sky)
            A = build_design_matrix(m, b, w, J, NPIX)
            assert np.all(A >= 0.0), f"Negative entries for open_sky={open_sky}"

    def test_forward_model_exact(self):
        """
        A @ x_true must reproduce noiseless simulate_observations output exactly.
        Uses a partial mask (half sky blocked) so all three parameter types
        (sky, regolith, sun) contribute non-trivially.
        """
        orbits = [MockOrbitPartial()] * N_ORBIT
        rots = [_identity_rots(N_OBS)] * N_ORBIT
        masks, beams, omega_B = compute_masks_and_beams(
            orbits, _obs_times(), rots, _u_body(), _kh(), NSIDE, verbose=False
        )
        J_SUN = np.zeros(N_OBS, dtype=int)  # sun at pixel 0 (open in partial mask)

        gsm_map = np.random.default_rng(0).uniform(1000, 5000, NPIX)
        T_REG, T_SUN = 250.0, 1e4

        _, y = simulate_observations(
            masks, beams, omega_B, gsm_map, T_REG, T_SUN,
            J_SUN, sigma_noise=[0.0, 0.0],
        )
        A = build_design_matrix(masks, beams, omega_B, J_SUN, NPIX)
        x_true = np.concatenate([gsm_map, [T_REG, T_SUN]])

        np.testing.assert_allclose(A @ x_true, y, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# svd_solve
# ──────────────────────────────────────────────────────────────────────────────

class TestSvdSolve:

    def _simple_system(self, n_rows=50, n_sky=8, seed=42):
        """
        Small synthetic over-constrained linear system for testing the
        solve arithmetic independently of the sky-simulation geometry.
        """
        rng = np.random.default_rng(seed)
        n_cols = n_sky + 2
        A = rng.standard_normal((n_rows, n_cols))
        x_true = rng.standard_normal(n_cols)
        y = A @ x_true
        return A, y, x_true, n_sky

    def _bloom_system(self):
        """Full BLOOM pipeline: masks → beams → A → y (noiseless)."""
        orbits = [MockOrbitPartial()] * N_ORBIT
        rots = [_identity_rots(N_OBS)] * N_ORBIT
        masks, beams, omega_B = compute_masks_and_beams(
            orbits, _obs_times(), rots, _u_body(), _kh(), NSIDE, verbose=False
        )
        J_SUN = np.zeros(N_OBS, dtype=int)
        gsm_map = np.random.default_rng(3).uniform(1000, 5000, NPIX)
        T_REG, T_SUN = 250.0, 1e4
        _, y = simulate_observations(
            masks, beams, omega_B, gsm_map, T_REG, T_SUN,
            J_SUN, sigma_noise=[0.0, 0.0],
        )
        A = build_design_matrix(masks, beams, omega_B, J_SUN, NPIX)
        return A, y, gsm_map, T_REG, T_SUN

    def test_result_keys(self):
        A, y, _, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        assert set(result) == {"sky_map", "t_regolith", "t_sun",
                               "U", "sv", "Vt", "rank", "unobserved"}

    def test_svd_shapes(self):
        A, y, _, n_sky = self._simple_system(n_rows=50, n_sky=8)
        result = svd_solve(A, y, n_sky)
        n_unknowns = n_sky + 2
        assert result["sv"].shape  == (n_unknowns,)
        assert result["Vt"].shape  == (n_unknowns, n_unknowns)
        assert result["U"].shape   == (50, n_unknowns)

    def test_svd_reconstructs_A(self):
        """U * diag(sv) * Vt = A (thin SVD identity)."""
        A, y, _, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        A_rec = result["U"] * result["sv"] @ result["Vt"]
        np.testing.assert_allclose(A_rec, A, atol=1e-8)

    def test_exact_recovery_overdetermined(self):
        """Exact recovery when the system is over-constrained and noise-free."""
        A, y, x_true, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        obs = ~result["unobserved"]
        np.testing.assert_allclose(result["sky_map"][obs], x_true[:n_sky][obs],
                                   rtol=1e-6)
        np.testing.assert_allclose(result["t_regolith"], x_true[n_sky],
                                   rtol=1e-6)
        np.testing.assert_allclose(result["t_sun"], x_true[n_sky + 1],
                                   rtol=1e-6)

    def test_rank_positive(self):
        A, y, _, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        assert result["rank"] > 0

    def test_rank_le_n_unknowns(self):
        A, y, _, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        assert result["rank"] <= n_sky + 2

    def test_unobserved_pixels_are_nan(self):
        """sky_map entries flagged unobserved must be NaN; others must not be."""
        A, y, _, n_sky = self._simple_system()
        result = svd_solve(A, y, n_sky)
        sky = result["sky_map"]
        unobs = result["unobserved"]
        np.testing.assert_array_equal(np.isnan(sky), unobs)

    def test_blocked_sky_recovers_t_regolith(self):
        """
        With fully blocked sky (A[:,npix]=1, all other columns=0) and noiseless
        data, t_regolith is recovered exactly.
        """
        T_REG = 275.0
        masks, beams, omega_B = _run_cmb(open_sky=False)
        J_SUN = np.zeros(N_OBS, dtype=int)
        _, y = simulate_observations(
            masks, beams, omega_B,
            gsm_map=np.zeros(NPIX), t_regolith=T_REG, t_sun=0.0,
            J_SUN=J_SUN, sigma_noise=[0.0, 0.0],
        )
        A = build_design_matrix(masks, beams, omega_B, J_SUN, NPIX)
        result = svd_solve(A, y, NPIX)
        np.testing.assert_allclose(result["t_regolith"], T_REG, rtol=1e-6)

    def test_bloom_pipeline_residual(self):
        """
        End-to-end: noiseless BLOOM pipeline residual ‖A x̂ − y‖ should be
        small (near machine precision for observed modes).
        """
        A, y, _, _, _ = self._bloom_system()
        result = svd_solve(A, y, NPIX)
        x_hat = np.where(
            result["unobserved"],
            0.0,
            result["sky_map"],
        )
        x_hat = np.concatenate([x_hat, [result["t_regolith"], result["t_sun"]]])
        residual = np.linalg.norm(A @ x_hat - y)
        # Residual should be small relative to ‖y‖
        assert residual / np.linalg.norm(y) < 1e-4
