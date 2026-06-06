"""Focused tests for reusable lunar campaign infrastructure."""

import healpy
import numpy as np
import pytest
from astropy.coordinates import (
    BarycentricMeanEcliptic,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time
import astropy.units as u

from eigsep_sim import Beam, ForwardModel, LunarOrbit, Sky
from eigsep_sim.lunar import (
    LunarCampaign,
    angular_momentum_for_spin_period,
    crossed_rod_inertia,
    integrate_torque_free,
    make_ecliptic_orbit_normals,
)
from eigsep_sim.lunar_recovery import LunarRecoveryAdapter


def _simple_objects(nside=4, nfreq=3):
    freqs_hz = np.linspace(55e6, 95e6, nfreq)
    npix = healpy.nside2npix(nside)
    sky_map = 800.0 + np.arange(npix)[:, None] + np.linspace(0.0, 20.0, nfreq)
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=nfreq)
    sky_coeffs = sky.basis.project(sky_map).astype(np.float32)
    beam = Beam.from_dipole(nside, freqs_hz, [6.0, 4.0], K=nfreq)
    observer = LunarOrbit(100e3, [0, 0, 1], [0, 0, 1])
    return freqs_hz, sky_map, sky, sky_coeffs, beam, observer


def _campaign_config():
    return {
        "spacecraft": {
            "opening_angle_deg": 90.0,
            "arm_lengths_m": [6.0, 4.0],
            "arm_masses_kg": [1.0, 2.25],
            "angular_momentum_direction_gal": [1.0, 0.0, 0.2],
            "spin_period_s": 60.0,
            "attitude_phase_offsets_deg": [0.0, 45.0],
        },
        "orbit": {
            "altitude_km": 100.0,
            "inclinations_deg": [-15.0, 15.0],
            "ascending_node_lon_deg": 0.0,
            "equinox": "J2000",
            "epoch": "2025-01-01",
        },
        "antenna": {"nside": 4, "n_modes": 3},
        "sky": {"nside": 4, "n_modes": 3},
        "frequency": {"min_mhz": 55.0, "max_mhz": 95.0, "nchan": 3},
        "integration": {
            "duration_hours": 2.0,
            "ntimes": 4,
            "attitude_step_s": 10.0,
        },
        "surface": {"T_regolith_K": 300.0, "reflectivity_enabled": False},
        "signal_21cm": {"enabled": False, "model_index": 0},
        "sources": {"earth": {"enabled": False}, "sun": {"enabled": False}},
    }


def test_ecliptic_orbit_normals_have_common_node_and_inclinations():
    normals_gal = make_ecliptic_orbit_normals([-15.0, 15.0], 0.0, "J2000")
    assert np.allclose(np.linalg.norm(normals_gal, axis=1), 1.0)
    coords = SkyCoord(
        CartesianRepresentation(normals_gal.T, unit=u.one), frame="galactic"
    )
    normals_ecl = coords.transform_to(
        BarycentricMeanEcliptic(equinox=Time("J2000"))
    ).cartesian.xyz.value.T
    inclinations = np.rad2deg(np.arccos(normals_ecl[:, 2]))
    np.testing.assert_allclose(inclinations, [15.0, 15.0], atol=1e-6)
    assert np.allclose(
        normals_ecl[:, 1], [-normals_ecl[1, 1], normals_ecl[1, 1]], atol=1e-10
    )
    np.testing.assert_allclose(normals_ecl[:, 0], 0.0, atol=1e-10)


def test_angular_momentum_direction_is_scaled_to_spin_period():
    inertia = crossed_rod_inertia()
    direction = np.array([2.0, 0.0, 0.4])
    momentum = angular_momentum_for_spin_period(inertia, direction, 60.0)
    omega = np.linalg.solve(inertia, momentum)
    np.testing.assert_allclose(
        momentum / np.linalg.norm(momentum),
        direction / np.linalg.norm(direction),
    )
    np.testing.assert_allclose(np.linalg.norm(omega), 2.0 * np.pi / 60.0)


def test_angular_momentum_spin_period_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        angular_momentum_for_spin_period(np.eye(3), [1.0, 0.0, 0.0], 0.0)


def test_torque_free_conserves_norm_momentum_and_energy():
    inertia = crossed_rod_inertia()
    sim = integrate_torque_free(
        inertia, [1.0, 0.2, 0.4], np.linspace(0.0, 30.0, 301)
    )
    np.testing.assert_allclose(
        np.linalg.norm(sim["quaternions_wxyz"], axis=1), 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        sim["momentum_inertial"],
        np.broadcast_to(
            sim["momentum_inertial"][0], sim["momentum_inertial"].shape
        ),
        atol=2e-8,
    )
    np.testing.assert_allclose(
        sim["kinetic_energy"],
        np.broadcast_to(sim["kinetic_energy"][0], sim["kinetic_energy"].shape),
        rtol=2e-8,
    )


def test_forward_model_uniform_surface_matches_scalar_fallback_and_reduction():
    _, _, sky, sky_coeffs, beam, observer = _simple_objects()
    times = observer.t0 + np.array([0.0, 900.0]) * u.s
    scalar = ForwardModel(observer, beam, sky)

    _, _, _, _, _, emitting_observer = _simple_objects()
    emitting_observer.occultation_temperature_K = 300.0
    surface = ForwardModel(emitting_observer, beam, sky)

    scalar_geom = scalar.precompute_geometry(times=times)
    surface_geom = surface.precompute_geometry(times=times)
    scalar_truth = np.asarray(
        scalar.simulate(sky_coeffs, beam.coeffs, geom=scalar_geom, T_gnd=300.0)
    )
    surface_truth = np.asarray(
        surface.simulate(
            sky_coeffs, beam.coeffs, geom=surface_geom, T_gnd=999.0
        )
    )
    np.testing.assert_allclose(
        surface_truth, scalar_truth, rtol=2e-5, atol=2e-4
    )
    sky_mask = surface.build_sky_mask(times=times)
    reduced_geom = surface.precompute_geometry(times=times, sky_mask=sky_mask)
    reduced_truth = np.asarray(
        surface.simulate(
            sky_coeffs, beam.coeffs, geom=reduced_geom, T_gnd=999.0
        )
    )
    np.testing.assert_allclose(
        reduced_truth, surface_truth, rtol=2e-5, atol=2e-4
    )


def test_campaign_shapes_tracks_phase_and_recovery_adapter():
    freqs_hz, sky_map, sky, sky_coeffs, _, _ = _simple_objects()
    campaign = LunarCampaign(
        _campaign_config(), sky=sky, sky_coeffs=sky_coeffs
    )
    result = campaign.run()
    np.testing.assert_allclose(
        campaign.initial_angular_speed_rad_s, 2.0 * np.pi / 60.0
    )
    assert result.spectra_K.shape == (2, 4, 2, len(freqs_hz))
    assert result.masks.shape == (2, 4, sky.npix)
    assert not np.array_equal(result.masks[0], result.masks[1])
    assert not np.allclose(result.body_rots[0], result.body_rots[1])
    adapter = LunarRecoveryAdapter(campaign, result)
    sky_recon = campaign.sky.basis.deproject(campaign.sky_coeffs)
    adapter.verify_noiseless(sky_recon[:, 1], 300.0, 1, rtol=3e-5, atol=3e-4)
    assert adapter.source_registry["earth"]["enabled"] is False
    assert adapter.source_registry["sun"]["enabled"] is False
