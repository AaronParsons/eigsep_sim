"""Tests for Phase-1 external-source injection (sources.py + the
ext_source_dirs_gal/ext_source_temps hooks on ForwardModel).

Mirrors the existing transmitter tests in test_simulate.py (same
beam-weighted-point-source machinery), but exercises the *time-varying*,
galactic-frame, occultation-gated path used for real Sun/Earth ephemeris,
as opposed to the fixed-topocentric ``transmitters=`` constructor arg.
"""

import numpy as np
import healpy
import pytest
from astropy.time import Time
import astropy.units as u

from eigsep_sim.simulate import ForwardModel
from eigsep_sim.beam import Beam
from eigsep_sim.sky import Sky
from eigsep_sim.observer import EarthSurface, LunarOrbit
from eigsep_sim.sources import (
    quiet_sun_temperature_K,
    solar_activity_envelope,
    sun_temperature_K,
    inject_solar_bursts,
    flag_bursts,
    earth_rfi_temperature_K,
    FM_BAND_HZ,
)
from eigsep_sim.ephemeris import body_directions_gal, body_occulted_by_moon
from eigsep_sim.param_recovery import t21_template, t21_matched_filter


@pytest.fixture
def simple_fwd():
    """Minimal 1-dipole ForwardModel: nside=4, 3 freqs (matches test_simulate.py)."""
    rng = np.random.default_rng(0)
    freqs_hz = np.linspace(50e6, 150e6, 3)
    nside = 4
    npix = healpy.nside2npix(nside)
    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky_map = np.abs(rng.standard_normal((npix, 3))) + 1.0
    sky = Sky.from_map(nside, freqs_hz, sky_map, n_modes=2)
    observer = EarthSurface(lat=45.0, lon=0.0)
    observer.set_time("2000-01-01")
    return ForwardModel(observer, beam, sky)


@pytest.fixture
def simple_coeffs(simple_fwd):
    rng = np.random.default_rng(1)
    fwd = simple_fwd
    sky_c = rng.standard_normal((fwd.sky.npix, 2)).astype(np.float32)
    beam_c = np.abs(
        rng.standard_normal((fwd.beam.coeffs.shape[0], fwd.beam.npix, 2))
    ).astype(np.float32)
    return sky_c, beam_c


def test_ext_source_absent_matches_prior_behavior(simple_fwd, simple_coeffs):
    """No ext_source_dirs_gal / ext_source_temps -> geometry/simulate unchanged.

    Regression guard for the kernel-signature change: the new ext_px/wgts/T
    args must contribute exactly zero when unused.
    """
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    R = fwd.observer.rot_gal2top().astype(np.float32)

    geom = fwd.precompute_geometry(rots=[R] * 3)
    assert geom["ext_px_jax"].shape == (3, 4, 0)
    assert geom["ext_wgts_jax"].shape == (3, 4, 0)

    T = np.array(fwd.simulate(sky_c, beam_c, geom=geom))
    T_again = np.array(fwd.simulate(sky_c, beam_c, geom=geom, ext_source_temps=None))
    np.testing.assert_array_equal(T, T_again)


def test_ext_source_zero_temp_no_effect(simple_fwd, simple_coeffs):
    """An ext source with temp=0 at every step gives the same Tant as none."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    ntimes = 3
    R = fwd.observer.rot_gal2top().astype(np.float32)
    rots = [R] * ntimes

    geom_no = fwd.precompute_geometry(rots=rots)
    T_no = np.array(fwd.simulate(sky_c, beam_c, geom=geom_no))

    src_dir = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (ntimes, 1, 1))
    geom_src = fwd.precompute_geometry(rots=rots, ext_source_dirs_gal=src_dir)
    zero_temp = np.zeros((ntimes, 1, len(fwd.beam.freqs_hz)), dtype=np.float32)
    T_zero = np.array(
        fwd.simulate(sky_c, beam_c, geom=geom_src, ext_source_temps=zero_temp)
    )
    np.testing.assert_allclose(T_no, T_zero, atol=1e-6)


def test_ext_source_channel_selectivity(simple_fwd):
    """Ext source at freq[1] only raises Tant at that channel (zenith dir,
    zero sky/T_gnd to isolate the contribution, physical beam coefficients
    so the sign is unambiguous -- mirrors test_transmitter_channel_selectivity."""
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    nfreq = len(freqs_hz)
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 1

    sky_c = np.zeros((fwd.sky.npix, 2), dtype=np.float32)
    beam_c = fwd.beam.coeffs

    src_dir = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)  # (1,1,3) zenith
    temp = np.zeros((ntimes, 1, nfreq), dtype=np.float32)
    temp[0, 0, 1] = 1e4  # middle channel only

    geom_no = fwd.precompute_geometry(rots=[R] * ntimes)
    T_no = np.array(fwd.simulate(sky_c, beam_c, geom=geom_no, T_gnd=0.0))

    geom_src = fwd.precompute_geometry(
        rots=[R] * ntimes, ext_source_dirs_gal=src_dir
    )
    T_src = np.array(
        fwd.simulate(sky_c, beam_c, geom=geom_src, T_gnd=0.0, ext_source_temps=temp)
    )

    delta = T_src - T_no  # (1, n_dipoles, nfreq)
    np.testing.assert_allclose(delta[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta[..., 2], 0.0, atol=1e-6)
    assert np.all(delta[..., 1] > 0)


def test_ext_source_occultation_gating(simple_fwd):
    """Per-timestep zeroing (the caller's occultation gate) blocks exactly
    the gated timesteps and leaves the rest unaffected -- this is how real
    Moon-occultation of the Sun/Earth is meant to be applied."""
    fwd = simple_fwd
    freqs_hz = fwd.beam.freqs_hz
    nfreq = len(freqs_hz)
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 4

    sky_c = np.zeros((fwd.sky.npix, 2), dtype=np.float32)
    beam_c = fwd.beam.coeffs

    src_dir = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (ntimes, 1, 1))
    temp = np.full((ntimes, 1, nfreq), 1e4, dtype=np.float32)
    occulted = np.array([False, True, False, True])
    temp[occulted] = 0.0  # caller-side occultation gate

    geom_no = fwd.precompute_geometry(rots=[R] * ntimes)
    T_no = np.array(fwd.simulate(sky_c, beam_c, geom=geom_no, T_gnd=0.0))

    geom_src = fwd.precompute_geometry(rots=[R] * ntimes, ext_source_dirs_gal=src_dir)
    T_src = np.array(
        fwd.simulate(sky_c, beam_c, geom=geom_src, T_gnd=0.0, ext_source_temps=temp)
    )

    delta = T_src - T_no  # (ntimes, n_dipoles, nfreq)
    for t in range(ntimes):
        if occulted[t]:
            np.testing.assert_allclose(delta[t], 0.0, atol=1e-6)
        else:
            assert np.any(delta[t] > 1e-6)


def test_ext_source_geom_shapes(simple_fwd):
    fwd = simple_fwd
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 5
    n_ext = 2
    rng = np.random.default_rng(2)
    dirs = rng.standard_normal((ntimes, n_ext, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    geom = fwd.precompute_geometry(rots=[R] * ntimes, ext_source_dirs_gal=dirs)
    assert geom["ext_px_jax"].shape == (ntimes, 4, n_ext)
    assert geom["ext_wgts_jax"].shape == (ntimes, 4, n_ext)


def test_ext_source_linearity(simple_fwd, simple_coeffs):
    """Two identical ext sources produce double the contribution of one."""
    fwd = simple_fwd
    sky_c, beam_c = simple_coeffs
    nfreq = len(fwd.beam.freqs_hz)
    R = fwd.observer.rot_gal2top().astype(np.float32)
    ntimes = 1

    dir1 = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    dir2 = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    temp1 = np.full((ntimes, 1, nfreq), 1e4, dtype=np.float32)
    temp2 = np.full((ntimes, 2, nfreq), 1e4, dtype=np.float32)

    geom_0 = fwd.precompute_geometry(rots=[R] * ntimes)
    geom_1 = fwd.precompute_geometry(rots=[R] * ntimes, ext_source_dirs_gal=dir1)
    geom_2 = fwd.precompute_geometry(rots=[R] * ntimes, ext_source_dirs_gal=dir2)

    T_0 = np.array(fwd.simulate(sky_c, beam_c, geom=geom_0))
    T_1 = np.array(fwd.simulate(sky_c, beam_c, geom=geom_1, ext_source_temps=temp1))
    T_2 = np.array(fwd.simulate(sky_c, beam_c, geom=geom_2, ext_source_temps=temp2))

    np.testing.assert_allclose(T_2 - T_0, 2.0 * (T_1 - T_0), rtol=1e-5)


def test_quiet_sun_temperature_rises_toward_lower_frequency():
    freqs_hz = np.array([50e6, 80e6, 100e6, 150e6])
    T = quiet_sun_temperature_K(freqs_hz)
    assert np.all(np.diff(T) < 0)  # monotonically decreasing with frequency
    # right decade per the LOFAR anchor points (~6e5 K @ 150 MHz, ~1-2e6 K lower)
    assert 3e5 < T[-1] < 1e6
    assert 8e5 < T[0] < 3e6


def test_quiet_sun_end_to_end_through_real_ephemeris_and_occultation():
    """Full Phase-1 pipeline, wired together: real Sun ephemeris ->
    occultation-gated quiet-Sun brightness -> injected into simulate().

    Uses a low lunar orbit over a few hours (Sun direction barely moves in
    that span -- real orbital motion of the spacecraft around the Moon is
    what drives the occultation on/off, exactly as it would for a real
    far-side-transitioning mission orbit).  With zero sky/T_gnd, any
    nonzero antenna temperature is entirely the injected Sun term, so the
    occultation gating can be checked exactly (not just "changes").
    """
    freqs_hz = np.linspace(50e6, 150e6, 4)
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(
        nside, freqs_hz, np.zeros((npix_sky, len(freqs_hz))), n_modes=2
    )

    orbit = LunarOrbit(
        altitude=100e3, rot_orbit_vec=[0, 1, 0], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    fwd = ForwardModel(orbit, beam, sky)

    times = Time("2030-01-01") + np.linspace(0, 6 * 3600, 40) * u.s
    sun_dirs, _ = body_directions_gal(times, bodies=("sun",))
    occulted = body_occulted_by_moon(orbit, times, sun_dirs["sun"])
    assert 0.0 < occulted.mean() < 1.0, "test orbit should both see and lose the Sun"

    T_sun = quiet_sun_temperature_K(freqs_hz)  # (nfreq,)
    ntimes = len(times)
    ext_dirs = sun_dirs["sun"][:, None, :]  # (ntimes, 1, 3)
    ext_temps = np.broadcast_to(T_sun, (ntimes, 1, len(freqs_hz))).copy()
    ext_temps[occulted] = 0.0

    geom = fwd.precompute_geometry(times=times, ext_source_dirs_gal=ext_dirs)
    T_ant = np.array(
        fwd.simulate(
            np.zeros((npix_sky, 2), dtype=np.float32),
            beam.coeffs,
            geom=geom,
            T_gnd=0.0,
            ext_source_temps=ext_temps,
        )
    )  # (ntimes, n_dipoles, nfreq)

    excess = np.abs(T_ant).max(axis=(1, 2))  # (ntimes,)
    np.testing.assert_allclose(excess[occulted], 0.0, atol=1e-6)
    assert np.all(excess[~occulted] > 1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Solar slow variability
# ─────────────────────────────────────────────────────────────────────────


def test_solar_activity_envelope_bounded_and_varying():
    times = Time("2030-01-01") + np.linspace(0, 400, 200) * u.day
    env = solar_activity_envelope(times, cycle_min=0.5, cycle_max=1.5,
                                  rotation_amplitude=0.15)
    # bounded within cycle range inflated by the rotational modulation
    assert np.all(env > 0.5 * (1 - 0.15) - 1e-6)
    assert np.all(env < 1.5 * (1 + 0.15) + 1e-6)
    assert env.std() > 0.01  # not constant


def test_solar_activity_envelope_reproducible():
    times = Time("2030-01-01") + np.linspace(0, 10, 20) * u.day
    env1 = solar_activity_envelope(times)
    env2 = solar_activity_envelope(times)
    np.testing.assert_array_equal(env1, env2)  # deterministic, no RNG


def test_sun_temperature_K_shape_and_scaling():
    freqs_hz = np.array([50e6, 100e6, 150e6])
    times = Time("2030-01-01") + np.linspace(0, 60, 10) * u.day
    T = sun_temperature_K(freqs_hz, times)
    assert T.shape == (10, 3)
    env = solar_activity_envelope(times)
    T_ref = quiet_sun_temperature_K(freqs_hz)
    np.testing.assert_allclose(T, env[:, None] * T_ref[None, :])


# ─────────────────────────────────────────────────────────────────────────
# Solar bursts: injection + flagging
# ─────────────────────────────────────────────────────────────────────────


def test_inject_solar_bursts_zero_rate_gives_no_bursts():
    times = Time("2030-01-01") + np.linspace(0, 6, 40) * u.hour
    freqs_hz = np.linspace(50e6, 150e6, 5)
    rng = np.random.default_rng(0)
    burst_temp, is_contaminated = inject_solar_bursts(
        times, freqs_hz, rng, rate_per_hour=0.0
    )
    np.testing.assert_array_equal(burst_temp, 0.0)
    assert not is_contaminated.any()


def test_inject_solar_bursts_shape_and_nonnegative():
    times = Time("2030-01-01") + np.linspace(0, 6, 200) * u.hour
    freqs_hz = np.linspace(50e6, 150e6, 5)
    rng = np.random.default_rng(1)
    burst_temp, is_contaminated = inject_solar_bursts(
        times, freqs_hz, rng, rate_per_hour=10.0
    )
    assert burst_temp.shape == (200, 5)
    assert np.all(burst_temp >= 0.0)
    assert is_contaminated.any(), "high rate over 6 hr should produce >=1 burst"
    assert is_contaminated.shape == (200,)


def test_inject_solar_bursts_spectral_index_scaling():
    """Every burst shares the same frequency shape (f/ref_freq)**index, so
    the ratio holds exactly at any time with nonzero power, even with
    several overlapping bursts (the sum is still separable in time/freq
    since freq_shape doesn't depend on which burst). rate_per_hour is set
    high enough that P(zero bursts) is negligible."""
    times = Time("2030-01-01") + np.linspace(0, 1, 500) * u.hour
    freqs_hz = np.array([50e6, 100e6, 150e6])
    rng = np.random.default_rng(2)
    burst_temp, _ = inject_solar_bursts(
        times, freqs_hz, rng, rate_per_hour=50.0,
        ref_freq_hz=100e6, ref_amplitude_K=(1e6, 1e6),  # fixed amplitude
        duration_s=(5.0, 5.0),
    )
    assert burst_temp.max() > 0, "expected >=1 burst at this rate"
    peak_t = burst_temp.max(axis=1).argmax()
    shape = burst_temp[peak_t] / burst_temp[peak_t, 1]  # normalize to 100 MHz bin
    expected = (freqs_hz / 100e6) ** -2.9
    np.testing.assert_allclose(shape, expected, rtol=1e-6)


def test_flag_bursts_recovers_injected_bursts_with_low_false_positive_rate():
    rng = np.random.default_rng(3)
    ntimes, nfreq = 500, 4
    times = Time("2030-01-01") + np.linspace(0, 12, ntimes) * u.hour
    freqs_hz = np.linspace(50e6, 150e6, nfreq)

    noise = rng.standard_normal((ntimes, nfreq)) * 0.05
    burst_temp, truth = inject_solar_bursts(
        times, freqs_hz, rng, rate_per_hour=3.0,
        ref_amplitude_K=(1e6, 5e7), duration_s=(1.0, 30.0),
    )
    data = noise + burst_temp

    flagged = flag_bursts(data, threshold_sigma=6.0)

    if truth.any():
        recall = (flagged & truth).sum() / truth.sum()
        assert recall > 0.8
    # false-positive rate on genuinely clean samples should be small
    clean = ~truth
    if clean.any():
        fpr = (flagged & clean).sum() / clean.sum()
        assert fpr < 0.08  # 6-sigma across 4 channels combined via "any" inflates this a bit


def test_flag_bursts_axis_and_shape():
    data = np.zeros((50, 3, 4))
    data[10, 1, 2] = 100.0  # single huge outlier
    flagged = flag_bursts(data, threshold_sigma=5.0, axis=0)
    assert flagged.shape == (50,)
    assert flagged[10]
    assert flagged.sum() == 1


# ─────────────────────────────────────────────────────────────────────────
# Earth RFI
# ─────────────────────────────────────────────────────────────────────────


def test_earth_rfi_temperature_fm_band_only():
    freqs_hz = np.array([60e6, 90e6, 100e6, 108e6, 120e6, 150e6])
    T = earth_rfi_temperature_K(freqs_hz, in_band_temp_K=5e5, out_of_band_temp_K=0.0)
    in_band = (freqs_hz >= FM_BAND_HZ[0]) & (freqs_hz <= FM_BAND_HZ[1])
    np.testing.assert_array_equal(T[in_band], 5e5)
    np.testing.assert_array_equal(T[~in_band], 0.0)


def test_earth_rfi_end_to_end_through_real_ephemeris_and_occultation():
    """Same pattern as the quiet-Sun end-to-end test, for Earth RFI."""
    freqs_hz = np.array([95e6, 60e6, 130e6])  # index 0 in-FM-band, rest not
    nside = 4
    npix_sky = healpy.nside2npix(nside)

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(
        nside, freqs_hz, np.zeros((npix_sky, len(freqs_hz))), n_modes=2
    )

    orbit = LunarOrbit(
        altitude=100e3, rot_orbit_vec=[0, 1, 0], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    fwd = ForwardModel(orbit, beam, sky)

    times = Time("2030-01-01") + np.linspace(0, 6 * 3600, 40) * u.s
    earth_dirs, _ = body_directions_gal(times, bodies=("earth",))
    occulted = body_occulted_by_moon(orbit, times, earth_dirs["earth"])
    assert 0.0 < occulted.mean() < 1.0, "test orbit should both see and lose Earth"

    T_rfi = earth_rfi_temperature_K(freqs_hz)  # (nfreq,), zero outside FM band
    ntimes = len(times)
    ext_dirs = earth_dirs["earth"][:, None, :]
    ext_temps = np.broadcast_to(T_rfi, (ntimes, 1, len(freqs_hz))).copy()
    ext_temps[occulted] = 0.0

    geom = fwd.precompute_geometry(times=times, ext_source_dirs_gal=ext_dirs)
    T_ant = np.array(
        fwd.simulate(
            np.zeros((npix_sky, 2), dtype=np.float32),
            beam.coeffs,
            geom=geom,
            T_gnd=0.0,
            ext_source_temps=ext_temps,
        )
    )  # (ntimes, n_dipoles, nfreq)

    excess = np.abs(T_ant).max(axis=1)  # (ntimes, nfreq)
    np.testing.assert_allclose(excess[occulted], 0.0, atol=1e-6)
    # visible timesteps: nonzero at the in-FM-band channel, zero elsewhere
    assert np.all(excess[~occulted, 0] > 1e-6)
    np.testing.assert_allclose(excess[~occulted, 1:], 0.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Burst flagging protects T21 recovery (validation per the realism plan)
# ─────────────────────────────────────────────────────────────────────────


def test_burst_flagging_protects_t21_recovery():
    """Injecting unmodelled solar bursts biases a matched-filter T21
    recovery; excising flagged timesteps brings it back toward the
    no-burst baseline. Uses the matched-filter machinery from
    param_recovery.py directly (no fit involved) so the test isolates the
    data-selection question flagging is meant to answer."""
    freqs_hz = np.linspace(50e6, 150e6, 6)
    nside = 4
    npix_sky = healpy.nside2npix(nside)
    ntimes = 300

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=3.0, K=2)
    sky = Sky.from_map(
        nside, freqs_hz, np.zeros((npix_sky, len(freqs_hz))), n_modes=2
    )
    orbit = LunarOrbit(
        altitude=100e3, rot_orbit_vec=[0, 1, 0], rot_spin_vec=[0, 0, 1],
        spin_period=0.0, t0=Time("2030-01-01"),
    )
    fwd = ForwardModel(orbit, beam, sky)
    times = Time("2030-01-01") + np.linspace(0, 12 * 3600, ntimes) * u.s
    geom = fwd.precompute_geometry(times=times)

    T21_true = (-0.1 * np.exp(-0.5 * ((freqs_hz / 1e6 - 75) / 15) ** 2))
    sky_coeffs = np.zeros((npix_sky, 2), dtype=np.float32)

    template, terrain = t21_template(fwd, geom, beam.coeffs @ beam.basis.A.T)
    data_clean = np.asarray(
        fwd.simulate(sky_coeffs, beam.coeffs, geom=geom, T_iso=T21_true)
    )

    sun_dirs, _ = body_directions_gal(times, bodies=("sun",))
    occulted = body_occulted_by_moon(orbit, times, sun_dirs["sun"])
    rng = np.random.default_rng(4)
    burst_temp, truth = inject_solar_bursts(
        times, freqs_hz, rng, rate_per_hour=4.0,
        ref_amplitude_K=(5e5, 2e6), duration_s=(5.0, 60.0),
    )
    burst_temp[occulted] = 0.0  # bursts also occulted when Sun is blocked
    ext_dirs = sun_dirs["sun"][:, None, :]
    geom_sun = fwd.precompute_geometry(times=times, ext_source_dirs_gal=ext_dirs)
    burst_contrib = np.asarray(
        fwd.simulate(
            sky_coeffs, beam.coeffs, geom=geom_sun,
            ext_source_temps=burst_temp[:, None, :],
        )
    ) - np.asarray(fwd.simulate(sky_coeffs, beam.coeffs, geom=geom_sun))
    data_with_bursts = data_clean + burst_contrib

    T21_baseline = t21_matched_filter(data_clean - terrain, template)
    T21_all = t21_matched_filter(data_with_bursts - terrain, template)

    flagged = flag_bursts(data_with_bursts, threshold_sigma=6.0, axis=0)
    keep = ~flagged
    assert keep.sum() < ntimes, "test should actually excise some timesteps"
    T21_flagged = t21_matched_filter(
        data_with_bursts[keep] - terrain[keep], template[keep]
    )

    err_all = np.max(np.abs(T21_all - T21_baseline))
    err_flagged = np.max(np.abs(T21_flagged - T21_baseline))
    assert err_flagged < err_all
