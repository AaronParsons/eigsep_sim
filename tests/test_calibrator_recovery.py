#!/usr/bin/env python
"""Accuracy regression tests for joint sky+beam recovery.

These tests guard against the two failure modes diagnosed in the
EIGSEP_Recovery_v001 plateau investigation (2026-06):

1. Solver plateau: damped Jacobi fixed-point steps stall on the bilinear
   sky x beam problem; exact conditional Newton-CG solves (solver='fast-cg')
   must reach the noise floor in a handful of iterations.
2. Objective bias: with the harmonic beam regularizer anchored at the
   initial beam, a high-ell beam perturbation makes the truth heavily
   penalized.  Each scenario pairs perturbation and regularizer
   consistently: i.i.d. scatter init with lam_beam_harmonic=0 (the
   notebook configuration), or a smooth low-ell perturbation with a
   modest harmonic prior.
"""

import numpy as np
import pytest
from astropy.time import Time

from eigsep_sim import Beam, Sky, ForwardModel, NullTerrain, Calibrator
from eigsep_sim.observer import EarthSurface


def build_recovery_problem(noise_snr=1e4, beam_perturb="scatter"):
    """Small EIGSEP-like joint recovery problem with known truth.

    beam_perturb='scatter' applies i.i.d. +-10% scatter to every beam
    coefficient (the notebook's init; pair with lam_beam_harmonic=0).
    beam_perturb='smooth' applies a low-ell multiplicative pattern,
    consistent with a harmonic beam-shape prior.
    """
    freqs_hz = np.linspace(60e6, 140e6, 8)
    nside = 4

    beam = Beam.from_dipole(nside, freqs_hz, arm_lengths_m=[2.0], K=3)
    sky = Sky.from_gsm(nside, freqs_hz, n_modes=3, include_flat=True)
    observer = EarthSurface(lat=39.2, lon=-113.4)
    fwd = ForwardModel(observer, beam, sky, terrain=NullTerrain())

    sky_coeffs = sky.init_coeffs()
    beam_coeffs = beam.coeffs.copy()

    # A few sidereal times x a few beam orientations (az/alt scan).
    times = [Time("2025-01-01") + i * 0.25 for i in range(6)]
    base_rots = observer.rot_gal2top_stack(times)
    orient_rots = np.stack(
        [
            Beam.top2body(az, alt)
            for alt in (0.0, 0.8)
            for az in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
    )
    n_orient = orient_rots.shape[0]
    rots = np.repeat(base_rots, n_orient, axis=0)
    body_rots = np.tile(orient_rots, (len(times), 1, 1))
    geom = fwd.precompute_geometry(rots=rots, body_rots=body_rots)

    truth = np.asarray(fwd.simulate(sky_coeffs, beam_coeffs, geom=geom))
    sigma = np.abs(truth).mean() / noise_snr
    rng = np.random.default_rng(seed=0)
    data = truth + rng.normal(scale=sigma, size=truth.shape)
    inv_noise_var = np.full_like(data, 1.0 / sigma**2)

    import healpy

    prng = np.random.default_rng(seed=1)
    if beam_perturb == "scatter":
        # i.i.d. +-10% scatter on every beam coefficient.
        beam_ini = beam_coeffs * prng.uniform(
            0.9, 1.1, size=beam_coeffs.shape
        ).astype(beam_coeffs.dtype)
    elif beam_perturb == "smooth":
        # ell<=1 pattern: constant + dipole along a random direction.
        pix_vec = np.array(
            healpy.pix2vec(nside, np.arange(beam.npix))
        )  # (3, npix)
        direction = prng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pattern = 0.5 + (direction @ pix_vec)
        pattern /= np.sqrt(np.mean(pattern**2))
        beam_ini = beam_coeffs * (1.0 + 0.1 * pattern[None, :, None]).astype(
            beam_coeffs.dtype
        )
    else:
        raise ValueError(f"unknown beam_perturb: {beam_perturb}")
    sky_ini = sky_coeffs * prng.uniform(
        0.9, 1.1, size=sky_coeffs.shape
    ).astype(sky_coeffs.dtype)

    return {
        "fwd": fwd,
        "geom": geom,
        "data": data,
        "inv_noise_var": inv_noise_var,
        "prms_tru": {"sky_coeffs": sky_coeffs, "beam_coeffs": beam_coeffs},
        "prms_ini": {"sky_coeffs": sky_ini, "beam_coeffs": beam_ini},
        "sigma": sigma,
    }


def test_fast_cg_reaches_noise_floor():
    """fast-cg must reach the noise floor; adaptive steps plateau here.

    The loss is mean(inv_noise_var * resid**2), so at the truth it is ~1.
    A correct solver gets within a small factor of that in a few exact
    conditional solves.  The damped Jacobi solvers stall orders of
    magnitude higher on this same problem.

    Uses the notebook configuration: i.i.d. coefficient scatter init and
    no harmonic beam prior (which would otherwise anchor to the scattered
    init and make the truth score worse than the init).
    """
    prob = build_recovery_problem(beam_perturb="scatter")
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._beam_nom = np.asarray(prob["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=prob["geom"])

    loss_ini = float(cal._loss(prob["prms_ini"]))
    loss_tru = float(cal._loss(prob["prms_tru"]))
    # Objective consistency: the truth must score far better than the
    # init, otherwise the recovery benchmark is meaningless.
    assert loss_tru < 0.01 * loss_ini

    result = cal.fit(
        params={k: v.copy() for k, v in prob["prms_ini"].items()},
        geom=prob["geom"],
        max_iter=8,
        tol=1e-4,
        verbose=False,
        solver="fast-cg",
    )
    data_chi2 = float(cal.data_loss(result["params"]))
    # Truth-level data chi2 is ~1; allow slack for finite iterations and
    # regularization pull, but stay far below the init loss scale.
    assert data_chi2 < 10.0, f"data_chi2={data_chi2:.3e}"
    assert data_chi2 < 1e-3 * loss_ini


def test_fast_cg_beam_error_does_not_grow():
    """Recovered beam must not drift away from truth while fitting data.

    At this problem scale the sky basis can absorb nearly any smooth beam
    perturbation, so joint recovery cannot pull the beam all the way to
    truth (its error is dominated by data-degenerate directions pinned to
    the init by the ridge/harmonic priors).  The regression guarded here
    is the solver riding a degeneracy to a *worse* beam while chi2 drops.
    """
    prob = build_recovery_problem(beam_perturb="smooth")
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=1e2,
    )
    cal._beam_nom = np.asarray(prob["prms_ini"]["beam_coeffs"]).copy()
    cal._resolve_geom(geom=prob["geom"])

    result = cal.fit(
        params={k: v.copy() for k, v in prob["prms_ini"].items()},
        geom=prob["geom"],
        max_iter=8,
        tol=1e-4,
        verbose=False,
        solver="fast-cg",
    )

    fwd = prob["fwd"]
    A_beam = fwd.beam.basis.A
    beam_tru = prob["prms_tru"]["beam_coeffs"] @ A_beam.T
    beam_ini = prob["prms_ini"]["beam_coeffs"] @ A_beam.T
    beam_fit = np.asarray(result["params"]["beam_coeffs"]) @ A_beam.T

    # Remove the per-frequency multiplicative sky/beam gauge before
    # comparing (the notebook removes the same gauge via ScaleDegeneracy).
    def align(b):
        scale = np.sum(b * beam_tru, axis=(0, 1)) / np.maximum(
            np.sum(b**2, axis=(0, 1)), 1e-30
        )
        return b * scale[None, None, :]

    err_fit = np.linalg.norm(align(beam_fit) - beam_tru)
    err_ini = np.linalg.norm(align(beam_ini) - beam_tru)
    assert err_fit < 1.02 * err_ini, (
        f"beam error grew while fitting data: "
        f"fit={err_fit:.3e} init={err_ini:.3e}"
    )


def test_conditional_linear_operators_match_simulate():
    """The fast-path linear operators must reproduce the forward model.

    sky_step/beam_cg_step solve the conditional problems through
    precomputed operators (G for sky with beam fixed, W for beam with sky
    fixed).  Their predictions must match ForwardModel.simulate exactly
    (up to float32 roundoff) including the affine emission offset.
    """
    import jax.numpy as jnp

    prob = build_recovery_problem()
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._resolve_geom(geom=prob["geom"])
    ops = cal._ensure_linear_ops()
    assert ops is not None

    fwd = prob["fwd"]
    s = prob["prms_ini"]["sky_coeffs"]
    b = prob["prms_ini"]["beam_coeffs"]
    ref = np.asarray(fwd.simulate(s, b, geom=prob["geom"]))  # (T, D, F)
    scale = np.abs(ref).max()

    # Sky operator: pred = G . sky_recon + simulate(0, beam).
    offset = np.asarray(fwd.simulate(np.zeros_like(s), b, geom=prob["geom"]))
    g_op = np.asarray(ops["build_g"](jnp.asarray(b)))  # (D, F, T, P)
    pred_sky = np.einsum("dftp,pf->tdf", g_op, s @ fwd.sky.basis.A.T) + offset
    assert np.max(np.abs(pred_sky - ref)) < 1e-4 * scale

    # Beam operator: pred = W . beam_recon (strictly linear in beam).
    w_op = np.asarray(ops["build_w"](jnp.asarray(s)))  # (F, T, Q)
    pred_beam = np.einsum("ftq,dqf->tdf", w_op, b @ fwd.beam.basis.A.T)
    assert np.max(np.abs(pred_beam - ref)) < 1e-4 * scale


def test_direct_sky_solve_beats_truncated_cg():
    """The exact Cholesky sky solve must reach a lower conditional loss.

    The conditional sky Hessian is ill-conditioned (kappa ~ 1e8 from sky
    coverage), so truncated CG stalls well above the conditional minimum.
    The direct normal-equations solve reaches that minimum in one step;
    here it must beat a 50-iteration CG solve from the same point.
    """
    prob = build_recovery_problem()
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._resolve_geom(geom=prob["geom"])

    params = {k: v.copy() for k, v in prob["prms_ini"].items()}
    loss_before = float(cal._loss(params))

    sky_direct = cal._sky_step_direct(params, step_size=1.0)
    assert sky_direct is not None
    loss_direct = float(
        cal._loss(
            {"sky_coeffs": sky_direct, "beam_coeffs": params["beam_coeffs"]}
        )
    )

    sky_cg = cal._sky_step_linear(params, n_cg=50, lam=1e-4, step_size=1.0)
    loss_cg = float(
        cal._loss(
            {
                "sky_coeffs": np.asarray(sky_cg),
                "beam_coeffs": params["beam_coeffs"],
            }
        )
    )

    assert loss_direct < loss_before
    assert loss_direct <= loss_cg + 1e-6


def test_direct_sky_solve_falls_back_when_too_large():
    """Direct solve declines (returns None) above the dense-size budget.

    The Cholesky is O((npix*nmodes)^3); _sky_step_direct must defer to the
    CG path when the system exceeds max_unknowns so high-nside problems
    don't attempt an intractable dense factorization.
    """
    prob = build_recovery_problem()
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._resolve_geom(geom=prob["geom"])
    params = {k: v.copy() for k, v in prob["prms_ini"].items()}
    assert cal._sky_step_direct(params, step_size=1.0, max_unknowns=1) is None


def test_direct_sky_solve_with_terrain_masking():
    """The factorization handles terrain masking via the same operators.

    The sky/beam conditional operators (build_g, build_w) fold in the
    time-varying terrain visibility mask and the terrain emission offset,
    so the direct sky solve and the full fast-cg fit work unchanged when a
    large fraction of the sky is blocked (the EIGSEP canyon case).
    """
    import healpy
    import jax.numpy as jnp
    from astropy.time import Time

    from eigsep_sim.observer import EarthSurface
    from eigsep_sim.terrain import HorizonTerrain

    nside = 4
    npix = healpy.nside2npix(nside)
    freqs = np.linspace(60e6, 140e6, 8)

    # Topocentric horizon map: finite => terrain-blocked, NaN => open sky.
    vec = np.array(healpy.pix2vec(nside, np.arange(npix)))
    horizon = np.where(vec[2] < 0.3, 100.0, np.nan).astype(np.float32)
    terr = HorizonTerrain(nside, horizon, T_terrain=300.0)

    beam = Beam.from_dipole(nside, freqs, arm_lengths_m=[2.0], K=3)
    sky = Sky.from_gsm(nside, freqs, n_modes=3, include_flat=True)
    obs = EarthSurface(lat=39.2, lon=-113.4)
    fwd = ForwardModel(obs, beam, sky, terrain=terr)
    sc, bc = sky.init_coeffs(), beam.coeffs.copy()

    times = [Time("2025-01-01") + i * 0.25 for i in range(6)]
    base = obs.rot_gal2top_stack(times)
    orient = np.stack(
        [
            Beam.top2body(a, h)
            for h in (0.0, 0.8)
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
    )
    rots = np.repeat(base, orient.shape[0], axis=0)
    body = np.tile(orient, (len(times), 1, 1))
    geom = fwd.precompute_geometry(rots=rots, body_rots=body)
    # Terrain blocks a large fraction of the sky in this configuration.
    assert float(np.mean(np.asarray(geom["terrain_masks_jax"]))) < 0.6

    truth = np.asarray(fwd.simulate(sc, bc, geom=geom))
    sigma = np.abs(truth).mean() / 1e4
    rng = np.random.default_rng(0)
    data = (truth + rng.normal(scale=sigma, size=truth.shape)).reshape(-1, 8)
    inv = np.full_like(data, 1.0 / sigma**2)
    cal = Calibrator(
        fwd, data, inv_noise_var=inv, lam_beam=0.01, lam_beam_harmonic=0.0
    )
    cal._resolve_geom(geom=geom)
    ops = cal._ensure_linear_ops()

    # Conditional operators reproduce simulate WITH terrain mask + emission.
    scale = np.abs(truth).max()
    const = np.asarray(fwd.simulate(np.zeros_like(sc), bc, geom=geom))
    g = np.asarray(ops["build_g"](jnp.asarray(bc)))  # (D, F, T, P)
    pred_sky = np.einsum("dftp,pf->tdf", g, sc @ sky.basis.A.T) + const
    assert np.max(np.abs(pred_sky - truth)) < 1e-4 * scale
    w = np.asarray(ops["build_w"](jnp.asarray(sc)))  # (F, T, Q)
    pred_beam = np.einsum("ftq,dqf->tdf", w, bc @ beam.basis.A.T)
    assert np.max(np.abs(pred_beam - truth)) < 1e-4 * scale

    # Direct sky solve and full fit work with the terrain present.
    prng = np.random.default_rng(1)
    ini = {
        "sky_coeffs": sc * prng.uniform(0.9, 1.1, sc.shape).astype(sc.dtype),
        "beam_coeffs": bc * prng.uniform(0.9, 1.1, bc.shape).astype(bc.dtype),
    }
    loss_init = float(cal._loss(ini))
    sky_direct = cal._sky_step_direct(ini, step_size=1.0)
    assert sky_direct is not None
    loss_direct = float(
        cal._loss(
            {"sky_coeffs": sky_direct, "beam_coeffs": ini["beam_coeffs"]}
        )
    )
    assert loss_direct < 0.1 * loss_init

    result = cal.fit(
        params={k: v.copy() for k, v in ini.items()},
        geom=geom,
        max_iter=8,
        tol=0.0,
        verbose=False,
        solver="fast-cg",
    )
    assert float(cal.data_loss(result["params"])) < 5.0


def test_fit_keyboard_interrupt_returns_truncated():
    """A KeyboardInterrupt mid-fit returns the last completed iteration.

    The fit must not propagate the interrupt; it returns the usual result
    dict truncated to the completed iterations, with converged=False and
    interrupted=True, and params improved over the starting point.
    """
    prob = build_recovery_problem()
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._resolve_geom(geom=prob["geom"])
    params = {k: v.copy() for k, v in prob["prms_ini"].items()}
    loss_init = float(cal._loss(params))

    # Interrupt at the start of the second outer iteration's sky step, so
    # exactly one iteration completes before the interrupt.
    orig_sky_step = cal.sky_step
    calls = {"n": 0}

    def interrupting_sky_step(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt
        return orig_sky_step(*args, **kwargs)

    cal.sky_step = interrupting_sky_step

    result = cal.fit(
        params=params,
        geom=prob["geom"],
        max_iter=10,
        tol=0.0,
        verbose=False,
        solver="fast-cg",
    )

    assert result["interrupted"] is True
    assert result["converged"] is False
    assert result["n_iter"] == 1
    assert len(result["losses"]) == 1
    # The returned params are the completed-iteration state and improve on
    # the starting point; the input dict is left unmutated.
    assert np.isfinite(float(cal._loss(result["params"])))
    assert float(cal._loss(result["params"])) < loss_init
    assert float(cal._loss(params)) == loss_init


def test_fit_reports_not_interrupted_on_clean_run():
    """A normal fit reports interrupted=False."""
    prob = build_recovery_problem()
    cal = Calibrator(
        prob["fwd"],
        prob["data"],
        inv_noise_var=prob["inv_noise_var"],
        lam_beam=0.01,
        lam_beam_harmonic=0.0,
    )
    cal._resolve_geom(geom=prob["geom"])
    result = cal.fit(
        params={k: v.copy() for k, v in prob["prms_ini"].items()},
        geom=prob["geom"],
        max_iter=2,
        tol=0.0,
        verbose=False,
        solver="fast-cg",
    )
    assert result["interrupted"] is False
    assert result["n_iter"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
