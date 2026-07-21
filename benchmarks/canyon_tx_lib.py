"""TX-anchored joint sky+beam recovery for the EIGSEP canyon — prototype lib.

A ground transmitter directly below the antenna (nadir) emits in every Nth
channel with isolation in adjacent channels. Combined with terrain masking and
many antenna orientations, it anchors the beam (absolute scale + meridian
profile) and breaks the sky x beam degeneracy.

This module builds the forward linear operators by hand (single dipole) and
provides exact direct (Cholesky) conditional solves for sky and beam, with the
TX term folded into the beam operator. It runs against live ``src`` via a
PYTHONPATH shim; it does not modify or reinstall the package.
"""

from functools import partial

import numpy as np
import healpy
import jax
import jax.numpy as jnp
import astropy.units as u
from astropy.time import Time

from eigsep_sim import Sky, Beam, ForwardModel, HorizonTerrain
from eigsep_sim.observer import EarthSurface

T_GND = 300.0


def synthetic_horizon(nside, alt_block_deg=10.0, seed=0):
    """Topocentric horizon: block pixels below alt_block plus a rough skyline."""
    vec = np.array(healpy.pix2vec(nside, np.arange(healpy.nside2npix(nside))))
    alt = np.degrees(np.arcsin(np.clip(vec[2], -1, 1)))
    rng = np.random.default_rng(seed)
    # Wandering skyline: per-azimuth blocking altitude.
    az = np.arctan2(vec[1], vec[0])
    skyline = alt_block_deg + 15.0 * (
        0.5 * np.sin(2 * az) + 0.5 * np.sin(3 * az + 1.0)
    )
    blocked = alt < skyline
    horizon = np.where(blocked, 100.0, np.nan).astype(np.float32)
    return horizon


def build_canyon(
    nside_sky=8,
    nside_beam=8,
    n_sky_modes=3,
    k_beam=5,
    n_freq=24,
    n_times=8,
    hours=8.0,
    n_az=12,
    n_alt=6,
    alt_max_deg=80.0,
    tx_every=8,
    tx_power_K=1e3,
    terrain="synthetic",
    scan="tumble",
    seed=0,
):
    """Return a dict describing one canyon configuration and its geometry.

    The campaign nests a fast antenna-orientation scan (az x alt body
    rotations) within slow sidereal time sampling (n_times over `hours`).
    Sidereal rotation moves the galactic sky through the fixed topocentric
    terrain horizon, which is what makes the sky solvable at resolution.
    """
    freqs = np.linspace(50e6, 200e6, n_freq)
    obs = EarthSurface(lat=39.2, lon=-113.4, height=1600.0)
    t0 = Time("2025-06-21T04:00:00")
    times = t0 + np.linspace(0, hours * 3600.0, n_times) * u.s
    R_times = obs.rot_gal2top_stack(times).astype(np.float32)  # (n_times,3,3)

    if terrain == "synthetic":
        terr = HorizonTerrain(
            nside_sky, synthetic_horizon(nside_sky, seed=seed), T_terrain=T_GND
        )
    elif terrain == "packaged":
        terr = HorizonTerrain.from_packaged_model(height=100.0, T_terrain=T_GND)
    elif terrain is None:
        terr = None
    else:
        raise ValueError(terrain)

    beam = Beam.from_dipole(
        nside_beam, freqs, arm_lengths_m=2.0,
        u_body=np.array([[1.0, 0.0, 0.0]], dtype=np.float32), K=k_beam,
    )
    sky = Sky.from_gsm(nside_sky, freqs, n_modes=n_sky_modes)

    # TX channels: every tx_every-th channel.
    tx_idx = np.arange(0, n_freq, tx_every)
    tx_freqs = freqs[tx_idx]
    tx_pow = tx_power_K * np.ones_like(tx_freqs)
    tx_dir_top = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    fwd = ForwardModel(
        obs, beam, sky, terrain=terr,
        transmitters=[(tx_dir_top, tx_freqs, tx_pow)],
    )

    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    alt = np.linspace(0, np.radians(alt_max_deg), n_alt)
    if scan == "tumble":
        # Azimuth applied as the OUTER rotation to the tilted body (a hanging
        # antenna spinning about vertical and swinging about a fixed
        # horizontal axis): the nadir TX sweeps a 2-D region of the beam.
        orient = [Beam.rot_z(-a) @ Beam.rot_x(-h) for a in az for h in alt]
    elif scan == "altaz":
        # Alt-az with azimuth about the vertical axis: nadir stays on that
        # axis, so the TX traces only a 1-D meridian (set by tilt).
        orient = [Beam.top2body(a, h) for a in az for h in alt]
    else:
        raise ValueError(scan)
    # Nest the orientation scan within sidereal time sampling.
    rots, body = [], []
    for ti in range(n_times):
        for o in orient:
            rots.append(R_times[ti])
            body.append(o)
    sky_mask = fwd.build_sky_mask(rots=rots)
    geom = fwd.precompute_geometry(rots=rots, body_rots=body, sky_mask=sky_mask)

    tx_mask = np.zeros(n_freq, dtype=bool)
    tx_mask[tx_idx] = True

    return {
        "freqs": freqs, "fwd": fwd, "beam": beam, "sky": sky, "terr": terr,
        "geom": geom, "sky_mask": sky_mask, "tx_mask": tx_mask,
        "sky_coeffs": sky.init_coeffs(), "beam_coeffs": beam.coeffs.copy(),
        "n_orient": len(body), "n_freq": n_freq,
        "A_sky": np.asarray(sky.basis.A), "A_beam": np.asarray(beam.basis.A),
        "tx_T": np.asarray(fwd._tx_T_internal[0]),  # (F,)  internal-scaled
    }


# ---------------------------------------------------------------------------
# Hand-built forward operators (single dipole), JAX.
# ---------------------------------------------------------------------------

def _geom_arrays(cfg):
    g = cfg["geom"]
    return dict(
        bpx=jnp.asarray(g["beam_px_jax"]),               # (T,4,P)
        bwg=jnp.asarray(g["beam_wgts_jax"]),             # (T,4,P)
        mask=jnp.asarray(g["terrain_masks_jax"]),        # (T,P)
        emit=jnp.asarray(g["terrain_emissions_jax"]),    # (T,P,F)
        dmask=jnp.asarray(g["default_emission_masks_jax"]),  # (T,P)
        ubw=jnp.asarray(g["unresolved_beam_weights_jax"]),   # (T,Q)
        uem=jnp.asarray(g["unresolved_emission_jax"]),       # (F,)
        udef=jnp.asarray(g["unresolved_default_emission_jax"]),  # (F,)
        tpx=jnp.asarray(g["tx_px_jax"]),                 # (T,4,nsrc)
        twg=jnp.asarray(g["tx_wgts_jax"]),               # (T,4,nsrc)
    )


def make_ops(cfg, t_gnd=T_GND, tx_on=True):
    ga = _geom_arrays(cfg)
    A_sky = jnp.asarray(cfg["A_sky"])     # (F,Ms)
    A_beam = jnp.asarray(cfg["A_beam"])   # (F,Kb)
    txT = jnp.asarray(cfg["tx_T"]) * (1.0 if tx_on else 0.0)   # (F,)
    Q = int(cfg["beam"].npix)
    nfreq = int(cfg["n_freq"])
    sky_idx = cfg["geom"].get("sky_indices_jax")
    sky_idx = None if sky_idx is None else jnp.asarray(sky_idx)
    g = ga

    @jax.jit
    def beam_at_sky(beam_recon):  # beam_recon (Q,F) -> (T,P,F)
        return jax.lax.fori_loop(
            0, 4,
            lambda k, acc: acc + beam_recon[g["bpx"][:, k, :]] * g["bwg"][:, k, :][..., None],
            jnp.zeros((g["bpx"].shape[0], g["bpx"].shape[2], nfreq), beam_recon.dtype),
        )

    @jax.jit
    def predict(sky_recon_vis, beam_coeffs):
        if beam_coeffs.ndim == 3:
            beam_coeffs = beam_coeffs[0]
        beam_recon = beam_coeffs @ A_beam.T   # (Q,F)
        bas = beam_at_sky(beam_recon)         # (T,P,F)
        T_sky = jnp.einsum("tpf,pf->tf", bas * g["mask"][..., None], sky_recon_vis)
        T_terr = jnp.einsum("tpf,tpf->tf", bas, g["emit"])
        T_gnd = t_gnd * jnp.einsum("tpf,tp->tf", bas, g["dmask"])
        T_unres = (g["ubw"] @ beam_recon) * (g["uem"] + t_gnd * g["udef"])[None, :]
        # TX (sum over sources, here one)
        bat = jax.lax.fori_loop(
            0, 4,
            lambda k, acc: acc + beam_recon[g["tpx"][:, k, :]] * g["twg"][:, k, :][..., None],
            jnp.zeros((g["tpx"].shape[0], g["tpx"].shape[2], nfreq), beam_recon.dtype),
        )  # (T,nsrc,F)
        T_tx = jnp.einsum("tsf,f->tf", bat, txT)
        return T_sky + T_terr + T_gnd + T_unres + T_tx

    def simulate(sky_coeffs, beam_coeffs):
        sc = jnp.asarray(sky_coeffs)
        sc_vis = sc if sky_idx is None else sc[sky_idx]
        sky_recon_vis = sc_vis @ A_sky.T
        return predict(sky_recon_vis, jnp.asarray(beam_coeffs))

    # --- beam operator W (F,T,Q): T[t,f] = sum_q W[t,f,q] beam_recon[q,f] ---
    @jax.jit
    def build_W(sky_recon_vis):  # (P,F) -> (F,T,Q)
        T = g["bpx"].shape[0]
        s_eff = (
            sky_recon_vis[None] * g["mask"][..., None]
            + g["emit"]
            + t_gnd * g["dmask"][..., None]
        )  # (T,P,F)
        flat = (jnp.arange(T)[:, None, None] * Q + g["bpx"]).reshape(-1)
        vals = (g["bwg"][..., None] * s_eff[:, None, :, :]).reshape(-1, nfreq)
        w = jnp.zeros((T * Q, nfreq), s_eff.dtype).at[flat].add(vals).reshape(T, Q, nfreq)
        w = w + g["ubw"][..., None] * (g["uem"] + t_gnd * g["udef"])[None, None, :]
        # TX scatter
        tx_flat = (jnp.arange(T)[:, None] * Q + g["tpx"][:, :, 0]).reshape(-1)
        tx_vals = (g["twg"][:, :, 0][..., None] * txT[None, None, :]).reshape(-1, nfreq)
        w = w.reshape(T * Q, nfreq).at[tx_flat].add(tx_vals).reshape(T, Q, nfreq)
        return jnp.transpose(w, (2, 0, 1))  # (F,T,Q)

    return dict(predict=predict, simulate=simulate, build_W=build_W,
                beam_at_sky=beam_at_sky, A_sky=A_sky, A_beam=A_beam,
                txT=txT, Q=Q, nfreq=nfreq, sky_idx=sky_idx, ga=ga, t_gnd=t_gnd)


# ---------------------------------------------------------------------------
# Exact conditional solves (direct Cholesky, per-frequency Gram structure).
# ---------------------------------------------------------------------------

def _chol_solve(H, b, ridge_rel=1e-8):
    import scipy.linalg as sla
    H = np.asarray(H, np.float64)
    H = 0.5 * (H + H.T)
    H[np.diag_indices_from(H)] += ridge_rel * np.trace(H) / H.shape[0]
    c, low = sla.cho_factor(H, check_finite=False)
    return sla.cho_solve((c, low), np.asarray(b, np.float64).ravel(), check_finite=False)


def eig_system(H):
    """Eigendecomposition of a symmetric Gram, returned high-to-low."""
    H = np.asarray(H, np.float64)
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    return w[::-1], V[:, ::-1]


def solve_reg(H, b, lam, w=None, V=None, prior=None):
    """Tikhonov / MAP solve  (H + lam I) x = b + lam*prior  via eigen-basis.

    lam is absolute (same units as Gram eigenvalues). If w,V (eig of H) are
    given they are reused (cheap lambda sweeps). prior is the flat prior mean.
    """
    b = np.asarray(b, np.float64).ravel()
    if w is None:
        w, V = eig_system(H)
    rhs = b if prior is None else b + lam * np.asarray(prior, np.float64).ravel()
    coeff = (V.T @ rhs) / (w + lam)
    return V @ coeff


def n_modes_above(w, lam):
    """Number of Gram modes with eigenvalue above the regularization floor."""
    return int(np.sum(np.asarray(w) > lam))


def sky_system(ops, cfg, data, inv_var, beam_coeffs):
    """Build the conditional sky normal-equations (H, b) given the beam.

    The sky is linear given the beam: data_eff = data - simulate(sky=0, beam).
    Returns H (P*Ms, P*Ms), b (P*Ms,), and (P, Ms).
    """
    A_sky = ops["A_sky"]
    P = int(cfg["geom"]["beam_px_jax"].shape[2])
    Ms = A_sky.shape[1]
    bc = jnp.asarray(beam_coeffs)
    if bc.ndim == 3:
        bc = bc[0]
    beam_recon = bc @ ops["A_beam"].T
    bas = ops["beam_at_sky"](beam_recon)
    Gmask = bas * ops["ga"]["mask"][..., None]
    zero_sky = jnp.zeros((P, ops["nfreq"]), beam_recon.dtype)
    offset = ops["predict"](zero_sky, beam_coeffs)
    data_eff = np.asarray(data, np.float64) - np.asarray(offset, np.float64)
    iv = np.asarray(inv_var, np.float64).T
    A = np.asarray(A_sky, np.float64)
    G = np.asarray(jnp.transpose(Gmask, (2, 0, 1)), np.float64)   # (F,T,P)
    b = (np.einsum("ftp,ft->fp", G, iv * data_eff.T).T @ A).ravel()
    Bf = np.einsum("ftp,ft,ftq->fpq", G, iv, G)
    H = np.einsum("fm,fn,fpq->pmqn", A, A, Bf).reshape(P * Ms, P * Ms)
    return H, b, P, Ms


def beam_system(ops, cfg, data, inv_var, sky_coeffs, beam_nom=None):
    """Build the conditional beam normal-equations (H, b) given the sky.

    The whole prediction is linear in beam given the sky (TX folded into W).
    If beam_nom is given, b is for the residual about beam_nom (so a ridge
    pulls toward the nominal beam, not toward zero).
    Returns H (Q*Kb, Q*Kb), b, prior (Q*Kb,), and (Q, Kb).
    """
    A_beam = ops["A_beam"]
    Kb = A_beam.shape[1]
    Q = ops["Q"]
    sc = jnp.asarray(sky_coeffs)
    sc_vis = sc if ops["sky_idx"] is None else sc[ops["sky_idx"]]
    sky_recon_vis = sc_vis @ ops["A_sky"].T
    W = np.asarray(ops["build_W"](sky_recon_vis), np.float64)   # (F,T,Q)
    iv = np.asarray(inv_var, np.float64).T
    data_f = np.asarray(data, np.float64).T
    A = np.asarray(A_beam, np.float64)
    b = (np.einsum("ftq,ft->fq", W, iv * data_f).T @ A).ravel()
    Bf = np.einsum("ftq,ft,ftr->fqr", W, iv, W)
    H = np.einsum("fk,fl,fqr->qkrl", A, A, Bf).reshape(Q * Kb, Q * Kb)
    prior = None
    if beam_nom is not None:
        bn = np.asarray(beam_nom, np.float64)
        if bn.ndim == 3:
            bn = bn[0]
        prior = bn.reshape(Q * Kb)
    return H, b, prior, Q, Kb


def isolate_tx(cfg, data, deg=2, window=3):
    """Isolate the transmitter by LOCAL differencing against adjacent channels.

    The sky+terrain emission is smooth in frequency; for each TX channel we fit
    a low-order polynomial to the nearby TX-free (science) channels within
    +/-window and evaluate it at the TX channel. Subtracting it leaves the pure
    TX signal (= beam at the nadir body direction x P_tx), sky-independent.
    Local fitting cancels the smooth emission far better than a global fit
    (only the local curvature leaks, which is below a bright TX tone).
    Returns (iso, tx_idx), iso shape (n_orient, n_tx_channels).
    """
    data = np.asarray(data, np.float64)
    tx_mask = np.asarray(cfg["tx_mask"])
    sci = np.where(~tx_mask)[0]
    tx_idx = np.where(tx_mask)[0]
    x = np.log(cfg["freqs"])
    iso = np.zeros((data.shape[0], len(tx_idx)))
    for j, f in enumerate(tx_idx):
        near = sci[np.abs(sci - f) <= window]
        d = min(deg, len(near) - 1)
        V = np.vander(x[near], d + 1)
        coef = np.linalg.lstsq(V, data[:, near].T, rcond=None)[0]  # (d+1, T)
        iso[:, j] = data[:, f] - np.vander([x[f]], d + 1) @ coef
    return iso, tx_idx


def tx_beam_system(ops, cfg, iso, tx_idx, inv_tx):
    """Normal equations for beam_coeffs from isolated TX measurements only.

    iso[t,i] = (P_tx[f_i]) * sum_k tx_wgt[t,k] beam_recon[tx_px[t,k], f_i],
    linear in beam_coeffs and sky-INDEPENDENT. Returns (H, b, Q, Kb).
    """
    ga = ops["ga"]
    A_beam = np.asarray(ops["A_beam"], np.float64)         # (F, Kb)
    Q, Kb = ops["Q"], A_beam.shape[1]
    tpx = np.asarray(ga["tpx"])[:, :, 0]                    # (T,4)
    twg = np.asarray(ga["twg"], np.float64)[:, :, 0]       # (T,4)
    txT = np.asarray(ops["txT"], np.float64)               # (F,) internal-scaled
    T = tpx.shape[0]
    n_tx = len(tx_idx)
    # W_tx[t,i,q]: TX interp weights scattered onto beam pixels (i indexes
    # TX channel; weights are channel-independent here, P_tx folds in below).
    W = np.zeros((T, Q), np.float64)
    for k in range(4):
        np.add.at(W, (np.arange(T)[:, None], tpx[:, k][:, None]), twg[:, k][:, None])
    # per-TX-channel pixel Gram weighted by inv and (P_tx)^2
    a = A_beam[tx_idx]                                      # (n_tx, Kb)
    p = txT[tx_idx]                                         # (n_tx,)
    iv = np.asarray(inv_tx, np.float64)                    # (T, n_tx)
    H = np.zeros((Q * Kb, Q * Kb))
    b = np.zeros((Q, Kb))
    for i in range(n_tx):
        Bi = (W * iv[:, i][:, None]).T @ W                 # (Q,Q)
        H += np.kron(Bi, (p[i] ** 2) * np.outer(a[i], a[i]))
        rhs_q = (W * (iv[:, i] * iso[:, i])[:, None]).sum(0)  # (Q,)
        b += p[i] * np.outer(rhs_q, a[i])
    return H, b.ravel(), Q, Kb


def beam_from_tx(ops, cfg, data, lam=1e-6, deg=3, sigma_iso=None):
    """Recover beam_coeffs from the transmitter alone (sky-independent).

    Differences out the smooth emission, then solves the linear TX system.
    Pixels not seen by the TX trajectory are left near zero (ridge).
    """
    iso, tx_idx = isolate_tx(cfg, data, deg=deg)
    if sigma_iso is None:
        sigma_iso = np.std(iso) * 1e-3 + 1e-12
    inv_tx = np.full_like(iso, 1.0 / sigma_iso**2)
    H, b, Q, Kb = tx_beam_system(ops, cfg, iso, tx_idx, inv_tx)
    w, V = eig_system(H)
    x = solve_reg(H, b, lam * w[0], w=w, V=V)
    return np.asarray(x.reshape(Q, Kb), np.float32), iso, tx_idx


def _scatter_full_sky(cfg, ops, sky_vis, Ms):
    full = np.zeros((cfg["sky"].npix, Ms), np.float32)
    if ops["sky_idx"] is not None:
        full[np.asarray(ops["sky_idx"])] = sky_vis
        return full
    return np.asarray(sky_vis, np.float32)


def sky_solve(ops, cfg, data, inv_var, beam_coeffs, lam=0.0):
    """Regularized conditional sky solve. lam is RELATIVE to the top Gram
    eigenvalue (config-independent); lam=0 does an unregularized Cholesky."""
    H, b, P, Ms = sky_system(ops, cfg, data, inv_var, beam_coeffs)
    if lam == 0.0:
        x = _chol_solve(H, b)
    else:
        w, V = eig_system(H)
        x = solve_reg(H, b, lam * w[0], w=w, V=V)
    return _scatter_full_sky(cfg, ops, x.reshape(P, Ms), Ms)


def beam_solve(ops, cfg, data, inv_var, sky_coeffs, lam=0.0, beam_nom=None):
    """Regularized conditional beam solve (TX folded into W). lam is RELATIVE
    to the top Gram eigenvalue; the ridge pulls toward beam_nom."""
    H, b, prior, Q, Kb = beam_system(ops, cfg, data, inv_var, sky_coeffs,
                                     beam_nom=beam_nom)
    if lam == 0.0:
        x = _chol_solve(H, b)
    else:
        w, V = eig_system(H)
        x = solve_reg(H, b, lam * w[0], w=w, V=V, prior=prior)
    return np.asarray(x.reshape(Q, Kb), np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def chi2(ops, data, inv_var, sky_coeffs, beam_coeffs):
    pred = np.asarray(ops["simulate"](sky_coeffs, beam_coeffs))
    return float(np.mean(np.asarray(inv_var) * (pred - np.asarray(data)) ** 2))


def map_err_sky(cfg, sky_fit, vis_only=True):
    A = cfg["A_sky"].T
    tru = cfg["sky_coeffs"] @ A
    fit = np.asarray(sky_fit) @ A
    m = np.asarray(cfg["sky_mask"]) if vis_only else np.ones(tru.shape[0], bool)
    return float(np.linalg.norm((fit - tru)[m]) / np.linalg.norm(tru[m]))


def map_err_beam(cfg, beam_fit, gauge=True):
    A = cfg["A_beam"].T
    tru = cfg["beam_coeffs"][0] @ A           # (Q,F)
    fit = np.asarray(beam_fit) @ A            # (Q,F)
    if gauge:
        s = np.sum(fit * tru, axis=0) / np.maximum(np.sum(fit**2, axis=0), 1e-30)
        fit = fit * s[None, :]
    return float(np.linalg.norm(fit - tru) / np.linalg.norm(tru))
