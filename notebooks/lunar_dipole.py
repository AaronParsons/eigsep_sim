"""lunar_dipole.py — Dipole diagnostic summary tool for BLOOM-21cm.

Antenna physics functions live in eigsep_sim.beam and are re-exported here
for convenience.  This module adds only summarize_lengths(), which prints a
human-readable performance table.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

from eigsep_sim.beam import (
    gsm_like_tsky_K,
    short_dipole_radiation_resistance_ohm,
    realized_efficiency,
    antenna_temperature_K,
    receiver_margin_factor,
)

__all__ = [
    "gsm_like_tsky_K",
    "short_dipole_radiation_resistance_ohm",
    "realized_efficiency",
    "antenna_temperature_K",
    "receiver_margin_factor",
    "summarize_lengths",
]


def summarize_lengths(
    lengths_m=(6.0, 4.0, 3.0, 2.0),
    freq_mhz=None,
    trx_K: float = 100.0,
    r_loss_ohm: float = 5.0,
    z_rx_ohm: float = 50.0,
    x_scale: float = 120.0,
):
    """
    Print a compact performance summary for a set of dipole lengths.

    Reports η, Tant, and 2·Tant/Trx at 30 MHz, plus the lowest frequency
    where the receiver-noise criterion (Trx < 2·Tant, i.e. margin > 1) is met.
    """
    if freq_mhz is None:
        freq_mhz = np.linspace(30.0, 200.0, 400)
    f = np.asarray(freq_mhz, dtype=float)

    print("Assumptions:")
    print(f"  Trx = {trx_K:.1f} K")
    print(f"  Rloss = {r_loss_ohm:.1f} ohm")
    print(f"  Zrx = {z_rx_ohm:.1f} ohm")
    print(f"  reactance scale = {x_scale:.1f}")
    print()

    for L in lengths_m:
        eta    = realized_efficiency(L, f, r_loss_ohm=r_loss_ohm, z_rx_ohm=z_rx_ohm, x_scale=x_scale)
        tant   = antenna_temperature_K(L, f, r_loss_ohm=r_loss_ohm, z_rx_ohm=z_rx_ohm, x_scale=x_scale)
        margin = receiver_margin_factor(L, f, trx_K=trx_K, r_loss_ohm=r_loss_ohm, z_rx_ohm=z_rx_ohm, x_scale=x_scale)

        i30     = np.argmin(np.abs(f - 30.0))
        passing = np.where(margin > 1.0)[0]
        fmin    = f[passing[0]] if len(passing) else np.nan

        print(f"L = {L:.1f} m")
        print(f"  eta(30 MHz)     = {eta[i30]:.4f}")
        print(f"  Tant(30 MHz)    = {tant[i30]:.1f} K")
        print(f"  2*Tant/Trx @30  = {margin[i30]:.2f}   (>1 passes)")
        if np.isfinite(fmin):
            print(f"  criterion passes from {fmin:.1f} MHz upward")
        else:
            print("  criterion never passes in this band")
        print()


if __name__ == "__main__":
    from eigsep_sim.lunar_orbit import OrbiterMission
    _cfg = OrbiterMission(os.path.join(_HERE, "bloom_config.yaml"))
    summarize_lengths(
        lengths_m=_cfg.antenna.arm_lengths,
        freq_mhz=np.linspace(30.0, 200.0, 400),
        trx_K=_cfg.antenna.t_rx,
        r_loss_ohm=_cfg.antenna.r_loss_ohm,
        z_rx_ohm=_cfg.antenna.z_rx_ohm,
        x_scale=_cfg.antenna.x_scale,
    )
