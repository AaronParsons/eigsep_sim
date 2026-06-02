"""Shared beam sampling and linear recovery matrix construction."""

from __future__ import annotations

from collections.abc import Mapping

import healjax
import numpy as np


def sample_beam_weights(fwd, geom, freq_index, beam_coeffs=None):
    """Sample normalized body-frame beams onto Galactic sky pixels.

    Parameters
    ----------
    fwd : ForwardModel
        Forward model providing sky and beam descriptors.
    geom : dict
        Geometry returned by ``ForwardModel.precompute_geometry``.
    freq_index : int
        Frequency channel to sample.
    beam_coeffs : ndarray, optional
        Beam coefficients. Defaults to ``fwd.beam.coeffs``.

    Returns
    -------
    ndarray
        Normalized weights with shape ``(ntime, ndipole, npix)``.
    """
    if beam_coeffs is None:
        beam_coeffs = fwd.beam.coeffs
    sky_dirs = np.asarray(geom["crds_gal_jax"])
    rots = np.asarray(geom["rots_jax"])
    body_rots = np.asarray(geom["body_rots_jax"])
    body_dirs = np.einsum("tij,tjk,kn->tin", body_rots, rots, sky_dirs)
    beam_maps = fwd.beam.basis.deproject(beam_coeffs)[:, :, freq_index]
    weights = np.empty(
        (len(rots), beam_maps.shape[0], sky_dirs.shape[1]), dtype=float
    )
    scale = fwd.sky.npix / fwd.beam.npix
    for time_index, directions in enumerate(body_dirs):
        theta, phi = healjax.vec2ang(
            directions[0], directions[1], directions[2]
        )
        pixels, interp_weights = healjax.get_interp_weights(
            theta, phi, fwd.beam.nside
        )
        for dipole_index, beam_map in enumerate(beam_maps):
            sampled = sum(
                beam_map[np.asarray(pixels[k])] * np.asarray(interp_weights[k])
                for k in range(4)
            )
            weights[time_index, dipole_index] = sampled / (
                beam_map.sum() * scale
            )
    return weights


def build_surface_design_matrix(
    weights,
    masks,
    unresolved_surface_weight=None,
    source_columns=None,
    include_receiver_offsets=False,
):
    """Build a generic sky, blocked-surface, and optional-source matrix.

    Column ordering is ``[sky pixels | surface | sources | receiver offsets]``.
    Source columns preserve the insertion order of ``source_columns``.

    Parameters
    ----------
    weights : ndarray, shape (nobs, ndipole, npix)
        Normalized beam weights for sky pixels.
    masks : ndarray, shape (nobs, npix)
        Visibility factors, where one means visible sky.
    unresolved_surface_weight : ndarray, shape (nobs, ndipole), optional
        Additional beam weight assigned to unresolved blocked-surface emission.
    source_columns : mapping or sequence of ndarray, optional
        Extra columns with shape ``(nobs, ndipole)``.
    include_receiver_offsets : bool
        Append one offset column per dipole.
    """
    weights = np.asarray(weights, dtype=float)
    masks = np.asarray(masks, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (nobs, ndipole, npix)")
    nobs, ndipole, npix = weights.shape
    if masks.shape != (nobs, npix):
        raise ValueError(
            f"masks must have shape {(nobs, npix)}, got {masks.shape}"
        )
    if unresolved_surface_weight is None:
        unresolved_surface_weight = np.zeros((nobs, ndipole), dtype=float)
    unresolved_surface_weight = np.asarray(
        unresolved_surface_weight, dtype=float
    )
    if unresolved_surface_weight.shape != (nobs, ndipole):
        raise ValueError(
            "unresolved_surface_weight must have shape "
            f"{(nobs, ndipole)}, got {unresolved_surface_weight.shape}"
        )
    if source_columns is None:
        source_columns = []
    elif isinstance(source_columns, Mapping):
        source_columns = list(source_columns.values())
    else:
        source_columns = list(source_columns)
    source_columns = [
        np.asarray(column, dtype=float) for column in source_columns
    ]
    for column in source_columns:
        if column.shape != (nobs, ndipole):
            raise ValueError(
                f"source columns must have shape {(nobs, ndipole)}, "
                f"got {column.shape}"
            )

    ncols = npix + 1 + len(source_columns)
    if include_receiver_offsets:
        ncols += ndipole
    matrix = np.zeros((nobs, ndipole, ncols), dtype=float)
    matrix[:, :, :npix] = weights * masks[:, None, :]
    matrix[:, :, npix] = (
        np.sum(weights * (1.0 - masks[:, None, :]), axis=2)
        + unresolved_surface_weight
    )
    for source_index, column in enumerate(source_columns):
        matrix[:, :, npix + 1 + source_index] = column
    if include_receiver_offsets:
        offset_start = npix + 1 + len(source_columns)
        for dipole_index in range(ndipole):
            matrix[:, dipole_index, offset_start + dipole_index] = 1.0
    return matrix.reshape(nobs * ndipole, ncols)
