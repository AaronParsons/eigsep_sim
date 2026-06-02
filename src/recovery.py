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


class ScaleDegeneracy:
    """Multiplicative gauge coupled across parameter arrays.

    ``responses`` maps parameter names to their power-law response to one
    scale gauge. For example ``{"sky": 1, "beam": -1}`` applies
    ``sky *= scale`` and ``beam /= scale``. ``group_axes`` retains independent
    gauges, such as one scale per frequency channel. A sequence of keys is
    accepted as shorthand for unit responses.

    With no reference solution, the scale sets the joint geometric-mean gauge
    to unity. With a reference, a log-amplitude least-squares fit replaces the
    degenerate subspace with the reference gauge while preserving products
    between parameters with opposite responses.
    """

    def __init__(self, responses, group_axes=()):
        if hasattr(responses, "items"):
            self.responses = dict(responses)
        else:
            self.responses = {key: 1.0 for key in responses}
        if not self.responses or any(
            response == 0 for response in self.responses.values()
        ):
            raise ValueError("scale responses must be non-zero")
        self.group_axes = tuple(group_axes)

    def project(self, params, reference=None):
        numerator = None
        denominator = None
        for key, response in self.responses.items():
            array = np.asarray(params[key])
            axes = _reduction_axes(array, self.group_axes)
            if reference is None:
                valid = np.abs(array) > 0
                log_ratio = -np.log(
                    np.abs(array),
                    where=valid,
                    out=np.zeros_like(array, dtype=float),
                )
            else:
                target = np.asarray(reference[key])
                if target.shape != array.shape:
                    raise ValueError(
                        f"parameter {key!r} has shape {array.shape}, expected {target.shape}"
                    )
                valid = (array * target) > 0
                ratio = np.divide(
                    np.abs(target),
                    np.abs(array),
                    out=np.ones_like(array, dtype=float),
                    where=valid,
                )
                log_ratio = np.log(ratio)
            term = response * np.sum(
                np.where(valid, log_ratio, 0.0), axis=axes
            )
            weight = response**2 * np.sum(valid, axis=axes)
            numerator = term if numerator is None else numerator + term
            denominator = (
                weight if denominator is None else denominator + weight
            )
        if np.any(denominator <= 0):
            raise ValueError(
                "cannot project scale gauge without non-zero matching values"
            )
        scale = np.exp(numerator / denominator)
        for key, response in self.responses.items():
            array = np.asarray(params[key])
            params[key] = array * _expand_group_value(
                scale**response, array.ndim, self.group_axes
            )


class AdditiveDegeneracy:
    """Additive gauge coupled across parameter arrays.

    ``coefficients`` maps parameter names to their response to one additive
    gauge value. For example ``{"sky": 1, "ground": 1, "receiver": -1}``
    preserves a radiometer model when a common temperature offset moves from
    receiver temperature into sky and ground emission. ``group_axes`` retains
    independent gauges, such as one offset per frequency channel.
    """

    def __init__(self, coefficients, group_axes=()):
        self.coefficients = dict(coefficients)
        self.group_axes = tuple(group_axes)

    def project(self, params, reference=None):
        numerator = None
        denominator = None
        for key, coefficient in self.coefficients.items():
            array = np.asarray(params[key])
            axes = _reduction_axes(array, self.group_axes)
            target = 0.0 if reference is None else np.asarray(reference[key])
            term = coefficient * np.sum(target - array, axis=axes)
            weight = coefficient**2 * np.prod(
                [array.shape[axis] for axis in axes]
            )
            numerator = term if numerator is None else numerator + term
            denominator = (
                weight if denominator is None else denominator + weight
            )
        offset = numerator / denominator
        for key, coefficient in self.coefficients.items():
            array = np.asarray(params[key])
            params[key] = array + coefficient * _expand_group_value(
                offset, array.ndim, self.group_axes
            )


class RecoverySolution:
    """Parameter dictionary with redcal-style degeneracy projection.

    ``remove_degen()`` changes only explicitly registered degenerate subspaces.
    A supplied ``degen_sol`` replaces those gauges with the reference gauges;
    otherwise each degeneracy chooses its canonical zero or unit gauge.
    """

    def __init__(self, params, degeneracies=()):
        self.params = {
            key: np.array(value, copy=True) for key, value in params.items()
        }
        self.degeneracies = tuple(degeneracies)

    def remove_degen(self, degen_sol=None, inplace=True):
        """Remove degeneracies or replace them with gauges from ``degen_sol``."""
        if degen_sol is not None:
            reference = (
                degen_sol.params
                if isinstance(degen_sol, RecoverySolution)
                else degen_sol
            )
        else:
            reference = None
        solution = (
            self
            if inplace
            else RecoverySolution(self.params, self.degeneracies)
        )
        for degeneracy in solution.degeneracies:
            degeneracy.project(solution.params, reference=reference)
        if not inplace:
            return solution


def _normalize_group_axes(ndim, group_axes):
    axes = tuple(axis + ndim if axis < 0 else axis for axis in group_axes)
    if len(set(axes)) != len(axes) or any(
        axis < 0 or axis >= ndim for axis in axes
    ):
        raise ValueError(f"group_axes {group_axes} invalid for {ndim}-D array")
    return tuple(sorted(axes))


def _reduction_axes(array, group_axes):
    group_axes = _normalize_group_axes(array.ndim, group_axes)
    return tuple(axis for axis in range(array.ndim) if axis not in group_axes)


def _expand_group_value(value, ndim, group_axes):
    group_axes = _normalize_group_axes(ndim, group_axes)
    shape = [1] * ndim
    for size, axis in zip(np.shape(value), group_axes):
        shape[axis] = size
    return np.reshape(value, shape)


def relative_rms(estimate, reference):
    """Return RMS error relative to the RMS amplitude of a reference array."""
    estimate = np.asarray(estimate)
    reference = np.asarray(reference)
    return float(
        np.sqrt(np.mean((estimate - reference) ** 2))
        / np.sqrt(np.mean(reference**2))
    )
