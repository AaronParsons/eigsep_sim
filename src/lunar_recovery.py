"""Linear recovery adapter using lunar campaign geometry."""

from __future__ import annotations

import numpy as np

from .recovery import build_surface_design_matrix


class LunarRecoveryAdapter:
    """Build linear recovery matrices from LunarCampaign geometry."""

    def __init__(self, campaign, result):
        self.campaign = campaign
        self.result = result
        self.source_registry = {
            "earth": {"enabled": False},
            "sun": {"enabled": False},
        }

    def beam_weights(self, spacecraft_index, freq_index):
        """Return normalized Galactic-pixel weights.

        The shape is ``(time, dipole, pixel)``.
        """
        fwd = self.campaign.forward_models[spacecraft_index]
        return fwd.sample_beam_weights(
            self.result.geometry[spacecraft_index], freq_index
        )

    def build_design_matrix(self, freq_index, include_receiver_offsets=False):
        """Build sky, lunar-disk, and optional receiver-offset columns."""
        blocks = []
        for spacecraft_index in range(len(self.campaign.observers)):
            weights = self.beam_weights(spacecraft_index, freq_index)
            masks = self.result.masks[spacecraft_index]
            blocks.append(
                build_surface_design_matrix(
                    weights,
                    masks,
                    unresolved_surface_weight=1.0 - np.sum(weights, axis=2),
                    include_receiver_offsets=include_receiver_offsets,
                )
            )
        return np.vstack(blocks)

    def predict(self, sky_map_K, T_regolith_K, freq_index):
        """Predict flattened spectra from a sky map and uniform lunar disk."""
        parameters = np.concatenate(
            [np.asarray(sky_map_K, dtype=float), [float(T_regolith_K)]]
        )
        return self.build_design_matrix(freq_index) @ parameters

    def truth_vector(self, freq_index):
        """Return stacked ForwardModel truth in design-matrix row order."""
        return self.result.spectra_K[:, :, :, freq_index].reshape(-1)

    def verify_noiseless(self, sky_map_K, T_regolith_K, freq_index, **kwargs):
        """Assert that the adapter reproduces JAX ForwardModel truth."""
        np.testing.assert_allclose(
            self.predict(sky_map_K, T_regolith_K, freq_index),
            self.truth_vector(freq_index),
            **kwargs,
        )
