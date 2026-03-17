import os
import numpy as np
from scipy.interpolate import interp1d

MODELS_FILE = os.path.join(os.path.dirname(__file__), "data", "models_21cm.npz")

class T21cmModel:
    """
    Interpolate 21cm signal models onto requested frequencies.

    Assumes the NPZ contains:
      - freqs: model frequencies in GHz
      - models: model temperatures in mK

    Internally stores:
      - freqs in Hz
      - models in K
    """

    def __init__(self, filename=MODELS_FILE, dtype="float32", kind="cubic"):
        npz = np.load(filename)
        self._freqs = (npz["freqs"] * 1e9).astype(dtype)   # GHz -> Hz
        self._mdls = (npz["models"] * 1e-3).astype(dtype)  # mK -> K
        self._kind = kind

    def __len__(self):
        return len(self._mdls)

    def interp(self, model_index=None):
        """
        Return an interp1d object for one model or all models.

        Parameters
        ----------
        model_index : int or None
            If int, interpolate just that model.
            If None, interpolate all models along axis 0.

        Returns
        -------
        scipy.interpolate.interp1d
        """
        if model_index is None:
            y = self._mdls
            axis = -1
        else:
            y = self._mdls[model_index]
            axis = -1

        return interp1d(
            self._freqs,
            y,
            kind=self._kind,
            axis=axis,
            fill_value=0,
            bounds_error=False,
        )

    def __getitem__(self, i):
        return self.interp(i)

    def __call__(self, freqs, model_index=None):
        """
        Evaluate one model or all models at `freqs`.

        Parameters
        ----------
        freqs : array-like
            Frequencies in Hz.
        model_index : int or None
            If int, return shape matching `freqs`.
            If None, return shape (n_models, len(freqs)).

        Returns
        -------
        ndarray
            Interpolated model temperature(s) in K.
        """
        return self.interp(model_index=model_index)(freqs)
