import os
import numpy as np
from scipy.interpolate import interp1d

MODELS_FILE = os.path.join(os.path.dirname(__file__), 'models_21cm.npz')


class T21cmModel:
    def __init__(self, filename=MODELS_FILE, dtype='float32'):
        npz = np.load(filename)
        self._freqs = (npz['freqs'] * 1e9).astype(dtype)
        self._mdls = npz['models'].astype(dtype)

    def __len__(self):
        return len(self._mdls)

    def __getitem__(self, i):
        return interp1d(
            self._freqs, self._mdls[i], fill_value=0, bounds_error=False
        )

    def __call__(self, i, freqs):
        return self[i](freqs)
