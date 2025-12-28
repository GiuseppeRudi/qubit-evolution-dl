from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal, Sequence
import numpy as np


InferenceMode = Literal["teacher_forcing", "free_running"]
StartMode = Literal["zeros", "last_x"]
# TODO remember to update the start mode with add the last x and this is the reason why the plots start wrong 

def as_index_list(v) -> list[int]:
    """Accept int or list/tuple/ndarray and return a list[int]."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [int(i) for i in v]
    return [int(v)]


def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure array is (N,T,D). If (T,D), add batch dim."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape={x.shape}")


class AutoregressiveAdapter(Protocol):
    """
    Adapter minimalista per decoding autoregressivo.
    Implementazioni diverse (LSTM/GRU/Transformer) cambiano SOLO encode/step/state.
    """

    @property
    def feature_dim(self) -> int: ...

    def encode(self, X: np.ndarray, *, batch_size: int) -> object:
        """Encode X -> state (opaque object)."""

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray: ...
        # """Return initial decoder input of shape (N,1,D). Must NOT use labels."""

    def step(
        self,
        dec_t: np.ndarray,  # (N,1,D)
        state: object,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, object]: ...
        # """
        # One autoregressive step:
        # (dec_t, state) -> (y_t, new_state)
        # where y_t has shape (N,1,D).
        # """


def decode_autoregressive(
    adapter: AutoregressiveAdapter,
    X: np.ndarray,
    *,
    out_steps: int,
    start_mode: StartMode = "zeros",
    batch_size: int = 64,
    dtype=np.float32,
) -> np.ndarray:
    """
    Generic free-running decoding.
    Returns pred of shape (N, out_steps, D).
    """
    X = ensure_3d(X).astype(dtype, copy=False)
    N = X.shape[0]
    D = adapter.feature_dim

    state = adapter.encode(X, batch_size=batch_size)
    dec_t = adapter.init_decoder_input(X, start_mode=start_mode).astype(dtype, copy=False)
    
    if dec_t.shape != (N, 1, D):
        raise ValueError(f"init_decoder_input must return shape (N,1,D)={(N,1,D)}, got {dec_t.shape}")

    out = np.empty((N, out_steps, D), dtype=dtype)

    for t in range(out_steps):
        y_t, state = adapter.step(dec_t, state, batch_size=batch_size)
        y_t = np.asarray(y_t)
        if y_t.shape != (N, 1, D):
            raise ValueError(f"step() must return y_t shape (N,1,D)={(N,1,D)}, got {y_t.shape}")
        out[:, t:t+1, :] = y_t
        dec_t = y_t

    return out
