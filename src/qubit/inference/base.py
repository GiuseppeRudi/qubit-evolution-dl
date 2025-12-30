from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal, Sequence
import numpy as np

InferenceMode = Literal["teacher_forcing", "free_running"]
StartMode = Literal["zeros", "last_x"]
# TODO remember to update the start mode with add the last x and this is the reason why the plots start wrong 

class AutoregressiveAdapter(Protocol):
    """Adapter protocol for autoregressive inference."""

    @property
    def feature_dim(self) -> int: ...

    def encode(self, X: np.ndarray, *, batch_size: int) -> object:
        """Encode X -> state (object)."""

    def init_decoder_input(self, X: np.ndarray, *, start_mode: StartMode) -> np.ndarray: ...

    def step(
        self,
        dec_t: np.ndarray,  # (N,1,D)
        state: object,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, object]: ...
   

def decode_autoregressive(
    adapter: AutoregressiveAdapter,
    X: np.ndarray,
    *,
    out_steps: int,
    start_mode: StartMode,
    batch_size: int,
) -> np.ndarray:
 
    
    N = X.shape[0]
    D = adapter.feature_dim

    # provide the initial state from encoder 
    # state is an object because different models have different state structure
    state = adapter.encode(X, batch_size=batch_size)

    # initialize the first decoder input because at the t=0 we have no previous prediction
    # dec_t shape = (n_traj, 1, feature_dim)
    dec_t = adapter.init_decoder_input(X, start_mode=start_mode).astype(dtype=np.float32, copy=False)
    

    out = np.empty((N, out_steps, D), dtype=np.float32)

    for t in range(out_steps):

        # return the prediction at time t and the new state
        y_t, state = adapter.step(dec_t, state, batch_size=batch_size)
        y_t = np.asarray(y_t)

        if y_t.shape != (N, 1, D):
            raise ValueError(f"step() must return y_t shape (N,1,D)={(N,1,D)}, got {y_t.shape}")
        
        out[:, t:t+1, :] = y_t

        # the next decoder input is the current prediction
        dec_t = y_t

    return out
