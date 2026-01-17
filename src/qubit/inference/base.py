from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from ..enums.start_mode import StartMode
import numpy as np

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
   

# def decode_autoregressive(
#     adapter: AutoregressiveAdapter,
#     X: np.ndarray,
#     *,
#     out_steps: int,
#     start_mode: StartMode,
#     batch_size: int,
# ) -> np.ndarray:
 
    
#     N = X.shape[0]
#     D = adapter.feature_dim

#     # provide the initial state from encoder 
#     # state is an object because different models have different state structure
#     state = adapter.encode(X, batch_size=batch_size)

#     # initialize the first decoder input because at the t=0 we have no previous prediction
#     # dec_t shape = (n_traj, 1, feature_dim)
#     dec_t = adapter.init_decoder_input(X, start_mode=start_mode).astype(dtype=np.float32, copy=False)
    

#     out = np.empty((N, out_steps, D), dtype=np.float32)

#     for t in range(out_steps):

#         # return the prediction at time t and the new state
#         y_t, state = adapter.step(dec_t, state, batch_size=batch_size)
#         y_t = np.asarray(y_t)

#         if y_t.shape != (N, 1, D):
#             raise ValueError(f"step() must return y_t shape (N,1,D)={(N,1,D)}, got {y_t.shape}")
        
#         out[:, t:t+1, :] = y_t

#         # the next decoder input is the current prediction
#         dec_t = y_t

#     return out


from typing import Optional
import numpy as np
from ..enums.inference_mode import InferenceMode

def decode(
    adapter,
    X: np.ndarray,
    *,
    out_steps: int,
    start_mode,
    batch_size: int,
    mode : InferenceMode,                       # InferenceMode.FREE_RUNNING / TEACHER_FORCING
    y_true: Optional[np.ndarray] = None,   # richiesto se TEACHER_FORCING
) -> np.ndarray:
    N = X.shape[0]
    D = adapter.feature_dim

    state = adapter.encode(X, batch_size=batch_size)

    dec_t = adapter.init_decoder_input(X, start_mode=start_mode).astype(np.float32, copy=False)  # (N,1,D)

    out = np.empty((N, out_steps, D), dtype=np.float32)

    if mode == InferenceMode.TEACHER_FORCING :
        if y_true is None:
            raise ValueError("y_true deve essere fornito in TEACHER_FORCING.")
        if y_true.shape != (N, out_steps, D):
            raise ValueError(f"y_true atteso shape (N,out_steps,D)={(N,out_steps,D)}, got {y_true.shape}")

        y_tf = y_true.astype(np.float32, copy=False)

        for t in range(out_steps):
            y_t_pred, state = adapter.step(dec_t, state, batch_size=batch_size)
            y_t_pred = np.asarray(y_t_pred)

            if y_t_pred.shape != (N, 1, D):
                raise ValueError(f"step() must return (N,1,D)={(N,1,D)}, got {y_t_pred.shape}")

            out[:, t:t+1, :] = y_t_pred

            # TEACHER FORCING: il prossimo input è la GT del tempo t
            dec_t = y_tf[:, t:t+1, :]

        return out

    # AUTOREGRESSIVO (standard)
    for t in range(out_steps):
        y_t_pred, state = adapter.step(dec_t, state, batch_size=batch_size)
        y_t_pred = np.asarray(y_t_pred)

        if y_t_pred.shape != (N, 1, D):
            raise ValueError(f"step() must return (N,1,D)={(N,1,D)}, got {y_t_pred.shape}")

        out[:, t:t+1, :] = y_t_pred

        # autoregressione: il prossimo input è la predizione corrente
        dec_t = y_t_pred

    return out
