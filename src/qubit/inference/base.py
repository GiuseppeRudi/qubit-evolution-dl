from __future__ import annotations
from typing import Protocol, Any, Optional
import tensorflow as tf

from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode
from ..enums.start_mode import StartMode

class AutoregressiveAdapter(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def encode(self, X: tf.Tensor, *, batch_size: int) -> Any:
        """X: (batch_size, input_seq_len, feature_dim) -> state"""

    def init_decoder_input(self, X: tf.Tensor, *, start_mode: StartMode) -> tf.Tensor:
        """Return dec0: (batch_size, 1, feature_dim)"""

    def step(
        self,
        dec_t: tf.Tensor,   # (batch_size, 1, feature_dim)
        state: Any,
        *,
        batch_size: int,
    ) -> tuple[tf.Tensor, Any]:
        """Return (y_t_pred: (batch_size,1,feature_dim), new_state)"""


# def decode(
#     adapter,
#     X: np.ndarray,
#     *,
#     out_steps: int,
#     start_mode,
#     batch_size: int,
#     mode : InferenceMode,                       # InferenceMode.FREE_RUNNING / TEACHER_FORCING
#     y_true: Optional[np.ndarray] = None,   # richiesto se TEACHER_FORCING
# ) -> np.ndarray:
#     N = X.shape[0]
#     D = adapter.feature_dim

#     state = adapter.encode(X, batch_size=batch_size)

#     dec_t = adapter.init_decoder_input(X, start_mode=start_mode).astype(np.float32, copy=False)  # (N,1,D)

#     out = np.empty((N, out_steps, D), dtype=np.float32)

#     if mode == InferenceMode.TEACHER_FORCING :
#         if y_true is None:
#             raise ValueError("y_true deve essere fornito in TEACHER_FORCING.")
#         if y_true.shape != (N, out_steps, D):
#             raise ValueError(f"y_true atteso shape (N,out_steps,D)={(N,out_steps,D)}, got {y_true.shape}")

#         y_tf = y_true.astype(np.float32, copy=False)

#         for t in range(out_steps):
#             y_t_pred, state = adapter.step(dec_t, state, batch_size=batch_size)
#             y_t_pred = np.asarray(y_t_pred)

#             if y_t_pred.shape != (N, 1, D):
#                 raise ValueError(f"step() must return (N,1,D)={(N,1,D)}, got {y_t_pred.shape}")

#             out[:, t:t+1, :] = y_t_pred

#             # TEACHER FORCING: il prossimo input è la GT del tempo t
#             dec_t = y_tf[:, t:t+1, :]

#         return out

#     # AUTOREGRESSIVO (standard)
#     for t in range(out_steps):
#         y_t_pred, state = adapter.step(dec_t, state, batch_size=batch_size)
#         y_t_pred = np.asarray(y_t_pred)

#         if y_t_pred.shape != (N, 1, D):
#             raise ValueError(f"step() must return (N,1,D)={(N,1,D)}, got {y_t_pred.shape}")

#         out[:, t:t+1, :] = y_t_pred

#         # autoregressione: il prossimo input è la predizione corrente
#         dec_t = y_t_pred

#     return out
