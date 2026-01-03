import numpy as np
from ..strategies.teacher_forcing import make_decoder_inputs
import tensorflow as tf
import keras
from typing import cast


from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..inference.base import  decode_autoregressive
from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode
from ..model.training_config import TrainingConfig

class FreeRunningEvalCallback(keras.callbacks.Callback):
    """
    Calculate the free-running loss (autoregressive) based on the every_epoch variable,
    and inserting it in the history (es: history.history['val_fr_loss']).
    """

    def __init__(
        self,
        X_eval: np.ndarray,
        Y_eval: np.ndarray,
        *,
        verbose: VerboseMode,
        start_mode: StartMode,         
        training_cfg : TrainingConfig       
    ):
        super().__init__()
        self.X_eval = X_eval
        self.Y_eval = Y_eval
        self.batch_size = training_cfg.batch_size
        self.start_mode = start_mode
        self.every_epochs = training_cfg.fr_eval.every_epochs
        self.p_eval = training_cfg.fr_eval.p_eval
        self.log_prefix = training_cfg.fr_eval.split
        self.verbose = verbose
        self.adapter = None
        self.loss_fn = None

    def on_train_begin(self, logs=None):
        if self.model is None:
            raise RuntimeError("Callback not related to a model (self.model is None).")

        trained_model = cast(keras.Model, self.model)
        self.adapter = Seq2SeqLSTM2LayerAdapter(trained_model, verbose= self.verbose)

        loss = self.model.loss
        self.loss_fn = keras.losses.get(loss) if isinstance(loss, str) else loss
        if self.loss_fn is None:
            raise RuntimeError("Loss function could not be determined")


    def _scalar_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_t = tf.convert_to_tensor(y_true)
        y_pred_t = tf.convert_to_tensor(y_pred)
        if self.loss_fn is not None:
            v = self.loss_fn(y_true_t, y_pred_t)     
        return float(tf.reduce_mean(v).numpy())

    def _slice_eval(self):
        if self.p_eval is None or self.p_eval <= 0 or self.p_eval > 1:
            raise ValueError("P_eval must be in (0,1].")
        n = int(self.X_eval.shape[0] * self.p_eval)
        return self.X_eval[:n], self.Y_eval[:n]

    def on_epoch_end(self, epoch, logs=None):
        
        # epoch è 0-based
        if (epoch + 1) % self.every_epochs != 0:
            return

        logs = logs or {}
        X, Y = self._slice_eval()

        if self.adapter is None:
            raise RuntimeError("Adapter not initialized.")
        
        pred_fr = decode_autoregressive(
            self.adapter,
            X,
            out_steps=Y.shape[1],
            start_mode=cast(StartMode, self.start_mode),
            batch_size=self.batch_size,
        )

        fr_loss = self._scalar_loss(Y, pred_fr)
        logs[f"{self.log_prefix}_fr_loss"] = fr_loss

