import numpy as np
from ..strategies.teacher_forcing import make_decoder_inputs
import tensorflow as tf
import keras
from typing import cast


from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..inference.base import  decode
from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode
from ..enums.inference_mode import InferenceMode
from ..model.training_config import TrainingConfig

from ..rnn.Seq2SeqLSTM2LayerStepWiseModel import Seq2SeqLSTM2LayerStepWiseModel
from ..inference.step_wise_rnn_adapter import StepWiseSeq2SeqAdapter

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
        inference_mode : InferenceMode,   
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
        self.global_epoch = 0
        self.inference_mode = inference_mode

    def on_train_begin(self, logs=None):
        print("FreeRunningEvalCallback: on_train_begin")
        if self.model is None:
            raise RuntimeError("Callback not related to a model (self.model is None).")

        trained_model = cast(keras.Model, self.model)
        
        if isinstance(self.model, Seq2SeqLSTM2LayerStepWiseModel):
            self.adapter = StepWiseSeq2SeqAdapter(trained_model, verbose=self.verbose)
        else: self.adapter = Seq2SeqLSTM2LayerAdapter(trained_model, verbose=self.verbose)
        # self.adapter = Seq2SeqLSTM2LayerAdapter(trained_model, verbose= self.verbose)

        loss = self.model.loss
        self.loss_fn = keras.losses.get(loss) if isinstance(loss, str) else loss
        if self.loss_fn is None:
            raise RuntimeError("Loss function could not be determined")


    def _scalar_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_t = tf.convert_to_tensor(y_true)
        y_pred_t = tf.convert_to_tensor(y_pred)
        if self.loss_fn is not None:
            v = self.loss_fn(y_true_t, y_pred_t)     
        return float(tf.reduce_mean(cast(tf.Tensor,v)).numpy()) 

    def _slice_eval(self):
        if self.p_eval is None or self.p_eval <= 0 or self.p_eval > 1:
            raise ValueError("P_eval must be in (0,1].")
        n = int(self.X_eval.shape[0] * self.p_eval)
        return self.X_eval[:n], self.Y_eval[:n]

    def on_epoch_end(self, epoch, logs=None):
        print("\nFreeRunningEvalCallback: on_epoch_end")
        
        # epoch è 0-based
        self.global_epoch += 1
        if self.global_epoch % self.every_epochs != 0:
            return

        logs = logs or {}
        X, Y = self._slice_eval()

        if self.adapter is None:
            raise RuntimeError("Adapter not initialized.")
        
        pred_fr = decode(
            self.adapter,
            X,
            out_steps=Y.shape[1],
            start_mode=cast(StartMode, self.start_mode),
            batch_size=self.batch_size,
            mode=self.inference_mode,
            y_true= Y
        )

        Xtf = tf.convert_to_tensor(X)
        Ytf = tf.convert_to_tensor(Y)
        Ptf = tf.convert_to_tensor(pred_fr)

        model = cast(keras.Model,self.model)

        fr_loss = model.compute_loss(x=Xtf, y=Ytf, y_pred=Ptf, training=False)
        
        logs[f"{self.log_prefix.value.lower()}_fr_loss"] = fr_loss

        print("trained sum:", self.sum_trainable_tf(model))

        adapter_model = None
        if (hasattr(self.adapter, "model") ) and isinstance(self.adapter,StepWiseSeq2SeqAdapter ):
            adapter_model = cast(keras.Model, self.adapter.model)

        if adapter_model is None:
            print("adapter has no .model; can't compare weights")
            return

        print("adapter sum:", self.sum_trainable_tf(adapter_model))

        print("same object:", adapter_model is model)

        print("same first var object:",
            adapter_model.trainable_variables[0] is model.trainable_variables[0])


    def sum_trainable_tf(self,m: keras.Model) -> float:
        # evita get_weights() (più pesante) e somma direttamente i tensori
        return float(tf.add_n([tf.reduce_sum(v) for v in m.trainable_variables]).numpy())
