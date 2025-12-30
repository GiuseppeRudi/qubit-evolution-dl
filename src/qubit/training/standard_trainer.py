from typing import cast
import numpy as np
from ..registry import register_trainer

import tensorflow as tf
import keras
from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..inference.base import StartMode, decode_autoregressive

# TODO : implement a strategies in one approach and create a different type of approach 

# teacher forcing technique
def make_decoder_inputs(Y: np.ndarray) -> np.ndarray:
    # dec_in shape = (N, output_seq_len, feature_dim)

    # initialize array with all zeros
    dec_in = np.zeros_like(Y, dtype=Y.dtype)

    # shift Y by one time step => so start the time step 0 with zeros 
    dec_in[:, 0, :] = 0.0

    # fill the rest of the decoder input at timestep t with the target value at timestep t-1
    dec_in[:, 1:, :] = Y[:, :-1, :]

    return dec_in

#TODO change the signature because we use in the standard trainer the teacher forcing approach
@register_trainer("standard")
class StandardTrainer:
    def __init__(self, model, model_cfg, eval_cfg=None):
        self.model = model
        self.cfg = model_cfg
        self.eval_cfg = eval_cfg or {}

    def fit(self, splits):

        # Y_train shape = (N, output_seq_len, feature_dim)
        # Y_val shape = (N, output_seq_len, feature_dim)
        callbacks = []

        callbacks.append(
            FreeRunningEvalCallback(
                splits.X_test,
                splits.Y_test,
                batch_size=self.cfg.training.batch_size,
                start_mode=self.eval_cfg.get("start_mode", "zeros"),
                every_epochs=1,
                n_eval=None,
                compute_tf_subset=False,
            )
        )

        dec_in_train = make_decoder_inputs(splits.Y_train)
        dec_in_val   = make_decoder_inputs(splits.Y_val)
        return self.model.fit(
            [splits.X_train, dec_in_train],   # teacher forcing training 
            splits.Y_train,
            epochs=self.cfg.training.epochs,
            batch_size=self.cfg.training.batch_size,
            validation_data=([splits.X_val, dec_in_val], splits.Y_val),
            verbose=1,
            callbacks=callbacks,
        )


    def predict_all_test(self, splits):
        mode = self.eval_cfg.get("inference_mode", "free_running")
        X = splits.X_test
        Y = splits.Y_test
        bs = self.cfg.training.batch_size

        if mode == "teacher_forcing":
            dec_in = make_decoder_inputs(Y)  
            pred = self.model.predict([X, dec_in], batch_size=bs, verbose=0)
            return X, Y, pred

        # free-running (no helper inputs)
        adapter = Seq2SeqLSTM2LayerAdapter(self.model)
        pred = decode_autoregressive(
            adapter,
            X,
            out_steps=Y.shape[1],                      
            start_mode=self.eval_cfg.get("start_mode", "zeros"),
            batch_size=bs,
        )
        return X, Y, pred

    def compare_losses(self, splits, split="val"):
        
        # select the correct X,Y based on the type of split 
        X = getattr(splits, f"X_{split}")
        Y = getattr(splits, f"Y_{split}")

        bs = self.cfg.training.batch_size

        # teacher forcing loss 
        dec_in = make_decoder_inputs(Y)
        eval_out = self.model.evaluate([X, dec_in], Y, batch_size=bs, verbose=0)
        tf_loss = float(eval_out[0] if isinstance(eval_out, (list, tuple)) else eval_out)

        # free-running loss 
        adapter = Seq2SeqLSTM2LayerAdapter(self.model)
        pred_fr = decode_autoregressive(
            adapter,
            X,
            out_steps=Y.shape[1],
            start_mode=self.eval_cfg.get("start_mode", "zeros"),
            batch_size=bs,
        )
        fr_loss = self._scalar_loss(Y, pred_fr)

        return {"teacher_forcing_loss": tf_loss, "free_running_loss": fr_loss}


    def _scalar_loss(self, y_true, y_pred) -> float:
        v = self.model.loss(y_true, y_pred)        
        return float(tf.reduce_mean(v).numpy())

    def report_sample(self, sample_x, sample_y, pred):
        
        # TODO refactor for a correct indexing 
        steps = self.eval_cfg.get("print_steps", 5)
        print(f"  X shape: {sample_x.shape}")
        print(f"  Y shape: {sample_y.shape}")
        print(f"  Pred shape: {pred.shape}")

        np.set_printoptions(suppress=True, precision=16)

        print("\n step | target                | pred                  | abs_err")
        print("------|-----------------------|-----------------------|----------------------")

        for t in range(pred.shape[1] // 2,(pred.shape[1] // 2) + steps):
            if t >= pred.shape[1]: break
            y_t = sample_y[0, t]
            p_t = pred[0, t]
            err = np.abs(p_t - y_t)

            print(
                f"{t:>5} | "
                f"{np.array2string(y_t, precision=6)} | "
                f"{np.array2string(p_t, precision=6)} | "
                f"{np.array2string(err, precision=6)}"
            )



import numpy as np
import tensorflow as tf
import keras

from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..inference.base import decode_autoregressive

class FreeRunningEvalCallback(keras.callbacks.Callback):
    """
    Calcola la free-running loss (autoregressiva) a fine epoca e la inserisce nei logs,
    così finisce dentro history.history (es: history.history['val_fr_loss']).
    """

    def __init__(
        self,
        X_eval: np.ndarray,
        Y_eval: np.ndarray,
        *,
        batch_size: int,
        start_mode: str = "zeros",         # "zeros" | "last_x"
        every_epochs: int = 1,
        n_eval: int | None = None,         # per velocizzare (es. 256)
        log_prefix: str = "val",           # "val" o "test"
        compute_tf_subset: bool = False,   # opzionale: rifare anche TF loss su subset
    ):
        super().__init__()
        self.X_eval = X_eval
        self.Y_eval = Y_eval
        self.batch_size = batch_size
        self.start_mode = start_mode
        self.every_epochs = max(1, int(every_epochs))
        self.n_eval = n_eval
        self.log_prefix = log_prefix
        self.compute_tf_subset = compute_tf_subset

        self.adapter = None
        self.loss_fn = None

    def on_train_begin(self, logs=None):
        if self.model is None:
            raise RuntimeError("Callback non associata a un modello (self.model è None).")

        trained_model = cast(keras.Model, self.model)
        self.adapter = Seq2SeqLSTM2LayerAdapter(trained_model)
        # Loss coerente col compile (string o oggetto Loss)
        loss = self.model.loss
        self.loss_fn = keras.losses.get(loss) if isinstance(loss, str) else loss
        if self.loss_fn is None:
            # fallback (meglio comunque compilare sempre con loss!)
            self.loss_fn = keras.losses.MeanSquaredError()

    def _scalar_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_t = tf.convert_to_tensor(y_true)
        y_pred_t = tf.convert_to_tensor(y_pred)
        if self.loss_fn is not None:
            v = self.loss_fn(y_true_t, y_pred_t)      # può essere per-sample o per-timestep
        return float(tf.reduce_mean(v).numpy())

    def _slice_eval(self):
        if self.n_eval is None:
            return self.X_eval, self.Y_eval
        return self.X_eval[: self.n_eval], self.Y_eval[: self.n_eval]

    def on_epoch_end(self, epoch, logs=None):
        # epoch è 0-based
        if (epoch + 1) % self.every_epochs != 0:
            return

        logs = logs or {}
        X, Y = self._slice_eval()

        # FREE-RUNNING prediction
        if self.adapter is None:
            raise RuntimeError("Adapter non inizializzato.")
        
        pred_fr = decode_autoregressive(
            self.adapter,
            X,
            out_steps=Y.shape[1],
            start_mode=cast(StartMode, self.start_mode),
            batch_size=self.batch_size,
        )

        fr_loss = self._scalar_loss(Y, pred_fr)
        logs[f"{self.log_prefix}_fr_loss"] = fr_loss

        # opzionale: teacher forcing loss su subset (in genere non serve perché hai già val_loss)
        if self.compute_tf_subset:
            dec_in = make_decoder_inputs(Y)
            out = cast(keras.Model, self.model).evaluate([X, dec_in], Y, batch_size=self.batch_size)
            tf_loss = float(out[0] if isinstance(out, (list, tuple)) else out)
            logs[f"{self.log_prefix}_tf_loss_subset"] = tf_loss
