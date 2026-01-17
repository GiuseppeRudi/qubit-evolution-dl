import numpy as np
from ..strategies.teacher_forcing import make_decoder_inputs
import tensorflow as tf
import keras
from typing import cast

from ..model.fr_eval_config import FrEvalProbeConfig, OutStepsSpec

from ..inference.full_seq_lstm_adapter import FullSeqLstmAdapter
from ..inference.base import  decode
from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode
from ..enums.inference_mode import InferenceMode
from ..model.training_config import TrainingConfig

from ..rnn.step_wise_lstm_model import StepWiseLstmModel
from ..inference.step_wise_lstm_adapter import StepWiseLstmAdapter
from ..transformer.step_wise_model import StepWiseTrnModel
from ..inference.step_wise_trn_adapter import StepWiseTrnAdapter
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

        self.log_prefix = training_cfg.fr_eval.split
        self.verbose = verbose
        self.adapter = None
        self.loss_fn = None
        self.phase_epoch = 0
        self.inference_mode = inference_mode

        self.phase_horizon = None 
        self.end_of_phase = False

        self.probes = training_cfg.fr_eval.probes

    def set_phase_horizon(self, *, phase_horizon: int):
        self.phase_horizon = phase_horizon
    
    def set_end_of_phase(self, *, end_of_phase: bool):
        self.end_of_phase = end_of_phase

    def on_train_begin(self, logs=None):
        if self.model is None:
            raise RuntimeError("Callback not related to a model (self.model is None).")

        trained_model = cast(keras.Model, self.model)
        
        if isinstance(self.model, StepWiseLstmModel):
            self.adapter = StepWiseLstmAdapter(trained_model, verbose=self.verbose)
        elif isinstance(self.model, StepWiseTrnModel):
            self.adapter = StepWiseTrnAdapter(trained_model, verbose=self.verbose)
        else: self.adapter = FullSeqLstmAdapter(trained_model, verbose=self.verbose)
        # self.adapter = Seq2SeqLSTM2LayerAdapter(trained_model, verbose= self.verbose)

        loss = self.model.loss
        self.loss_fn = keras.losses.get(loss) if isinstance(loss, str) else loss
        if self.loss_fn is None:
            raise RuntimeError("Loss function could not be determined")


    def _scalar_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_t = tf.convert_to_tensor(y_true)
        y_pred_t = tf.convert_to_tensor(y_pred)
        print(f"\n{y_true_t.shape}")
        print(y_pred_t.shape)
        if self.loss_fn is not None:
            v = self.loss_fn(y_true_t, y_pred_t)     
        return float(tf.reduce_mean(cast(tf.Tensor,v)).numpy()) 

    def _should_run_probe(self, probe: FrEvalProbeConfig, phase_epoch: int) -> bool:
        every = probe.every_epochs
        if (every == "end_of_phase" and probe.name!="fr_phase"): return bool(self.end_of_phase)
        return (phase_epoch % int(every) == 0)

    def _resolve_steps(self, spec: OutStepsSpec) -> list[int]:
        if isinstance(spec, list):
            return spec
        elif spec == "phase":
            if self.phase_horizon is None:
                raise RuntimeError("phase_horizon is None: call set_phase_context() from Trainer.")
            return [int(self.phase_horizon)]
        else:
            return [int(self.Y_eval.shape[1])]

    def _slice_eval(self, p_eval: float):
        if p_eval <= 0 or p_eval > 1:
            raise ValueError("p_eval must be in (0,1].")
        n = int(self.X_eval.shape[0] * p_eval)
        return self.X_eval[:n], self.Y_eval[:n]

    def on_epoch_end(self, epoch, logs=None):
    
        logs = logs or {}

        if self.adapter is None:
            raise RuntimeError("Adapter not initialized.")

        # choose the active probe in this epoch and take the requested horizon
        active_probes: list[tuple[FrEvalProbeConfig, list[int]]] = []
        requested_steps: list[int] = []
        self.phase_epoch += 1

        for probe in self.probes:

            # check if the probe must be do
            if not self._should_run_probe(probe, self.phase_epoch):
                continue

            # out_steps of probe => list or string
            steps = self._resolve_steps(probe.out_steps)

            steps = [int(k) for k in steps if int(k) > 0]

            active_probes.append((probe, steps))
            requested_steps.extend(steps)

        
        if not active_probes: return

        p_eval = max(p.p_eval for p in self.probes)
        # one slice for max p_eval (after each probe use our slie)
        X_big, Y_big = self._slice_eval(p_eval)
        n_big = X_big.shape[0]

        # decode once for a max requested horizon and after each probe take our part
        max_steps = int(max(requested_steps))
        pred_max = decode(
            self.adapter,
            X_big,
            out_steps=max_steps,
            start_mode=cast(StartMode, self.start_mode),
            batch_size=self.batch_size,
            mode=self.inference_mode,
            y_true=Y_big,
        )

        prefix = self.log_prefix.value.lower()

        # log for each prob with specific p_eval and horizon
        for probe, steps in active_probes:
           
            n_probe = int(self.X_eval.shape[0] * probe.p_eval)
            n_probe = min(n_probe, n_big)  # safety

            Yp = Y_big[:n_probe]
            Pp = pred_max[:n_probe]

            for k in steps:
                y_k = Yp[:, :k, :]
                p_k = Pp[:, :k, :]
                fr_loss = self._scalar_loss(y_k, p_k)  # float

                logs[f"{prefix}_{probe.name}_loss_{k}"] = fr_loss

    def sum_trainable_tf(self,m: keras.Model) -> float:
        return float(tf.add_n([tf.reduce_sum(v) for v in m.trainable_variables]).numpy())
