import numpy as np
import tensorflow as tf
import keras
from typing import cast

from ..dataclasses.fr_eval_config import FrEvalProbeConfig, OutStepsSpec

from ..enums.start_mode import StartMode
from ..enums.verbose_mode import VerboseMode
from ..enums.inference_mode import InferenceMode
from ..enums.prediction_mode import PredictionMode
from ..dataclasses.training_config import TrainingConfig

from ..models.rnn.step_wise_lstm_model import StepWiseLstmModel
from ..inference.step_wise_lstm_adapter import StepWiseLstmAdapter

from ..models.rnn.full_seq_lstm_model import FullSeqLstmModel
from ..inference.full_seq_lstm_adapter import FullSeqLstmAdapter

from ..models.trn.step_wise_model import StepWiseTrnModel
from ..inference.step_wise_trn_adapter import StepWiseTrnAdapter

class FreeRunningEvalCallback(keras.callbacks.Callback):
    """
    Calculate the free-running losses (autoregressive) [fr_curve, fr_target, fr_phase] based on the
    every_epoch variable to verify if there is exposure bias
    """

    def __init__(
        self,
        X_eval: np.ndarray,
        Y_eval: np.ndarray,
        *,
        verbose: VerboseMode,
        start_mode: StartMode,   
        inference_mode : InferenceMode,   
        training_cfg : TrainingConfig,
    ):
        super().__init__()
        self.X_eval = X_eval
        self.Y_eval = Y_eval

        # X_eval => X_test or X_val => .shape(num_windows, t_in, feature_dim)
        # Y_eval => Y_test or Y_val => .shape(num_windows, t_out, feature_dim)

        self.batch_size = training_cfg.fr_eval.batch_size

        # Start_mode : ZEROS or LAST_X
        self.start_mode = start_mode

        # Split : VAL or TEST
        self.log_prefix = training_cfg.fr_eval.split

        self.verbose = verbose
        self.inference_adapter = None
        self.phase_epoch = 0
        self.inference_mode = inference_mode
        self.phase_horizon = None 
        self.end_of_phase = False
        self.probes = training_cfg.fr_eval.probes
        self.prediction_mode = training_cfg.prediction_mode

    def on_train_begin(self, logs=None):
        # This function is called before the training starting

        if self.prediction_mode == PredictionMode.HORIZON: return
        
        self.inference_adapter = self._create_adapter(self.Y_eval.shape[1])

    def _slice_eval(self, p_eval: float):
        n = int(self.X_eval.shape[0] * p_eval)
        return self.X_eval[:n], self.Y_eval[:n]

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # choose the active probe in this epoch and take the requested horizon
        active_probes: list[tuple[FrEvalProbeConfig, list[int]]] = []
        requested_steps: list[int] = []
        self.phase_epoch += 1

        prefix = self.log_prefix.value.lower()

        for probe in self.probes:

            # probe.out_steps => list or string
            outsteps = self._outsteps(probe.out_steps)

            for step in outsteps : 
                # prefix => test or val 
                # probe.name => fr_curve, fr_target, fr_phase
                key = f"{prefix}_{probe.name}_loss_{step}"
                logs.setdefault(key, np.nan)

            # check if the probe must be do
            if not self._should_run_probe(probe, self.phase_epoch):
                continue

            active_probes.append((probe, outsteps))
            requested_steps.extend(outsteps)

        
        if not active_probes: return

        # one slice for max p_eval (after each probe use our slice)
        p_eval_max = max(p.p_eval for p in self.probes)

        # p_eval_max is the maximum values from all probes
        # that bring the least possible reduction of windows

        # num_reduced_windows => p_eval_max * num_windows
        X_reduced, Y_reduced = self._slice_eval(p_eval_max)

        # X_reduced => X_test or X_val => .shape(num_reduced_windows, t_in, feature_dim)

        # if prediction_mode == ALL so t => Y.shape[1] == output_seq_len
        # if prediction_mode == HORIZON so t => max_steps
        # Y_reduced => Y_test or Y_val => .shape(num_reduced_windows, t, feature_dim)

        # decode once for a max requested horizon and after each probe take our part
        max_steps = int(max(requested_steps))

        if self.prediction_mode == PredictionMode.HORIZON:    
            self.inference_adapter = self._create_adapter(max_steps)

            # TF needs the ground truth until max_steps if prediction_mode == HORIZON
            Y_reduced  = Y_reduced[:,:max_steps,:]

        if self.inference_adapter is None : 
            raise ValueError("Adapter is None")
        
        # number of batch =>  num_reduced_windows / batch_size
        if self.inference_mode == InferenceMode.FREE_RUNNING:
            Y_pred_max = self.inference_adapter.predict(X_reduced, batch_size=self.batch_size)
            
        elif self.inference_mode == InferenceMode.TEACHER_FORCING:
            Y_pred_max = self.inference_adapter.predict((X_reduced, Y_reduced), batch_size=self.batch_size)

        # Y_pred_max.shape(n_reduced, t, feature_dim )
        # if prediction_mode == ALL so t => Y.shape[1] == output_seq_len
        # if prediction_mode == HORIZON so t => max_steps

        # log for each prob with specific p_eval and horizon
        for probe, outsteps in active_probes:
           
            n_reduced_probe = int(self.X_eval.shape[0] * probe.p_eval)

            Y_reduced_probe = Y_reduced[:n_reduced_probe]
            # Y_reduced_probe.shape(n_reduced_probe, t, feature_dim)
            Y_pred_reduced_probe = Y_pred_max[:n_reduced_probe]
            # Y_pred_reduced_probe.shape(n_reduced_probe, t, feature_dim)

            for k in outsteps:
                Y_per_step = Y_reduced_probe[:, :k, :]
                # Y_per_step.shape(n_reduced_probe, k, feature_dim)

                Y_pred_per_step = Y_pred_reduced_probe[:, :k, :]
                # Y_pred_per_step.shape(n_reduced_probe, k, feature_dim)

                trained_model = cast(keras.Model, self.model)
                fr_loss = trained_model.compute_loss(y=Y_per_step, y_pred=Y_pred_per_step)

                logs[f"{prefix}_{probe.name}_loss_{k}"] = float(fr_loss.numpy())  # type: ignore[arg-type]


    def _should_run_probe(self, probe: FrEvalProbeConfig, phase_epoch: int) -> bool:
        
        # need for the fr_target or fr_curve at each end_of_phase
        if (probe.every_epochs == "end_of_phase" and probe.name!="fr_phase"): return bool(self.end_of_phase)
        
        # need for fr_phase at probe.every_epochs
        return (phase_epoch % int(probe.every_epochs) == 0)


    def _outsteps(self, spec: OutStepsSpec) -> list[int]:
        
        if isinstance(spec, list): return spec

        # Phase horizon
        elif spec == "phase":

            if self.phase_horizon is None:
                raise RuntimeError("phase_horizon is None: call set_phase_context() from Trainer.")
            
            return [int(self.phase_horizon)]
        
        # output_seq_len => Global horizon
        else: return [int(self.Y_eval.shape[1])]

    def _create_adapter(self, outsteps: int):

        # timesteps => max horizon 
        trained_model = cast(keras.Model, self.model)
        if isinstance(self.model, StepWiseLstmModel):
            return StepWiseLstmAdapter(trained_model, verbose=self.verbose)
        elif isinstance(self.model, StepWiseTrnModel):
            return StepWiseTrnAdapter(trained_model, verbose=self.verbose)
        elif isinstance(self.model, FullSeqLstmModel):
            return FullSeqLstmAdapter(trained_model, 
                                            verbose = self.verbose,
                                            start_mode = self.start_mode,
                                            inference_mode = self.inference_mode,
                                            out_steps = outsteps 
            )
        else:
            raise ValueError("Unsupported model type")
