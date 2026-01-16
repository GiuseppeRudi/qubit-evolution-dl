from abc import ABC, abstractmethod
from typing import cast
import numpy as np
import keras


from ..model.dataset_splits import DatasetSplits
from ..model.model_config import ModelConfig
from ..model.training_config import TrainingConfig
from ..model.phase_config import Phase
from ..model.custom_history import CustomHistory

from ..rnn.Seq2SeqLSTM2LayerStepWiseModel import Seq2SeqLSTM2LayerStepWiseModel

from .free_running_eval import FreeRunningEvalCallback
from ..strategies.strategy_factory import create_strategy
from ..strategies.decoder_utils import make_decoder_inputs
from ..enums.split_name import SplitName
from ..enums.inference_mode import InferenceMode

from ..inference.base import decode

class BaseTrainer(ABC):

    def __init__(self, model, model_cfg: ModelConfig, training_cfg: TrainingConfig):
        self.model = model
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        
        
        self.phases = self._build_phases()
    
    def _build_phases(self) -> list[Phase] :

        """Convert Phase Config in executable strategies"""
        phases : list[Phase]  = []
        for phase in self.training_cfg.phases:

            # call the factory method to choose the correct class 
            strategy = create_strategy(phase)
            phases.append(Phase(phase,strategy))
        return phases
    

    @abstractmethod
    def _prepare_model_inputs(self, X, Y, strategy, epoch, total_epochs) -> tuple:
        """
        Args:
            X: Input encoder
            Y: Target sequences
            strategy: current training strategy
            epoch: Current epoch in the current strategy 
            total_epochs: total epoch in the current strategy
            
        Returns:
            (model_inputs, targets) dove model_inputs è quello che viene passato a model.fit()
        """
        pass
    
    @abstractmethod
    def _create_inference_adapter(self):
        """
        Returns:
            Adapter object (es. Seq2SeqLSTM2LayerAdapter o TransformerAdapter)
        """
        pass
    

    
    def fit(self, splits: DatasetSplits):

        callbacks = self._prepare_callbacks(splits)

        curriculum_no_dupe = list(dict.fromkeys(
            splits.Y_train.shape[1] if h == -1 else int(h)
            for h in self.training_cfg.curriculum
        ))
        
        fr_target = self.training_cfg.fr_eval.split.value + '_fr_target_loss_' + str(splits.Y_train.shape[1])
        fr_curve = self.training_cfg.fr_eval.split.value + '_fr_curve_loss_'
        fr_phase = self.training_cfg.fr_eval.split.value + '_fr_phase_loss_'
        out_steps_curve = next(p.out_steps for p in self.training_cfg.fr_eval.probes if p.name == "fr_curve")

        history_combined = {
            'loss': [],
            **{fr_curve + str(h): [] for h in out_steps_curve},
            **{fr_phase + str(h): [] for h in curriculum_no_dupe},
            fr_target: [],
            'val_loss': [],
            'phase_names': []
        }

        current_epoch = 0
        lr_global = self.model_cfg.compile.learning_rate
        clip_norm_global = self.model_cfg.compile.clip_norm
        
        # for each phases (strategy)
        for phase_idx, phase in enumerate(self.phases):

            strategy = phase.strategy
            phase_epochs = phase.cfg.epochs
            horizon = self.training_cfg.curriculum[phase_idx] if self.training_cfg.curriculum[phase_idx] != -1 else splits.Y_train.shape[1]
            
            lr_local = self.phases[phase_idx].cfg.learning_rate
            clip_norm_local = self.phases[phase_idx].cfg.clip_norm
            
            lr_to_use = lr_global if lr_local is None else lr_local
            clip_norm_to_use = clip_norm_global if clip_norm_local is None else clip_norm_local

            self.model.optimizer.learning_rate.assign(float(lr_to_use))
            self.model.current_clip_norm = clip_norm_to_use

            # clip_norm is used only for a custom model (not for full_seq model)
            self.model.current_clip_norm = clip_norm_to_use

            print(f"\n{'='*70}")
            print(f"   PHASE {phase_idx+1}/{len(self.phases)}: {strategy.get_name()}")
            for key, value in phase.cfg.__dict__.items():
                if (key != "epochs" and key != "name" and key != "learning_rate" and key != "clip_norm"): print(f"   {key.replace("_"," ").title()}: {value if not isinstance(value,str) else str(value)}")
            print(f"   Epochs: {phase_epochs}")
            print(f"   Learning Rate: {lr_to_use}")
            print(f"   Clipping Norm: {clip_norm_to_use}")
            print(f"   Horizon (train loss): {horizon}")
            print(f"   Output timesteps (val_loss and fr_loss): {splits.Y_train.shape[1]}")
            print(f"{'='*70}\n")


            callbacks[0].phase_horizon = horizon

            # for each epoch of each phase 
            for epoch in range(phase_epochs):
                callbacks[0].end_of_phase = (epoch == phase_epochs-1)

                print(f"Epoch {current_epoch + 1}/{self.training_cfg.epochs} ")
       
                if  isinstance(self.model,Seq2SeqLSTM2LayerStepWiseModel):
                    self.model.set_context(strategy=strategy, epoch=epoch, total_epochs=phase_epochs,horizon=horizon)
                    train_inputs, train_targets = splits.X_train, splits.Y_train[:, :horizon, :]
                    val_inputs, val_targets = splits.X_val, splits.Y_val[:, :horizon, :]
                else:
                    train_inputs, train_targets = strategy.prepare_inputs(splits.X_train, splits.Y_train,epoch, phase_epochs,horizon)
                    val_inputs, val_targets = strategy.prepare_inputs(splits.X_val, splits.Y_val,epoch, phase_epochs, horizon)

                
                # training for 1 epoch because there are strategies that need results for each epoch
                history = self.model.fit(
                    train_inputs,
                    train_targets,
                    epochs=1,
                    batch_size=self.training_cfg.batch_size,
                    validation_data=(val_inputs, val_targets),
                    verbose=self.training_cfg.verbose,
                    callbacks=callbacks,
                )
                
                # object useful to obtain a custom history for plotting
                history_combined['loss'].extend(history.history['loss'])

                if (fr_target in history.history):
                    history_combined[fr_target].extend(history.history[fr_target])
                else : history_combined[fr_target].append(None)

                for h in out_steps_curve:
                    curve = fr_curve + str(h)
                    if (curve in history.history):
                        history_combined[curve].extend(history.history[curve])
                    else: history_combined[curve].append(None)

                phase = fr_phase + str(horizon)

                if (phase in history.history):
                    history_combined[phase].extend(history.history[phase])
                else : history_combined[phase].append(None)  
            
                history_combined['val_loss'].extend(history.history['val_loss'])
                history_combined['phase_names'].append(strategy.get_name())
                
                current_epoch += 1
            
            callbacks[0].phase_epoch = 0
        
        # this functions is important to return a history like keras 
        # we used this because we fit the model for 1 epoch and we dont have history.epoch = total epoch  
        return self._create_history_object(history_combined)
    
    def _prepare_callbacks(self, splits):

        callbacks = []

        if self.training_cfg.fr_eval.enabled:
            is_test = self.training_cfg.fr_eval.split == SplitName.TEST
            X = splits.X_test if is_test else splits.X_val
            Y = splits.Y_test if is_test else splits.Y_val
            
            callbacks.append(
                FreeRunningEvalCallback(
                    X, Y,
                    start_mode=self.model_cfg.inference.start_mode,
                    verbose=self.model_cfg.inference.verbose,
                    inference_mode=self.model_cfg.inference.mode,
                    training_cfg=self.training_cfg
                )
            )
        
        return callbacks
    
    def _create_history_object(self, history_dict):
        n_epochs = len(history_dict["loss"])
        return CustomHistory(
            history=history_dict,
            epoch=list(range(n_epochs)),
        )

    
    def predict_all_test(self, splits):
        """
        Inference in fre-running mode - uses the specific adapter
        """

        # TODO : resolve the problem that occurs when we are using the step wise
        # because this model currently doesn't have the inference model 
        # for this reason decode_autoregressive and the adapter doesnt exits
        print("Predicting all test samples in " + str(self.model_cfg.inference.mode) + " mode...")

        X = splits.X_test
        Y = splits.Y_test

        
        
        # this function is implemented in the concrete classes 
        adapter = self._create_inference_adapter()
        
        # safety check
        if adapter is None :
            raise TypeError("Adapter is None ")

        if self.model_cfg.inference.mode == InferenceMode.TEACHER_FORCING:
            pred = decode(adapter, X,
                        out_steps=Y.shape[1],
                        start_mode=self.model_cfg.inference.start_mode,
                        batch_size=self.training_cfg.batch_size,
                        mode=InferenceMode.TEACHER_FORCING,
                        y_true=Y)
        else:
            pred = decode(adapter, X,
                        out_steps=Y.shape[1],
                        start_mode=self.model_cfg.inference.start_mode,
                        batch_size=self.training_cfg.batch_size,
                        mode=InferenceMode.FREE_RUNNING)

        return X, Y, pred
    
    def report_sample(self, sample_x, sample_y, pred):
        """Report dei risultati - IDENTICO per tutti"""
        print(f"  X shape: {sample_x.shape}")
        print(f"  Y shape: {sample_y.shape}")
        print(f"  Pred shape: {pred.shape}")

        np.set_printoptions(suppress=True, precision=16)

        print("\n step | target                | pred                  | abs_err")
        print("------|-----------------------|-----------------------|----------------------")

        for t in range(pred.shape[1] // 2, (pred.shape[1] // 2) + 5):
            if t >= pred.shape[1]: 
                break
            y_t = sample_y[0, t]
            p_t = pred[0, t]
            err = np.abs(p_t - y_t)

            print(
                f"{t:>5} | "
                f"{np.array2string(y_t, precision=6)} | "
                f"{np.array2string(p_t, precision=6)} | "
                f"{np.array2string(err, precision=6)}"
            )

