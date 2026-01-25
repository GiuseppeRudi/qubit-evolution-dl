from abc import ABC, abstractmethod
from typing import cast
import numpy as np
import keras
import tensorflow as tf

from ..dataclasses.dataset_splits import DatasetSplits
from ..dataclasses.model_config import ModelConfig
from ..dataclasses.training_config import TrainingConfig

from ..callbacks.free_running_eval import FreeRunningEvalCallback
from ..callbacks.phase_scheduler import PhaseSchedulerCallback
from ..callbacks.filtered_history import FilteredProgbar

from ..enums.split_name import SplitName
from ..enums.inference_mode import InferenceMode


class BaseTrainer(ABC):

    def __init__(self, model, model_cfg: ModelConfig, training_cfg: TrainingConfig):
        self.model = model
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.phases = self.training_cfg.phases
    
    @abstractmethod
    def _create_inference_adapter(self, outsteps: int):
        """
        Returns:
            Adapter object (es. Seq2SeqLSTM2LayerAdapter o TransformerAdapter)
        """
        pass
    
    
    def fit(self, splits : DatasetSplits):

        total_epochs = sum(p.epochs for p in self.phases)

        callbacks = self._prepare_callbacks(splits)

        # X_train and X_val.shape => (num_windows , input_seq_len , feauture_dim)
        # Y_train and Y_val.shape => (num_windows , output_seq_len , feauture_dim)

        # the fit function transform the previous shape in the shape below
        # num_windows / batch_size = num_batch 
        # internally each train_step and test_step function 
        # X_train and X_val .shape => (batch_size, input_seq_len , feature_dim)
        # Y_train and Y_val .shape => (batch_size, output_seq_len, feauture_dim)
        
        history = self.model.fit(
            splits.X_train, splits.Y_train,     
            epochs=total_epochs,
            batch_size=self.training_cfg.batch_size,
            validation_data=(splits.X_val, splits.Y_val),
            callbacks=callbacks,
            verbose=self.training_cfg.verbose,
        )

        return history

    
    def _prepare_callbacks(self, splits):

        callbacks = []

        fr_eval = None
        
        if self.training_cfg.fr_eval.enabled:
            is_test = self.training_cfg.fr_eval.split == SplitName.TEST
            X = splits.X_test if is_test else splits.X_val
            Y = splits.Y_test if is_test else splits.Y_val

            # X.shape(num_windows, input_seq_len, feature_dim)
            # Y.shape(num_windows, output_seq_len, feature_dim)
            
            fr_eval = FreeRunningEvalCallback(
                    X, Y,
                    start_mode=self.model_cfg.inference.start_mode,
                    verbose=self.model_cfg.inference.verbose,
                    inference_mode=self.model_cfg.inference.mode,
                    training_cfg=self.training_cfg,
                )
         
        phase_scheduler = PhaseSchedulerCallback(
            phases=self.phases,
            curriculum=self.training_cfg.curriculum,
            lr_global=self.model_cfg.compile.learning_rate,
            clip_global=self.model_cfg.compile.clip_norm,
            fr_eval=fr_eval
        )
        
        callbacks.append(phase_scheduler)

        if fr_eval is not None: callbacks.append(fr_eval)

        callbacks.append(FilteredProgbar())
        
        return callbacks
    
    def predict_all_test(self, splits):
        """
        Inference in fre-running mode - uses the specific adapter
        """

        print("Predicting all test samples in " + str(self.model_cfg.inference.mode) + " mode...")

        X = splits.X_test
        Y = splits.Y_test

        # X_test.shape(num_windows, input_seq_len, feature_dim)
        # Y_test.shape(num_windows, output_seq_len, feature_dim)

        # this function is implemented in the concrete classes 
        inference_adapter = self._create_inference_adapter(outsteps= Y.shape[1])
        
        # safety check
        if inference_adapter is None :
            raise TypeError("Adapter is None ")

        if self.model_cfg.inference.mode == InferenceMode.TEACHER_FORCING:
            pred = inference_adapter.predict((X, Y), batch_size=self.training_cfg.fr_eval.batch_size)
       
        else:
            pred = inference_adapter.predict(X, batch_size=self.training_cfg.fr_eval.batch_size)

        # pred.shape(num_windows, output_seq_len,feature_dim)
         
        return X, Y, pred
    
    def report_sample(self, sample_x, sample_y, pred):
        """Report"""
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

