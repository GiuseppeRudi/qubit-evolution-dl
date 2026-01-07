from abc import ABC, abstractmethod
import numpy as np

from ..model.dataset_splits import DatasetSplits
from ..model.model_config import ModelConfig
from ..model.training_config import TrainingConfig
from ..model.phase_config import Phase
from ..model.custom_history import CustomHistory

from .free_running_eval import FreeRunningEvalCallback
from ..strategies.strategy_factory import create_strategy
from ..enums.split_name import SplitName
from ..inference.base import decode_autoregressive

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
        

        history_combined = {
            'loss': [], 
            'test_fr_loss': [], 
            'val_loss': [],
            'phase_names': [],  
            'phase_boundaries': [] 
        }

        current_epoch = 0
 
        
        # Esegui ogni fase
        for phase_idx, phase in enumerate(self.phases):

            strategy = phase.strategy
            phase_epochs = phase.cfg.epochs
            
            print(f"\n{'='*70}")
            print(f"   PHASE {phase_idx+1}/{len(self.phases)}: {strategy.get_name()}")
            print(f"   Epochs: {phase_epochs}")
            print(f"{'='*70}\n")
            
            history_combined['phase_boundaries'].append(current_epoch)
            
            # Custom training loop per questa fase
            for epoch in range(phase_epochs):

                print(f"Epoch {current_epoch + 1}/{self.training_cfg.epochs} ")
              
    
                    # se è un modello step-wise
                if hasattr(self.model, "set_context"):
                    self.model.set_context(strategy=strategy, epoch=epoch, total_epochs=phase_epochs)
                    train_inputs, train_targets = splits.X_train, splits.Y_train
                    val_inputs, val_targets = splits.X_val, splits.Y_val
                else:
                    train_inputs, train_targets = self._prepare_model_inputs(
                        splits.X_train, splits.Y_train, 
                        strategy, epoch, phase_epochs
                    )
                        
                    val_inputs, val_targets = self._prepare_model_inputs(
                        splits.X_val, splits.Y_val,
                        strategy, epoch, phase_epochs
                    )
                
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
                # TODO remove the comment when resolve the call back function
                # history_combined['test_fr_loss'].extend(history.history['test_fr_loss'])
                history_combined['val_loss'].extend(history.history['val_loss'])
                history_combined['phase_names'].append(strategy.get_name())
                
                current_epoch += 1
        
        # this functions is important to return a history like keras 
        # we used this because we fit the model for 1 epoch and we dont have history.epoch = total epoch  
        return self._create_history_object(history_combined)
    
    def _prepare_callbacks(self, splits):

        callbacks = []

        if self.training_cfg.fr_eval.enabled:
            is_test = self.training_cfg.fr_eval.split == SplitName.TEST
            X = splits.X_test if is_test else splits.X_val
            Y = splits.Y_test if is_test else splits.Y_val
            
            # callbacks.append(
            #     FreeRunningEvalCallback(
            #         X, Y,
            #         start_mode=self.model_cfg.inference.start_mode,
            #         verbose=self.model_cfg.inference.verbose,
            #         training_cfg=self.training_cfg
            #     )
            # )
        
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

        X = splits.X_test
        Y = splits.Y_test
        
        # this function is implemented in the concrete classes 
        adapter = self._create_inference_adapter()
        
        # safety check
        if adapter is None :
            raise TypeError("Adapter is None ")

        pred = decode_autoregressive(
            adapter,
            X,
            out_steps=Y.shape[1],
            start_mode=self.model_cfg.inference.start_mode,
            batch_size=self.training_cfg.batch_size,
        )
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

