
from abc import ABC, abstractmethod

from ..registry import register_trainer
import numpy as np
import keras
from ..model.dataset_splits import DatasetSplits
from ..model.model_config import ModelConfig
from ..model.training_config import TrainingConfig
from .free_running_eval import FreeRunningEvalCallback
from .base_trainer import BaseTrainer
from ..strategies.teacher_forcing import make_decoder_inputs
from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..enums.model_type import ModelType

from ..rnn.Seq2SeqLSTM2LayerStepWiseModel import Seq2SeqLSTM2LayerStepWiseModel
from ..inference.step_wise_rnn_adapter import StepWiseSeq2SeqAdapter

# ============================================
# RNN/LSTM TRAINER
# ============================================
@register_trainer(ModelType.LSTM)
class RNNTrainer(BaseTrainer):
 
    def _prepare_model_inputs(self, X, Y, strategy, epoch, total_epochs):
     
        # the corrispondent strategy prepare the input based on the current epoch in some cases
        model_inputs, targets = strategy.prepare_inputs(X, Y, epoch, total_epochs)

        # model inputs is tupla formed by encoder inputs , decoder inputs 
        # targets is the ground truth
        return model_inputs, targets
    
    def _create_inference_adapter(self):
        if isinstance(self.model, Seq2SeqLSTM2LayerStepWiseModel):
            return StepWiseSeq2SeqAdapter(self.model, verbose=self.model_cfg.inference.verbose)
        return Seq2SeqLSTM2LayerAdapter(self.model, verbose=self.model_cfg.inference.verbose)

    
    def _get_model_state_for_scheduled_sampling(self, X, prev_predictions, timestep):
        """
        Metodo specifico RNN per scheduled sampling.
        Ottiene le predizioni del modello per il timestep corrente.
        """
        # RNN mantiene hidden state interno
        # Qui potresti implementare logica per accedere agli stati nascosti
        pass

