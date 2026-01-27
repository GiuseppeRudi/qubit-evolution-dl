from ..utils.registry import register_trainer
from .base_trainer import BaseTrainer
from ..inference.full_seq_lstm_adapter import FullSeqLstmAdapter
from ..enums.model_type import ModelType

from ..models.rnn.step_wise_lstm_model import StepWiseLstmModel
from ..inference.step_wise_lstm_adapter import StepWiseLstmAdapter

@register_trainer(ModelType.LSTM)
class RNNTrainer(BaseTrainer):
    
    def _create_inference_adapter(self, outsteps: int):
        if isinstance(self.model, StepWiseLstmModel):
            return StepWiseLstmAdapter(self.model,
                                  out_steps=outsteps,
                                  inference_mode=self.model_cfg.inference.mode)
        
        return FullSeqLstmAdapter(self.model,
                                  out_steps=outsteps,
                                  inference_mode=self.model_cfg.inference.mode)
