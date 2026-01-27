from ..utils.registry import register_trainer
import numpy as np

from .base_trainer import BaseTrainer
from ..inference.step_wise_trn_adapter import StepWiseTrnAdapter
from ..enums.model_type import ModelType


@register_trainer(ModelType.TRN)
class TransformerTrainer(BaseTrainer):    
    def _create_inference_adapter(self, outsteps: int):
        #if isinstance(self.model, Seq2SeqTransformerStepWiseModel):
            return StepWiseTrnAdapter(self.model,
                                  out_steps=outsteps,
                                  inference_mode=self.model_cfg.inference.mode)
        #return Seq2SeqLSTM2LayerAdapter(self.model, verbose=self.model_cfg.inference.verbose)
