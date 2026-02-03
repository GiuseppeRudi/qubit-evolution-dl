from ..utils.registry import register_trainer
import numpy as np

from .base_trainer import BaseTrainer
from ..enums.model_type import ModelType

from ..models.trn.hybrid_trn_model import HybridTrnModel
from ..inference.hybrid_trn_adapter import HybridTrnAdapter

from ..models.trn.sr_trn_model import SrTrnModel
from ..inference.sr_trn_adapter import SrTrnAdapter

@register_trainer(ModelType.TRN)
class TransformerTrainer(BaseTrainer):  
      
    def _create_inference_adapter(self, outsteps: int, feature_dim: int):
        if isinstance(self.model, HybridTrnModel):
            return HybridTrnAdapter(self.model,
                                  out_steps=outsteps,
                                  feature_dim=feature_dim,
                                  inference_mode=self.model_cfg.inference.mode)
        elif isinstance(self.model,SrTrnModel):
            return SrTrnAdapter(self.model)
        else:
            raise ValueError("Unsupported model type")