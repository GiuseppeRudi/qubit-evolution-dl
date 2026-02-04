from ..utils.registry import register_trainer
import numpy as np
import tensorflow as tf 
from .base_trainer import BaseTrainer
from ..enums.model_type import ModelType

from ..models.trn.hybrid_trn_model import HybridTrnModel
from ..inference.hybrid_trn_adapter import HybridTrnAdapter

from ..models.trn.sr_trn_model import SrTrnModel
from ..inference.sr_trn_adapter import SrTrnAdapter

from ..dataclasses.dataset_splits import DatasetSplits

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
        

    def extract_attention_maps(self, splits: DatasetSplits, *, sample_index: int = 0):
        
        print("Creating attention maps")
        if not isinstance(self.model, HybridTrnModel):
            raise TypeError("Attention maps are supported only for HybridTrnModel (TRN forecasting).")

        # use 1 sample 
        X1 = splits.X_test[sample_index:sample_index+1]
        Y1 = splits.Y_test[sample_index:sample_index+1]


        # forward TF to get maps
        y_pred, attn = self.model.forward_with_attn(tf.convert_to_tensor(X1), tf.convert_to_tensor(Y1), training=False)
        
        attn_np: dict[str, np.ndarray] = {k: v.numpy() for k, v in attn.items()}

        return y_pred, attn_np
