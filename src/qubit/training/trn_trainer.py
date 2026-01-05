
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

@register_trainer(ModelType.TRN)
class TransformerTrainer(BaseTrainer):
    """
    Trainer specifico per modelli Transformer.
    Gestisce attention masks, positional encoding, causal masking.
    """

    ModelInputs = list[np.ndarray] | np.ndarray
    Targets = np.ndarray 
    
    def _prepare_model_inputs(self, X, Y, strategy, epoch, total_epochs):
        """
        Prepara input per Transformer: [encoder_input, decoder_input, masks]
        
        Transformer richiede:
        - Encoder input: X
        - Decoder input: shifted Y
        - Encoder padding mask
        - Decoder causal mask (look-ahead mask)
        - Decoder padding mask
        """
        # Ottieni input base dalla strategia
        base_inputs, targets = strategy.prepare_inputs(X, Y, epoch, total_epochs)
        
        # Estrai encoder e decoder input
        if isinstance(base_inputs, list):
            encoder_input = base_inputs[0]
            decoder_input = base_inputs[1]
        else:
            encoder_input = X
            decoder_input = make_decoder_inputs(Y)
        
        # ⭐ LOGICA SPECIFICA TRANSFORMER: crea attention masks
        # encoder_mask = self._create_padding_mask(encoder_input)
        # decoder_mask = self._create_causal_mask(decoder_input)
        
        # Combina input con masks
        model_inputs = [
            encoder_input,
            decoder_input,
            # encoder_mask,
            # decoder_mask
        ]
        
        return model_inputs, targets
    
    # def _create_padding_mask(self, sequence):
        # """
        # Crea padding mask per nascondere padding tokens.
        # Shape: (batch_size, 1, 1, seq_len)
        # """
        # # Assumi che padding = 0
        # mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        # return mask[:, tf.newaxis, tf.newaxis, :]
    
    # def _create_causal_mask(self, sequence):
    #     """
    #     Crea look-ahead mask per impedire al decoder di vedere il futuro.
    #     Shape: (seq_len, seq_len)
    #     """
    #     seq_len = tf.shape(sequence)[1]
    #     mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    #     return mask
    
    # def _create_inference_adapter(self):
    #     """Crea adapter per inferenza Transformer"""
    #     return TransformerAdapter(
    #         self.model,
    #         verbose=self.model_cfg.inference.verbose
    #     )
    
    # def _apply_positional_encoding(self, inputs):
    #     """
    #     Applica positional encoding (se non già nel modello).
    #     Transformer-specific logic.
    #     """
    #     # Implementa se necessario
    #     pass
