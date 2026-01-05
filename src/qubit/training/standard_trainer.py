import numpy as np
from ..registry import register_trainer

import tensorflow as tf
import keras
from ..enums.start_mode import StartMode
from ..enums.inference_mode import InferenceMode
from ..enums.split_name import SplitName

from ..model.model_config import ModelConfig
from ..model.dataset_splits import DatasetSplits
from ..model.training_config import TrainingConfig
from ..inference.seq2seq_rnn import Seq2SeqLSTM2LayerAdapter
from ..inference.base import decode_autoregressive
from .free_running_eval import FreeRunningEvalCallback
from ..strategies.teacher_forcing import make_decoder_inputs
# TODO : implement a strategies in one approach and create a different type of approach 



#TODO change the signature because we use in the standard trainer the teacher forcing approach

class StandardTrainer:
    def __init__(self, model, model_cfg : ModelConfig, training_cfg: TrainingConfig):
        self.model = model
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
    

    def fit(self, splits : DatasetSplits):

        # Y_train shape = (N, output_seq_len, feature_dim)
        # Y_val shape = (N, output_seq_len, feature_dim)
        callbacks = []

        if self.training_cfg.fr_eval.enabled:
    
            is_test = self.training_cfg.fr_eval.split == SplitName.TEST
            X = splits.X_test if is_test else splits.X_val
            Y = splits.Y_test if is_test else splits.Y_val
                
            callbacks.append(
                FreeRunningEvalCallback(
                    X,
                    Y,
                    start_mode= self.model_cfg.inference.start_mode,
                    verbose = self.model_cfg.inference.verbose,
                    training_cfg = self.training_cfg
                )
            )

        dec_in_train = make_decoder_inputs(splits.Y_train)
        dec_in_val   = make_decoder_inputs(splits.Y_val)
        return self.model.fit(
            [splits.X_train, dec_in_train],   # teacher forcing training 
            splits.Y_train,
            epochs=self.training_cfg.epochs,
            batch_size=self.training_cfg.batch_size,
            validation_data=([splits.X_val, dec_in_val], splits.Y_val),
            verbose=self.training_cfg.verbose,
            callbacks=callbacks,
        )


    def predict_all_test(self, splits):
        mode = InferenceMode.FREE_RUNNING
        X = splits.X_test
        Y = splits.Y_test
        bs = self.training_cfg.batch_size
        verbose = self.model_cfg.inference.verbose

        if mode == InferenceMode.TEACHER_FORCING:
            dec_in = make_decoder_inputs(Y)  
            pred = self.model.predict([X, dec_in], batch_size=bs, verbose=verbose)
            return X, Y, pred

        # free-running (no helper inputs)
        adapter = Seq2SeqLSTM2LayerAdapter(self.model, verbose = verbose)
        pred = decode_autoregressive(
            adapter,
            X,
            out_steps=Y.shape[1],                      
            start_mode=self.model_cfg.inference.start_mode,
            batch_size=bs,
        )
        return X, Y, pred

    def report_sample(self, sample_x, sample_y, pred):
        
        # TODO refactor for a correct indexing 
        print(f"  X shape: {sample_x.shape}")
        print(f"  Y shape: {sample_y.shape}")
        print(f"  Pred shape: {pred.shape}")

        np.set_printoptions(suppress=True, precision=16)

        print("\n step | target                | pred                  | abs_err")
        print("------|-----------------------|-----------------------|----------------------")

        for t in range(pred.shape[1] // 2,(pred.shape[1] // 2) + 5):
            if t >= pred.shape[1]: break
            y_t = sample_y[0, t]
            p_t = pred[0, t]
            err = np.abs(p_t - y_t)

            print(
                f"{t:>5} | "
                f"{np.array2string(y_t, precision=6)} | "
                f"{np.array2string(p_t, precision=6)} | "
                f"{np.array2string(err, precision=6)}"
            )

