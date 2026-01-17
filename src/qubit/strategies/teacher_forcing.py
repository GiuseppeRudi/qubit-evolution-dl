import numpy as np 

from .decoder_utils import make_decoder_inputs
from .base_strategy import TrainingStrategy

class TeacherForcingStrategy(TrainingStrategy):


    def prepare_inputs_full_seq(self, X, Y, epoch, total_epochs, horizon):
        dec_in = make_decoder_inputs(Y,horizon)
        Y_h = Y[:, :horizon, :]  
        return [X, dec_in], Y_h
    
    def get_name(self) -> str:
        return f"TeacherForcing"
    
    def prepare_inputs_step_wise(self, *, y_true_t, y_pred_t, epoch, total_epochs):
            return y_true_t