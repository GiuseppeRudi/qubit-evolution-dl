import numpy as np 

from .decoder_utils import make_decoder_inputs
from .base_strategy import TrainingStrategy

class TeacherForcingStrategy(TrainingStrategy):


    def prepare_inputs(self, X, Y, epoch, total_epochs):
        dec_in = make_decoder_inputs(Y)
        return [X, dec_in], Y
    
    def get_name(self) -> str:
        return f"TeacherForcing"
    
    def next_dec_input(self, *, y_true_t, y_pred_t, epoch, total_epochs):
            return y_true_t