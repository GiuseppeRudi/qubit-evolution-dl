import numpy as np
from ..registry import register_trainer

# teacher forcing technique
def make_decoder_inputs(Y: np.ndarray) -> np.ndarray:
    # dec_in shape = (N, output_seq_len, feature_dim)

    # initialize array with all zeros
    dec_in = np.zeros_like(Y, dtype=Y.dtype)

    # shift Y by one time step => so start the time step 0 with zeros 
    dec_in[:, 0, :] = 0.0

    # fill the rest of the decoder input at timestep t with the target value at timestep t-1
    dec_in[:, 1:, :] = Y[:, :-1, :]

    return dec_in

#TODO change the signature because we use in the standard trainer the teacher forcing approach
@register_trainer("standard")
class StandardTrainer:
    def __init__(self, model, model_cfg, eval_cfg=None):
        self.model = model
        self.cfg = model_cfg
        self.eval_cfg = eval_cfg or {}

    def fit(self, splits):

        # Y_train shape = (N, output_seq_len, feature_dim)
        # Y_val shape = (N, output_seq_len, feature_dim)

        dec_in_train = make_decoder_inputs(splits.Y_train)
        dec_in_val   = make_decoder_inputs(splits.Y_val)
        return self.model.fit(
        [splits.X_train, dec_in_train],   # teacher forcing training 
        splits.Y_train,
        epochs=self.cfg.training.epochs,
        batch_size=self.cfg.training.batch_size,
        validation_data=([splits.X_val, dec_in_val], splits.Y_val),
        verbose=1,
        )

    def predict_from_train(self, splits):
        idx = int(self.eval_cfg.get("sample_index", 0))
        sample_x = splits.X_train[idx:idx+1]
        sample_y = splits.Y_train[idx:idx+1]

        dec_in = make_decoder_inputs(sample_y)
        pred = self.model.predict([sample_x, dec_in], verbose=0)
        return sample_x, pred
    
    def predict_from_test(self, splits):
        idx = int(self.eval_cfg.get("sample_index", 0))
        sample_x = splits.X_test[idx:idx+1]
        sample_y = splits.Y_test[idx:idx+1]

        dec_in = make_decoder_inputs(sample_y)
        pred = self.model.predict([sample_x, dec_in], verbose=0)
        return sample_x, pred

    # TODO change the method or create another predict because use the Xtrain eYtrain for predict and this is not correct only for debug 
    def report_sample(self, sample_x, pred):
        steps = int(self.eval_cfg.get("print_steps", 5))
        print(f"  X shape: {sample_x.shape}")
        print(f"  Pred shape: {pred.shape}")
        np.set_printoptions(suppress=True, precision=16)
        print(pred[0, :steps, :])
