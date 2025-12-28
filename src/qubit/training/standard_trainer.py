import numpy as np
from ..registry import register_trainer

# TODO : implement a strategies in one approach and create a different type of approach 

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


    def predict_all_test(self, splits):
        # TODO dont use teacher forcing for iference this is only for debug
        X = splits.X_test                 
        Y = splits.Y_test                

        dec_in = make_decoder_inputs(Y)  
        pred = self.model.predict([X, dec_in], batch_size=self.cfg.training.batch_size, verbose=0)
        return X, Y , pred                 




    def report_sample(self, sample_x, sample_y, pred):
        
        # TODO refactor for a correct indexing 
        steps = self.eval_cfg.get("print_steps", 5)
        print(f"  X shape: {sample_x.shape}")
        print(f"  Y shape: {sample_y.shape}")
        print(f"  Pred shape: {pred.shape}")

        np.set_printoptions(suppress=True, precision=16)

        print("\n step | target                | pred                  | abs_err")
        print("------|-----------------------|-----------------------|----------------------")

        for t in range(steps):
            y_t = sample_y[0, t]
            p_t = pred[0, t]
            err = np.abs(p_t - y_t)

            print(
                f"{t:>5} | "
                f"{np.array2string(y_t, precision=6)} | "
                f"{np.array2string(p_t, precision=6)} | "
                f"{np.array2string(err, precision=6)}"
            )


