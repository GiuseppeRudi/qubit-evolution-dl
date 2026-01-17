import numpy as np


def make_decoder_inputs(Y: np.ndarray, horizon : int ) -> np.ndarray:
    # dec_in shape = (N, output_seq_len, feature_dim)

    Y_horizon = Y[:, :horizon, :]
    # initialize array with all zeros
    dec_in = np.zeros_like(Y_horizon, dtype=Y.dtype)

    # shift Y by one time step => so start the time step 0 with zeros 
    dec_in[:, 0, :] = 0.0
    print(f"Y_horizon.shape = {Y_horizon.shape} (make_decoder_inputs)")
    print(f"dec_in.shape = {dec_in.shape} (make_decoder_inputs)")
    print(f"horizon = {horizon} (make_decoder_inputs)")

    # fill the rest of the decoder input at timestep t with the target value at timestep t-1
    dec_in[:, 1:, :] = Y_horizon[:, :-1, :]

    return dec_in