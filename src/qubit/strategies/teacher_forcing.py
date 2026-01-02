import numpy as np 
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