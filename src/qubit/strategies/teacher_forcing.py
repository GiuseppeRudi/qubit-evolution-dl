import tensorflow as tf

@tf.function
def teacher_forcing_full_seq(
    Y: tf.Tensor,
    dec0: tf.Tensor,
):
    # Y if prediction mode => ALL => Y.shape = (B, t = t_out == output_seq_len, F)
    # Y if prediction mode => HORIZON => Y.shape = (B, t = t_hor, F)

    # dec0.shape(batch_size , 1 , feature_dim)
    
    # we lose Y[-1] not used because at time t we use Y[t-1] as input
    Y_truncated = Y[:, :-1, :]  # (batch_size , t - 1 , feature_dim)

    # dec_in.shape[1] = dec0.shape[1] + Y_truncated.shape[1]  => t
    dec_in = tf.concat([dec0, Y_truncated], 1) # (batch_size, t, feature_dim)
    
    return dec_in


@tf.function
def teacher_forcing_step_wise(
    y_true_t: tf.Tensor, ) -> tf.Tensor:
    tf.print("\nRUNTIME teacher_forcing_step_wise")

    # y_true_t.shape(batch_size, 1, feature_dim)
    return y_true_t
