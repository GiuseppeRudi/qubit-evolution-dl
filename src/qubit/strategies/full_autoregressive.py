import tensorflow as tf

@tf.function
def full_autoregressive_step_wise(
    *,
    y_true_t: tf.Tensor,   # (B,1,F) unused
    y_pred_t: tf.Tensor,   # (B,1,F)
    gradient_through_time: tf.Tensor | bool,
    epoch: tf.Tensor | int = 0,        # unused
    total_epochs: tf.Tensor | int = 1, # unused
) -> tf.Tensor:
    tf.print("\nRUNTIME full_autoregressive_step_wise")
    gtt = tf.cast(gradient_through_time, tf.bool)

    # If you want NO gradient through time, stop it here
    return tf.cond(gtt, lambda: y_pred_t, lambda: tf.stop_gradient(y_pred_t))
