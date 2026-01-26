import tensorflow as tf

@tf.function
def full_autoregressive_step_wise(
    y_pred_t: tf.Tensor,   # (batch_size, 1, feature_dim) 
    gradient_through_time: tf.Tensor,
) -> tf.Tensor:
    tf.print("\nRUNTIME full_autoregressive_step_wise")

    # if gradient_through_time is False, we use the prediction as the next decoder input
    # but we stop gradients through it. This avoids backpropagating through the
    # autoregressive feedback loop (more stable training).
    # if True, gradients can flow through y_pred_t into future steps.

    return tf.cond(gradient_through_time, lambda: y_pred_t, lambda: tf.stop_gradient(y_pred_t))
