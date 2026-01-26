import tensorflow as tf

@tf.function
def linear_ratio(epoch, total_epochs, start, end) -> tf.Tensor:
    # epoch 0-based
    den = tf.maximum(tf.cast(total_epochs, tf.int32) - 1, 1)
    alpha = tf.cast(epoch, tf.float32) / tf.cast(den, tf.float32)
    start = tf.cast(start, tf.float32)
    end = tf.cast(end, tf.float32)
    return start + (end - start) * alpha


@tf.function
def scheduled_sampling_next(
    y_true_t: tf.Tensor,
    y_pred_t: tf.Tensor,
    ratio: tf.Tensor,
    *,
    scope_id: tf.Tensor,          # 0 -> broadcast (B,1,1), 1 -> per-feature (B,1,F)
    stop_grad_pred: bool = True,
) -> tf.Tensor:
    if stop_grad_pred:
        y_pred_t = tf.stop_gradient(y_pred_t)

    b = tf.shape(y_true_t)[0]
    f = tf.shape(y_true_t)[2]

    scope_id = tf.cast(scope_id, tf.int32)
    scope_id = tf.reshape(scope_id, [])   # forza scalare

    shape = tf.switch_case(
        scope_id,
        branch_fns={
            0: lambda: tf.stack([b, 1, 1]),  # stessa scelta per tutte le feature
            1: lambda: tf.stack([b, 1, f]),  # scelta diversa per ogni feature
        },
        default=lambda: tf.stack([b, 1, 1]),
    )

    use_teacher = tf.random.uniform(shape, 0.0, 1.0) < ratio
    return tf.where(use_teacher, y_true_t, y_pred_t)


@tf.function
def scheduled_sampling_strategy(
    y_true_t: tf.Tensor, # y_true_t.shape()
    y_pred_t: tf.Tensor,
    *,
    epoch: tf.Tensor,
    total_epochs: tf.Tensor,
    tf_ratio_start: tf.Tensor,
    tf_ratio_end: tf.Tensor,
    scope_id: tf.Tensor,
    stop_grad_pred: bool = True,
) -> tf.Tensor:
    
    tf.print("\nRUNTIME scheduled_sampling_strategy")

    ratio = linear_ratio(epoch, total_epochs, tf_ratio_start, tf_ratio_end)
    return scheduled_sampling_next(
        y_true_t,
        y_pred_t,
        ratio,
        scope_id=scope_id,
        stop_grad_pred=stop_grad_pred,
    )
