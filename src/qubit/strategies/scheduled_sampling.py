import tensorflow as tf
import math


@tf.function
def _alpha(epoch, total_epochs) -> tf.Tensor:
    # epoch 0-based, alpha in [0,1]
    den = tf.maximum(tf.cast(total_epochs, tf.int32) - 1, 1)
    return tf.cast(epoch, tf.float32) / tf.cast(den, tf.float32)


@tf.function
def linear_ratio(epoch, total_epochs, start, end) -> tf.Tensor:

    # LINEAR: ratio changes with a constant step from start to end.
    # Same amount of change every epoch (straight line).

    a = _alpha(epoch, total_epochs)
    return start + (end - start) * a

@tf.function
def cosine_ratio(epoch, total_epochs, start, end) -> tf.Tensor:

    # COSINE: ratio changes smoothly from start to end following a cosine curve.
    # It moves slowly at the beginning and at the end, and faster in the middle.

    a = _alpha(epoch, total_epochs)
    pi = tf.constant(math.pi, dtype=tf.float32)
    s = 0.5 * (1.0 - tf.cos(pi * a))
    return start + (end - start) * s

@tf.function
def power_ratio(epoch, total_epochs, start, end, power_value) -> tf.Tensor:

    # POWER: ratio follows a power curve from start to end.
    # power_value > 1 -> slow change early, faster later.
    # power_value < 1 -> fast change early, slower later.

    a = _alpha(epoch, total_epochs)
    s = tf.pow(a, power_value)
    return start + (end - start) * s


@tf.function
def sigmoid_ratio(epoch, total_epochs, start, end, mid_point, sharpness) -> tf.Tensor:
   
    # SIGMOID: ratio follows an S-shaped curve from start to end.
    # mid_point decides when the change happens (0..1 of training).
    # sharpness decides how fast the change is (bigger = more sudden).

    a = _alpha(epoch, total_epochs)
    z = (a - mid_point) * sharpness
    sig = tf.sigmoid(z)
    return start + (end - start) * sig


@tf.function
def scheduled_sampling_next(
    y_true_t: tf.Tensor,  # y_true_t.shape(batch_size, 1, feature_dim)
    y_pred_t: tf.Tensor,  # y_true_t.shape(batch_size, 1, feature_dim)
    ratio: tf.Tensor,
    *,
    per_feature: tf.Tensor, 
    stop_grad_pred: tf.Tensor,
) -> tf.Tensor:

    if stop_grad_pred:
        y_pred_t = tf.stop_gradient(y_pred_t)

    batch_size = tf.shape(y_true_t)[0]
    feature_dim = tf.shape(y_true_t)[2]

    # per_feature = 0 -> broadcast (batch_size, 1, feature_dim = 1),
    # per_feature = 1 -> per-feature (batch_size, 1, feature_dim)
   
    shape = tf.switch_case(
        per_feature,
        branch_fns={
            0: lambda: tf.stack([batch_size, 1, 1]),  # same choice for all feature
            1: lambda: tf.stack([batch_size, 1, feature_dim]),  # different choice for each feature
        },
        default=lambda: tf.stack([batch_size, 1, 1]),
    )

    use_teacher = tf.random.uniform(shape, 0.0, 1.0) < ratio
    return tf.where(use_teacher, y_true_t, y_pred_t)

@tf.function
def scheduled_sampling_strategy(
    y_true_t: tf.Tensor,  # y_true_t.shape(batch_size, 1, feature_dim)
    y_pred_t: tf.Tensor,  # y_true_t.shape(batch_size, 1, feature_dim)
    *,
    epoch: tf.Tensor,
    total_epochs: tf.Tensor,
    tf_ratio_start: tf.Tensor,
    tf_ratio_end: tf.Tensor,
    per_feature: tf.Tensor, # True = 1 , False = 0 
    stop_grad_pred: tf.Tensor,
    ratio_mode_id: tf.Tensor,
    mid_point: tf.Tensor,
    sharpness: tf.Tensor,
    power_value: tf.Tensor
) -> tf.Tensor:
    
    # tf.print("\nRUNTIME scheduled_sampling_strategy")

    def _linear_ratio():
        return linear_ratio(epoch,total_epochs,tf_ratio_start,tf_ratio_end)
    
    def _cosine_ratio():
        return cosine_ratio(epoch,total_epochs,tf_ratio_start,tf_ratio_end)
    
    def _sigmoid_ratio():
        return sigmoid_ratio(epoch,total_epochs,tf_ratio_start,tf_ratio_end,mid_point,sharpness)
    
    def _power_ratio():
        return power_ratio(epoch,total_epochs,tf_ratio_start,tf_ratio_end,power_value)

    ratio = tf.switch_case(
        ratio_mode_id,
        branch_fns={
            0: _linear_ratio,
            1: _cosine_ratio,
            2: _sigmoid_ratio,
            3: _power_ratio
        },
        default=_linear_ratio,
    )

    return scheduled_sampling_next(
        y_true_t,
        y_pred_t,
        ratio,
        per_feature=per_feature,
        stop_grad_pred=stop_grad_pred,
    )
