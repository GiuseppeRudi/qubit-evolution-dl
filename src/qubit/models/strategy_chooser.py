import abc
import tensorflow as tf
import keras

from ..strategies.strategy_layer import StrategyLayer
from ..strategies.scheduled_sampling import scheduled_sampling_strategy
from ..strategies.teacher_forcing import teacher_forcing_step_wise, teacher_forcing_full_seq
from ..strategies.masked_modelling import masked_modeling_step_wise, masked_modeling_full_seq
from ..strategies.full_autoregressive import full_autoregressive_step_wise


class StrategyChooserModel(keras.Model, abc.ABC):
    def __init__(self, *, t_out: int, prediction_mode_id: int, **kwargs):
        super().__init__(**kwargs)
        self.rt = StrategyLayer(t_out=t_out, prediction_mode_id= prediction_mode_id)

    def apply_strategy_step_wise(self, 
                y_true_t: tf.Tensor,  # y_true_t.shape(batch_size,1,feature_dim)
                y_pred_t: tf.Tensor) -> tf.Tensor: # y_pred_t.shape(batch_size,1,feature_dim)

        # index of strategy 
        phase_index = tf.convert_to_tensor(self.rt.phase_id)
        
        def teacher_forcing():
            return teacher_forcing_step_wise(y_true_t = y_true_t)
        
        def masked_modeling():
            return masked_modeling_step_wise(
                y_true_t=y_true_t,
                mask_prob=tf.convert_to_tensor(self.rt.mask_prob, dtype=tf.float32),
                mask_mode_id=tf.convert_to_tensor(self.rt.mask_mode_id, dtype=tf.int32),
                mask_scope_id=tf.convert_to_tensor(self.rt.mask_scope_id, dtype=tf.int32),
                mask_value=tf.convert_to_tensor(self.rt.mask_value, dtype=tf.float32),
                noise_sigma=tf.convert_to_tensor(self.rt.noise_sigma, dtype=tf.float32),
                noise_replace=tf.convert_to_tensor(self.rt.noise_replace, dtype=tf.bool)
            )

        def scheduled_sampling():
            return scheduled_sampling_strategy(
                y_true_t,
                y_pred_t,
                epoch=tf.convert_to_tensor(self.rt.epoch_in_phase, dtype=tf.int32),
                total_epochs=tf.convert_to_tensor(self.rt.phase_epochs, dtype=tf.int32),
                tf_ratio_start=tf.convert_to_tensor(self.rt.tf_ratio_start, dtype=tf.float32),
                tf_ratio_end=tf.convert_to_tensor(self.rt.tf_ratio_end, dtype=tf.float32),
                per_feature=tf.convert_to_tensor(self.rt.per_feature, dtype=tf.int32),
                stop_grad_pred=tf.convert_to_tensor(self.rt.stop_grad_pred, dtype=tf.bool),
                ratio_mode_id=tf.convert_to_tensor(self.rt.ratio_mode, dtype=tf.int32),
                mid_point=tf.convert_to_tensor(self.rt.mid_point, dtype=tf.float32),
                sharpness=tf.convert_to_tensor(self.rt.sharpness, dtype=tf.float32),
                power_value=tf.convert_to_tensor(self.rt.power_value, dtype=tf.float32),
            )

        def full_autoregressive():
            return full_autoregressive_step_wise(
                y_pred_t=y_pred_t,
                gradient_through_time=tf.convert_to_tensor(self.rt.gradient_through_time, dtype=tf.bool),
            )

        out = tf.switch_case(
            phase_index,
            branch_fns={
                0: teacher_forcing,
                1: masked_modeling,
                2: scheduled_sampling,
                3: full_autoregressive
            }
        )

        # tf.print(out[0, :5, :5])

        out = out[:, :1, :]
        out = tf.ensure_shape(out, [None, 1, self.feature_dim])
        return out
    

    def apply_strategy_full_seq(
        self,
        y_true: tf.Tensor,              #  (batch_size , t_out, feature_dim)
        dec0: tf.Tensor            # initialization of decoder input at timestep = 0 
                                   # dec_in.shape(batch_size , 1 , feaure_dim)
    ) -> tf.Tensor:
        
        # index of strategy 
        phase_index = tf.convert_to_tensor(self.rt.phase_id)

        def teacher_forcing():
            return teacher_forcing_full_seq(
                Y = y_true,
                dec0 = dec0
            )

        def masked_modeling():
            return masked_modeling_full_seq(
                Y = y_true,
                dec0 = dec0,
                mask_prob=tf.convert_to_tensor(self.rt.mask_prob, dtype=tf.float32),
                mask_mode_id=tf.convert_to_tensor(self.rt.mask_mode_id, dtype=tf.int32),
                mask_scope_id=tf.convert_to_tensor(self.rt.mask_scope_id, dtype=tf.int32),
                mask_value=tf.convert_to_tensor(self.rt.mask_value, dtype=tf.float32),
                noise_sigma=tf.convert_to_tensor(self.rt.noise_sigma, dtype=tf.float32),
                noise_replace=tf.convert_to_tensor(self.rt.noise_replace, dtype=tf.bool)
            )
        
        dec_in = tf.switch_case(
            phase_index,
            branch_fns={
                0: teacher_forcing,
                1: masked_modeling
            },
            default=teacher_forcing,
        )
        # tf.print(dec_in[0, :5, :4])

        
        return dec_in