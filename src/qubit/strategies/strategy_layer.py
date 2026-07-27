import tensorflow as tf
from keras import layers


# this layer is used to store the runtime strategy parameters as non-trainable weights
class StrategyLayer(layers.Layer):
    def __init__(self, *, t_out: int, prediction_mode_id: int, name="strategy_runtime"):
        super().__init__(name=name)

        # keras.backend.Variable 
        # each of this "weights" => non-trainable variable => state variable => wrapper of tf.Variable

        # output_seq_len  
        self.t_out = self.add_weight(
            name = "t_out" , shape = (), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(t_out),
            trainable = False
        )

        # prediction_mode_id => ALL (0) | HORIZON (1)
        # used to choose if we want to do prediction on T_out (prediction_mode = "all")
        # or to choose if we want to do prediction on horizon (prediction_mode = "horizon")
        self.prediction_mode_id = self.add_weight(
            name = "prediction_mode_id" , shape = (), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(prediction_mode_id),
            trainable = False
        )
        
        # decoder_mode_id => FULL_SEQ (0) | STEP_WISE (1)
        # used to choose if decoder model work with full_seq or step_wise based on the current type of strategy
        # decoder_mod_id.assign(1) => currerent strategy : Teacher Forcing or Masked Modeling 
        # decoder_mod_id.assign(0) => currerent strategy : Scheduled Sampling  or Autoregressive
        self.decoder_mode_id = self.add_weight(
            name = "decoder_mode_id" , shape = (), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(0),
            trainable = False
        )
        
        # used to choose the different type of strategy using the id in the graph mode 
        self.phase_id = self.add_weight(
            name="phase_id", shape=(), dtype=tf.int32,
            initializer="zeros", trainable=False
        )

        # horizon related to the current strategy => if it is equal to -1 we use the output_seq_len
        self.horizon = self.add_weight(
            name="horizon", shape=(), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(0),
            trainable=False
        )

        # total number of epoch for the current strategy => Es. Teacher forcing => 5 epochs 
        self.phase_epochs = self.add_weight(
            name="phase_epochs", shape=(), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(0),
            trainable=False
        )

        # current number of epoch for current strategy 
        # Es. Teacher forcing => total epoch => 5 but current is the 2/5
        self.epoch_in_phase = self.add_weight(
            name="epoch_in_phase", shape=(), dtype=tf.int32,
            initializer="zeros", trainable=False
        )

        #######################################################
        # Scheduled Sampling

        # for the first epoch we start use a percentage of high teacher forcing => more help 
        self.tf_ratio_start = self.add_weight(
            name="tf_ratio_start", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        # epoch by epoch we reduce the help tf_ratio, based on the current epoch, 
        # to arrive to the final epoch where the tf_ratio = tf_ratio_end
        self.tf_ratio_end = self.add_weight(
            name="tf_ratio_end", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        # true = different choice for each feature
        # false = same choice for all the features
        self.per_feature = self.add_weight(
            name="per_feature", shape=(), dtype=tf.int32,
            initializer="zeros",
            trainable=False
        )

        self.stop_grad_pred = self.add_weight(
            name="stop_grad_pred", shape=(), dtype=tf.bool,
            initializer="zeros",
            trainable=False
        )

        self.ratio_mode = self.add_weight(
            name="ratio_mode", shape=(), dtype=tf.int32,
            initializer=tf.keras.initializers.Constant(0),
            trainable=False
        )

        self.mid_point = self.add_weight(
            name="mid_point", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        self.sharpness = self.add_weight(
            name="sharpness", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        self.power_value = self.add_weight(
            name="power_value", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        #######################################################
        # Masked Modeling

        # pencentage of masking 
        self.mask_prob = self.add_weight(
            name="mask_prob", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        # mask_mode can be  zero (index : 0) | constant (index: 1) | noise (index: 2)
        # we used the index for a tf.switch_case 
        self.mask_mode_id = self.add_weight(  # enum -> int
            name="mask_mode_id", shape=(), dtype=tf.int32,
            initializer="zeros", trainable=False
        )

        # mask_scope can be  time (index : 0) | feature (index: 1) | element (index: 2)
        # we used the index for a tf.switch_case 
        self.mask_scope_id = self.add_weight(
            name="mask_scope_id", shape=(), dtype=tf.int32,
            initializer="zeros", trainable=False
        )

        # if mask_mode is equal to COSTANT so the mask_value variable store the constant 
        self.mask_value = self.add_weight(
            name="mask_value", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        # if mask_mode is equal to NOISE the noise_sigma variable store the pencentage of noise 
        self.noise_sigma = self.add_weight(
            name="noise_sigma", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False
        )

        self.noise_replace = self.add_weight(
            name="noise_replace", shape=(), dtype=tf.bool,
            initializer="zeros",
            trainable=False
        )

        #######################################################
        # Autoregressive 

        # GTT true or false 
        self.gradient_through_time = self.add_weight(
            name="gradient_through_time", shape=(), dtype=tf.bool,
            initializer="zeros",
            trainable=False
        )
