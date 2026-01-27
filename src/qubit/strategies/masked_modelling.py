import tensorflow as tf


# TODO possibility to change the step_wise to choose if we want to apply the mask also the dec0 
# in full seq this change is easier 


@tf.function
def make_mask(
    dec_in: tf.Tensor, # (batch_size , t, feature_dim)
    mask_prob: tf.Tensor, # scalar value beetwen (0,1)
    mask_scope_id: tf.Tensor,  # id of mask scope 
) -> tf.Tensor:
    
    B = tf.shape(dec_in)[0]
    T = tf.shape(dec_in)[1]
    F = tf.shape(dec_in)[2]

    # Mask Scope => time (0) , feature (1) , element (2)

    shape = tf.switch_case(
        mask_scope_id,
        branch_fns={
            0: lambda: tf.stack([B, T, 1]),  # TIME: variable mask for all the timesteps and the same for each feature
            1: lambda: tf.stack([B, 1, F]),  # FEATURE: variable mask for all the features and the same for each timestep
            2: lambda: tf.stack([B, T, F]),  # ELEMENT: variable mask for all
        }, # the dimensions with 1, after are broadcasted into a specific dimension of dec_in
        default = lambda: tf.stack([B, T, F]),
    )

    # create a float array with uniform random values in [0,1)
    u = tf.random.uniform(shape, 0.0, 1.0) 

    # create the bool mask array where each element is True with probability approx mask_prob

    # u[i] < mask_prob => m[i] = True else false
    m = u < mask_prob
    return m
    

@tf.function
def apply_mask(
    dec_in: tf.Tensor,             
    m: tf.Tensor,              # broadcastable to x
    mask_mode_id: tf.Tensor,   # scalar int
    mask_value: tf.Tensor,     # scalar float
    noise_sigma: tf.Tensor,    # scalar float
    noise_replace : tf.Tensor  # bool 
) -> tf.Tensor:

    # dec_in.shape(batch_size, timesteps, feature_dim) if use FULL_SEQ
    # dec_in.shape(batch_size, 1, feature_dim)         if use STEP_WISE

    # m.shape(batch_size, timesteps, 1) => if use TIME mode
    # m.shape(batch_size, 1 , feature_dim) => if use FEAUTURE mode
    # m.shape(batch_size, timesteps, feature_dim) => if use ELEMENT mode
    
    # ? we apply the mask also in timestep = 0 so in dec0

    def mode_zero():
        # for each element where  m[i] == True so dec_in[i] =  0
        # for each element where  m[i] == False so dec_in[i] =  dec_in[i] (original value)

        return tf.where(m, tf.zeros_like(dec_in), dec_in)

    def mode_constant():
        # for each element where  m[i] == True so dec_in[i] =  mask_value
        # for each element where  m[i] == False so dec_in[i] =  dec_in[i] (original value)

        return tf.where(m, tf.ones_like(dec_in) * mask_value, dec_in)
    
    def mode_noise():
        noise = tf.random.normal(tf.shape(dec_in), 0.0, noise_sigma, dtype=dec_in.dtype)

        def replace():
            # for each element where  m[i] == True so dec_in[i] =  noise
            # for each element where  m[i] == False so dec_in[i] =  dec_in[i] (original value)

            return tf.where(m, noise, dec_in)

        def additive():
            # for each element where  m[i] == True so dec_in[i] =  dec_in[i] + noise
            # for each element where  m[i] == False so dec_in[i] =  dec_in[i] (original value)

            return tf.where(m, dec_in + noise, dec_in)

        return tf.cond(noise_replace, replace, additive)

    return tf.switch_case(
        mask_mode_id,
        branch_fns={
            0: mode_zero,
            1: mode_constant,
            2: mode_noise,
        },
        default=lambda: dec_in,
    )


@tf.function
def masked_modeling_full_seq(
    Y: tf.Tensor,
    dec0: tf.Tensor,
    mask_prob: tf.Tensor,
    mask_mode_id: tf.Tensor,
    mask_scope_id: tf.Tensor,
    mask_value: tf.Tensor,
    noise_sigma: tf.Tensor,
    noise_replace : tf.Tensor
):
    
    # ! REMEMBER the masking modeling use always the groun truth for the input 
    # Y if prediction mode => ALL => Y.shape = (B, t = t_out == output_seq_len, F)
    # Y if prediction mode => HORIZON => Y.shape = (B, t = t_hor, F)

    # dec0.shape(batch_size , 1 , feature_dim)
    
    # we lose Y[-1] not used because at time t we use Y[t-1] as input
    Y_truncated = Y[:, :-1, :]  # (batch_size , t - 1 , feature_dim)

    # dec_in.shape[1] = dec0.shape[1] + Y_truncated.shape[1]  => t
    dec_in = tf.concat([dec0, Y_truncated], 1) # (batch_size, t, feature_dim)

    m = make_mask(dec_in, mask_prob, mask_scope_id)
    return apply_mask(dec_in, m, mask_mode_id, mask_value, noise_sigma,noise_replace)




@tf.function
def masked_modeling_step_wise(
    y_true_t: tf.Tensor,       # (batch_size , 1 , feature_dim)
    mask_prob: tf.Tensor,
    mask_mode_id: tf.Tensor,
    mask_scope_id: tf.Tensor,
    mask_value: tf.Tensor,
    noise_sigma: tf.Tensor,
    noise_replace : tf.Tensor,
) -> tf.Tensor:
    # tf.print("\nRUNTIME masked_modeling_step_wise")

    # ! REMEMBER the masking modeling use always the groun truth for the input 

    m = make_mask(y_true_t, mask_prob, mask_scope_id)
    return apply_mask(y_true_t, m, mask_mode_id, mask_value, noise_sigma, noise_replace)

