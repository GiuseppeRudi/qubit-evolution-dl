import tensorflow as tf

def get_device():
    """Returns '/GPU:0' if at least one GPU is available, otherwise '/CPU:0'."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU found, using GPU:")
        for gpu in gpus:
            print("   -", gpu)
        return "/GPU:0"
    else:
        print("⚠️ No GPU found, running on CPU.")
        return "/CPU:0"
