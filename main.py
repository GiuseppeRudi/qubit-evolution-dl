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


def demo_matmul(device_str: str):
    """Small computation example to verify that everything works."""
    print(f"Running demo_matmul on {device_str}...")
    with tf.device(device_str):
        a = tf.random.normal((2000, 2000))
        b = tf.random.normal((2000, 2000))
        c = tf.matmul(a, b)
    print("Operation completed. Result shape:", c.shape)


def main():
    device = get_device()
    demo_matmul(device)


if __name__ == "__main__":
    main()
