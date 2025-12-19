import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


def get_device():
    """Returns '/GPU:0' if at least one GPU is available, otherwise '/CPU:0'."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("🖥️ No GPU found, using CPU.")
        return "/CPU:0"

    print("✅ GPU found:")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)

        cc = details.get("compute_capability")  # spesso è tipo (8, 6)
        if isinstance(cc, (tuple, list)) and len(cc) == 2:
            cc_str = f"{cc[0]}.{cc[1]}"
        elif cc is not None:
            cc_str = str(cc)
        else:
            cc_str = "unknown"

        print(f"   - {name} (compute capability {cc_str})")

    return "/GPU:0"
