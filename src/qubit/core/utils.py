import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--run-cfg",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None
    )

    parser.add_argument(
        "--no-training",
        action="store_false",
        dest="training"
    )
    
    parser.set_defaults(training=True)

    return parser.parse_args()

def get_device():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found, using CPU.")
        return "/CPU:0"

    print("GPU found:")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)

        cc = details.get("compute_capability")  
        if isinstance(cc, (tuple, list)) and len(cc) == 2:
            cc_str = f"{cc[0]}.{cc[1]}"
        elif cc is not None:
            cc_str = str(cc)
        else:
            cc_str = "unknown"

        print(f"   - {name} (compute capability {cc_str})")

    return "/GPU:0"


import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

class Tee:
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream

    def write(self, s: str):
        # 1) console: lascia tutto com’è
        self.console.write(s)
        self.console.flush()

        # 2) file: ripulisci ANSI + ignora gli update di progress bar (carriage return senza newline)
        s2 = _ANSI_RE.sub("", s).replace("\b", "")
        if "\r" in s2 and "\n" not in s2:
            return  # skip “refresh” della progressbar
        s2 = s2.replace("\r", "")
        self.file.write(s2)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

