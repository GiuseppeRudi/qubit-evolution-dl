import tensorflow as tf
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser()

    # name of yaml file from we load the configuration
    parser.add_argument(
        "--run-cfg",
        type=str,
        required=True
    )

    # optional reuse of pre trained model weights
    parser.add_argument(
        "--model",
        type=str,
        default=None
    )

    # by default we perform the training if --no-training is not specified
    parser.add_argument(
        "--no-training",
        action="store_false",
        dest="training",
        default = True
    )

    args = parser.parse_args()

    # used only when argument --model is specified 
    # to indicate if we want to perform another training or only the inference 
    if not args.training and args.model is None:
        parser.error("--no-training can be used only together with --model ")
 
    return args


def get_device():

    # give the list of device 
    gpus = tf.config.list_physical_devices("GPU")
    
    if not gpus:
        print("No GPU found, using CPU.")

    print("GPU found:")
    for gpu in gpus:
        
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)
        print(f"   - {name} ")


# regex to match ANSI escape and remove that for log files
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# write to both console and file
class Logger:
    def __init__(self, console_stream, file_stream):

        # console => original stdout or stderr
        # file => log file
        self.console = console_stream
        self.file = file_stream

    def write(self, s: str):
        
        # console : write as it is 
        self.console.write(s)
        self.console.flush()

        # clean the ANSI codes for the file 
        s2 = _ANSI_RE.sub("", s).replace("\b", "")

        # ignore the update of progress bar 
        if "\r" in s2 and "\n" not in s2:
            return  # skip the code line with the carriage return only
        s2 = s2.replace("\r", "")
        
        self.file.write(s2)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

