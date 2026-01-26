import tensorflow as tf
import argparse
import sys
import re
from pathlib import Path

from  .config_values import LOG_PATH

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
 
class BufferedLogger:

    def __init__(self, console_stream):
        
        # console => original stdout or stderr
        self.console = console_stream

        # buffer
        self.lines: list[str] = []
 
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

        # append to the buffer
        self.lines.append(s2)
 
    def flush(self):
        self.console.flush()
 
    def dump_to_file(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(self.lines)

 
def start_log() -> BufferedLogger:
    logger = BufferedLogger(sys.__stdout__)
    sys.stdout = logger
    # se vuoi anche stderr:
    # sys.stderr = logger
    return logger
 
 
def finish_log(logger: BufferedLogger, run_dir: Path):
    log_path = run_dir / LOG_PATH
    logger.dump_to_file(log_path)

    # restore
    sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__