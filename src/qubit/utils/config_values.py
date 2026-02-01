from pathlib import Path
from typing import Literal

# training.fr_eval.probes[] item values
END_OF_PHASE = Literal["end_of_phase"] #######
OUT_STEPS_SPEC = Literal["phase", "global"] #######

# logs
LOG_PATH = "train.log"
PREDICTION_PATH = "predictions"
RUN_PATH = "runs"