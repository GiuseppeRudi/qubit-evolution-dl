# -------------------------
# Top-level sections (CONFIGS)
# -------------------------
DATA = "data" ########
MODEL = "model" ########
TRAINING = "training" ########
PLOT = "plot" ########

# -------------------------
# data.*
# -------------------------
DATASET = "dataset" ########
WINDOWING = "windowing" ########
DATA_SPLIT = "split" ########
TASK = "task" ########
SUPER_RESOLUTION = "sr" ########

# data.dataset.*
CSV_PATH = "csv_path"
TOTAL_QUBITS = "total_qubits"
USED_QUBITS = "used_qubits"
TIME_STEPS = "time_steps"
N_TRAJ = "n_traj"
TRAJ_FRACTION = "traj_fraction"

# data.windowing.*
INPUT_SEQ_LEN = "input_seq_len"
OUTPUT_SEQ_LEN = "output_seq_len"
STRIDE = "stride"

# data.split.*
SEED = "seed"
VAL_RATIO = "val_ratio"
TEST_RATIO = "test_ratio"

# -------------------------
# model.*
# -------------------------
MODEL_NAME = "name" ########
SAVE_MODEL = "save_model" ########
RETURN_ATTENTIONS = "return_attentions"
MODEL_TYPE = "type" ########
VARIANT = "variant" ########
DECODER_MODE = "decoder_mode" ########
PARAMS = "params" ########
COMPILE = "compile" ########
INFERENCE = "inference" ########

# model.params.*
LATENT_DIM = "latent_dim"

# model.compile.*
OPTIMIZER = "optimizer"
LOSS = "loss"
METRICS = "metrics"
RUN_EAGERLY = "run_eagerly"
LEARNING_RATE = "learning_rate" ########
CLIP_NORM = "clip_norm" ########

# model.inference.*
MODE = "mode"
START_MODE = "start_mode"
INFERENCE_VERBOSE = "verbose"

# -------------------------
# training.*
# -------------------------
PREDICTION_MODE = "prediction_mode" ########
TRAINING_VERBOSE = "verbose" ########
BATCH_SIZE = "batch_size" ########
CURRICULUM = "curriculum" ########
PHASES = "phases" ########
FR_EVAL = "fr_eval" ########

# training.phases[] item keys
PHASE_NAME = "name" ########
EPOCHS = "epochs"
MASK_PROB = "mask_prob"
MASK_SCOPE = "mask_scope"
MASK_MODE = "mask_mode"
MASK_VALUE = "mask_value"
NOISE_SIGMA = "noise_sigma"

# training.fr_eval.*
ENABLED = "enabled" ########
FR_EVAL_SPLIT = "split" ########
PROBES = "probes" ########
FR_BATCH_SIZE = "batch_size" ########

# training.fr_eval.probes[] item keys
PROBE_NAME = "name" ########
OUT_STEPS = "out_steps" ########
EVERY_EPOCHS = "every_epochs" ########
P_EVAL = "p_eval" ########

# -------------------------
# plot.*
# -------------------------
SAMPLE_INDEX = "sample_index" ########
SAVE_PLOTS = "save_plots" ########
SAVE_ARTIFACTS = "save_artifacts" ########

# -------------------------
# sr.*
# -------------------------
SR_STRIDE = "stride" ########
OFFSET = "offset" ########
SR_MASK_VALUE = "mask_value" ########
LOSS_ON = "loss_on" ########
OBSERVED_WEIGHT = "observed_weight" ########

##########################################################
# TUNING

TUNING = "tuning" ########
OPTUNA_PATH = "optuna" ########
# -------------------------
# Top-level sections (TUNING)
# -------------------------
STUDY_NAME = "study_name" ########
SEED = "seed" ########
N_TRIALS = "n_trials" ########
LEVEL = "level" ########
BASE_NAME = "base_name" ########
MONITORS = "monitors" ########
OUTPUT = "output" ########
SAMPLER = "sampler" ########
PRUNER = "pruner" ########

# -------------------------
# output.*
# -------------------------
ROOT_DIR = "root_dir" ########
STORAGE_FILENAME = "storage_filename" ########
REPORT_FILENAME = "report_filename" ########

# -------------------------
# sampler.*
# -------------------------
SAMPLER_TYPE = "type" ########

# -------------------------
# pruner.*
# -------------------------
PRUNER_TYPE = "type" ########
N_STARTUP_TRIALS = "n_startup_trials" ########
N_WARMUP_STEPS = "n_warmup_steps" ########