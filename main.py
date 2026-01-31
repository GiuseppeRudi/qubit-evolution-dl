import tensorflow as tf
import os 

# REDUCE THE OOM 
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")



from src.qubit.enums.model_type import ModelType
from src.qubit.core.data import prepare_dataset
from src.qubit.utils.utils import *
from src.qubit.utils.config_loader import load_run_config, load_model_config, load_training_config, load_plot_config, load_data_config
from src.qubit.utils.registry import get_builder,get_trainer
import src.qubit.utils.config_keys as cfg_keys

from src.qubit.core.save import save_outputs
from src.qubit.utils.error import check_lstm_correctness, check_trn_correctness

# needed to register the models and trainers
import src.qubit.models.rnn.builders  
import src.qubit.models.trn.builders
import src.qubit.training.rnn_trainer
import src.qubit.training.trn_trainer

def run_experiment(
    run_cfg_path: str,
    *,
    override: dict | None = None,
    out_dir: str | None = None,
    optuna_callback: list[tf.keras.callbacks.Callback] | None = None,
    do_predict: bool = True,
    training: bool = True
) -> dict :
    """
    Esegue UN run completo (come main), ma:
    - accetta override dict (Optuna)
    - accetta extra_callbacks (pruning ecc.)
    - ritorna metriche finali in dict (per Optuna)
    """

    # --- tutto quello che oggi fai in main() ---
    run_cfg = load_run_config(run_cfg_path)


    if override:
        print(override)
        run_cfg = deep_merge_dict(run_cfg, override)

    data_cfg = load_data_config(run_cfg[cfg_keys.DATA])
    model_cfg = load_model_config(run_cfg[cfg_keys.MODEL])
    training_cfg = load_training_config(run_cfg[cfg_keys.TRAINING])
    plot_cfg = load_plot_config(run_cfg[cfg_keys.PLOT])

    if model_cfg.type == ModelType.LSTM:
        check_lstm_correctness(model_cfg,training_cfg,data_cfg)
    else:
        check_trn_correctness(model_cfg,training_cfg,data_cfg)

    splits, feat_names = prepare_dataset(data_cfg)

    logger = start_log()  # se vuoi supportare out_dir
    builder = get_builder(model_cfg.type, model_cfg.variant, model_cfg.decoder_mode)
    model = builder(splits.X_train, splits.Y_train, model_cfg, training_cfg.prediction_mode, model_path=None)

    TrainerCls = get_trainer(model_cfg.type)
    trainer = TrainerCls(model, model_cfg, training_cfg)

    history_dict : dict = {}  

    print(f"--- Number of Windows for Train = {splits.X_train.shape[0]}")
    print(f"--- Number of Windows for Val = {splits.X_val.shape[0]}")
    print(f"--- Number of Windows for Test = {splits.X_test.shape[0]}")

    if training:  # o args.training
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type.value}/{model_cfg.variant.value}/{model_cfg.decoder_mode.value}]  ---")
        history = trainer.fit(splits, optuna_callback)
        # print the final loss taken by the history object 
        history_dict = history.history
        print(f"Final loss: {history.history['loss'][-1]:.4f}")


    # predictions opzionali (in tuning puoi metterlo False per velocizzare)
    if do_predict:
        sample_x, sample_y, pred = trainer.predict_all_test(splits.X_test, splits.Y_test)
        trainer.report_sample(sample_x, sample_y, pred)

        save_outputs(splits, pred, model_cfg, feat_names, history, training_cfg.fr_eval.split + "_fr_", plot_cfg, training_cfg, splits.Y_train.shape[1], logger, run_cfg_path, model)  

    # IMPORTANT: fai ritornare un dict di metriche finali (FR + val_loss ecc.)
   
    return history_dict
    

def main():
    # print device information
    get_device()

    args = parse_args()
    run_experiment(args.run_cfg, do_predict=True, training=args.training)

if __name__ == "__main__":
    main()


# TODO scheduled sampling con noise injection 
# - name: scheduled_sampling_with_noise
#   epochs: 20
#   tf_ratio_start: 1.0
#   tf_ratio_end: 0.0
#   noise_injection:
#     enabled: true
#     noise_type: "adaptive"  # Cresce con la distanza dall'inizio
#     sigma_start: 0.0
#     sigma_end: 0.5  # ← Simula errori di magnitudine ~0.5

# TODO try the different parameters changing the yaml


# TODO check in all model if the flag training is correctly used 

