import tensorflow as tf
import os 

# REDUCE THE OOM 
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

from src.qubit.enums.model_type import ModelType
from src.qubit.core.data import prepare_dataset
from src.qubit.utils.utils import *
from src.qubit.utils.config_loader import load_run_config, load_model_config, load_training_config, load_plot_config, load_data_config, load_sr_config
from src.qubit.utils.registry import get_builder,get_trainer
from src.qubit.utils.config_values import PREDICTION_PATH
import src.qubit.utils.config_keys as cfg_keys

from src.qubit.core.save import save_outputs
from src.qubit.utils.error import check_correctness

from src.qubit.enums.model_variant import ModelVariant
# needed to register the models and trainers
import src.qubit.models.rnn.builders  
import src.qubit.models.trn.builders
import src.qubit.training.rnn_trainer
import src.qubit.training.trn_trainer

def run_experiment(
    yaml_name: str,
    *,
    override: dict | None = None,
    out_dir: str,
    optuna_callback: list[tf.keras.callbacks.Callback] | None = None,
    do_predict: bool = True,
    training: bool = True
) -> dict :

    run_cfg = load_run_config(yaml_name)

    if override:
        print(override)
        run_cfg = deep_merge_dict(run_cfg, override)

    data_cfg = load_data_config(run_cfg[cfg_keys.DATA])
    model_cfg = load_model_config(run_cfg[cfg_keys.MODEL])

    sr_cfg = None
    if model_cfg.variant == ModelVariant.SUPER_RESOLUTION: 
        sr_cfg = load_sr_config(run_cfg[cfg_keys.SUPER_RESOLUTION])

    training_cfg = load_training_config(run_cfg[cfg_keys.TRAINING],model_cfg.variant)
    
    plot_cfg = load_plot_config(run_cfg[cfg_keys.PLOT])

    check_correctness(model_cfg,training_cfg,data_cfg)

    splits, feat_names,mean, std = prepare_dataset(data_cfg, sr_cfg)

    logger = start_log() 
    builder = get_builder(model_cfg.type, model_cfg.variant, model_cfg.decoder_mode)

    model = builder(splits.X_train, splits.Y_train, model_cfg, sr_cfg or training_cfg.prediction_mode , model_path=None)

    TrainerCls = get_trainer(model_cfg.type)
    trainer = TrainerCls(model, model_cfg, training_cfg)

    history_dict : dict = {}  

    print(f"--- Number of Windows for Train = {splits.X_train.shape[0]}")
    print(f"--- Number of Windows for Val = {splits.X_val.shape[0]}")
    print(f"--- Number of Windows for Test = {splits.X_test.shape[0]}")
    print(f"--- Curriculum = {[round(float(x) * data_cfg.windowing.output_seq_len) for x in training_cfg.curriculum]}")
    if training_cfg.fr_eval.enabled:
        for p in training_cfg.fr_eval.probes:
            if p.name == "fr_curve":
                curve_steps = [round(float(x) * data_cfg.windowing.output_seq_len) for x in p.out_steps]
        print(f"--- fr_curve.outsteps = {curve_steps}")
    
    # TODO put ifs for the prints (if superesolution, if fr_eval etc.)

    if training:  
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type.value}/{model_cfg.variant.value}/{model_cfg.decoder_mode.value}]  ---")
        history = trainer.fit(splits, optuna_callback)
        history_dict = history.history
        print(f"Final loss: {history.history['loss'][-1]:.4f}")

    if model_cfg.return_attentions: 
        _, attn = trainer.extract_attention_maps(splits, sample_index=plot_cfg.sample_index[0])

    if do_predict:
        sample_x, sample_y, pred = trainer.predict_all_test(splits.X_test, splits.Y_test)
        trainer.report_sample(sample_x, sample_y, pred)

        save_outputs(splits, pred, model_cfg, feat_names, 
                     history, plot_cfg, 
                     training_cfg, logger, 
                     yaml_name, out_dir, 
                     mean, std, attn or None, model)  
   
    return history_dict
    

def main():
    # print device information
    get_device()

    args = parse_args()
    out_dir = PREDICTION_PATH
    run_experiment(args.run_cfg, do_predict=True, out_dir=out_dir, training=args.training)

if __name__ == "__main__":
    main()

# TODO check in all model if the flag training is correctly used 

# TODO try comparison between models