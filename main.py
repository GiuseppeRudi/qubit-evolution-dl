import keras
from src.qubit.core.data import prepare_dataset
from src.qubit.utils.utils import get_device, parse_args, start_log
from src.qubit.utils.config_loader import load_run_config, load_model_config, load_training_config, load_plot_config, load_data_config
from src.qubit.utils.registry import get_builder,get_trainer
import src.qubit.utils.config_keys as cfg_keys

from src.qubit.core.save import save_outputs, make_run_output_dir
from src.qubit.utils.error import check_correctness

# needed to register the models and trainers
import src.qubit.models.rnn.builders  
import src.qubit.models.trn.builders
import src.qubit.training.rnn_trainer
import src.qubit.training.trn_trainer


def main():
    # print device information
    get_device()

    # parse command line arguments 
    args = parse_args()

    run_cfg = load_run_config(args.run_cfg)
 
    data_cfg = load_data_config(run_cfg[cfg_keys.DATA])
    model_cfg = load_model_config(run_cfg[cfg_keys.MODEL])
    training_cfg = load_training_config(run_cfg[cfg_keys.TRAINING])
    plot_cfg = load_plot_config(run_cfg[cfg_keys.PLOT])

    check_correctness(model_cfg,training_cfg,data_cfg)

    splits, feat_names = prepare_dataset(data_cfg)

    logger = start_log()

    # use the get_builder function to get the specific model builder function (Callable)
    builder = get_builder(model_cfg.type, model_cfg.variant,model_cfg.decoder_mode)
    
    # after we call the function with the parameters to build the model 
    model = builder(splits.X_train, splits.Y_train, model_cfg, training_cfg.prediction_mode, args.model)

    # use get_trainer to get the specific trainer class 
    TrainerCls = get_trainer(model_cfg.type)

    # after we instantiate the trainer 
    trainer = TrainerCls(model, model_cfg, training_cfg)
    
    # useful because if we used the pretrained model there is a possibility to don't perform again the training phase
    history = None
    
    print(f"--- Number of Windows for Train = {splits.X_train.shape[0]}")
    print(f"--- Number of Windows for Val = {splits.X_val.shape[0]}")
    print(f"--- Number of Windows for Test = {splits.X_test.shape[0]}")
    
    # default it is true if we don't specify 
    if args.training is True :
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type.value}/{model_cfg.variant.value}/{model_cfg.decoder_mode.value}]  ---")
        
        history = trainer.fit(splits)

        # print the final loss taken by the history object 
        print(f"Final loss: {history.history['loss'][-1]:.4f}")

    sample_x, sample_y, pred = trainer.predict_all_test(splits.X_test, splits.Y_test)

    # Referred to test split 
    # sample_x.shape(num_windows, input_seq_len, feature_dim)
    # sample_y.shape(num_windows, output_seq_len, feature_dim)
    # pred.shape(num_windows, output_seq_len, feature_dim)

    # print some predictions  
    trainer.report_sample(sample_x, sample_y, pred)
    
    save_outputs(splits, pred, model_cfg, feat_names, history, training_cfg.fr_eval.split + "_fr_", plot_cfg, training_cfg, splits.Y_train.shape[1], logger, args.run_cfg, model)  

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


