import keras
from src.qubit.core.data import load_or_prepare_dataset
from src.qubit.core.utils import get_device, parse_args
from src.qubit.core.config_loader import load_run_config, load_model_config, load_training_config, load_plot_config
from src.qubit.registry import get_builder,get_trainer
from src.qubit.core.save import save_outputs

import src.qubit.rnn.builders  
import src.qubit.transformer.builders
import src.qubit.training.standard_trainer
import src.qubit.training.rnn_trainer
import src.qubit.training.trn_trainer


def main():

    get_device()

    args = parse_args()
    run_cfg = load_run_config(args.run_cfg)
 
    splits, feat_names = load_or_prepare_dataset(run_cfg["data"])

    model_cfg = load_model_config(run_cfg["model"])
    
    if args.model is None:
        builder = get_builder(model_cfg.type, model_cfg.variant,model_cfg.decoder_mode)
        model = builder(splits.X_train, splits.Y_train, model_cfg)
    else: model = keras.models.load_model(args.model)

    training_cfg = load_training_config(run_cfg["training"])

    TrainerCls = get_trainer(model_cfg.type)

    trainer = TrainerCls(model, model_cfg, training_cfg)
    history = None
    
    if args.training is True :
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type.value}/{model_cfg.variant.value}]  ---")
        history = trainer.fit(splits)

        print(f"Final loss: {history.history['loss'][-1]:.4f}")

    sample_x,sample_y, pred = trainer.predict_all_test(splits)
    trainer.report_sample(sample_x, sample_y, pred)
    

    plot_cfg = load_plot_config(run_cfg["plot"])

  
    save_outputs(splits, pred, model_cfg, feat_names, history, plot_cfg, model)  

if __name__ == "__main__":
    main()



# TODO
# create a model seq2seq lstm one seq using a global rnn infernece model with a wrapper 
# implement the diffent options for the masked modelling strategy 

# TODO create functions in error.py that will check if in the config loader insert the correct parameters otherwise catch the error

# TODO  create a different plot configurations because we changed the history object with more informations 