import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os

from src.qubit.core.data import load_or_prepare_dataset
from src.qubit.core.utils import get_device, parse_args
from src.qubit.core.config_loader import load_run_config
from src.qubit.registry import get_builder,get_trainer
from src.qubit.core.config_loader import load_model_config
from src.qubit.core.save import save_outputs
import src.qubit.rnn.builders  
import src.qubit.transformer.builders
import qubit.training.standard_trainer

def main():

    get_device()

    args = parse_args()
    run_cfg = load_run_config(args.run_cfg)
 
    splits = load_or_prepare_dataset(run_cfg["data"])

    model_cfg = load_model_config(run_cfg["model"])

    builder = get_builder(model_cfg.type, model_cfg.variant)
    model = builder(splits.X_train, splits.Y_train, model_cfg)


    strategy = run_cfg["training"]["strategy"]
    TrainerCls = get_trainer(strategy)

    trainer = TrainerCls(model, model_cfg, eval_cfg=run_cfg.get("evaluation", {}))
    
    print(f"\n--- Training: {model_cfg.name} [{model_cfg.type}/{model_cfg.variant}] strategy={strategy} ---")
    history = trainer.fit(splits)

    print(f"Final loss: {history.history['loss'][-1]:.4f}")

    sample_x, pred = trainer.predict_sample(splits)
    trainer.report_sample(sample_x, pred)
    eval_cfg = run_cfg.get("evaluation", {})


    save_outputs(splits, pred, model_cfg, eval_cfg )


    #TODO create a mini plot for the validation error 

if __name__ == "__main__":
    main()