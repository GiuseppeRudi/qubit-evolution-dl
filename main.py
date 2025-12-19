import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os

from src.qubit.core.data import load_or_prepare_dataset
from src.qubit.core.utils import get_device, parse_args
from src.qubit.core.config_loader import load_run_config
from src.qubit.registry import get_builder,get_trainer
from src.qubit.core.config_loader import load_model_config
import src.qubit.rnn.builders  
import src.qubit.transformer.builders
import src.qubit.training.trainers

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
    '''
    # 5) plots
    eval_cfg = run_cfg.get("evaluation", {})
    if eval_cfg.get("save_plots", True):
        pred_dir = eval_cfg.get("predictions_dir", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        generate_all_plots(splits, transformer_prediction=pred if model_cfg.type=="TRN" else None,
                           rnn_prediction=pred if model_cfg.type=="RNN" else None,
                           sample_index=int(eval_cfg.get("sample_index", 0)))
    '''


if __name__ == "__main__":
    main()