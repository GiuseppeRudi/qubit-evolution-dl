
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import keras
from src.qubit.core.data import load_or_prepare_dataset
from src.qubit.core.utils import get_device, parse_args
from src.qubit.core.config_loader import load_run_config
from src.qubit.registry import get_builder,get_trainer
from src.qubit.core.config_loader import load_model_config
from src.qubit.core.save import save_outputs

import src.qubit.rnn.builders  
import src.qubit.transformer.builders
import src.qubit.training.standard_trainer

def main():

    get_device()

    args = parse_args()
    run_cfg = load_run_config(args.run_cfg)
 
    splits, feat_names = load_or_prepare_dataset(run_cfg["data"])

    model_cfg = load_model_config(run_cfg["model"])
    
    if args.model is None:
        builder = get_builder(model_cfg.type, model_cfg.variant)
        model = builder(splits.X_train, splits.Y_train, model_cfg)
    else: model = keras.models.load_model(args.model)

    
    strategy = run_cfg["training"]["strategy"]
    TrainerCls = get_trainer(strategy)

    trainer = TrainerCls(model, model_cfg, eval_cfg=run_cfg.get("evaluation", {}))
    history = None
    
    if args.training is True :
        print(f"\n--- Training: {model_cfg.name} [{model_cfg.type}/{model_cfg.variant}] strategy={strategy} ---")
        history = trainer.fit(splits)

        print(f"Final loss: {history.history['loss'][-1]:.4f}")

    sample_x,sample_y, pred = trainer.predict_all_test(splits)
    trainer.report_sample(sample_x, sample_y, pred)
    eval_cfg = run_cfg.get("evaluation", {})

    import matplotlib.pyplot as plt
    if history is not None:
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss (TF)")
        plt.plot(history.history.get("val_loss", []), label="val_loss (TF)")

        # se callback abilitata:
        if "val_fr_loss" in history.history:
            plt.plot(history.history["val_fr_loss"], label="val_fr_loss (free-running)")

        plt.legend()
        plt.show()

    cmp = trainer.compare_losses(splits, split="val") 
    print("VAL compare:", cmp)

    cmp_test = trainer.compare_losses(splits, split="test")
    print("TEST compare:", cmp_test)

    save_outputs(splits, pred, model_cfg, feat_names, history, eval_cfg, model, model_cfg.save_model)  

if __name__ == "__main__":
    main()

# TODO RESOLVE ERROR : 

# TODO CAPIRE CHE FA CALL BACK IN STANDARD TRAINER
# CAMBIARE I MODEL PER IL FILE YAML USANGO TRAINING CON DIVERSE STRATEGIE 
# SISTEMARE STRATEIGWE 
# PLOTTARE IL CODICE DI MATPLOT CHE ABBIAMO SUL MAIN E LO SPOSTIAMO NEL FILE DATA FROM PLOT 


#     9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# Traceback (most recent call last):
#   File "/home/giu20/projects/qubit-evolution-dl/main.py", line 72, in <module>
#     main()
#   File "/home/giu20/projects/qubit-evolution-dl/main.py", line 63, in main
#     cmp = trainer.compare_losses(splits, split="val") 
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/giu20/projects/qubit-evolution-dl/src/qubit/training/standard_trainer.py", line 110, in compare_losses
#     fr_loss = self._scalar_loss(Y, pred_fr)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/giu20/projects/qubit-evolution-dl/src/qubit/training/standard_trainer.py", line 116, in _scalar_loss
#     v = self.model.loss(y_true, y_pred)        
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: 'str' object is not callable