from __future__ import annotations

from pathlib import Path
import optuna
from qubit.utils.config_keys import MODEL, TYPE
import tensorflow as tf
import gc
import optuna
import tensorflow as tf
from .suggest import suggest_level
from .score import compute_score
from tensorflow.keras.callbacks import Callback
from qubit.callbacks.optuna_pruning import OptunaPruningCallback
from main import run_experiment
from qubit.utils.config_loader import load_yaml

def main():
    
    # ! set the seed important
    seed = 42

    # name of the original config yaml 
    base_name = "debug_rudi"

    # name of the tuning experiment 
    study_name = "study_lstm_prova"
    
    # path of the experiments results
    out_root = Path("runs/tuning") / study_name
    out_root.mkdir(parents=True, exist_ok=True)

    # save the current state of study 
    storage = f"sqlite:///{(out_root / 'optuna.db').as_posix()}"

    # ? we can also try different samplers and pruners 
    
    # samplers tell us which type of parameter to try for the next trial 
    # bayesian approach
    sampler = optuna.samplers.TPESampler(seed=seed)

    # pruners decide to interrupt a trial if is going badly
    
    # MedianPruner compares the 'intermediate' performance of the current trial
    # with the median of trials already completed at the same epoch

    # n_startup_trials => tell at the pruner after how many trial start to prune
    # n_warmup_steps => number of epoch after that start to prune a current trial
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # container object
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # main configuration 
    base_cfg = load_yaml(base_name)

    def objective(trial):
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            model_type = base_cfg[MODEL][TYPE]
            override = suggest_level(trial, model_type=model_type, level=1)

            monitors: list[str] = ["_fr_phase_loss_"]

            callbacks : list[Callback] = [OptunaPruningCallback(trial, monitors)]

            metrics = run_experiment(
                base_name,
                override=override,
                out_dir=f"runs/tuning/study/trial_{trial.number:04d}",
                optuna_callback=callbacks,
                do_predict=False,
            )

            score = compute_score(metrics, base_cfg)

            return score

        except tf.errors.ResourceExhaustedError:
            trial.set_user_attr("failed_reason", "OOM")
            raise optuna.TrialPruned("OOM")
        
        except ValueError as e:
            trial.set_user_attr("failed_reason", f"ValueError: {e}")
            raise optuna.TrialPruned(f"Bad metrics/config: {e}")

        finally:
            tf.keras.backend.clear_session()
            gc.collect()


    study.optimize(objective, n_trials=40)

    df = study.trials_dataframe(("number", "value", "duration", "params", "user_attrs", "system_attrs", "state"))
    df.to_csv(out_root / "report.csv", index=False)

    print("Best trial:", study.best_trial.number, "score:", study.best_value)

if __name__ == "__main__":
    main()
