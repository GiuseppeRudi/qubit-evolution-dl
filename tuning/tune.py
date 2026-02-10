from __future__ import annotations

from pathlib import Path
from typing import Any
import optuna
from qubit.utils.config_keys import BASE_NAME, LEVEL, MODEL, MODEL_TYPE, MONITORS, N_STARTUP_TRIALS, N_TRIALS, N_WARMUP_STEPS, OUTPUT, PRUNER, PRUNER_TYPE, ROOT_DIR, SAMPLER, SAMPLER_TYPE, SEED, STORAGE_FILENAME, STUDY_NAME, REPORT_FILENAME, OPTUNA_PATH, TUNING, TRAINING, CURRICULUM,DATA,WINDOWING,OUTPUT_SEQ_LEN
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



def build_sampler(cfg: dict[str, Any], seed: int, level: int) -> optuna.samplers.BaseSampler:
    sampler_cfg = cfg[SAMPLER]
    sampler_type = sampler_cfg[SAMPLER_TYPE]

    if level == 1:
        if sampler_type != "tpe":
            raise ValueError("Level1 supports sampler.type='tpe' only (for now)")
        return optuna.samplers.TPESampler(seed=seed)

    if sampler_type in ("nsga2"):
        return optuna.samplers.NSGAIISampler(seed=seed)

    raise ValueError("Level2 supports sampler.type='nsga2' only (for now)")


def build_pruner(cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_cfg = cfg[PRUNER]
    pruner_type = pruner_cfg[PRUNER_TYPE]

    if pruner_type != "median":
        raise ValueError(f"Unsupported pruner.type='{pruner_type}'. Allowed: 'median'")

    n_startup_trials  = pruner_cfg[N_STARTUP_TRIALS]
    n_warmup_steps  = pruner_cfg[N_WARMUP_STEPS]


    if not isinstance(n_startup_trials, int) or n_startup_trials < 0:
        raise ValueError("pruner.n_startup_trials must be a non-negative int")
    if not isinstance(n_warmup_steps, int) or n_warmup_steps < 0:
        raise ValueError("pruner.n_warmup_steps must be a non-negative int")

    return optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )


def load_level1_best_params(*, storage: str, study_name: str) -> dict[str, float]:
    """
    Returns dict with keys like 'lr' and 'clip_norm' if present in best trial params.
    """
    s = optuna.load_study(storage=storage, study_name=study_name)
    return dict(s.best_trial.params)

def main():

    tune_cfg = load_yaml(OPTUNA_PATH)   

    t = tune_cfg[TUNING]

    # ! set the seed important
    seed = t[SEED]

    # name of the original config yaml 
    base_name = t[BASE_NAME]

    # name of the tuning experiment 
    study_name = t[STUDY_NAME]

    n_trials = t[N_TRIALS]
    level = t[LEVEL]
    monitors = t[MONITORS]

    # path of the experiments results
    out_root = Path("runs/tuning") / study_name
    out_root.mkdir(parents=True, exist_ok=True)

    # save the current state of study 
    storage = f"sqlite:///{(out_root / "optuna.db").as_posix()}"

    report_name = t[OUTPUT][REPORT_FILENAME]


    # ? we can also try different samplers and pruners
    
    # samplers tell us which type of parameter to try for the next trial 
    # bayesian approach
    sampler = build_sampler(t, seed, level)

    # pruners decide to interrupt a trial if is going badly
    
    # MedianPruner compares the 'intermediate' performance of the current trial
    # with the median of trials already completed at the same epoch

    # n_startup_trials => tell at the pruner after how many trial start to prune
    # n_warmup_steps => number of epoch after that start to prune a current trial
    pruner = build_pruner(t)

    # container object
    level1_best = None
    interval = None
    
    if level == 1:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    elif level == 2:
        study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "maximize"],  # (loss, output_seq_len)
            sampler=sampler,
            pruner=pruner, 
            storage=storage,
            load_if_exists=True,
        )
        ref = t["level1_ref"]
        if ref:
            ref_storage = f"sqlite:///{(Path("runs/tuning") / ref['study_name'] / "optuna.db").as_posix()}"
            level1_best = load_level1_best_params(storage=ref_storage, study_name=ref["study_name"])
            interval = ref["interval"]


    # main configuration 
    base_cfg = load_yaml(base_name)


    def objective(trial):
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            model_type = base_cfg[MODEL][MODEL_TYPE]
            override = suggest_level(trial, model_type = model_type, level = level, level1_best = level1_best, interval = interval)

            callbacks : list[Callback] = [OptunaPruningCallback(trial, monitors)]

            metrics = run_experiment(
                base_name,
                override=override,
                out_dir=f"tuning/{study_name}/trial_{trial.number:04d}",
                optuna_callback=callbacks,
                do_predict=True,
            )

            score = compute_score(metrics, base_cfg,override, level)

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


    study.optimize(objective, n_trials=n_trials)

    df = study.trials_dataframe(("number", "value", "duration", "params", "user_attrs", "system_attrs", "state"))
    df.to_csv(out_root / report_name, index=False)

    print("Best trial:", study.best_trial.number, "score:", study.best_value)

if __name__ == "__main__":
    main()