# Optuna - Hyperparameter Tuning

This project includes an **Optuna-based tuning pipeline** inside the `tuning/` package.

It automatically runs multiple training trials, changes selected parameters, and saves:

- the full **trial report** (`report.csv`)
- the Optuna **SQLite database** (`optuna.db`)
- one output folder per trial with logs/metrics

You launch tuning with:

`PYTHONPATH=src python -m tuning.tune`

The tuning behavior is controlled by a YAML configuration file located in:

`configs/optuna.yaml`

---

## How it works

1. You choose a **base config** (example: `configs/lstm_full_seq.yaml`)
2. Optuna samples new hyperparameters using the **search space** in `tuning/search_space.py`
3. For each trial, the code builds an `override` dictionary and calls
4. The model trains normally (like `main.py`), but each trial has different parameters.
5. The trial is evaluated using a **score function** (`tuning/score.py`).
6. If the trial is performing badly, Optuna can **prune** it early (stop training).

---

## The tuning configuration file: `configs/optuna.yaml`

This YAML controls the Optuna study:

`tuning.seed`               ⇒  Controls reproducibility for the sampler.

`tuning.study_name`   ⇒  Name of the Optuna study. It also becomes the name of the output folder.

`tuning.n_trials`       ⇒  How many trials Optuna will run.

---

`tuning.level`             ⇒  Selects which group of parameters Optuna is allowed to modify.

Your code uses:

- `level1`, `level2`, `level3` inside `SEARCH_SPACE` (in `tuning/search_space.py`)

Example:

- `level: 1` → tune only basic training/model hyperparameters
- `level: 2` → tune window lengths and stride
- `level: 3` → WIP (strategies, curriculum)

---

### `tuning.base_name`

The base YAML experiment configuration to start from.

Important:

- You write the name **without `.yaml`**
- The loader resolves it as: `configs/<base_name>.yaml`

Example:

- `base_name: lstm_full_seq` → uses `configs/lstm_full_seq.yaml`

This base config defines the dataset, model type, training phases, etc.

Optuna only changes what is defined in the selected search space level.

---

### `tuning.monitors`

Defines which metric names the pruning callback should look at.

Typical usage:

- **Level 1**: prune based on free-running phase metrics `"_fr_phase_loss_"`
- **Level 2**: prune based on `val_loss` (because output length changes and FR curve is not comparable)

---

### `tuning.sampler`

Selects how Optuna chooses the next hyperparameters.

Your code supports:

- `type: tpe`

TPE is a Bayesian sampler and usually works well for neural networks.

---

### `tuning.pruner`

Controls early stopping of bad trials.

Your code supports:

- `type: median`

Parameters:

- `n_startup_trials`: number of completed trials before pruning starts
- `n_warmup_steps`: number of epochs before pruning is allowed inside one trial

---

## Output structure

After tuning finishes, you will find:

**1) Study outputs**

- `runs/tuning/<study_name>/optuna.db`
    
    (Optuna database: all trial history)
    
- `runs/tuning/<study_name>/report.csv`
    
    (table with trial number, score, parameters, duration, prune info)
    

**2) Per-trial experiment folders**

Each trial is executed like a normal experiment, and results are saved in:

- `tuning/<study_name>/trial_XXXX/`

This includes logs and (depending on your settings) saved artifacts.

---

## Score formula

- **Level 1**: weighted combination of free-running metrics
    
    (fr_target + fr_curve + fr_phase)
    
- **Level 2**: fr_target + val_loss
    
    (because output length changes, comparisons must stay fair)
    

Optuna minimizes this score.