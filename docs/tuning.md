# Optuna – Hyperparameter Tuning

This project includes an **tuning pipeline with Optuna** inside the `tuning/` package.

It automatically runs multiple training trials, changes selected parameters, and saves:

- the full **trial report** (`report.csv`)
- the Optuna **SQLite database** (`optuna.db`)
- one output folder per trial with logs/metrics

You launch tuning with:

```bash
PYTHONPATH=src python -m tuning.tune
```

The tuning behaviour is controlled by:

```
configs/optuna.yaml
```

---

## How it works

1. You choose a **base config** (example: `configs/lstm_full_seq.yaml`)
2. Optuna samples new hyperparameters using the **search space** in `tuning/search_space.py`
3. For each trial, the code builds an `override` dictionary and calls `run_experiment(...)`
4. The model trains normally (like `main.py`), but each trial has different parameters.
5. The trial is evaluated using a **score function** (`tuning/score.py`).
6. If the trial is performing badly, Optuna can **prune** it early, stopping the training.

---

## Search Space Levels

The search space is defined in `tuning/search_space.py` as:

- `level1`: model/optimizer/training hyperparameters (LR, clip norm, batch size, dims, etc.)
- `level2`: windowing hyperparameters (`input_seq_len`, `output_seq_len`, `stride`)
- `level3`: WIP

---

## Score


$$
\textbf{score} =
0.70 \cdot fr\_target +
0.25 \cdot fr\_curve +
0.05 \cdot fr\_phase
$$

Optuna **minimizes** this score.

---

## Level 1 – Single objective (minimize)

### lr scaling with batch size

When Level 1 tunes both `batch_size` and `learning_rate`, it is common to avoid "wasted" trials by connecting lr to batch size.

Instead of directly using the sampled LR, Optuna samples a **reference lr** (`lr_ref`) and you compute the **effective lr** used in training:

$$
lr_{eff} = lr_{ref}\cdot\left(\frac{batch}{B_0}\right)^{\alpha}
$$

Default:

- $B_0 = 64$
- $\alpha = 0.5$

---

## Level 2 – Multi objective

Level 2 tunes windowing hyperparameters:

- `input_seq_len`
- `output_seq_len`
- `stride`

But since longer `output_seq_len` is naturally more difficult, comparing trials using a single scalar loss will systematically prefer short horizons.

### Multi-objective formulation

Level 2 is handled as a **multi-objective optimization**:

- **Objective 1 (minimize)**: a scalar score (see below)
- **Objective 2 (maximize)**: `output_seq_len`

Optuna returns a **Pareto front** (set of non-dominated trials). There is no single best trial by definition—each Pareto point is a trade-off between error and horizon (point of maximum efficiency).


## Level 2: Minimal re-tuning of lr and clip norm

Level 2 **re-tune** `learning_rate` and `clip_norm` in a **minimal interval** around the best Level 1 solution.

Level 2 takes from level 1 the best `lr` and `clip norm` and samples:

$$
lr \in [lr_{best}(1-\delta),\; lr_{best}(1+\delta)]
$$

$$
clip \in [clip_{best}(1-\delta),\; clip_{best}(1+\delta)]
$$

where $\delta$ = `tuning.level1_ref.interval` (e.g. `0.30` for $\pm$ 30%).

- `lr_best` refer to the **effective lr** (`lr_eff`) actually used during training.

---

## Pruning

Pruning is still based on a **single monitored metric** (even in multi-objective mode).

Recommended monitors:

- **Level 1**: prune using free-running phase loss
    - `"_fr_phase_loss_"`
- **Level 2**: prune using a comparable scalar metric:
    - `"val_loss"`