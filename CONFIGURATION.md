# Configuration and Execution Guide

This document contains the operational instructions for Qubit Evolution DL.
For the research motivation, architecture, and representative results, see the
[project overview](README.md).

## Choose an execution environment

| Environment | Recommended for | Notes |
|---|---|---|
| Docker | Reproducible CPU execution | Uses the committed Python 3.12 container |
| Conda + official TensorFlow | Local development and supported GPUs | TensorFlow installation is selected by `install_tf.py` |
| Conda + custom TensorFlow wheel | NVIDIA RTX 50-series systems | Follow the prompt produced by `install_tf.py` |

You will need Git, Docker Desktop or a compatible Docker installation for the
container workflow, or Conda for local execution.

## Docker setup

From the repository root, build the image and open an interactive shell:

```bash
docker compose build
docker compose run --rm app bash
```

Inside the container, download the dataset and start an experiment:

```bash
python scripts/download_data.py
python main.py --run-cfg trn_hybrid
```

Other ready-to-run experiments include:

```bash
python main.py --run-cfg lstm_full_seq
python main.py --run-cfg lstm_step_wise
python main.py --run-cfg lstm_hybrid
python main.py --run-cfg trn_sr
```

Leave the container with:

```bash
exit
```

The committed Docker image is CPU-oriented. For local GPU acceleration, use the
Conda workflow and a TensorFlow build compatible with the host operating
system, drivers, and GPU.

## Local Conda setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate qubit
```

Install the appropriate TensorFlow distribution:

```bash
python install_tf.py
```

The installer distinguishes between the official TensorFlow package and the
custom path used for supported RTX 50-series environments. Follow its terminal
instructions if additional platform-specific steps are required.

Download the dataset and run an experiment:

```bash
python scripts/download_data.py
python main.py --run-cfg trn_hybrid
```

## Available experiment configurations

Configuration names map to YAML files under `configs/`.

| Run configuration | Model/task |
|---|---|
| `lstm_full_seq` | Full-sequence LSTM forecasting |
| `lstm_step_wise` | Step-wise autoregressive LSTM forecasting |
| `lstm_hybrid` | Hybrid LSTM forecasting |
| `trn_hybrid` | Hybrid Transformer forecasting |
| `trn_sr` | Transformer super-resolution |
| `optuna` | Hyperparameter-search defaults |

The main command requires a run configuration:

```bash
python main.py --run-cfg <configuration-name>
```

To load a saved model, pass its path with `--model`. Add `--no-training` to run
evaluation without further training:

```bash
python main.py \
  --run-cfg trn_hybrid \
  --model runs/<run-name>/model.keras \
  --no-training
```

`--no-training` is valid only when `--model` is also provided.

## YAML configuration

Each experiment YAML controls the complete run. The principal groups cover:

| Area | Typical settings |
|---|---|
| Data | dataset path, input/output length, stride, selected observables |
| Model | architecture name, hidden size, layers, heads, dropout |
| Training | epochs, batch size, optimizer, learning rate |
| Strategy | teacher forcing, scheduled sampling, curriculum or masking |
| Evaluation | prediction mode and free-running probes |
| Output | run and prediction directories |

Use [`docs/example.yaml`](docs/example.yaml) as the complete annotated
configuration reference. Prefer copying an existing file in `configs/` when
starting a new experiment so model-specific defaults remain consistent.

## Hyperparameter optimization

The Optuna workflow must be launched with `src` on `PYTHONPATH`.

Linux or macOS:

```bash
PYTHONPATH=src python -m tuning.tune
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m tuning.tune
```

See the [tuning guide](docs/tuning.md) for the two-stage search workflow,
multi-objective optimization, and study settings.

## Plotting predictions

Generate plots from a completed run:

```bash
python -m scripts.plot_from_data \
  --run runs/<run-name> \
  --label <label> \
  --feature <feature-index> \
  --out-dir runs/<run-name>/plots
```

Inspect Transformer attention:

```bash
python -m scripts.plot_attention_maps \
  --run runs/<run-name> \
  --attn-file <attention-file> \
  --out-dir runs/<run-name>/attention \
  --sample 0 \
  --head 0
```

Consult the command help for defaults and accepted values:

```bash
python -m scripts.plot_from_data --help
python -m scripts.plot_attention_maps --help
```

Additional context is available in the [plotting guide](docs/plotting.md).

## Generated outputs

Run artifacts are written below `runs/` and are intentionally excluded from
version control. A typical experiment produces a structure similar to:

```text
runs/
└── <run-name>/
    ├── checkpoints or saved model
    ├── logs and metrics
    ├── predictions/
    └── generated plots
```

The downloaded trajectory dataset under `data/` is also local-only. The
repository retains `data/.gitkeep` so the directory exists in a fresh clone.

## Troubleshooting

- **Dataset not found:** run `python scripts/download_data.py` from the
  repository root.
- **Configuration not found:** pass the YAML stem, for example
  `--run-cfg trn_hybrid`, and verify that the corresponding file exists under
  `configs/`.
- **TensorFlow cannot access the GPU:** verify the supported TensorFlow/Python
  combination, NVIDIA driver, CUDA requirements, and—on Windows—the recommended
  WSL/Linux execution path.
- **Saved model cannot be evaluated:** provide both `--model` and
  `--no-training`, and keep the run configuration compatible with the stored
  architecture.
- **Import failure in tuning:** set `PYTHONPATH=src` using the syntax for your
  shell before running `python -m tuning.tune`.
