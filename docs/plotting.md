# Plotting

After you run the main training script (`main.py`), the project saves a **run directory** containing artifacts like:

- `data_splits.npz` (test split, and optionally `mean/std`)
- `predictions.npz` (model predictions)
- `meta.json` (metadata such as feature names, sample indices)
- `attn_maps.npz` (only if you enabled attention saving in the Transformer)

The two scripts below are **post-processing tools**: they do not train anything, they only **read saved artifacts** and generate plots.

---

## 1) `scripts/plot_attention_maps.py`

### Purpose

This script loads `attn_maps.npz` from a run folder and produces **attention heatmaps** for every key inside the file.

Each attention map is stored as a 4D array:

- shape `(B, H, Tq, Tk)`
    - `B` = batch size used when extracting attention maps (often 1)
    - `H` = number of attention heads
    - `Tq` = number of query timesteps
    - `Tk` = number of key timesteps
    

The script converts one attention map into a 2D matrix `(Tq, Tk)` by:

- selecting one `sample` inside the batch (`-sample`)
- selecting one head (`-head`) or averaging all heads (`-head -1`)

Then it is saved in:

- `run_dir/attn_plots/`

### Command to run

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps
```

### Useful options

If you want to choose a specific run directory:

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps --run runs/predictions/TRN/Hybrid/FULL_SEQ/run__20260208_120501
```

Select a specific head:

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps --head 0
```

Average all heads (default):

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps --head -1
```

Select another sample in the batch (if `B > 1`):

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps --sample 3
```

---

## 2) `scripts/plot_from_data.py`

### Purpose

This script loads:

- `data_splits.npz` (X_test, Y_test, and mean/std)
- `predictions.npz` (predictions)
- `meta.json` (feature names and sample indices)

Then it generates plots for **forecasting** or **super-resolution**, depending on the shape of `X_test`.

### How it detects the mode

### Forecasting mode

- `X_test.shape = (num_windows, input_len, feature_dim)`
- `Y_test.shape = (num_windows, output_len, feature_dim)`

The plot shows:

- the full ground truth trajectory: input + output
- the prediction only over the output range
- a vertical line separating input and predicted region

### Super-resolution mode

- `X_test.shape = (num_windows, window_len, feature_dim + 1)`
    - the +1 is for the **mask channel**
- `Y_test.shape = (num_windows, window_len, feature_dim)`

The plot shows:

- ground truth as a line
- missing indices (holes) highlighted using the mask channel
- model predictions on the missing points
- optional comparison with a second run

### What it outputs

By default, it saves plots into:

- Single run: `run_dir/prediction_plots/`
- Two runs (comparison): `predictions/compare/compare__<timestamp>/`

Inside the output folder it creates subfolders like:

- `s(0)/` for sample index 0
- filenames like: `s(0)_f(1)_<feature_name>.jpg`

### Command to run

```bash
PYTHONPATH=src python -m scripts.plot_from_data
```

### Useful options

Plot from a specific run:

```bash
PYTHONPATH=src python -m scripts.plot_from_data --run-a runs/predictions/LSTM/FullSeq/run__20260208_121010
```

Compare two runs (same test split shapes required):

```bash
PYTHONPATH=src python -m scripts.plot_from_data \
  --run-a runs/predictions/LSTM/FullSeq/run__20260208_121010 \
  --run-b runs/predictions/TRN/Hybrid/run__20260208_122200 \
  --label-a"LSTM" \
  --label-b"Transformer"
```

Plot only one feature (0-based index):

```bash
PYTHONPATH=src python -m scripts.plot_from_data --feature
```

---

## Typical workflow

1. Train a model:

```bash
python main.py --run-cfg trn_hybrid
```

1. Generate prediction plots:

```bash
PYTHONPATH=src python -m scripts.plot_from_data
```

1. If attention maps were saved (Transformer):

```bash
PYTHONPATH=src python -m scripts.plot_attention_maps
```