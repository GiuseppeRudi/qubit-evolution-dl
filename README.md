# Qubit Evolution DL: Neural Modeling of Multi-Qubit Dynamics

Qubit Evolution DL is a research-oriented project that uses **deep learning**
(RNNs and Transformers) to model the **time evolution of multi-qubit systems**.

The main goal is to learn how the state (or measurement statistics) of a
quantum system changes as we:

- increase the **number of timesteps** (deeper circuits / longer evolution),
- increase the **number of qubits** (larger quantum registers).

The project is meant as a bridge between **quantum computing** and **sequence
modeling**: we treat the evolution of a quantum circuit as a sequence and use
sequence models to reconstruct and predict qubit behavior.

---

## Objectives

- Represent **quantum circuits** and **qubit states** in a form suitable for
  deep learning (e.g. time series of measurement outcomes, amplitudes,
  expectation values, etc.).
- Train:
  - a **Recurrent Neural Network (RNN)** baseline, and  
  - a **Transformer-based model**  
  to predict future qubit behavior from past timesteps.
- Study how model performance scales with:
  - number of qubits (2, 4, 8, ...),
  - number of timesteps (circuit depth / sequence length).
- Compare architectures in terms of:
  - prediction accuracy,
  - generalization across different quantum circuits,
  - robustness to noise (optional / future work).
 
## Quick Start

### Local (Conda)

```bash
conda env create -f environment.yml
conda activate qubit
python install_tf.py
python main.py
Docker (recommended)
bash
Copia codice
# RTX 50 image
docker build -t tf-app-rtx50 --build-arg TF_VARIANT=rtx50 .
docker run --gpus all -it tf-app-rtx50

# Official TF image
docker build -t tf-app-official --build-arg TF_VARIANT=official .
docker run --gpus all -it tf-app-official
```

# Docker (recommended)
###  RTX 50 image
```bash
docker build -t tf-app-rtx50 --build-arg TF_VARIANT=rtx50 .
docker run --gpus all -it tf-app-rtx50
```

### Official TF image
```bash
docker build -t tf-app-official --build-arg TF_VARIANT=official .
docker run --gpus all -it tf-app-official
```

# Documentation

- `docs/SETUP.md` → detailed setup and configuration
- `docs/report.tex` + `docs/report.pdf` → full technical report
