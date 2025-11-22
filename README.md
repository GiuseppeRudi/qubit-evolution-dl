# TensorFlow RTX 50 Demo

This project demonstrates how to run a TensorFlow-based workload on
NVIDIA GPUs, including the RTX 50 series, or on CPU only.

## Quick Start

### Local (Conda)

```bash
conda env create -f environment.yml
conda activate tf-project
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
