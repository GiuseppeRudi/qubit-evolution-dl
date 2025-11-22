# ----------------------------------------------------------
# Base image: NVIDIA CUDA 12.8 + cuDNN runtime + Ubuntu 24.04
# ----------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------
# Install system dependencies
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y \
        python3 \
        python3-venv \
        python3-dev \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Create virtual environment inside the container
# ----------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------
# Copy project files
# ----------------------------------------------------------
WORKDIR /app
COPY . /app

# ----------------------------------------------------------
# Install base Python dependencies (without TensorFlow)
# ----------------------------------------------------------
RUN pip install -r requirements.txt

# ----------------------------------------------------------
# Select TensorFlow variant (custom RTX 50 vs official)
# ----------------------------------------------------------
ARG TF_VARIANT=rtx50

RUN if [ "$TF_VARIANT" = "rtx50" ]; then \
      echo "Installing custom TensorFlow for RTX 50..."; \
      python -m pip install \
        "https://github.com/mypapit/tensorflowRTX50/releases/download/2.20dev-ubuntu-24.04-avx-too/tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl" \
        --no-deps && \
      python -m pip install -r requirements-rtx50.txt; \
    else \
      echo "Installing official TensorFlow..."; \
      python -m pip install tensorflow; \
    fi

# ----------------------------------------------------------
# Default command: just run the demo script
# ----------------------------------------------------------
CMD ["python", "main.py"]
