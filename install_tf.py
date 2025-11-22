import subprocess
import os
from pathlib import Path

def safe_nvidia_smi():
    """Return the output of nvidia-smi or None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
        return out.stdout
    except FileNotFoundError:
        print("⚠nvidia-smi not found · probably running without GPU.")
        return None


# -----------------------------------------------------------------------------------
# Detect GPU
# -----------------------------------------------------------------------------------
gpu_info = safe_nvidia_smi()

is_rtx50 = False
if gpu_info:
    # A detection that works also for 5060 / 5070 / 5080 / 5090
    if any(model in gpu_info for model in ["5060", "5070", "5080", "5090"]):
        is_rtx50 = True

# Path del file requirements-rtx50.txt (stessa cartella di install_tf.py)
BASE_DIR = Path(__file__).resolve().parent
RTX_REQ = BASE_DIR / "requirements-rtx50.txt"

# -----------------------------------------------------------------------------------
# Install TensorFlow
# -----------------------------------------------------------------------------------
if is_rtx50:
    print(" Detected RTX 50-series GPU → installing custom TensorFlow build...")

    wheel_url = (
        "https://github.com/mypapit/tensorflowRTX50/releases/download/"
        "2.20dev-ubuntu-24.04-avx-too/"
        "tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl"
    )

    # Install custom TensorFlow (no deps)
    subprocess.run(
        ["python", "-m", "pip", "install", wheel_url, "--no-deps"],
        check=True
    )

    print("📦 Installing missing TensorFlow dependencies for custom build...")
    subprocess.run(
        ["python", "-m", "pip", "install", "-r", str(RTX_REQ)],
        check=True
    )

    print("✅ Custom TensorFlow for RTX 50 installed successfully.")

else:
    print("Installing official TensorFlow... (CPU or supported GPU)")
    subprocess.run(["python", "-m", "pip", "install", "tensorflow"], check=True)
    print("✅ Official TensorFlow installed successfully.")

print(" Setup complete — you can now run your project!")
