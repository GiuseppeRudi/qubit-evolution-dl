import subprocess
from pathlib import Path


def get_nvidia_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None
    except FileNotFoundError:
        print("nvidia-smi not found: installing the official TensorFlow package.")
        return None


def is_rtx_50_series(gpu_info: str | None) -> bool:
    if not gpu_info:
        return False

    rtx_50_models = ["5060", "5070", "5080", "5090"]
    return any(model in gpu_info for model in rtx_50_models)


BASE_DIR = Path(__file__).resolve().parent
RTX_REQ = BASE_DIR / "requirements-rtx50.txt"

gpu_info = get_nvidia_info()

if is_rtx_50_series(gpu_info):
    print("RTX 50 GPU detected: installing the dedicated TensorFlow build.")

    wheel_url = (
        "https://github.com/mypapit/tensorflowRTX50/releases/download/"
        "2.20dev-ubuntu-24.04-avx-too/"
        "tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl"
    )

    subprocess.run(
        ["python", "-m", "pip", "install", wheel_url, "--no-deps"],
        check=True,
    )

    subprocess.run(
        ["python", "-m", "pip", "install", "-r", str(RTX_REQ)],
        check=True,
    )

    print("Custom TensorFlow build for RTX 50 installed.")

else:
    print("Installing the official TensorFlow package.")

    subprocess.run(
        ["python", "-m", "pip", "install", "tensorflow"],
        check=True,
    )

    print("Official TensorFlow package installed.")

print("Setup completed.")