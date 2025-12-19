from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast
import yaml

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]  # core -> qubit -> src -> qubit-evolution-dl


def load_yaml(path: Path | str) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping (dict) in {path}, got {type(data).__name__}")

    # opzionale: se vuoi anche assicurarti che le chiavi siano str
    if not all(isinstance(k, str) for k in data.keys()):
        raise TypeError(f"Expected dict with str keys in {path}")

    return cast(Dict[str, Any], data)
