from __future__ import annotations

from typing import Any
import optuna

from .search_space import SEARCH_SPACE, ParamSpec


def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:

    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _suggest_one(trial: optuna.Trial, spec: ParamSpec) -> Any:

    if spec.type == "float":
        assert spec.low is not None and spec.high is not None
        return trial.suggest_float(spec.name, float(spec.low), float(spec.high), log=spec.log)

    if spec.type == "int":
        assert spec.low is not None and spec.high is not None

        if spec.step is None:
            return trial.suggest_int(spec.name, int(spec.low), int(spec.high), log=spec.log)
        return trial.suggest_int(spec.name, int(spec.low), int(spec.high), step=int(spec.step), log=spec.log)

    if spec.type == "categorical":

        if not spec.choices:
            raise ValueError(f"No available choice for {spec.name} (path={spec.path})")
        return trial.suggest_categorical(spec.name, spec.choices)

    raise ValueError(f"Not supported type: {spec.type}")


def suggest_level(trial: optuna.Trial, *, model_type: str, level: int) -> dict[str, Any]:
    
    lvl = f"level{level}"
    
    if model_type not in SEARCH_SPACE or lvl not in SEARCH_SPACE[model_type]:
        raise ValueError(f"Missing search space for model_type={model_type} level={lvl}")

    override: dict[str, Any] = {}

    # a list of hyperparameters to tune 
    specs =  SEARCH_SPACE[model_type][lvl]
    
    for spec in specs:
        v = _suggest_one(trial, spec)
        
        _set_nested(override, spec.path, v)
    return override
