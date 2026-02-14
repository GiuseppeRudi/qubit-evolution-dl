from __future__ import annotations

from typing import Any
import optuna

from qubit.enums.model_type import ModelType

from .search_space import SEARCH_SPACE, ParamSpec
from qubit.utils.config_keys import BATCH_SIZE, TRAINING, MODEL , COMPILE , LEARNING_RATE



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



def suggest_level(
    trial: optuna.Trial,
    *,
    model_type: str,
    level: int,
    best_params_lvl1: dict[str, Any] | None,
    best_user_attr_lvl1: dict[str, Any] | None,  
    interval: float | None,
) -> dict[str, Any]:

    lvl = f"level{level}"
    if model_type not in SEARCH_SPACE or lvl not in SEARCH_SPACE[model_type]:
        raise ValueError(f"Missing search space for model_type={model_type} level={lvl}")

    override: dict[str, Any] = {}

    specs = list(SEARCH_SPACE[model_type][lvl])

    if level == 2 and best_params_lvl1 and best_user_attr_lvl1 and interval:
        lr0 = float(best_user_attr_lvl1["lr_eff"])
        specs.append(
            ParamSpec(
                name="lr",
                path="model.compile.learning_rate",
                type="float",
                low=lr0 * (1.0 - interval),
                high=lr0 * (1.0 + interval),
                log=True,
            )
        )

        c0 = float(best_params_lvl1["clip_norm"])
        specs.append(
            ParamSpec(
                name="clip_norm",
                path="model.compile.clip_norm",
                type="float",
                low=c0 * (1.0 - interval),
                high=c0 * (1.0 + interval),
                log=True,  
            )
        )

    for spec in specs:
        v = _suggest_one(trial, spec)
        _set_nested(override, spec.path, v)

    if level == 1:
        batch = override[TRAINING][BATCH_SIZE]
        lr_ref = override[MODEL][COMPILE][LEARNING_RATE]

        B0 = 64.0
        alpha = 0.5 

        lr_eff = float(lr_ref) * (float(batch) / B0) ** alpha
        
        print(f"Lr_ref : {lr_ref} , batch_size : {batch}, Lr_eff : {lr_eff}  \n" )
        _set_nested(override, "model.compile.learning_rate", lr_eff)

        trial.set_user_attr("lr_ref", float(lr_ref))
        trial.set_user_attr("lr_eff", float(lr_eff))
        trial.set_user_attr("lr_alpha", alpha)

    elif level == 2 and best_params_lvl1:
        print(f"batch_size: {best_params_lvl1["batch_size"]}, ",end="")
        _set_nested(override, "model.training.batch_size", best_params_lvl1["batch_size"])
        if model_type == ModelType.LSTM:
            print(f"latent_dim: {best_params_lvl1["latent_dim"]}")
            _set_nested(override, "model.params.latent_dim", best_params_lvl1["latent_dim"])
        elif model_type == ModelType.TRN:
            print(f"num_heads: {best_params_lvl1["num_heads"]}, ",end="")
            _set_nested(override, "model.params.num_heads", best_params_lvl1["num_heads"])
            print(f"dim_model: {best_params_lvl1["dim_model"]}, ",end="")
            _set_nested(override, "model.params.dim_model", best_params_lvl1["dim_model"])
            print(f"ff_dim: {best_params_lvl1["ff_dim"]}, ",end="")
            _set_nested(override, "model.params.ff_dim", best_params_lvl1["ff_dim"])
            print(f"num_layers: {best_params_lvl1["num_layers"]}, ",end="")
            _set_nested(override, "model.params.num_layers", best_params_lvl1["num_layers"])
            print(f"dropout: {best_params_lvl1["dropout"]}")
            _set_nested(override, "model.params.dropout", float(best_params_lvl1["dropout"]))

    return override