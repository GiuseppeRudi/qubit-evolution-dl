from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

Path = str

@dataclass(frozen=True)
class ParamSpec:
    name: str  # name of parameters in our study
    path: Path # where the variabial store in the yaml file 
    type: str # "int" | "float" | "categorical"
    # used when the type of parameter is int or float number
    low: float | int | None = None
    high: float | int | None = None

    # used for stride 
    step: int | None = None

    # verify 
    log: bool = False

    # if the type is categorical so we insert a list of numbers
    choices: list[Any] | None = None
    

    # choices_fn: Optional[Callable[[dict[str, Any]], list[Any]]] = None

SEARCH_SPACE: dict[str, dict[str, list[ParamSpec]]] = {
    "LSTM": {
        "level1": [
            ParamSpec(
                name="lr",
                path="model.compile.learning_rate",
                type="float",
                low=1e-5,
                high=3e-3,
                log=True,
            ),
            ParamSpec(
                name="clip_norm",
                path="model.compile.clip_norm",
                type="float",
                low=0.1,
                high=2.0,
                log=True,
            ),
            ParamSpec(
                name="batch_size",
                path="training.batch_size",
                type="categorical",
                choices=[32, 64, 128],
            ),
            ParamSpec(
                name="latent_dim",
                path="model.params.latent_dim",
                type="categorical",
                choices=[64, 96, 128, 192, 256],
            ),
        ],
        "level2": [
            # WIP
        ],
        "level3": [
            # WIP
        ],
    },

    "TRN": {
        "level1": [
            ParamSpec(
                name="lr",
                path="model.compile.learning_rate",
                type="float",
                low=1e-5,
                high=3e-3,
                log=True,
            ),
            ParamSpec(
                name="clip_norm",
                path="model.compile.clip_norm",
                type="float",
                low=0.1,
                high=2.0,
                log=True,
            ),
            ParamSpec(
                name="batch_size",
                path="training.batch_size",
                type="categorical",
                choices=[16, 32, 64],
            ),
            ParamSpec(
                name="num_heads",
                path="model.params.num_heads",
                type="categorical",
                choices=[2, 4, 8],
            ),

            # attention dim_mode depends from num_heads (must be dividible)
            ParamSpec(
                name="dim_model",
                path="model.params.dim_model",
                type="categorical",
                choices=[64, 128, 192, 256, 384, 512],
                # choices_fn=lambda ctx: [d for d in [64, 128, 192, 256, 384, 512] if d % int(ctx["num_heads"]) == 0],
            ),
            ParamSpec(
                name="ff_dim",
                path="model.params.ff_dim",
                type="categorical",
                choices=[128, 256, 512, 1024, 1536],
            ),
            ParamSpec(
                name="num_layers",
                path="model.params.num_layers",
                type="categorical",
                choices=[1, 2, 3, 4, 6],
            ),
            ParamSpec(
                name="dropout",
                path="model.params.dropout",
                type="float",
                low=0.0,
                high=0.3,
                log=False,
            ),
        ],
        "level2": [
            # WIP
        ],
        "level3": [
            # WIP
        ],
    },
}
