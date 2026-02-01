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
            ParamSpec(
                name="input_seq_len",
                path="data.windowing.input_seq_len",
                type="categorical",
                choices=[100, 200, 250, 300, 350, 400, 450, 500, 600, 650, 700, 750, 800],
            ),
            ParamSpec(
                name="output_seq_len",
                path="data.windowing.output_seq_len",
                type="categorical",
                choices=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
            ),
            ParamSpec(
                name="stride",
                path="data.windowing.stride",
                type="categorical",
                choices=[10, 15, 20, 25, 20, 35, 40, 45, 50, 55],
            ),
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

            ParamSpec(
                name="dim_model",
                path="model.params.dim_model",
                type="categorical",
                choices=[64, 128, 192, 256, 384, 512],
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
            ParamSpec(
                name="input_seq_len",
                path="data.windowing.input_seq_len",
                type="categorical",
                choices=[100, 200, 250, 300, 350, 400, 450, 500, 600, 650, 700, 750, 800],
            ),
            ParamSpec(
                name="output_seq_len",
                path="data.windowing.output_seq_len",
                type="categorical",
                choices=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
            ),
            ParamSpec(
                name="stride",
                path="data.windowing.stride",
                type="categorical",
                choices=[10, 15, 20, 25, 20, 35, 40, 45, 50, 55],
            ),
        ],
        "level3": [
            # WIP
        ],
    },
}
