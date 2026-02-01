from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..enums.loss_on import LossOn

from ..utils.config_keys import SUPER_RESOLUTION, SR_STRIDE, OFFSET, SR_MASK_VALUE, LOSS_ON, OBSERVED_WEIGHT

@dataclass(frozen=True)
class SuperResolutionConfig:
    stride: int
    offset: int
    mask_value: float
    loss_on: LossOn
    observed_weight: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SuperResolutionConfig:
        if SUPER_RESOLUTION in d:
            d = d[SUPER_RESOLUTION]
        return SuperResolutionConfig(
            stride = d[SR_STRIDE],
            offset = d[OFFSET],
            mask_value = d[SR_MASK_VALUE],
            loss_on=d[LOSS_ON],
            observed_weight=d[OBSERVED_WEIGHT]
        )
