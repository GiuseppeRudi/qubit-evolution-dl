from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
