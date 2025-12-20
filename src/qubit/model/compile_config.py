from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CompileConfig:
    optimizer: str 
    loss: str