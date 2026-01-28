from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    dim_model: int
    num_heads: int
    ff_dim: int
    num_layers: int
    dropout: float