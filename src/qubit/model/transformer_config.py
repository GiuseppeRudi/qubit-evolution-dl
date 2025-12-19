from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    embed_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 128