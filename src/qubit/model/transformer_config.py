from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    embed_dim: int 
    num_heads: int 
    ff_dim: int 