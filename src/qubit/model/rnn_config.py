from dataclasses import dataclass

@dataclass(frozen=True)
class RNNConfig:
    latent_dim: int = 64
    cell : str = "LSTM"

