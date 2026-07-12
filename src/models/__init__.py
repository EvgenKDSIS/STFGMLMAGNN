import torch

from .gcn import GCN
from .gat import GAT


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.5,
) -> torch.nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    if name == "gat":
        return GAT(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Unknown model: {name}. Choose from: gcn, gat")


__all__ = ["GCN", "GAT", "build_model"]
