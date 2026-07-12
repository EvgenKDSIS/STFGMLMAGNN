from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    dataset: str = "Cora"
    num_clients: int = 5
    num_malicious: int = 1
    iid: bool = False
    dirichlet_alpha: float = 0.5

    model: str = "gcn"
    hidden_dim: int = 64
    dropout: float = 0.5

    rounds: int = 50
    local_epochs: int = 3
    lr: float = 0.01
    weight_decay: float = 5e-4

    aggregation: str = "fedavg"  # fedavg | median | trimmed_mean | krum
    trim_ratio: float = 0.2

    attack: str = "none"  # none | label_flip | scaled_update
    flip_ratio: float = 1.0
    scale_factor: float = 10.0

    seed: int = 42
    device: str = "cpu"
    log_every: int = 5

    malicious_client_ids: list[int] = field(default_factory=list)
