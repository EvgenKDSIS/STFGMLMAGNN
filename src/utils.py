import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_global(model: torch.nn.Module, data, device: torch.device) -> dict[str, float]:
    model.eval()
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    logits = model(x, edge_index)
    pred = logits.argmax(dim=-1)

    metrics = {}
    for split, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        mask = mask.to(device)
        if mask.sum() == 0:
            continue
        correct = (pred[mask] == y[mask]).sum().item()
        metrics[f"{split}_acc"] = correct / mask.sum().item()
    return metrics
