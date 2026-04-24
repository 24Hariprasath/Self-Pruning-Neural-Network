from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Determinism
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic for reproducibility of pruning behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Config
@dataclass
class Config:
    data_dir: str = "./data"
    output_dir: str = "./results"
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_val: float = 1e-4
    lambda_warmup_epochs: int = 5
    val_split: float = 0.1
    seed: int = 42
    early_stop_patience: int = 5
    no_early_stop: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gate_threshold: float = 1e-2
    sweep: bool = False

# Prunable Layer
class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def sparsity_loss(self) -> torch.Tensor:
        return torch.sum(torch.sigmoid(self.gate_scores))


# Network
class SelfPruningNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

    def sparsity_loss(self) -> torch.Tensor:
        return sum(m.sparsity_loss() for m in self.modules() if isinstance(m, PrunableLinear))

    def global_sparsity(self, threshold: float) -> float:
        total, zero = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                total += gates.numel()
                zero += torch.sum(gates < threshold).item()
        return 100.0 * zero / total


# Data
def get_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor()
    ])
    transform_test = T.Compose([T.ToTensor()])

    dataset = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform_train)
    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    test_set = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform_test)

    return (
        DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=cfg.batch_size),
        DataLoader(test_set, batch_size=cfg.batch_size),
    )


# Training
def train(cfg: Config) -> Dict:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data(cfg)

    model = SelfPruningNet().to(cfg.device)

    # Separate param groups: NO weight decay on gates
    gate_params = []
    weight_params = []
    for name, p in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(p)
        else:
            weight_params.append(p)

    optimizer = Adam([
        {"params": weight_params, "weight_decay": cfg.weight_decay},
        {"params": gate_params, "weight_decay": 0.0},  # critical correctness
    ], lr=cfg.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = 0
    patience = 0
    history = {"val_acc": [], "sparsity": []}

    for epoch in range(cfg.epochs):
        model.train()
        lambda_eff = cfg.lambda_val * min(1.0, epoch / cfg.lambda_warmup_epochs)

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss += lambda_eff * model.sparsity_loss()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        val_acc = evaluate(model, val_loader, cfg)
        sparsity = model.global_sparsity(cfg.gate_threshold)

        history["val_acc"].append(val_acc)
        history["sparsity"].append(sparsity)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pt"))
            patience = 0
        else:
            patience += 1
            if not cfg.no_early_stop and patience >= cfg.early_stop_patience:
                break

    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best.pt")))
    test_acc = evaluate(model, test_loader, cfg)

    results = {
        "test_acc": test_acc,
        "sparsity": model.global_sparsity(cfg.gate_threshold),
    }

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate(model: nn.Module, loader: DataLoader, cfg: Config) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--lambda_val", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--no_early_stop", action="store_true")

    args = parser.parse_args()
    cfg = Config(**vars(args))

    if cfg.sweep:
        lambdas = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        sweep_results = []
        for l in lambdas:
            cfg.lambda_val = l
            res = train(cfg)
            sweep_results.append({"lambda": l, **res})

        with open(os.path.join(cfg.output_dir, "sweep.json"), "w") as f:
            json.dump(sweep_results, f, indent=2)
    else:
        train(cfg)


if __name__ == "__main__":
    main()