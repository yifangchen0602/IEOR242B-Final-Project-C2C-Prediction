"""Deep representation and contrastive models for the final IEOR 242B dataset.

This version uses the group-final split files:

    dataset/train.csv
    dataset/val.csv
    dataset/test.csv
    dataset/test_original_label1.csv
    dataset/test_synthetic_label1.csv

It trains:

1. Embedding + MLP classifier.
2. Contrastive-pretrained encoder + MLP classifier.

The main comparison to baseline_results.csv should use the "original_test"
rows, which match the baseline data_loader.py convention:
all test label-0 rows plus the real label-1 test rows.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


CAT_COLS = [
    "country",
    "type",
    "language",
    "gender",
    "civilityTitle",
    "hasAnyApp",
    "hasIosApp",
    "hasProfilePicture",
    "countryCode",
    "popularity",
]

NUM_COLS = [
    "socialNbFollowers",
    "socialNbFollows",
    "socialProductsLiked",
    "productsListed",
    "productsSold",
    "productsPassRate",
    "productsWished",
    "seniority",
    "seniorityAsMonths",
    "seniorityAsYears",
    "truesold",
]

LOG_COLS = [
    "socialNbFollowers",
    "socialNbFollows",
    "socialProductsLiked",
    "productsListed",
    "productsSold",
    "productsPassRate",
    "productsWished",
    "truesold",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    project_dir = model_dir.parent if model_dir.name == "contrastive_model" else model_dir
    parser = argparse.ArgumentParser(description="Deep tabular contrastive model")
    parser.add_argument("--dataset-dir", type=Path, default=project_dir / "dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir / "runs" / "final",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument(
        "--contrastive-mode",
        choices=["corruption", "group"],
        default="corruption",
        help=(
            "corruption uses two randomly corrupted views of the same user. "
            "group uses the older behavior-group positive-pair definition."
        ),
    )
    parser.add_argument("--projection-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--cat-mask-prob", type=float, default=0.15)
    parser.add_argument("--num-mask-prob", type=float, default=0.15)
    parser.add_argument("--num-noise-std", type=float, default=0.05)
    parser.add_argument(
        "--loss-weight",
        action="store_true",
        help="Use pos_weight in BCE. Default is off because train/val/test are already SMOTE-NC augmented.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["bce", "focal"],
        default="bce",
        help="Supervised fine-tuning loss. Focal loss is useful for hard imbalanced examples.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.50,
        help="Positive-class alpha for focal loss. Ignored when --loss-type bce.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.00,
        help="Focusing parameter for focal loss. Ignored when --loss-type bce.",
    )
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-contrastive", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_split(dataset_dir: Path, name: str) -> pd.DataFrame:
    path = dataset_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    df = pd.read_csv(path)
    required = set(CAT_COLS + NUM_COLS + ["label"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[CAT_COLS + NUM_COLS + ["label"]].copy()


def load_final_splits(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    train = read_split(dataset_dir, "train.csv")
    val = read_split(dataset_dir, "val.csv")
    test_all = read_split(dataset_dir, "test.csv")
    test_synth_pos = read_split(dataset_dir, "test_synthetic_label1.csv")
    test_neg = test_all[test_all["label"] == 0].copy()

    return {
        "train": train,
        "val": val,
        "original_test": test_all,
        "synthetic_test": pd.concat([test_neg, test_synth_pos], ignore_index=True),
    }


def maybe_subsample_train(splits: Dict[str, pd.DataFrame], max_rows: int | None, seed: int) -> None:
    if max_rows is None or len(splits["train"]) <= max_rows:
        return
    train = splits["train"]
    sampled = (
        train.groupby("label", group_keys=False)
        .sample(
            frac=max_rows / len(train),
            random_state=seed,
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    splits["train"] = sampled


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=["label"]), df["label"].astype(int).to_numpy()


def encode_categoricals(df: pd.DataFrame) -> np.ndarray:
    # SMOTE-NC outputs nominal features as floats; embeddings need integer indices.
    cat = df[CAT_COLS].fillna(0).to_numpy(dtype=np.float32)
    cat = np.rint(cat).astype(np.int64)
    cat = np.clip(cat, a_min=0, a_max=None)
    return cat


def cat_cardinalities(frames: Iterable[pd.DataFrame]) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for col in CAT_COLS:
        max_value = 0
        for frame in frames:
            values = np.rint(frame[col].fillna(0).to_numpy(dtype=np.float32)).astype(np.int64)
            max_value = max(max_value, int(values.max(initial=0)))
        cardinality = max_value + 1
        dim = min(50, max(2, int(round(math.sqrt(cardinality))) + 1))
        sizes.append((cardinality, dim))
    return sizes


def transform_numeric_raw(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    out = df[NUM_COLS].copy()
    out = out.fillna(medians)
    for col in LOG_COLS:
        out[col] = np.log1p(np.clip(out[col].astype(float), a_min=0.0, a_max=None))
    return out.astype(np.float32)


def fit_numeric_scaler(train_x: pd.DataFrame) -> Tuple[pd.Series, StandardScaler]:
    medians = train_x[NUM_COLS].median(numeric_only=True)
    scaler = StandardScaler()
    scaler.fit(transform_numeric_raw(train_x, medians))
    return medians, scaler


def transform_numerics(df: pd.DataFrame, medians: pd.Series, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(transform_numeric_raw(df, medians)).astype(np.float32)


class TabularDataset(Dataset):
    def __init__(self, cat: np.ndarray, num: np.ndarray, y: np.ndarray):
        self.cat = torch.as_tensor(cat, dtype=torch.long)
        self.num = torch.as_tensor(num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.cat[idx], self.num[idx], self.y[idx]


def rank_bins(values: Iterable[float], n_bins: int = 4) -> np.ndarray:
    pct = pd.Series(values).rank(pct=True, method="average").fillna(0.0).to_numpy()
    return np.minimum((pct * n_bins).astype(int), n_bins - 1)


def build_contrastive_groups(train_df: pd.DataFrame) -> List[str]:
    y = train_df["label"].astype(int).to_numpy()
    engagement = (
        train_df["socialNbFollowers"].fillna(0)
        + train_df["socialNbFollows"].fillna(0)
        + train_df["socialProductsLiked"].fillna(0)
        + train_df["productsListed"].fillna(0)
        + train_df["productsSold"].fillna(0)
        + train_df["productsWished"].fillna(0)
    )
    wished_bin = rank_bins(train_df["productsWished"].fillna(0))
    engagement_bin = rank_bins(engagement)
    language = np.rint(train_df["language"].fillna(0).to_numpy(dtype=np.float32)).astype(int)

    groups = []
    for label, w_bin, e_bin, lang in zip(y, wished_bin, engagement_bin, language):
        groups.append(f"y={label}|wished={w_bin}|engagement={e_bin}|language={lang}")
    return groups


class PositivePairDataset(Dataset):
    def __init__(self, base: TabularDataset, groups: List[str]):
        self.base = base
        self.groups = groups
        self.group_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)

        labels = base.y.numpy().astype(int)
        for idx, group in enumerate(groups):
            self.group_to_indices[group].append(idx)
            self.label_to_indices[int(labels[idx])].append(idx)

    def __len__(self) -> int:
        return len(self.base)

    def _positive_index(self, idx: int) -> int:
        group = self.groups[idx]
        candidates = self.group_to_indices[group]
        if len(candidates) <= 1:
            label = int(self.base.y[idx].item())
            candidates = self.label_to_indices[label]
        if len(candidates) <= 1:
            return idx

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(candidates)
        return pos_idx

    def __getitem__(self, idx: int):
        pos_idx = self._positive_index(idx)
        cat_a, num_a, _ = self.base[idx]
        cat_p, num_p, _ = self.base[pos_idx]
        return cat_a, num_a, cat_p, num_p


class SelfPairDataset(Dataset):
    """Returns the same user twice; random corruption creates the two views."""

    def __init__(self, base: TabularDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        cat, num, _ = self.base[idx]
        return cat, num, cat, num


class TabularEncoder(nn.Module):
    def __init__(
        self,
        cat_sizes: List[Tuple[int, int]],
        n_num: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dim) for cardinality, dim in cat_sizes]
        )
        in_dim = sum(dim for _, dim in cat_sizes) + n_num
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        embedded = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([*embedded, num], dim=1)
        return self.mlp(x)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BuyerClassifier(nn.Module):
    def __init__(self, encoder: TabularEncoder, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        z = self.encoder(cat, num)
        return self.head(z).squeeze(1)


def make_dataset(
    df: pd.DataFrame,
    medians: pd.Series,
    scaler: StandardScaler,
) -> TabularDataset:
    x, y = split_xy(df)
    return TabularDataset(
        encode_categoricals(x),
        transform_numerics(x, medians, scaler),
        y,
    )


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.50, gamma: float = 2.00):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("focal alpha must be in [0, 1]")
        if gamma < 0.0:
            raise ValueError("focal gamma must be non-negative")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        return loss.mean()


def supervised_loss(
    y_train: np.ndarray,
    use_weight: bool,
    device: torch.device,
    loss_type: str,
    focal_alpha: float,
    focal_gamma: float,
) -> nn.Module:
    if loss_type == "focal":
        return BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if not use_weight:
        return nn.BCEWithLogitsLoss()
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def predict(model: BuyerClassifier, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    for cat, num, y in loader:
        logits = model(cat.to(device), num.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def metrics_from_prob(prob: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out: Dict[str, object] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def tune_threshold(model: BuyerClassifier, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    prob, y_true = predict(model, loader, device)
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def evaluate_loss(
    model: BuyerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for cat, num, y in loader:
            logits = model(cat.to(device), num.to(device))
            loss = criterion(logits, y.to(device))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate_named_sets(
    model: BuyerClassifier,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    rows = []
    for test_set, loader in loaders.items():
        prob, y_true = predict(model, loader, device)
        for threshold_type, threshold in thresholds.items():
            rows.append(
                {
                    "test_set": test_set,
                    "threshold_type": threshold_type,
                    **metrics_from_prob(prob, y_true, threshold),
                }
            )
    return rows


def collect_prediction_rows(
    model_name: str,
    model: BuyerClassifier,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, loader in loaders.items():
        prob, y_true = predict(model, loader, device)
        for row_id, (label, score) in enumerate(zip(y_true, prob)):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "row_id": row_id,
                    "y_true": int(label),
                    "prob": float(score),
                }
            )
    return rows


def corrupt_batch(
    cat: torch.Tensor,
    num: torch.Tensor,
    cat_sizes: List[Tuple[int, int]],
    cat_mask_prob: float,
    num_mask_prob: float,
    num_noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cat_aug = cat.clone()
    num_aug = num.clone()

    if cat_mask_prob > 0:
        for col_idx, (cardinality, _) in enumerate(cat_sizes):
            mask = torch.rand(cat_aug.size(0), device=cat_aug.device) < cat_mask_prob
            if mask.any():
                cat_aug[mask, col_idx] = torch.randint(
                    low=0,
                    high=cardinality,
                    size=(int(mask.sum().item()),),
                    device=cat_aug.device,
                )

    if num_mask_prob > 0:
        num_mask = torch.rand_like(num_aug) < num_mask_prob
        num_aug = num_aug.masked_fill(num_mask, 0.0)

    if num_noise_std > 0:
        num_aug = num_aug + torch.randn_like(num_aug) * num_noise_std

    return cat_aug, num_aug


def train_contrastive(
    encoder: TabularEncoder,
    projection_head: ProjectionHead,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    cat_sizes: List[Tuple[int, int]],
    contrastive_mode: str,
    cat_mask_prob: float,
    num_mask_prob: float,
    num_noise_std: float,
) -> List[Dict[str, float]]:
    encoder.to(device)
    projection_head.to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    history = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        projection_head.train()
        losses = []
        for cat_a, num_a, cat_p, num_p in loader:
            cat_a = cat_a.to(device)
            num_a = num_a.to(device)
            cat_p = cat_p.to(device)
            num_p = num_p.to(device)

            if contrastive_mode == "corruption":
                cat_a, num_a = corrupt_batch(
                    cat_a,
                    num_a,
                    cat_sizes,
                    cat_mask_prob,
                    num_mask_prob,
                    num_noise_std,
                )
                cat_p, num_p = corrupt_batch(
                    cat_p,
                    num_p,
                    cat_sizes,
                    cat_mask_prob,
                    num_mask_prob,
                    num_noise_std,
                )

            z_a = F.normalize(projection_head(encoder(cat_a, num_a)), dim=1)
            z_p = F.normalize(projection_head(encoder(cat_p, num_p)), dim=1)
            logits = torch.matmul(z_a, z_p.T) / temperature
            labels = torch.arange(logits.size(0), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "contrastive_loss": float(np.mean(losses)),
            "contrastive_mode": contrastive_mode,
        }
        history.append(row)
        print(f"[contrastive] epoch={epoch:03d} loss={row['contrastive_loss']:.4f}")
    return history


def train_supervised(
    model: BuyerClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    label: str,
) -> Tuple[BuyerClassifier, List[Dict[str, object]]]:
    model.to(device)
    criterion = supervised_loss(
        y_train,
        args.loss_weight,
        device,
        args.loss_type,
        args.focal_alpha,
        args.focal_gamma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    stale_epochs = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for cat, num, y in train_loader:
            logits = model(cat.to(device), num.to(device))
            loss = criterion(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_prob, val_true = predict(model, val_loader, device)
        val_metrics = metrics_from_prob(val_prob, val_true, threshold=0.5)
        score = float(val_metrics["pr_auc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)
        print(
            f"[{label}] epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[{label}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history


def save_training_plot(output_dir: Path, histories: Dict[str, List[Dict[str, object]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, rows in histories.items():
        if not rows or "train_loss" not in rows[0]:
            continue
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label=f"{model_name} train")
        axes[0].plot(epochs, [row["val_loss"] for row in rows], label=f"{model_name} val")
        axes[1].plot(epochs, [row["val_pr_auc"] for row in rows], label=f"{model_name} val PR-AUC")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    metadata: Dict[str, object],
    metric_rows: List[Dict[str, object]],
    histories: Dict[str, List[Dict[str, object]]],
    prediction_rows: List[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)

    history_rows = []
    for model_name, rows in histories.items():
        for row in rows:
            history_rows.append({"model": model_name, **row})
    pd.DataFrame(history_rows).to_csv(output_dir / "histories.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "metrics": metric_rows,
                "histories": histories,
                "prediction_file": "predictions.csv",
            },
            f,
            indent=2,
        )

    save_training_plot(output_dir, histories)


def write_baseline_comparison(project_dir: Path, output_dir: Path, metric_rows: List[Dict[str, object]]) -> None:
    baseline_path = project_dir / "baseline_method" / "baseline_results.csv"
    if not baseline_path.exists():
        return

    baseline = pd.read_csv(baseline_path)
    baseline["source"] = "baseline"
    baseline["test_set"] = "original_test"
    baseline["threshold_type"] = "model_default"

    contrastive = pd.DataFrame(metric_rows)
    contrastive = contrastive[contrastive["test_set"] == "original_test"].copy()
    contrastive["source"] = "contrastive_model"
    contrastive["model"] = contrastive["model"].map(
        {
            "embedding_mlp": "Embedding + MLP",
            "contrastive_mlp": "Contrastive + MLP",
        }
    )

    cols = [
        "source",
        "model",
        "test_set",
        "threshold_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    combined = pd.concat([baseline[cols], contrastive[cols]], ignore_index=True)
    combined.to_csv(output_dir / "vs_baselines.csv", index=False)


def print_split_summary(splits: Dict[str, pd.DataFrame]) -> None:
    for name, df in splits.items():
        counts = df["label"].value_counts().sort_index().to_dict()
        pos_rate = float(df["label"].mean())
        print(f"{name}: shape={df.shape}, labels={counts}, positive_rate={pos_rate:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    project_dir = model_dir.parent if model_dir.name == "contrastive_model" else model_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_final_splits(args.dataset_dir)
    maybe_subsample_train(splits, args.max_train_rows, args.seed)
    print_split_summary(splits)
    print(f"device={device}, loss_weight={args.loss_weight}")

    train_x, y_train = split_xy(splits["train"])
    medians, scaler = fit_numeric_scaler(train_x)
    cat_sizes = cat_cardinalities([df.drop(columns=["label"]) for df in splits.values()])

    datasets = {
        name: make_dataset(df, medians, scaler)
        for name, df in splits.items()
    }
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loaders = {
        name: DataLoader(datasets[name], batch_size=args.batch_size, shuffle=False)
        for name in ["original_test", "synthetic_test"]
    }

    histories: Dict[str, List[Dict[str, object]]] = {}
    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    prediction_loaders = {"val": val_loader, **test_loaders}

    def new_encoder() -> TabularEncoder:
        return TabularEncoder(
            cat_sizes=cat_sizes,
            n_num=len(NUM_COLS),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        )

    embedding_threshold = None
    contrastive_threshold = None

    if not args.skip_embedding:
        set_seed(args.seed)
        embedding_model = BuyerClassifier(
            new_encoder(),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        embedding_model, histories["embedding_mlp"] = train_supervised(
            embedding_model,
            train_loader,
            val_loader,
            y_train,
            device,
            args,
            label="embedding_mlp",
        )
        embedding_threshold, embedding_val_f1 = tune_threshold(embedding_model, val_loader, device)
        for row in evaluate_named_sets(
            embedding_model,
            test_loaders,
            device,
            {"default_0.5": 0.5, "tuned": embedding_threshold},
        ):
            row.update({"model": "embedding_mlp", "val_tuned_f1": embedding_val_f1})
            metric_rows.append(row)
        prediction_rows.extend(
            collect_prediction_rows("embedding_mlp", embedding_model, prediction_loaders, device)
        )
        torch.save(embedding_model.state_dict(), args.output_dir / "embedding_mlp.pt")

    if not args.skip_contrastive:
        set_seed(args.seed)
        if args.contrastive_mode == "group":
            pair_dataset = PositivePairDataset(datasets["train"], build_contrastive_groups(splits["train"]))
        else:
            pair_dataset = SelfPairDataset(datasets["train"])
        pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        contrastive_encoder = new_encoder()
        projection_head = ProjectionHead(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.projection_hidden_dim,
            projection_dim=args.projection_dim,
        )
        histories["contrastive_pretrain"] = train_contrastive(
            contrastive_encoder,
            projection_head,
            pair_loader,
            device,
            epochs=args.pretrain_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            cat_sizes=cat_sizes,
            contrastive_mode=args.contrastive_mode,
            cat_mask_prob=args.cat_mask_prob,
            num_mask_prob=args.num_mask_prob,
            num_noise_std=args.num_noise_std,
        )
        contrastive_model = BuyerClassifier(
            contrastive_encoder,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        contrastive_model, histories["contrastive_mlp"] = train_supervised(
            contrastive_model,
            train_loader,
            val_loader,
            y_train,
            device,
            args,
            label="contrastive_mlp",
        )
        contrastive_threshold, contrastive_val_f1 = tune_threshold(contrastive_model, val_loader, device)
        for row in evaluate_named_sets(
            contrastive_model,
            test_loaders,
            device,
            {"default_0.5": 0.5, "tuned": contrastive_threshold},
        ):
            row.update({"model": "contrastive_mlp", "val_tuned_f1": contrastive_val_f1})
            metric_rows.append(row)
        prediction_rows.extend(
            collect_prediction_rows("contrastive_mlp", contrastive_model, prediction_loaders, device)
        )
        torch.save(contrastive_model.state_dict(), args.output_dir / "contrastive_mlp.pt")

    metadata = {
        "dataset_dir": str(args.dataset_dir),
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "early_stopping_metric": "validation PR-AUC",
        "early_stopping_patience": args.patience,
        "contrastive_loss": "InfoNCE with cosine similarity",
        "contrastive_mode": args.contrastive_mode,
        "projection_head": {
            "hidden_dim": args.projection_hidden_dim,
            "projection_dim": args.projection_dim,
        },
        "cat_mask_prob": args.cat_mask_prob,
        "num_mask_prob": args.num_mask_prob,
        "num_noise_std": args.num_noise_std,
        "temperature": args.temperature,
        "loss": "BinaryFocalLoss" if args.loss_type == "focal" else "BCEWithLogitsLoss",
        "loss_type": args.loss_type,
        "loss_weight": args.loss_weight,
        "focal_alpha": args.focal_alpha if args.loss_type == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
        "threshold_tuning": "Threshold selected on validation set to maximize F1.",
        "embedding_mlp_tuned_threshold": embedding_threshold,
        "contrastive_mlp_tuned_threshold": contrastive_threshold,
        "skip_embedding": args.skip_embedding,
        "skip_contrastive": args.skip_contrastive,
    }
    write_outputs(args.output_dir, metadata, metric_rows, histories, prediction_rows)
    write_baseline_comparison(project_dir, args.output_dir, metric_rows)

    print("\nMain baseline-aligned results: original_test with tuned threshold")
    result_df = pd.DataFrame(metric_rows)
    if not result_df.empty:
        main_rows = result_df[
            (result_df["test_set"] == "original_test")
            & (result_df["threshold_type"].isin(["default_0.5", "tuned"]))
        ]
        print(
            main_rows[
                [
                    "model",
                    "threshold_type",
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
