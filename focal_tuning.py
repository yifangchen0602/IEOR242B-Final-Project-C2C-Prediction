"""Focal-loss sensitivity runs for the contrastive model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


CONFIGS = [
    {"name": "focal_a025_g1", "alpha": 0.25, "gamma": 1.0},
    {"name": "focal_a025_g2", "alpha": 0.25, "gamma": 2.0},
    {"name": "focal_a050_g1", "alpha": 0.50, "gamma": 1.0},
    {"name": "focal_a075_g2", "alpha": 0.75, "gamma": 2.0},
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    project_dir = model_dir.parent if model_dir.name == "contrastive_model" else model_dir
    parser = argparse.ArgumentParser(description="Run focal-loss experiments")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--model-script", type=Path, default=script_dir / "contrastive_model.py")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=model_dir / "runs" / "focal_tuning",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def run_config(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    output_dir = args.output_root / config["name"]
    cmd = [
        sys.executable,
        str(args.model_script),
        "--skip-embedding",
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--contrastive-mode",
        "corruption",
        "--dropout",
        "0.30",
        "--hidden-dim",
        "128",
        "--embedding-dim",
        "64",
        "--temperature",
        "0.10",
        "--cat-mask-prob",
        "0.10",
        "--num-mask-prob",
        "0.10",
        "--num-noise-std",
        "0.03",
        "--projection-hidden-dim",
        "128",
        "--projection-dim",
        "64",
        "--loss-type",
        "focal",
        "--focal-alpha",
        str(config["alpha"]),
        "--focal-gamma",
        str(config["gamma"]),
    ]
    if args.max_train_rows is not None:
        cmd.extend(["--max-train-rows", str(args.max_train_rows)])

    print(f"\n=== Running {config['name']} ===", flush=True)
    subprocess.run(cmd, cwd=args.project_dir, check=True)

    metrics = pd.read_csv(output_dir / "metrics.csv")
    metrics.insert(0, "config", config["name"])
    metrics.insert(1, "focal_alpha", config["alpha"])
    metrics.insert(2, "focal_gamma", config["gamma"])
    return metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows = [run_config(args, config) for config in CONFIGS]
    all_metrics = pd.concat(all_rows, ignore_index=True)
    all_metrics.to_csv(args.output_root / "focal_all_metrics.csv", index=False)

    summary = all_metrics[
        (all_metrics["test_set"] == "original_test")
        & (all_metrics["threshold_type"] == "default_0.5")
    ].copy()
    summary = summary.sort_values(["f1", "pr_auc"], ascending=False)
    summary.to_csv(args.output_root / "focal_summary.csv", index=False)

    display_cols = [
        "config",
        "focal_alpha",
        "focal_gamma",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    print("\n=== Focal summary: original_test, default threshold 0.5 ===")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved focal tuning outputs to: {args.output_root}")


if __name__ == "__main__":
    main()
