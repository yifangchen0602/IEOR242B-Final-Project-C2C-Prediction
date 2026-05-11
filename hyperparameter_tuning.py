"""Small hyperparameter search for the contrastive model.

The search focuses on the improved contrastive setup:

- tabular corruption views,
- projection head,
- InfoNCE pretraining,
- supervised fine-tuning.

It skips repeated Embedding + MLP training inside each run and summarizes the
Contrastive + MLP metrics on the baseline-aligned original_test split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SMALL_CONFIGS = [
    {
        "name": "corrupt_default",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.30",
            "--hidden-dim",
            "128",
            "--embedding-dim",
            "64",
            "--temperature",
            "0.20",
            "--cat-mask-prob",
            "0.15",
            "--num-mask-prob",
            "0.15",
            "--num-noise-std",
            "0.05",
            "--projection-hidden-dim",
            "128",
            "--projection-dim",
            "64",
        ],
    },
    {
        "name": "corrupt_light_aug",
        "args": [
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
        ],
    },
    {
        "name": "corrupt_more_capacity",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.30",
            "--hidden-dim",
            "256",
            "--embedding-dim",
            "64",
            "--temperature",
            "0.20",
            "--cat-mask-prob",
            "0.15",
            "--num-mask-prob",
            "0.15",
            "--num-noise-std",
            "0.05",
            "--projection-hidden-dim",
            "256",
            "--projection-dim",
            "64",
        ],
    },
    {
        "name": "corrupt_dropout02",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.20",
            "--hidden-dim",
            "128",
            "--embedding-dim",
            "64",
            "--temperature",
            "0.20",
            "--cat-mask-prob",
            "0.15",
            "--num-mask-prob",
            "0.15",
            "--num-noise-std",
            "0.05",
            "--projection-hidden-dim",
            "128",
            "--projection-dim",
            "64",
        ],
    },
]


EXTRA_CONFIGS = [
    {
        "name": "light_aug_dropout02",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.20",
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
        ],
    },
    {
        "name": "light_aug_dropout025",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.25",
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
        ],
    },
    {
        "name": "light_aug_temp007",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.30",
            "--hidden-dim",
            "128",
            "--embedding-dim",
            "64",
            "--temperature",
            "0.07",
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
        ],
    },
    {
        "name": "light_aug_temp015",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.30",
            "--hidden-dim",
            "128",
            "--embedding-dim",
            "64",
            "--temperature",
            "0.15",
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
        ],
    },
    {
        "name": "light_aug_lr0007_wd5e5",
        "args": [
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
            "--lr",
            "0.0007",
            "--weight-decay",
            "0.00005",
        ],
    },
    {
        "name": "light_aug_embed128",
        "args": [
            "--contrastive-mode",
            "corruption",
            "--dropout",
            "0.30",
            "--hidden-dim",
            "128",
            "--embedding-dim",
            "128",
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
        ],
    },
]


def get_configs(search: str) -> list[dict]:
    if search == "small":
        return SMALL_CONFIGS
    if search == "extra":
        return EXTRA_CONFIGS
    return SMALL_CONFIGS + EXTRA_CONFIGS


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    project_dir = model_dir.parent if model_dir.name == "contrastive_model" else model_dir
    parser = argparse.ArgumentParser(description="Run a small contrastive-model hyperparameter search")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--model-script", type=Path, default=script_dir / "contrastive_model.py")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=model_dir / "runs" / "hyperparameter_tuning",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--search", choices=["small", "extra", "all"], default="extra")
    parser.add_argument("--force", action="store_true", help="Rerun configs even if metrics.csv already exists.")
    return parser.parse_args()


def run_config(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    output_dir = args.output_root / config["name"]
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and not args.force:
        print(f"\n=== Reusing {config['name']} ===", flush=True)
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "config", config["name"])
        return metrics

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
        *config["args"],
    ]
    if args.max_train_rows is not None:
        cmd.extend(["--max-train-rows", str(args.max_train_rows)])

    print(f"\n=== Running {config['name']} ===", flush=True)
    subprocess.run(cmd, cwd=args.project_dir, check=True)

    metrics = pd.read_csv(output_dir / "metrics.csv")
    metrics.insert(0, "config", config["name"])
    return metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for config in get_configs(args.search):
        all_rows.append(run_config(args, config))

    all_metrics = pd.concat(all_rows, ignore_index=True)
    all_metrics.to_csv(args.output_root / "tuning_all_metrics.csv", index=False)

    summary = all_metrics[all_metrics["test_set"] == "original_test"].copy()
    summary = summary.sort_values(["f1", "pr_auc"], ascending=False)
    summary.to_csv(args.output_root / "tuning_summary.csv", index=False)

    default_summary = summary[summary["threshold_type"] == "default_0.5"].copy()
    default_summary.to_csv(args.output_root / "tuning_summary_default.csv", index=False)

    tuned_summary = summary[summary["threshold_type"] == "tuned"].copy()
    tuned_summary.to_csv(args.output_root / "tuning_summary_tuned.csv", index=False)

    display_cols = [
        "config",
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
    print("\n=== Tuning summary: original_test, sorted by F1 ===")
    print(summary[display_cols].head(12).to_string(index=False))
    print(f"\nSaved tuning outputs to: {args.output_root}")


if __name__ == "__main__":
    main()
