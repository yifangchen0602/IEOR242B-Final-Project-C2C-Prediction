"""Multi-seed ensemble and calibration for the contrastive model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


FINAL_CONFIG_ARGS = [
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
]


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    project_dir = model_dir.parent if model_dir.name == "contrastive_model" else model_dir
    parser = argparse.ArgumentParser(description="Run multi-seed ensemble and calibration.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--model-script", type=Path, default=script_dir / "contrastive_model.py")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=model_dir / "runs" / "multi_seed_ensemble",
    )
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("42,7,13,21,84"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun seed models even if predictions exist.")
    return parser.parse_args()


def run_seed(args: argparse.Namespace, seed: int) -> Path:
    seed_dir = args.output_root / "seeds" / f"seed_{seed}"
    predictions_path = seed_dir / "predictions.csv"
    if predictions_path.exists() and not args.force:
        print(f"=== Reusing seed {seed} ===", flush=True)
        return seed_dir

    cmd = [
        sys.executable,
        str(args.model_script),
        "--skip-embedding",
        "--output-dir",
        str(seed_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        *FINAL_CONFIG_ARGS,
    ]
    if args.max_train_rows is not None:
        cmd.extend(["--max-train-rows", str(args.max_train_rows)])

    print(f"=== Running seed {seed} ===", flush=True)
    subprocess.run(cmd, cwd=args.project_dir, check=True)
    return seed_dir


def metrics_from_prob(prob: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


def tune_fbeta_threshold(prob: np.ndarray, y_true: np.ndarray, beta: float = 1.0) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0

    beta2 = beta * beta
    denom = beta2 * precision[:-1] + recall[:-1] + 1e-12
    scores = (1 + beta2) * precision[:-1] * recall[:-1] / denom
    idx = int(np.nanargmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def threshold_for_precision_floor(
    prob: np.ndarray,
    y_true: np.ndarray,
    precision_floor: float,
) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0

    valid = np.where(precision[:-1] >= precision_floor)[0]
    if len(valid) == 0:
        idx = int(np.nanargmax(precision[:-1]))
    else:
        idx = int(valid[np.nanargmax(recall[:-1][valid])])
    return float(thresholds[idx]), float(recall[:-1][idx])


def threshold_for_recall_floor(
    prob: np.ndarray,
    y_true: np.ndarray,
    recall_floor: float,
) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0

    valid = np.where(recall[:-1] >= recall_floor)[0]
    if len(valid) == 0:
        idx = int(np.nanargmax(recall[:-1]))
    else:
        idx = int(valid[np.nanargmax(precision[:-1][valid])])
    return float(thresholds[idx]), float(precision[:-1][idx])


def load_seed_predictions(seed_dirs: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.removeprefix("seed_"))
        path = seed_dir / "predictions.csv"
        df = pd.read_csv(path)
        df = df[df["model"] == "contrastive_mlp"].copy()
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_ensemble_predictions(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["split", "row_id"]
    label_check = seed_predictions.groupby(index_cols)["y_true"].nunique()
    if int(label_check.max()) != 1:
        raise ValueError("Seed predictions have inconsistent labels for the same split/row_id.")

    labels = seed_predictions.groupby(index_cols, as_index=False)["y_true"].first()
    probs = (
        seed_predictions.pivot_table(
            index=index_cols,
            columns="seed",
            values="prob",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    prob_cols = [col for col in probs.columns if col not in index_cols]
    probs["prob_raw"] = probs[prob_cols].mean(axis=1)
    return labels.merge(probs[index_cols + ["prob_raw"]], on=index_cols, how="inner")


def calibrate_predictions(ensemble: pd.DataFrame) -> pd.DataFrame:
    out = ensemble.copy()
    val = out[out["split"] == "val"].copy()
    if val["y_true"].nunique() < 2:
        raise ValueError("Validation split needs both classes for calibration.")

    x_val = val["prob_raw"].to_numpy().reshape(-1, 1)
    y_val = val["y_true"].to_numpy()

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(x_val, y_val)
    out["prob_platt"] = platt.predict_proba(out["prob_raw"].to_numpy().reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val["prob_raw"].to_numpy(), y_val)
    out["prob_isotonic"] = isotonic.transform(out["prob_raw"].to_numpy())
    return out


def evaluate_methods(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_cols = {
        "raw_ensemble": "prob_raw",
        "platt": "prob_platt",
        "isotonic": "prob_isotonic",
    }
    test_sets = ["original_test", "synthetic_test"]

    for method, prob_col in method_cols.items():
        val = predictions[predictions["split"] == "val"].copy()
        val_prob = val[prob_col].to_numpy()
        val_true = val["y_true"].to_numpy()

        threshold_specs = {
            "default_0.5": (0.5, None),
            "val_f1_tuned": tune_fbeta_threshold(val_prob, val_true, beta=1.0),
            "val_f0.5_tuned": tune_fbeta_threshold(val_prob, val_true, beta=0.5),
            "val_f2_tuned": tune_fbeta_threshold(val_prob, val_true, beta=2.0),
            "val_precision_0.45_floor": threshold_for_precision_floor(val_prob, val_true, 0.45),
            "val_recall_0.55_floor": threshold_for_recall_floor(val_prob, val_true, 0.55),
        }

        for split in ["val", *test_sets]:
            split_df = predictions[predictions["split"] == split].copy()
            prob = split_df[prob_col].to_numpy()
            y_true = split_df["y_true"].to_numpy()
            for threshold_type, (threshold, val_score) in threshold_specs.items():
                row = {
                    "method": method,
                    "split": split,
                    "threshold_type": threshold_type,
                    "val_selection_score": val_score,
                    **metrics_from_prob(prob, y_true, threshold),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_individual_seeds(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_df in seed_predictions.groupby("seed"):
        val = seed_df[seed_df["split"] == "val"].copy()
        threshold, val_f1 = tune_fbeta_threshold(
            val["prob"].to_numpy(),
            val["y_true"].to_numpy(),
            beta=1.0,
        )
        for split in ["original_test", "synthetic_test"]:
            split_df = seed_df[seed_df["split"] == split].copy()
            row = {
                "seed": seed,
                "split": split,
                "threshold_type": "val_f1_tuned",
                "val_tuned_f1": val_f1,
                **metrics_from_prob(split_df["prob"].to_numpy(), split_df["y_true"].to_numpy(), threshold),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary(output_root: Path, seeds: List[int], ensemble_metrics: pd.DataFrame, seed_metrics: pd.DataFrame) -> None:
    original = ensemble_metrics[ensemble_metrics["split"] == "original_test"].copy()
    best_f1 = original.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
    best_pr = original.sort_values(["pr_auc", "f1"], ascending=False).iloc[0]

    seed_original = seed_metrics[seed_metrics["split"] == "original_test"].copy()
    seed_f1_mean = seed_original["f1"].mean()
    seed_f1_std = seed_original["f1"].std(ddof=0)
    seed_pr_mean = seed_original["pr_auc"].mean()
    seed_pr_std = seed_original["pr_auc"].std(ddof=0)

    lines = [
        "Multi-Seed Ensemble and Calibration Summary",
        "===========================================",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Individual seed stability on original_test:",
        f"- F1 mean +/- std = {seed_f1_mean:.4f} +/- {seed_f1_std:.4f}",
        f"- PR-AUC mean +/- std = {seed_pr_mean:.4f} +/- {seed_pr_std:.4f}",
        "",
        "Best ensemble/calibration result by F1 on original_test:",
        f"- method = {best_f1['method']}",
        f"- threshold_type = {best_f1['threshold_type']}",
        f"- threshold = {best_f1['threshold']:.6f}",
        f"- accuracy = {best_f1['accuracy']:.4f}",
        f"- precision = {best_f1['precision']:.4f}",
        f"- recall = {best_f1['recall']:.4f}",
        f"- F1 = {best_f1['f1']:.4f}",
        f"- ROC-AUC = {best_f1['roc_auc']:.4f}",
        f"- PR-AUC = {best_f1['pr_auc']:.4f}",
        "",
        "Best ensemble/calibration result by PR-AUC on original_test:",
        f"- method = {best_pr['method']}",
        f"- threshold_type = {best_pr['threshold_type']}",
        f"- F1 = {best_pr['f1']:.4f}",
        f"- PR-AUC = {best_pr['pr_auc']:.4f}",
        "",
        "Files:",
        "- seed_predictions.csv",
        "- individual_seed_metrics.csv",
        "- ensemble_predictions.csv",
        "- ensemble_metrics.csv",
        "- ensemble_summary.txt",
    ]
    (output_root / "ensemble_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    seed_dirs = [run_seed(args, seed) for seed in args.seeds]
    seed_predictions = load_seed_predictions(seed_dirs)
    seed_predictions.to_csv(args.output_root / "seed_predictions.csv", index=False)

    seed_metrics = summarize_individual_seeds(seed_predictions)
    seed_metrics.to_csv(args.output_root / "individual_seed_metrics.csv", index=False)

    ensemble = build_ensemble_predictions(seed_predictions)
    ensemble = calibrate_predictions(ensemble)
    ensemble.to_csv(args.output_root / "ensemble_predictions.csv", index=False)

    ensemble_metrics = evaluate_methods(ensemble)
    ensemble_metrics = ensemble_metrics.sort_values(
        ["split", "f1", "pr_auc"],
        ascending=[True, False, False],
    )
    ensemble_metrics.to_csv(args.output_root / "ensemble_metrics.csv", index=False)

    write_summary(args.output_root, args.seeds, ensemble_metrics, seed_metrics)

    original = ensemble_metrics[ensemble_metrics["split"] == "original_test"].copy()
    display_cols = [
        "method",
        "threshold_type",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    print("\n=== Ensemble/calibration: original_test sorted by F1 ===")
    print(original[display_cols].head(12).to_string(index=False))
    print(f"\nSaved ensemble outputs to: {args.output_root}")


if __name__ == "__main__":
    main()
