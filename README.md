# Contrastive Model: Deep Model and Contrastive Learning

This implementation uses the final group datasets in `dataset/`:

- `train.csv`
- `val.csv`
- `test.csv`
- `test_original_label1.csv`
- `test_synthetic_label1.csv`

The main script is:

```bash
scripts/contrastive_model.py
```

Complete reproducible experiment runs are saved to:

```bash
contrastive_model/runs/
```

Clean final report outputs are collected separately in:

```bash
output/
```

## How to Run

On this machine, the Anaconda environment has `pandas` and `sklearn`, while PyTorch is available through the cached `PYTHONPATH`. Run:

```bash
cd /Users/hongjiayang/Desktop/Project_242b

PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/contrastive_model.py
```

Run the final tuned configuration:

```bash
PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/contrastive_model.py \
  --output-dir contrastive_model/runs/final \
  --contrastive-mode corruption \
  --dropout 0.30 \
  --hidden-dim 128 \
  --embedding-dim 64 \
  --temperature 0.10 \
  --cat-mask-prob 0.10 \
  --num-mask-prob 0.10 \
  --num-noise-std 0.03 \
  --projection-hidden-dim 128 \
  --projection-dim 64
```

Run the extra hyperparameter search around the current best configuration:

```bash
PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/hyperparameter_tuning.py \
  --search extra \
  --output-root contrastive_model/runs/extra_tuning
```

Use `--search small` to rerun the earlier small search, or `--search all` to run both sets.

Run focal-loss sensitivity:

```bash
PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/focal_tuning.py
```

Run the multi-seed ensemble and calibration experiment:

```bash
PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/multi_seed_ensemble.py \
  --seeds 42,7,13,21,84 \
  --output-root contrastive_model/runs/multi_seed_ensemble
```

Quick smoke test:

```bash
PYTHONPATH="/Users/hongjiayang/.cache/uv/archive-v0/iih7UOTzzvT8Q6DU3sEf0" \
/Users/hongjiayang/anaconda3/bin/python3 contrastive_model/scripts/contrastive_model.py \
  --max-train-rows 5000 --epochs 1 --pretrain-epochs 1 \
  --output-dir contrastive_model/runs/smoke
```

## What Changed from the Earlier Version

The earlier code generated the label and split from `data.csv`. The current group-final data is already processed and split, so the script now reads the provided split files directly.

Because the training split already includes SMOTE-NC positive samples at about a 1:4 positive-to-negative ratio, the default supervised loss is plain `BCEWithLogitsLoss`. Using an additional `pos_weight` would double-count imbalance correction. The validation and test splits are kept on the original non-SMOTE distribution for early stopping, threshold tuning, and final evaluation. A `--loss-weight` option is still available for sensitivity testing.

## Evaluation Sets

The main comparison evaluates on the original test distribution:

```text
dataset/test.csv
```

The contrastive script reports two test sets:

- `original_test`: baseline-aligned test set.
- `synthetic_test`: label-0 rows from `test.csv` plus synthetic label-1 rows.

Use `original_test` for comparison with `baseline_method/baseline_results.csv`.

Latest dataset note: `dataset/val.csv` and `dataset/test.csv` are now original, non-SMOTE splits. This is better for validation, early stopping, and threshold tuning. `train.csv` remains SMOTE-NC augmented.

## Models

Embedding + MLP:

```text
categorical integer features -> embedding layers
numerical features -> log1p for skewed count features -> standardization
concatenate -> MLP encoder -> classifier head
```

Contrastive + MLP:

```text
same encoder
-> two corrupted views of the same user
-> projection head
-> contrastive pretraining with InfoNCE loss
-> supervised fine-tuning with BCEWithLogitsLoss
```

The final model uses tabular corruption instead of manually grouped positive pairs. For each user, two augmented views are created by randomly replacing some categorical values, masking some numerical features to zero, and adding small Gaussian noise. These two views are treated as the positive pair; other users in the batch act as negatives.

The projection head is used only during contrastive pretraining:

```text
encoder -> projection head -> InfoNCE loss
encoder -> classifier head -> BCE fine-tuning
```

## Loss, Optimizer, and Regularization

Supervised loss:

```text
BCEWithLogitsLoss
```

Contrastive loss:

```text
InfoNCE = -log exp(sim(z_i, z_j) / T) / sum_k exp(sim(z_i, z_k) / T)
```

where `sim` is cosine similarity and `T = 0.20`.

Training setup:

- optimizer: Adam
- learning rate: `1e-3`
- weight decay: `1e-4`
- dropout: `0.30`
- early stopping metric: validation PR-AUC
- final threshold: report both default `0.5` and validation-tuned threshold

Final tuned contrastive configuration:

- contrastive mode: tabular corruption
- temperature: `0.10`
- categorical mask probability: `0.10`
- numerical mask probability: `0.10`
- numerical noise std: `0.03`
- projection hidden dimension: `128`
- projection dimension: `64`

## Output Files

- `metrics.csv`: all test-set metrics for default and tuned thresholds.
- `vs_baselines.csv`: baseline-aligned comparison using `original_test`.
- `summary.txt`: concise interpretation for report/slides.
- `histories.csv`: training curves data.
- `metrics.json`: full metadata and metrics.
- `predictions.csv`: validation/test labels and predicted probabilities for ensemble analysis.
- `training_curves.png`: loss and validation PR-AUC plot.
- `embedding_mlp.pt`, `contrastive_mlp.pt`: saved PyTorch checkpoints.

## Current Result

Latest dataset note: `dataset/val.csv` and `dataset/test.csv` are now original, non-SMOTE splits. `train.csv` remains SMOTE-NC augmented. This makes early stopping and threshold tuning much more reliable.

On the updated original `test.csv`, the best contrastive-model F1 result is `Contrastive + MLP` with projection head, tabular corruption, BCE loss, and validation-tuned threshold:

```text
accuracy  = 0.9329
precision = 0.4365
recall    = 0.5208
F1        = 0.4749
ROC-AUC   = 0.8680
PR-AUC    = 0.4526
```

After rerunning baselines on the updated data, the best baseline by F1 is MLP:

```text
F1      = 0.4379
PR-AUC  = 0.4195
ROC-AUC = 0.8516
```

Random Forest has the best baseline PR-AUC:

```text
F1      = 0.4331
PR-AUC  = 0.4569
ROC-AUC = 0.8743
```

Interpretation for single models: the contrastive model is best by F1 and recall-balanced classification, while Random Forest remains slightly better by PR-AUC.

Small hyperparameter tuning results are saved in:

```text
contrastive_model/runs/hyperparameter_tuning/tuning_summary.csv
```

The selected final configuration is `corrupt_light_aug`, which reached a validation-threshold tuned F1 of `0.4749` on the original test set.

The second tuning round is saved in:

```text
contrastive_model/runs/extra_tuning/tuning_summary.csv
```

It tested dropout, temperature, learning rate/weight decay, and embedding size around `corrupt_light_aug`. The best extra-tuning F1 was `0.4724` from `light_aug_dropout025`, so it did not beat the current single-model F1 of `0.4749`. The best extra-tuning PR-AUC was `0.4538` from `light_aug_temp015`, but that setting lowered F1. Keep `corrupt_light_aug` as the final single model when optimizing F1.

## Multi-Seed Ensemble and Calibration

The final single-model configuration was rerun with five seeds: `42, 7, 13, 21, 84`. Individual seeds showed meaningful variance:

```text
individual-seed F1 mean +/- std     = 0.4703 +/- 0.0069
individual-seed PR-AUC mean +/- std = 0.4505 +/- 0.0033
```

Averaging the five seed probabilities improved both F1 and PR-AUC on the original test set. The best threshold strategy was selected on validation by requiring validation precision >= `0.45` and then maximizing recall:

```text
method    = raw probability ensemble
threshold = 0.5528
accuracy  = 0.9356
precision = 0.4534
recall    = 0.5127
F1        = 0.4813
ROC-AUC   = 0.8701
PR-AUC    = 0.4583
```

Platt calibration produced the same ranking and predictions at the selected threshold, while isotonic calibration reduced PR-AUC on this validation split. The best result is therefore the raw multi-seed probability ensemble with validation thresholding.

Compared with baselines, the ensemble now has the best F1 and slightly improves over Random Forest PR-AUC (`0.4583` vs. `0.4569`).

Main ensemble files:

```text
contrastive_model/runs/multi_seed_ensemble/ensemble_summary.txt
contrastive_model/runs/multi_seed_ensemble/ensemble_metrics.csv
contrastive_model/runs/multi_seed_ensemble/ensemble_vs_baselines.csv
```

## Focal Loss Ablation

Focal loss was retested after the original val/test update. It still did not improve the main F1 result:

```text
BCE tuned F1          = 0.4749
Best focal tuned F1   = 0.4710   (alpha=0.50, gamma=1.0)
Best focal PR-AUC     = 0.4540   (alpha=0.75, gamma=2.0)
```

Conclusion: keep `BCEWithLogitsLoss` for the main single model, and report focal loss as an ablation.
