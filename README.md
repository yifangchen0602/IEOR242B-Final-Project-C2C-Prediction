# Contrastive Representation Learning for High-Value Buyer Prediction

This project applies deep tabular modeling and contrastive representation learning to predict high-value buyers in a C2C fashion marketplace. The task is formulated as an imbalanced binary classification problem, where the model predicts whether a user belongs to the high-value buyer class.

The final model combines categorical embeddings, numerical feature normalization, contrastive pretraining, supervised fine-tuning, and multi-seed probability ensembling.

---

## Project Structure

```text
Project_242b/
│
├── dataset/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── test_original_label1.csv
│   └── test_synthetic_label1.csv
│
├── output/
│   ├── baseline_ablation_results.csv
│   ├── ensemble_metrics.csv
│   ├── ensemble_summary.txt
│   ├── final_best_comparison.csv
│   ├── final_best_comparison.txt
│   └── training_curves.png
│
├── 242b_proj.py
├── ablation_study.py
├── contrastive_model.py
├── data_loader.py
├── focal_tuning.py
└── README.md
```

---

## Dataset

The implementation uses the final group datasets stored in `dataset/`:

- `train.csv`
- `val.csv`
- `test.csv`
- `test_original_label1.csv`
- `test_synthetic_label1.csv`

The current data split follows the final experimental setting:

- `train.csv` contains SMOTE-NC augmented training samples.
- `val.csv` and `test.csv` use the original non-SMOTE distribution.
- Validation and test data are not oversampled, which makes early stopping, threshold tuning, and final evaluation more reliable.

The main evaluation is conducted on:

```text
dataset/test.csv
```

This test set represents the original test distribution and is used for comparison with baseline models.

---

## How to Run

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn torch matplotlib
```

Run the main contrastive learning model:

```bash
python contrastive_model.py
```

Run the baseline and ablation study:

```bash
python ablation_study.py
```

Run focal-loss sensitivity analysis:

```bash
python focal_tuning.py
```

The final outputs are saved in:

```text
output/
```

---

## Model Overview

### Embedding + MLP

The supervised deep tabular baseline uses categorical embeddings and normalized numerical features.

```text
categorical features -> embedding layers
numerical features -> preprocessing and standardization
concatenate features -> MLP encoder -> classifier head
```

### Contrastive + MLP

The proposed model adds contrastive pretraining before supervised fine-tuning.

```text
same encoder
-> two corrupted views of the same user
-> projection head
-> InfoNCE contrastive pretraining
-> supervised fine-tuning with BCEWithLogitsLoss
```

During contrastive pretraining, two augmented views are generated from the same user record. The augmentation is based on tabular corruption:

- random replacement of selected categorical values
- numerical feature masking
- small Gaussian noise added to numerical features

The two views of the same user are treated as a positive pair. Other users in the batch are treated as negative samples.

The projection head is used only during contrastive pretraining:

```text
encoder -> projection head -> InfoNCE loss
encoder -> classifier head -> BCE fine-tuning
```

---

## Loss Functions

### Supervised Loss

The main supervised objective is:

```text
BCEWithLogitsLoss
```

Because the training split already includes SMOTE-NC positive samples, the default setting does not use an additional positive-class weight. Adding `pos_weight` would double-count imbalance correction.

### Contrastive Loss

The contrastive pretraining stage uses InfoNCE loss:

```text
InfoNCE = -log exp(sim(z_i, z_j) / T) / sum_k exp(sim(z_i, z_k) / T)
```

where `sim` denotes cosine similarity and `T` is the temperature parameter.

---

## Final Contrastive Configuration

The final single-model contrastive configuration is:

```text
contrastive mode: tabular corruption
dropout: 0.30
hidden dimension: 128
embedding dimension: 64
temperature: 0.10
categorical mask probability: 0.10
numerical mask probability: 0.10
numerical noise std: 0.03
projection hidden dimension: 128
projection dimension: 64
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-4
early stopping metric: validation PR-AUC
threshold selection: validation-tuned threshold
```

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC

Because the task is highly imbalanced, F1 score and PR-AUC are the most important metrics. F1 measures threshold-dependent classification quality, while PR-AUC evaluates ranking performance under class imbalance.

---

## Experimental Results

### Best Baseline Model

After rerunning baselines on the updated data, the best baseline by F1 is the MLP model:

```text
F1      = 0.4379
PR-AUC  = 0.4195
ROC-AUC = 0.8516
```

Random Forest achieves the best baseline PR-AUC:

```text
F1      = 0.4331
PR-AUC  = 0.4569
ROC-AUC = 0.8743
```

### Best Single Contrastive Model

The best single contrastive model uses tabular corruption, a projection head, BCE loss, and validation-tuned thresholding:

```text
accuracy  = 0.9329
precision = 0.4365
recall    = 0.5208
F1        = 0.4749
ROC-AUC   = 0.8680
PR-AUC    = 0.4526
```

Compared with the best MLP baseline, the contrastive model improves F1 from `0.4379` to `0.4749`.

### Multi-Seed Ensemble

The final contrastive configuration was rerun with five random seeds:

```text
42, 7, 13, 21, 84
```

The individual seed results show moderate variance:

```text
individual-seed F1 mean +/- std     = 0.4703 +/- 0.0069
individual-seed PR-AUC mean +/- std = 0.4505 +/- 0.0033
```

Averaging predicted probabilities across the five models further improves performance. The best threshold is selected on the validation set by requiring validation precision >= `0.45` and then maximizing recall.

The final ensemble result is:

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

The raw multi-seed probability ensemble achieves the best overall result. It improves over both the best MLP baseline and the best single contrastive model by F1. It also slightly improves over Random Forest in PR-AUC.

---

## Ablation Study

The ablation study compares the following model variants:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Nonlinear tree-based baseline |
| MLP | Standard supervised neural network |
| Embedding + MLP | Deep tabular model with categorical embeddings |
| Contrastive + MLP | Contrastive pretraining followed by supervised fine-tuning |
| Multi-seed Contrastive Ensemble | Probability averaging across independently trained contrastive models |

The main improvement pattern is:

```text
MLP F1                         = 0.4379
Embedding + MLP F1             = 0.4653
Single Contrastive + MLP F1    = 0.4749
Multi-seed Contrastive F1      = 0.4813
```

This suggests that categorical embeddings improve tabular representation quality, contrastive pretraining provides additional representation-level gains, and multi-seed ensembling improves robustness.

---

## Focal Loss Sensitivity

Focal loss was also tested as an alternative supervised objective. It did not improve the main F1 result:

```text
BCE tuned F1          = 0.4749
Best focal tuned F1   = 0.4710
Best focal PR-AUC     = 0.4540
```

The best focal-loss F1 was obtained with:

```text
alpha = 0.50
gamma = 1.0
```

The best focal-loss PR-AUC was obtained with:

```text
alpha = 0.75
gamma = 2.0
```

Since focal loss does not outperform BCE on the main F1 metric, the final model keeps `BCEWithLogitsLoss`.

---

## Output Files

Final report-ready outputs are saved in `output/`:

| File | Description |
|---|---|
| `baseline_ablation_results.csv` | Baseline and ablation comparison results |
| `ensemble_metrics.csv` | Multi-seed ensemble metrics |
| `ensemble_summary.txt` | Summary of ensemble performance |
| `final_best_comparison.csv` | Final comparison across best-performing models |
| `final_best_comparison.txt` | Text summary of final model comparison |
| `training_curves.png` | Training loss and validation PR-AUC curves |

---

## Key Finding

The final multi-seed contrastive ensemble achieves the best overall classification performance:

```text
F1      = 0.4813
PR-AUC  = 0.4583
ROC-AUC = 0.8701
```

The results indicate that contrastive representation learning improves high-value buyer prediction beyond standard supervised tabular models. The strongest gains come from combining categorical embeddings, tabular corruption-based contrastive pretraining, validation-based threshold tuning, and probability averaging across multiple random seeds.
