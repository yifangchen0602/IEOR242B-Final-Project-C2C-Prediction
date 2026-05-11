import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, f1_score, precision_score, recall_score
from data_loader             import load_splits

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')

# ── data ──────────────────────────────────────────────────────────────────────
X_train, y_train, X_val, y_val, _, _ = load_splits()

test_orig_pos  = pd.read_csv(os.path.join(DATA_DIR, 'test_original_label1.csv'))
test_synth_pos = pd.read_csv(os.path.join(DATA_DIR, 'test_synthetic_label1.csv'))
test_neg       = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
test_neg       = test_neg[test_neg['label'] == 0]

def make_test(pos_df):
    df = pd.concat([test_neg, pos_df], ignore_index=True)
    return df.drop(columns=['label']).astype(float), df['label'].astype(int)

X_orig,  y_orig  = make_test(test_orig_pos)
X_synth, y_synth = make_test(test_synth_pos)


# ── metric row ────────────────────────────────────────────────────────────────
def metrics_row(model_name, y_true, y_pred):
    return {
        'model':     model_name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ── 1. Linear Regression ──────────────────────────────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

best_thresh, best_f1 = 0.5, 0.0
for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    f1 = f1_score(y_val, (lr_model.predict(X_val) >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

lr_orig_pred  = (np.clip(lr_model.predict(X_orig),  0, 1) >= best_thresh).astype(int)
lr_synth_pred = (np.clip(lr_model.predict(X_synth), 0, 1) >= best_thresh).astype(int)
print(f"[Linear Regression] best threshold={best_thresh}")


# ── 2. Logistic Regression ────────────────────────────────────────────────────
scaler_log = StandardScaler()
X_train_sl = scaler_log.fit_transform(X_train)
X_val_sl   = scaler_log.transform(X_val)

best_log, best_f1, best_C = None, 0.0, 1
for C in [0.01, 0.1, 1, 10]:
    m = LogisticRegression(C=C, max_iter=1000, random_state=42)
    m.fit(X_train_sl, y_train)
    f1 = f1_score(y_val, m.predict(X_val_sl), zero_division=0)
    if f1 > best_f1:
        best_f1, best_C, best_log = f1, C, m

log_orig_pred  = best_log.predict(scaler_log.transform(X_orig))
log_synth_pred = best_log.predict(scaler_log.transform(X_synth))
print(f"[Logistic Regression] best C={best_C}")


# ── 3. Random Forest ──────────────────────────────────────────────────────────
best_rf, best_f1, best_rf_params = None, 0.0, {}
for n_est, max_d in itertools.product([100, 200], [10, 20, None]):
    m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                               random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    f1 = f1_score(y_val, m.predict(X_val), zero_division=0)
    if f1 > best_f1:
        best_f1, best_rf_params, best_rf = f1, {'n_estimators': n_est, 'max_depth': max_d}, m

rf_orig_pred  = best_rf.predict(X_orig)
rf_synth_pred = best_rf.predict(X_synth)
print(f"[Random Forest] best params={best_rf_params}")


# ── 4. MLP ────────────────────────────────────────────────────────────────────
scaler_mlp = StandardScaler()
X_train_sm = scaler_mlp.fit_transform(X_train)
X_val_sm   = scaler_mlp.transform(X_val)

best_mlp, best_f1, best_layers = None, 0.0, None
for layers in [(64,), (128, 64), (256, 128)]:
    m = MLPClassifier(hidden_layer_sizes=layers, max_iter=300,
                      early_stopping=True, validation_fraction=0.1,
                      random_state=42)
    m.fit(X_train_sm, y_train)
    f1 = f1_score(y_val, m.predict(X_val_sm), zero_division=0)
    if f1 > best_f1:
        best_f1, best_layers, best_mlp = f1, layers, m

mlp_orig_pred  = best_mlp.predict(scaler_mlp.transform(X_orig))
mlp_synth_pred = best_mlp.predict(scaler_mlp.transform(X_synth))
print(f"[MLP] best layers={best_layers}")


# ── compile results ───────────────────────────────────────────────────────────
rows = []
for name, orig_pred, synth_pred in [
    ("Linear Regression",   lr_orig_pred,  lr_synth_pred),
    ("Logistic Regression", log_orig_pred, log_synth_pred),
    ("Random Forest",       rf_orig_pred,  rf_synth_pred),
    ("MLP",                 mlp_orig_pred, mlp_synth_pred),
]:
    r_orig  = metrics_row(name, y_orig,  orig_pred)
    r_synth = metrics_row(name, y_synth, synth_pred)
    rows.append({
        'model':              name,
        'orig_accuracy':      r_orig['accuracy'],
        'synth_accuracy':     r_synth['accuracy'],
        'accuracy_diff':      round(r_orig['accuracy'] - r_synth['accuracy'], 4),
        'orig_precision':     r_orig['precision'],
        'synth_precision':    r_synth['precision'],
        'orig_recall':        r_orig['recall'],
        'synth_recall':       r_synth['recall'],
        'orig_f1':            r_orig['f1'],
        'synth_f1':           r_synth['f1'],
        'f1_diff':            round(r_orig['f1'] - r_synth['f1'], 4),
    })

summary = pd.DataFrame(rows)

print(f"\n{'='*70}")
print("  ABLATION STUDY — Original vs Synthetic label-1 test samples")
print(f"{'='*70}")
print(summary[['model', 'orig_accuracy', 'synth_accuracy', 'accuracy_diff',
               'orig_f1', 'synth_f1', 'f1_diff']].to_string(index=False))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_results.csv')
summary.to_csv(out, index=False)
print(f"\nSaved → {out}")
