import os
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')


def load_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

    test_all      = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_orig_pos = pd.read_csv(os.path.join(DATA_DIR, 'test_original_label1.csv'))
    # original test = all label-0 rows (never synthesised) + original label-1 rows
    test = pd.concat([test_all[test_all['label'] == 0], test_orig_pos],
                     ignore_index=True)

    def xy(df):
        return df.drop(columns=['label']).astype(float), df['label'].astype(int)

    return (*xy(train), *xy(val), *xy(test))


def report(name, y_true, y_pred, y_prob=None):
    metrics = {
        'model':     name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob), 4)
        metrics['pr_auc']  = round(average_precision_score(y_true, y_prob), 4)

    print(f"\n{'='*45}")
    print(f"  {name} — Test Results")
    print(f"{'='*45}")
    for k, v in metrics.items():
        if k != 'model':
            print(f"  {k:12s}: {v}")
    return metrics


