import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from linear_regression    import run as lr_run
from logistic_regression  import run as logreg_run
from random_forest        import run as rf_run
from mlp                  import run as mlp_run

MODELS = [
    ("Linear Regression",   lr_run),
    ("Logistic Regression", logreg_run),
    ("Random Forest",       rf_run),
    ("MLP",                 mlp_run),
]

if __name__ == '__main__':
    all_results = []
    for name, fn in MODELS:
        print(f"\n{'#'*50}")
        print(f"# {name}")
        print(f"{'#'*50}")
        all_results.append(fn())

    summary = pd.DataFrame(all_results)
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    cols = [c for c in cols if c in summary.columns]

    print(f"\n{'='*60}")
    print("  BASELINE COMPARISON — Test Set")
    print(f"{'='*60}")
    print(summary[cols].to_string(index=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_results.csv')
    summary[cols].to_csv(out, index=False)
    print(f"\nSaved → {out}")
