import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

# Load data
data = pd.read_csv("data.csv")

# Feature selection
features = [
    'country', 'type', 'language', 'socialNbFollowers', 'socialNbFollows',
    'socialProductsLiked', 'productsListed', 'productsSold', 'productsPassRate',
    'productsWished', 'productsBought', 'gender', 'civilityTitle', 'hasAnyApp',
    'hasIosApp', 'hasProfilePicture', 'daysSinceLastLogin', 'seniority',
    'seniorityAsMonths', 'seniorityAsYears', 'countryCode'
]
df_processed = data[features].copy()

# Binary encoding for boolean columns
bool_cols = ['hasAnyApp', 'hasIosApp', 'hasProfilePicture']
for col in bool_cols:
    df_processed[col] = df_processed[col].astype(int)

# Gender encoding
unique_genders = df_processed['gender'].dropna().unique()
if len(unique_genders) == 2:
    gender_map = {unique_genders[0]: 0, unique_genders[1]: 1}
    df_processed['gender'] = df_processed['gender'].map(gender_map)
else:
    df_processed['gender'] = pd.factorize(df_processed['gender'])[0]

# Engineered feature: effective sales
df_processed['truesold'] = df_processed['productsPassRate'] * df_processed['productsSold']

# Engineered feature: login recency bucket
def categorize_login(days):
    if pd.isna(days):
        return np.nan
    elif days < 60:
        return 1
    elif days <= 365:
        return 2
    else:
        return 3

df_processed['popularity'] = df_processed['daysSinceLastLogin'].apply(categorize_login)
df_processed = df_processed.drop(columns=['daysSinceLastLogin'])

# Integer encoding for categorical columns
cat_cols = ['country', 'type', 'language', 'civilityTitle', 'countryCode']
for col in cat_cols:
    df_processed[col] = pd.factorize(df_processed[col])[0]

# Define target: top 20% buyers -> label 1
percentile_ranks = df_processed['productsBought'].rank(pct=True)
y = (percentile_ranks >= 0.8).astype(int)

print("Label distribution (Count):")
print(y.value_counts())
print("\nLabel distribution (Percentage):")
print(y.value_counts(normalize=True))

# Train / val / test split (70 / 15 / 15)
X = df_processed.drop(columns=['productsBought'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

# SMOTE-NC: categorical feature indices in X
# 0:country  1:type  2:language  10:gender  11:civilityTitle
# 12:hasAnyApp  13:hasIosApp  14:hasProfilePicture  18:countryCode  20:popularity
cat_idx = [0, 1, 2, 10, 11, 12, 13, 14, 18, 20]

def apply_smote(X, y, cat_idx, ratio=1/4):
    sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=ratio, random_state=42)
    X_arr, y_arr = sm.fit_resample(X.to_numpy(), y.to_numpy())
    X_res = pd.DataFrame(X_arr, columns=X.columns)
    y_res = pd.Series(y_arr, name=y.name)
    return X_res, y_res

print("\nApplying SMOTE-NC (1:4 ratio) ...")

X_train_res, y_train_res = apply_smote(X_train, y_train, cat_idx)
X_valid_res, y_valid_res = apply_smote(X_valid, y_valid, cat_idx)

# For test: track original vs synthetic rows
n_test_orig = len(X_test)
X_test_res, y_test_res = apply_smote(X_test, y_test, cat_idx)

print(f"Train after SMOTE: {X_train_res.shape}, label counts: {y_train_res.value_counts().to_dict()}")
print(f"Val   after SMOTE: {X_valid_res.shape}, label counts: {y_valid_res.value_counts().to_dict()}")
print(f"Test  after SMOTE: {X_test_res.shape},  label counts: {y_test_res.value_counts().to_dict()}")

# Save CSVs
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(out, exist_ok=True)

train_df = X_train_res.copy(); train_df['label'] = y_train_res.values
valid_df = X_valid_res.copy(); valid_df['label'] = y_valid_res.values
test_df  = X_test_res.copy();  test_df['label']  = y_test_res.values

train_df.to_csv(os.path.join(out, "train.csv"), index=False)
valid_df.to_csv(os.path.join(out, "val.csv"),   index=False)
test_df.to_csv( os.path.join(out, "test.csv"),  index=False)

# Split test into original-label-1 and synthetic-label-1
test_original  = test_df.iloc[:n_test_orig]
test_synthetic = test_df.iloc[n_test_orig:]

test_original[test_original['label'] == 1].to_csv(
    os.path.join(out, "test_original_label1.csv"), index=False)
test_synthetic[test_synthetic['label'] == 1].to_csv(
    os.path.join(out, "test_synthetic_label1.csv"), index=False)

print(f"\nSaved 5 CSVs to: {out}")
print(f"  train.csv               — {len(train_df)} rows")
print(f"  val.csv                 — {len(valid_df)} rows")
print(f"  test.csv                — {len(test_df)} rows")
print(f"  test_original_label1.csv  — {len(test_original[test_original['label']==1])} rows")
print(f"  test_synthetic_label1.csv — {len(test_synthetic[test_synthetic['label']==1])} rows")
