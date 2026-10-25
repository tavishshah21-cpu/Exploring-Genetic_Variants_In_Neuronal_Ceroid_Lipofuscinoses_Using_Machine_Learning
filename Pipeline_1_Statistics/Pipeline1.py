# ==========================================
# FINAL TRAINING PIPELINE (ROBUST & SAFE)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold

from xgboost import XGBClassifier

# --------------------------------------------------
# SAFE CLEANING FUNCTION
# --------------------------------------------------
def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Safely convert a column to numeric:
    - If object: handle '.', 'NA', '', multi-values 'a;b'
    - If already numeric: just coerce
    """
    if s.dtype == "object":
        s = s.replace([".", "NA", "nan", ""], np.nan)
        # ensure string ops only on strings
        s = s.astype(str)
        s = s.str.split(";").str[0]
        return pd.to_numeric(s, errors="coerce")
    else:
        return pd.to_numeric(s, errors="coerce")

# --------------------------------------------------
# LOAD DATA (USE YOUR FULL PATHS)
# --------------------------------------------------
train_df = pd.read_csv(
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2025_RAW.csv"
)
test_df = pd.read_csv(
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\CLN3_2026_RAW.csv"
)

train_df["__split__"] = "train"
test_df["__split__"] = "test"

df = pd.concat([train_df, test_df], axis=0)
print("📊 Combined shape:", df.shape)

# --------------------------------------------------
# CLEAN ALL FEATURES (SAFE)
# --------------------------------------------------
for col in df.columns:
    if col not in ["label", "__split__"]:
        df[col] = clean_numeric_series(df[col])

# --------------------------------------------------
# DROP HIGH-MISSING FEATURES (>80%)
# --------------------------------------------------
missing_ratio = df.isnull().mean()
df = df.loc[:, missing_ratio < 0.8]

# --------------------------------------------------
# FILL MISSING
# --------------------------------------------------
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

# --------------------------------------------------
# SPLIT BACK
# --------------------------------------------------
train_df = df[df["__split__"] == "train"].drop(columns="__split__")
test_df = df[df["__split__"] == "test"].drop(columns="__split__")

X = train_df.drop(columns=["label"])
y = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("✅ Features after cleaning:", X.shape[1])

# --------------------------------------------------
# REMOVE LEAKAGE FEATURES
# --------------------------------------------------
for col in ["Chromosome", "Start", "End"]:
    if col in X.columns:
        X = X.drop(columns=col)
        X_test = X_test.drop(columns=col)

# --------------------------------------------------
# REMOVE CONSTANT FEATURES
# --------------------------------------------------
var = VarianceThreshold(0)

X_var = var.fit_transform(X)
kept_cols = X.columns[var.get_support()]

X = pd.DataFrame(X_var, columns=kept_cols)
X_test = X_test[kept_cols]

print("✅ After variance filter:", X.shape[1])

# --------------------------------------------------
# FALLBACK IF TOO FEW FEATURES
# --------------------------------------------------
if X.shape[1] < 3:
    print("⚠️ Too few features — using curated dbNSFP set")

    important_features = [
        "LINSIGHT",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "SiPhy_29way_logOdds",
        "var_length"
    ]

    important_features = [f for f in important_features if f in train_df.columns]

    X = train_df[important_features].copy()
    X_test = test_df[important_features].copy()

    # Apply same safe cleaning
    for col in X.columns:
        X[col] = clean_numeric_series(X[col])
        X_test[col] = clean_numeric_series(X_test[col])

    X = X.fillna(X.median()).fillna(0)
    X_test = X_test.fillna(X.median()).fillna(0)

    print("✅ Using fallback features:", important_features)

# --------------------------------------------------
# FINAL NUMERIC GUARANTEE
# --------------------------------------------------
X = X.astype(float)
X_test = X_test.astype(float)

print("✅ Final feature count:", X.shape[1])

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for tr, val in kf.split(X_scaled, y):
    model.fit(X_scaled[tr], y.iloc[tr])
    y_prob = model.predict_proba(X_scaled[val])[:, 1]
    scores.append(roc_auc_score(y.iloc[val], y_prob))

print("📈 CV ROC-AUC:", np.mean(scores))

# --------------------------------------------------
# FINAL TRAIN
# --------------------------------------------------
model.fit(X_scaled, y)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
importance = model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\n🔍 Feature Importance:")
print(feat_imp)

plt.figure()
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (XGBoost)")
plt.show()

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

print("\n🚀 FINAL RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))