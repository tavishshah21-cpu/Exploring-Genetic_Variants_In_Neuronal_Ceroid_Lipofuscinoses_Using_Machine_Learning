
# ============================================================
# RENOVO ML PIPELINE - CLN3 TEMPORAL VALIDATION
# 2025 = training / threshold selection
# 2026 = untouched final test set
# ============================================================

import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
)

# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Pipeline_1_Statistics")

TRAIN_PATH = os.path.join(DATA_DIR, "CLN3_2025_CLN3_ML_READY.csv")
TEST_PATH = os.path.join(DATA_DIR, "CLN3_2026_CLN3_ML_READY.csv")

GENE = "CLN3"
RANDOM_STATE = 42
N_SPLITS = 5


def out(filename):
    return os.path.join(BASE_DIR, filename)


print("=" * 70)
print("RENOVO - CLN3 TEMPORAL VALIDATION")
print("2025 training + OOF threshold selection; 2026 final test")
print("=" * 70)

print("Input directory :", DATA_DIR)
print("Output directory:", BASE_DIR)

# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"2025 training file not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"2026 test file not found:\n{TEST_PATH}"
    )

df_train = pd.read_csv(TRAIN_PATH, low_memory=False)
df_test = pd.read_csv(TEST_PATH, low_memory=False)

print("\n2025 dataset shape:", df_train.shape)
print("2026 dataset shape:", df_test.shape)

if "label" not in df_train.columns or "label" not in df_test.columns:
    raise ValueError("Both datasets must contain a 'label' column.")

# Keep only rows with known binary labels.
for name, frame in (("2025", df_train), ("2026", df_test)):
    frame["label"] = pd.to_numeric(
        frame["label"], errors="coerce"
    )
    frame = frame.dropna(subset=["label"]).copy()
    frame = frame.loc[frame["label"].isin([0, 1])].copy()

    if name == "2025":
        df_train = frame
    else:
        df_test = frame

df_train["label"] = df_train["label"].astype(int)
df_test["label"] = df_test["label"].astype(int)

if df_train["label"].nunique() != 2:
    raise ValueError(
        "2025 training data must contain benign (0) "
        "and pathogenic (1) samples."
    )

if df_test["label"].nunique() != 2:
    raise ValueError(
        "2026 test data must contain both classes to calculate ROC-AUC."
    )

if df_train["label"].value_counts().min() < N_SPLITS:
    raise ValueError(
        f"Each class in 2025 must have at least {N_SPLITS} rows."
    )

print("\n2025 Pathogenic:", int((df_train["label"] == 1).sum()))
print("2025 Benign    :", int((df_train["label"] == 0).sum()))
print("2026 Pathogenic:", int((df_test["label"] == 1).sum()))
print("2026 Benign    :", int((df_test["label"] == 0).sum()))

# ============================================================
# 3. REMOVE EXACT VARIANT OVERLAP
# ============================================================
# If variant identifiers exist in both datasets, remove exact
# train/test overlaps from the test set.
#
# This uses variant identity only. It does not use 2026 labels
# to fit the model, select features, or select the threshold.

variant_key_candidates = ["Chr", "Start", "End", "Ref", "Alt"]

variant_key = [
    c for c in variant_key_candidates
    if c in df_train.columns and c in df_test.columns
]

overlap_count = 0

if all(c in variant_key for c in ["Chr", "Start", "Ref", "Alt"]):

    def make_variant_key(frame):
        cols = [
            c for c in ["Chr", "Start", "End", "Ref", "Alt"]
            if c in frame.columns
        ]

        return frame[cols].astype(str).apply(
            lambda row: "|".join(row.values), axis=1
        )

    train_keys = set(make_variant_key(df_train))
    test_keys = make_variant_key(df_test)

    overlap_mask = test_keys.isin(train_keys)
    overlap_count = int(overlap_mask.sum())

    if overlap_count:
        print(
            f"\nRemoving {overlap_count} 2026 rows "
            "whose exact variant is also in 2025."
        )

        df_test = df_test.loc[~overlap_mask].copy()
    else:
        print("\nNo exact variant overlap detected.")

else:
    print(
        "\nWARNING: Full variant identity columns are unavailable "
        "in both files. Exact overlap could not be checked."
    )

if df_test.empty or df_test["label"].nunique() != 2:
    raise ValueError(
        "After removing variant overlap, the 2026 test set is empty "
        "or does not contain both classes. Review the datasets."
    )

# ============================================================
# 4. SEPARATE LABELS AND FEATURES
# ============================================================

y_train = df_train["label"].astype(int).reset_index(drop=True)
y_test = df_test["label"].astype(int).reset_index(drop=True)

# Remove target, variant identifiers, and annotation fields that
# are not intended to be predictive features.
drop_cols = [
    "label",
    "Chr",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150",
]

X_train_raw = df_train.drop(
    columns=drop_cols, errors="ignore"
).reset_index(drop=True)

X_test_raw = df_test.drop(
    columns=drop_cols, errors="ignore"
).reset_index(drop=True)

# Training columns alone define the feature schema.
X_test_raw = X_test_raw.reindex(
    columns=X_train_raw.columns
)

print("\nRaw training features:", X_train_raw.shape)
print("Raw testing features :", X_test_raw.shape)

# ============================================================
# 5. DETERMINISTIC VALUE CLEANING
# ============================================================
# Median imputation and categorical encoding are fitted within
# each CV fold, rather than using information from validation rows.

def clean_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value in ("", ".", "NA", "N/A", "nan", "None"):
            return np.nan

        if ";" in value:
            value = value.split(";")[0].strip()

    return value


variant_col = "ExonicFunc.refGene"

for frame in (X_train_raw, X_test_raw):
    frame.replace(".", np.nan, inplace=True)

    for col in frame.columns:
        frame[col] = frame[col].map(clean_value)

categorical_cols = (
    [variant_col]
    if variant_col in X_train_raw.columns
    else []
)

numeric_cols = [
    c for c in X_train_raw.columns
    if c not in categorical_cols
]

for col in numeric_cols:
    X_train_raw[col] = pd.to_numeric(
        X_train_raw[col], errors="coerce"
    )

    X_test_raw[col] = pd.to_numeric(
        X_test_raw[col], errors="coerce"
    )

for col in categorical_cols:
    X_train_raw[col] = X_train_raw[col].astype(object)
    X_test_raw[col] = X_test_raw[col].astype(object)

# Remove features entirely missing in 2025.
all_missing_cols = [
    c for c in numeric_cols
    if X_train_raw[c].isna().all()
]

if all_missing_cols:
    print(
        "\nDropping features entirely missing in 2025:",
        all_missing_cols
    )

    X_train_raw.drop(columns=all_missing_cols, inplace=True)
    X_test_raw.drop(columns=all_missing_cols, inplace=True)

    numeric_cols = [
        c for c in numeric_cols
        if c not in all_missing_cols
    ]

# Remove constant numeric features based on 2025 only.
constant_cols = [
    c for c in numeric_cols
    if X_train_raw[c].nunique(dropna=True) <= 1
]

if constant_cols:
    print(
        "\nDropping constant numeric features based on 2025:",
        constant_cols
    )

    X_train_raw.drop(columns=constant_cols, inplace=True)
    X_test_raw.drop(columns=constant_cols, inplace=True)

    numeric_cols = [
        c for c in numeric_cols
        if c not in constant_cols
    ]

if not numeric_cols and not categorical_cols:
    raise ValueError("No usable feature columns remain.")

print("\nNumeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))

# ============================================================
# 6. CLASS WEIGHT FROM 2025 ONLY
# ============================================================

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())

if pos == 0 or neg == 0:
    raise ValueError("Both classes are required in 2025.")

scale_pos_weight = neg / pos

print("2025 pathogenic:", pos)
print("2025 benign    :", neg)
print(
    "scale_pos_weight (2025 only):",
    round(scale_pos_weight, 4)
)

# ============================================================
# 7. MODEL PIPELINE
# ============================================================

transformers = []

if numeric_cols:
    numeric_pipe = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ])

    transformers.append(
        ("numeric", numeric_pipe, numeric_cols)
    )

if categorical_cols:
    categorical_pipe = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers.append(
        ("categorical", categorical_pipe, categorical_cols)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False
)


def make_model():
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])


# ============================================================
# 8. OUT-OF-FOLD PREDICTIONS ON 2025
# ============================================================
# Every 2025 row receives a probability from a model that did
# not train on that row.
#
# These probabilities are used to choose the threshold.
# The 2026 data is not involved.

print("\n" + "=" * 70)
print("GENERATING 5-FOLD OUT-OF-FOLD PREDICTIONS ON 2025")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_prob = np.full(len(y_train), np.nan, dtype=float)
fold_auc = []

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(X_train_raw, y_train), start=1
):
    fold_model = make_model()

    fold_model.fit(
        X_train_raw.iloc[fit_idx],
        y_train.iloc[fit_idx]
    )

    fold_prob = fold_model.predict_proba(
        X_train_raw.iloc[val_idx]
    )[:, 1]

    oof_prob[val_idx] = fold_prob

    auc = roc_auc_score(
        y_train.iloc[val_idx],
        fold_prob
    )

    fold_auc.append(auc)

    print(f"Fold {fold} ROC-AUC: {auc:.4f}")

if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Some 2025 rows did not receive out-of-fold predictions."
    )

cv_mean = float(np.mean(fold_auc))
oof_auc = float(roc_auc_score(y_train, oof_prob))

print("Mean fold ROC-AUC:", round(cv_mean, 4))
print("Combined OOF ROC-AUC:", round(oof_auc, 4))

# ============================================================
# 9. SELECT THRESHOLD USING 2025 OOF PREDICTIONS ONLY
# ============================================================

print(
    "\nSelecting classification threshold "
    "from 2025 OOF predictions only..."
)

precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_prob
)

if len(thresholds) == 0:
    best_threshold = 0.5
else:
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1] + recall[:-1] + 1e-12
        )
    )

    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

print(
    "2025-derived threshold:",
    round(best_threshold, 6)
)

# ============================================================
# 10. FIT FINAL MODEL ON ALL 2025 DATA
# ============================================================
# No 2026 eval_set, labels, early stopping, or metrics are used
# to fit the final model.

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON ALL 2025 DATA")
print("=" * 70)

model = make_model()
model.fit(X_train_raw, y_train)

print("Final model fitted using 2025 data only.")

# ============================================================
# 11. FINAL EVALUATION ON 2026
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON 2026 TEST DATA")
print("=" * 70)

y_prob = model.predict_proba(X_test_raw)[:, 1]

# Apply the fixed threshold selected from 2025.
y_pred = (y_prob >= best_threshold).astype(int)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Pathogenic"],
    digits=4,
    zero_division=0
)

print("\nClassification report:\n")
print(report)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("=" * 70)
print("FINAL METRICS")
print("=" * 70)

print("Accuracy           :", round(accuracy, 4))
print("ROC-AUC            :", round(roc_auc, 4))
print("PR-AUC             :", round(pr_auc, 4))
print("Mean 2025 CV AUC   :", round(cv_mean, 4))
print("Combined 2025 OOF  :", round(oof_auc, 4))
print("Threshold from 2025:", round(best_threshold, 6))

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Pathogenic"]
)

disp.plot()

plt.title(
    "RENOVO - CLN3 Confusion Matrix "
    f"(Train 2025, Test 2026, "
    f"threshold={best_threshold:.3f})"
)

plt.savefig(
    out("RENOVO_CLN3_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved RENOVO_CLN3_confusion_matrix.png")

# ============================================================
# 13. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"RENOVO CLN3 (AUC={roc_auc:.4f})"
)

plt.title(
    "RENOVO - CLN3 ROC Curve "
    "(Trained 2025, Tested 2026)"
)

plt.savefig(
    out("RENOVO_CLN3_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved RENOVO_CLN3_roc_curve.png")

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

try:
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(
        "Importance",
        ascending=False
    )

    print(importance_df.head(20).to_string(index=False))

    importance_df.to_csv(
        out("RENOVO_CLN3_feature_importance.csv"),
        index=False
    )

    print("Saved RENOVO_CLN3_feature_importance.csv")

except Exception as exc:
    print("Could not create feature-importance table:", exc)
    importance_df = pd.DataFrame()

# ============================================================
# 15. SAVE 2026 PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Label": y_test.to_numpy(),

    "Actual_Class": np.where(
        y_test.to_numpy() == 1,
        "Pathogenic",
        "Benign"
    ),

    "Predicted_Label": y_pred,

    "Predicted_Class": np.where(
        y_pred == 1,
        "Pathogenic",
        "Benign"
    ),

    "Pathogenic_Probability": y_prob,
    "Predicted_Probability_Percent": y_prob * 100,

    "Classification_Threshold_From_2025_OOF": best_threshold
})

predictions_df.to_csv(
    out("RENOVO_CLN3_2026_predictions.csv"),
    index=False
)

print("Saved RENOVO_CLN3_2026_predictions.csv")

# Save 2025 out-of-fold predictions for auditing.
oof_df = pd.DataFrame({
    "Actual_Label_2025": y_train.to_numpy(),
    "OOF_Pathogenic_Probability": oof_prob,

    "OOF_Predicted_Label_At_Threshold": (
        oof_prob >= best_threshold
    ).astype(int),

    "Threshold_From_2025_OOF": best_threshold
})

oof_df.to_csv(
    out("RENOVO_CLN3_2025_oof_predictions.csv"),
    index=False
)

print("Saved RENOVO_CLN3_2025_oof_predictions.csv")

# ============================================================
# 16. SAVE FINAL MODEL AND METADATA
# ============================================================

joblib.dump(
    model,
    out("RENOVO_CLN3_model.pkl")
)

print("Saved RENOVO_CLN3_model.pkl")

metadata = {
    "gene": GENE,
    "algorithm": "XGBoost classifier in RENOVO pipeline",
    "training_year": 2025,
    "test_year": 2026,
    "threshold": best_threshold,

    "threshold_source": (
        "5-fold out-of-fold probabilities on 2025 training data"
    ),

    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,

    "feature_columns_before_preprocessing": list(
        X_train_raw.columns
    ),

    "n_train": int(len(X_train_raw)),
    "n_test": int(len(X_test_raw)),
    "random_state": RANDOM_STATE
}

joblib.dump(
    metadata,
    out("RENOVO_CLN3_model_metadata.pkl")
)

print("Saved RENOVO_CLN3_model_metadata.pkl")

# ============================================================
# 17. SAVE RESULTS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],

    "Algorithm": [
        "RENOVO pipeline - XGBoost implementation"
    ],

    "Train_year": [2025],
    "Test_year": [2026],

    "Accuracy": [round(accuracy, 4)],
    "ROC_AUC": [round(roc_auc, 4)],
    "PR_AUC": [round(pr_auc, 4)],

    "Mean_CV_ROC_AUC_2025": [round(cv_mean, 4)],
    "Combined_OOF_ROC_AUC_2025": [round(oof_auc, 4)],

    "Threshold": [best_threshold],
    "Threshold_Source": ["2025 5-fold OOF predictions"],

    "Train_size": [len(X_train_raw)],
    "Test_size": [len(X_test_raw)],

    "Train_Pathogenic": [int((y_train == 1).sum())],
    "Train_Benign": [int((y_train == 0).sum())],

    "Test_Pathogenic": [int((y_test == 1).sum())],
    "Test_Benign": [int((y_test == 0).sum())],

    "Numeric_Feature_Count": [len(numeric_cols)],
    "Categorical_Feature_Count": [len(categorical_cols)],

    "Exact_Overlap_Removed": [overlap_count]
})

summary.to_csv(
    out("RENOVO_CLN3_results_summary.csv"),
    index=False
)

print("Saved RENOVO_CLN3_results_summary.csv")

# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 70)
print("RENOVO CLN3 PIPELINE FINISHED")
print("=" * 70)

print("Training data       : 2025")
print("Threshold source    : 2025 out-of-fold predictions")
print("Final testing data  : 2026")
print("Accuracy            :", round(accuracy, 4))
print("ROC-AUC             :", round(roc_auc, 4))
print("PR-AUC              :", round(pr_auc, 4))
print("Mean 2025 CV AUC    :", round(cv_mean, 4))
print("Threshold           :", round(best_threshold, 6))

print("\nAll output files are saved in:")
print(BASE_DIR)

gc.collect()
