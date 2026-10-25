# ============================================================
# CLN2-SPECIFIC ML PIPELINE
# Train on 2025, Test on 2026
# Leakage-controlled temporal validation
# ============================================================

import os
import gc
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve
)
from sklearn.base import clone
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_cln2"

TRAIN_PATH = os.path.join(
    BASE_DIR, "CLN2_2025_CLN2_ML_READY.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "CLN2_2026_CLN2_ML_READY.csv"
)

GENE = "CLN2"
RANDOM_STATE = 42
N_SPLITS = 5

USE_CV = True
USE_SHAP = True

N_ESTIMATORS = 1200


def out(filename):
    return os.path.join(BASE_DIR, filename)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading 2025 CLN2 training data...")

df_train = pd.read_csv(TRAIN_PATH, low_memory=False)

print("2025 dataset:", df_train.shape)

print("Pathogenic:", (df_train["label"] == 1).sum())
print("Benign    :", (df_train["label"] == 0).sum())

print("\nLoading 2026 CLN2 test data...")

df_test = pd.read_csv(TEST_PATH, low_memory=False)

print("2026 dataset:", df_test.shape)

print("Pathogenic:", (df_test["label"] == 1).sum())
print("Benign    :", (df_test["label"] == 0).sum())


# ============================================================
# BASIC VALIDATION
# ============================================================

if "label" not in df_train.columns:
    raise ValueError("The 2025 dataset must contain a 'label' column.")

if "label" not in df_test.columns:
    raise ValueError("The 2026 dataset must contain a 'label' column.")

if df_train.empty or df_test.empty:
    raise ValueError("The training or test dataset is empty.")

if df_train["label"].isna().any() or df_test["label"].isna().any():
    raise ValueError("Missing target labels found.")

if not set(df_train["label"].unique()).issubset({0, 1}):
    raise ValueError("2025 labels must be 0 (Benign) or 1 (Pathogenic).")

if not set(df_test["label"].unique()).issubset({0, 1}):
    raise ValueError("2026 labels must be 0 (Benign) or 1 (Pathogenic).")

if df_train["label"].nunique() != 2:
    raise ValueError("2025 training data must contain both classes.")

if df_test["label"].nunique() != 2:
    raise ValueError("2026 test data must contain both classes.")


# ============================================================
# SEPARATE LABELS
# ============================================================

y_train = df_train["label"].astype(int).reset_index(drop=True)
y_test = df_test["label"].astype(int).reset_index(drop=True)

X_train = df_train.drop(columns=["label"]).copy()
X_test = df_test.drop(columns=["label"]).copy()

del df_train, df_test
gc.collect()


# ============================================================
# REMOVE IDENTIFIERS AND ANNOTATION TEXT
# ============================================================

# These columns are excluded from both datasets.
# Variant coordinates and identifiers can encourage memorization.
# Review this list if any column contains biologically meaningful
# information that you intentionally want to retain.

drop_cols = [
    "Chr",
    "Chromosome",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

X_train.drop(
    columns=[c for c in drop_cols if c in X_train.columns],
    inplace=True,
    errors="ignore"
)

X_test.drop(
    columns=[c for c in drop_cols if c in X_test.columns],
    inplace=True,
    errors="ignore"
)


# ============================================================
# REMOVE POTENTIAL TARGET-LEAKAGE COLUMNS
# ============================================================

# Never use a feature that directly encodes ClinVar's clinical
# classification or the target label.
#
# Adjust these names to match the actual columns in your data.
# Do not remove independent biological predictor scores simply
# because they correlate with pathogenicity.

target_leakage_columns = [
    "CLNSIG",
    "CLNSIGCONF",
    "ClinicalSignificance",
    "clinical_significance",
    "Pathogenicity",
    "Pathogenic",
    "label",
    "label_source",
    "Label",
    "classification",
    "Classification",
    "ReviewStatus",
    "ReviewStatus_stars"
]

X_train.drop(
    columns=[
        c for c in target_leakage_columns
        if c in X_train.columns
    ],
    inplace=True,
    errors="ignore"
)

X_test.drop(
    columns=[
        c for c in target_leakage_columns
        if c in X_test.columns
    ],
    inplace=True,
    errors="ignore"
)


# ============================================================
# ALIGN RAW FEATURES USING THE 2025 SCHEMA ONLY
# ============================================================

# The 2025 dataset determines the feature schema.
# 2026-only columns are not allowed to introduce new features.

common_columns = list(X_train.columns)

X_test = X_test.reindex(columns=common_columns)

# Encode categorical variant type using categories learned from
# 2025 only. Unseen categories in 2026 become all-zero indicators.

if "ExonicFunc.refGene" in X_train.columns:

    print("\nEncoding variant type using 2025 categories...")

    train_category = (
        X_train["ExonicFunc.refGene"]
        .fillna("__MISSING__")
        .astype(str)
    )

    test_category = (
        X_test["ExonicFunc.refGene"]
        .fillna("__MISSING__")
        .astype(str)
    )

    train_dummies = pd.get_dummies(
        train_category,
        prefix="vartype",
        dtype=np.int8
    )

    test_dummies = pd.get_dummies(
        test_category,
        prefix="vartype",
        dtype=np.int8
    )

    # The 2025 dummy columns are the only allowed columns.
    test_dummies = test_dummies.reindex(
        columns=train_dummies.columns,
        fill_value=0
    )

    X_train.drop(
        columns=["ExonicFunc.refGene"],
        inplace=True
    )

    X_test.drop(
        columns=["ExonicFunc.refGene"],
        inplace=True
    )

    X_train = pd.concat(
        [
            X_train.reset_index(drop=True),
            train_dummies.reset_index(drop=True)
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            X_test.reset_index(drop=True),
            test_dummies.reset_index(drop=True)
        ],
        axis=1
    )


# ============================================================
# CLEAN VALUES AND CONVERT TO NUMERIC
# ============================================================

print("\nCleaning values and converting features to numeric...")

def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in {"", ".", "NA", "N/A", "nan", "None"}:
            return np.nan

        # Use the first annotation value when a field contains
        # multiple semicolon-separated annotations.
        if ";" in value:
            value = value.split(";")[0].strip()

            if value in {"", ".", "NA", "N/A", "nan", "None"}:
                return np.nan

    return value


def convert_numeric(X):

    X = X.copy()

    for column in X.columns:

        X[column] = X[column].map(clean_value)

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X


X_train = convert_numeric(X_train)
X_test = convert_numeric(X_test)

# Remove features that contain no usable values in 2025.
# This decision is made without consulting 2026 values.

valid_columns = X_train.columns[
    X_train.notna().any(axis=0)
].tolist()

X_train = X_train.reindex(columns=valid_columns)

X_test = X_test.reindex(columns=valid_columns)

print("Biological/numeric features:", X_train.shape[1])


# ============================================================
# MISSING INDICATORS
# ============================================================

print("\nCreating missing-value indicators...")

# Indicators are created before imputation so that the model can
# distinguish missing annotations from actual numeric values.

train_missing = X_train.isna().astype(np.int8)
test_missing = X_test.isna().astype(np.int8)

train_missing.columns = [
    f"{c}_missing" for c in train_missing.columns
]

test_missing.columns = [
    f"{c}_missing" for c in test_missing.columns
]

# Use only missing-indicator columns derived from 2025 features.
test_missing = test_missing.reindex(
    columns=train_missing.columns,
    fill_value=0
)

X_train = pd.concat(
    [
        X_train.reset_index(drop=True),
        train_missing.reset_index(drop=True)
    ],
    axis=1
)

X_test = pd.concat(
    [
        X_test.reset_index(drop=True),
        test_missing.reset_index(drop=True)
    ],
    axis=1
)

# Ensure identical column names and ordering.
X_test = X_test.reindex(
    columns=X_train.columns
)

print("Final feature count:", X_train.shape[1])


# ============================================================
# IMPUTATION — FIT ON 2025 ONLY
# ============================================================

print("\nFitting missing-value imputation on 2025 only...")

# Calculate medians from training data only.
# An all-missing training feature has already been removed.

training_medians = X_train.median(axis=0)

X_train = X_train.fillna(training_medians)
X_test = X_test.fillna(training_medians)

# Any residual missing values are filled using training medians.
# Normally, there should be none after this step.

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Enforce numeric data types.
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

assert list(X_train.columns) == list(X_test.columns)
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

print("Training matrix:", X_train.shape)
print("Testing matrix :", X_test.shape)


# ============================================================
# CLASS IMBALANCE — 2025 ONLY
# ============================================================

print("\nCalculating class imbalance from 2025...")

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())

if pos == 0 or neg == 0:
    raise ValueError("Both classes must exist in 2025 training data.")

# Preserve the standard XGBoost positive-class weighting.
scale_pos_weight = neg / pos

print("Pathogenic:", pos)
print("Benign    :", neg)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def make_model(n_estimators=N_ESTIMATORS):

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# 5-FOLD OUT-OF-FOLD PREDICTIONS ON 2025
# ============================================================

print("\nGenerating out-of-fold probabilities on 2025...")

# Each row receives a probability from a model that did not train
# on that row. These predictions are used to select the threshold.

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_prob = np.full(
    len(X_train),
    np.nan,
    dtype=np.float64
)

fold_auc_scores = []

for fold, (fit_idx, valid_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1
):

    print(f"\nTraining fold {fold}/{N_SPLITS}...")

    X_fold_train = X_train.iloc[fit_idx]
    y_fold_train = y_train.iloc[fit_idx]

    X_fold_valid = X_train.iloc[valid_idx]
    y_fold_valid = y_train.iloc[valid_idx]

    fold_model = make_model()

    # No 2026 data is supplied to fit().
    # No 2026 early stopping or model selection is performed.

    fold_model.fit(
        X_fold_train,
        y_fold_train,
        verbose=False
    )

    fold_prob = fold_model.predict_proba(
        X_fold_valid
    )[:, 1]

    oof_prob[valid_idx] = fold_prob

    fold_auc = roc_auc_score(
        y_fold_valid,
        fold_prob
    )

    fold_auc_scores.append(fold_auc)

    print(f"Fold {fold} ROC-AUC: {fold_auc:.4f}")

    del fold_model
    gc.collect()


if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Some training rows did not receive out-of-fold predictions."
    )

cv_mean = float(np.mean(fold_auc_scores))

print("\n5-Fold ROC-AUC scores:", np.round(fold_auc_scores, 4))
print("Mean 5-Fold ROC-AUC:", round(cv_mean, 4))

oof_auc = roc_auc_score(y_train, oof_prob)

print("Combined OOF ROC-AUC:", round(oof_auc, 4))


# ============================================================
# SELECT THRESHOLD USING 2025 ONLY
# ============================================================

print("\nSelecting classification threshold from 2025 OOF predictions...")

precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_prob
)

# precision and recall contain one extra point compared with
# thresholds. Therefore, use precision[:-1] and recall[:-1].

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    /
    (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)

if len(thresholds) == 0:
    raise RuntimeError(
        "Unable to calculate a threshold from 2025 predictions."
    )

best_idx = int(np.nanargmax(f1_scores))

best_threshold = float(thresholds[best_idx])

print("Selected threshold from 2025:", round(best_threshold, 6))
print("2025 OOF F1 at selected threshold:", round(f1_scores[best_idx], 4))


# Save the threshold-selection record.

threshold_record = pd.DataFrame({
    "Training_year": [2025],
    "Method": ["5-fold out-of-fold predictions"],
    "Selection_metric": ["F1-score"],
    "Threshold": [best_threshold],
    "OOF_F1": [float(f1_scores[best_idx])],
    "OOF_ROC_AUC": [float(oof_auc)]
})

threshold_record.to_csv(
    out("CLN2_2025_threshold_selection.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL MODEL ON ALL 2025 DATA
# ============================================================

print("\nTraining final XGBoost model on all 2025 data...")

model = make_model()

# Crucially, no 2026 eval_set is supplied.
# The number of trees is fixed in advance.

model.fit(
    X_train,
    y_train,
    verbose=100
)

print("Final model trained on 2025 only.")


# ============================================================
# EVALUATE ON 2026 — THRESHOLD ALREADY FIXED
# ============================================================

print("\nEvaluating on 2026 test data...")

# Do not alter the threshold based on these results.

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(y_test, y_pred)

roc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

print("\nClassification Report — 2026")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Pathogenic"],
        zero_division=0
    )
)

print("Accuracy :", round(acc, 4))
print("ROC-AUC  :", round(roc, 4))
print("PR-AUC   :", round(pr_auc, 4))
print("Threshold:", round(best_threshold, 6))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Pathogenic"]
)

disp.plot()

plt.title(
    "CLN2 — 2026 Test Set "
    f"(2025 threshold={best_threshold:.3f})"
)

plt.savefig(
    out("CLN2_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN2_confusion_matrix.png")


# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"CLN2 XGBoost 2025 to 2026 (AUC={roc:.4f})"
)

plt.title("CLN2 ROC Curve — Trained 2025, Tested 2026")

plt.savefig(
    out("CLN2_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN2_roc_curve.png")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance")

feat = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feat = feat.sort_values(
    "Importance",
    ascending=False
)

print(feat.head(20).to_string(index=False))

feat.to_csv(
    out("CLN2_feature_importance.csv"),
    index=False
)

print("Saved CLN2_feature_importance.csv")


# ============================================================
# SHAP — EXPLANATION ONLY
# ============================================================

if USE_SHAP:

    try:

        import shap

        print("\nRunning SHAP...")

        # SHAP uses 2026 observations only to explain predictions.
        # They are not used to fit or select the model.

        shap_sample = (
            X_test
            if len(X_test) <= 500
            else X_test.sample(
                n=500,
                random_state=RANDOM_STATE
            )
        )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(shap_sample)

        shap.summary_plot(
            shap_values,
            shap_sample,
            show=False
        )

        plt.title("CLN2 SHAP Summary — 2026 Test Set")

        plt.tight_layout()

        plt.savefig(
            out("CLN2_shap_summary.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved CLN2_shap_summary.png")

    except Exception as e:

        print("SHAP failed:", e)


# ============================================================
# SAVE MODEL AND REQUIRED METADATA
# ============================================================

model_path = out("CLN2_xgb_model.pkl")

joblib.dump(model, model_path)

print("\nModel saved:", model_path)

# Save metadata required for applying the same feature schema,
# imputation values and threshold to future input data.

metadata = {
    "gene": GENE,
    "training_year": 2025,
    "test_year": 2026,
    "feature_columns": list(X_train.columns),
    "training_medians": training_medians.to_dict(),
    "threshold": best_threshold,
    "threshold_source": "2025 out-of-fold predictions",
    "threshold_metric": "F1-score",
    "n_estimators": N_ESTIMATORS,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE
}

joblib.dump(
    metadata,
    out("CLN2_xgb_metadata.pkl")
)

print("Saved CLN2_xgb_metadata.pkl")


# ============================================================
# SAVE RESULTS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Train_year": [2025],
    "Test_year": [2026],
    "Accuracy": [round(acc, 4)],
    "ROC_AUC": [round(roc, 4)],
    "PR_AUC": [round(pr_auc, 4)],
    "CV_ROC_2025": [round(cv_mean, 4)],
    "OOF_ROC_AUC_2025": [round(oof_auc, 4)],
    "Threshold": [round(best_threshold, 6)],
    "Threshold_source": ["2025 OOF predictions"],
    "Train_size": [len(X_train)],
    "Test_size": [len(X_test)],
    "Train_Pathogenic": [pos],
    "Train_Benign": [neg],
    "Test_Pathogenic": [int((y_test == 1).sum())],
    "Test_Benign": [int((y_test == 0).sum())]
})

summary.to_csv(
    out("CLN2_results_summary.csv"),
    index=False
)

print("Saved CLN2_results_summary.csv")


# ============================================================
# SAVE 2026 PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual_label": y_test,
    "Predicted_pathogenic_probability": y_prob,
    "Predicted_label": y_pred,
    "Threshold_used": best_threshold
})

predictions.to_csv(
    out("CLN2_2026_predictions.csv"),
    index=False
)

print("Saved CLN2_2026_predictions.csv")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("CLN2 PIPELINE FINISHED")
print("Training dataset : 2025")
print("Threshold source : 2025 OOF predictions")
print("Test dataset     : 2026")
print("2026 used for    : Final evaluation only")
print("=" * 60)