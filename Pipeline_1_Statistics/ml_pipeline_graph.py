
# ==========================================
# CLN3-SPECIFIC ML PIPELINE
# Train on 2025, Test on 2026
# Threshold selected using 2025 OOF predictions
# No 2026 data used during model fitting or
# threshold selection
# ==========================================

import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
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

from xgboost import XGBClassifier


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR, "CLN3_2025_CLN3_ML_READY.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "CLN3_2026_CLN3_ML_READY.csv"
)


def out(filename):
    return os.path.join(BASE_DIR, filename)


GENE = "CLN3"
RANDOM_STATE = 42

USE_CV = True
USE_SHAP = True

N_SPLITS = 5
FINAL_ESTIMATORS = 1200
CV_ESTIMATORS = 300
THRESHOLD_ESTIMATORS = 300

LABEL_COLUMN = "label"

# Identifiers and descriptive annotations are not used
# as model predictors.
DROP_COLUMNS = [
    "Chr",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

VARIANT_TYPE_COLUMN = "ExonicFunc.refGene"

# Identifier columns retained temporarily for overlap checks.
IDENTIFIER_COLUMNS = ["Chr", "Start", "Ref", "Alt"]


# ==========================================
# VALIDATE AND LOAD DATA
# ==========================================

print("\n" + "=" * 60)
print("CLN3 ML PIPELINE: 2025 TRAINING -> 2026 TEST")
print("=" * 60)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"2025 training file not found: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"2026 test file not found: {TEST_PATH}"
    )

print("\nLoading 2025 training data...")
df_train = pd.read_csv(TRAIN_PATH, low_memory=False)

print("Loading 2026 test data...")
df_test = pd.read_csv(TEST_PATH, low_memory=False)

for name, df in [
    ("2025 training", df_train),
    ("2026 test", df_test)
]:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{name} data does not contain '{LABEL_COLUMN}'."
        )

    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(
            f"{name} data contains missing labels."
        )

    labels = set(pd.to_numeric(
        df[LABEL_COLUMN], errors="coerce"
    ).dropna().unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{name} labels must be 0 (benign) or "
            f"1 (pathogenic). Found: {labels}"
        )

    if len(labels) != 2:
        raise ValueError(
            f"{name} data must contain both label classes."
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN], errors="raise"
    ).astype(int)

print("\n2025 dataset:", df_train.shape)
print("2025 pathogenic:", int(
    (df_train[LABEL_COLUMN] == 1).sum()
))
print("2025 benign:", int(
    (df_train[LABEL_COLUMN] == 0).sum()
))

print("\n2026 dataset:", df_test.shape)
print("2026 pathogenic:", int(
    (df_test[LABEL_COLUMN] == 1).sum()
))
print("2026 benign:", int(
    (df_test[LABEL_COLUMN] == 0).sum()
))


# ==========================================
# CHECK FOR OVERLAPPING VARIANTS
# ==========================================
# This check does not use labels to construct features.
# It warns if variants occur in both years.
# Overlap may make temporal evaluation less independent.

if all(
    col in df_train.columns and col in df_test.columns
    for col in IDENTIFIER_COLUMNS
):
    def variant_keys(df):
        return (
            df[IDENTIFIER_COLUMNS]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip())
            .agg("|".join, axis=1)
        )

    keys_train = set(variant_keys(df_train))
    keys_test = set(variant_keys(df_test))

    overlap = keys_train.intersection(keys_test)

    print("\nVariant overlap check:")
    print("Unique 2025 variants:", len(keys_train))
    print("Unique 2026 variants:", len(keys_test))
    print("Variants occurring in both:", len(overlap))

    if overlap:
        warnings.warn(
            "Some variants occur in both datasets. "
            "The 2026 results may not represent performance "
            "on entirely unseen variants. Inspect the overlap "
            "before interpreting temporal generalisation."
        )

    del keys_train, keys_test
else:
    print(
        "\nVariant overlap check skipped: "
        "one or more identifier columns are unavailable."
    )


# ==========================================
# PREPARE FEATURES
# ==========================================
# The test labels are kept separately for evaluation.
# No test labels are used in fitting, CV or threshold selection.

print("\nPreparing features...")

y_train = df_train[LABEL_COLUMN].copy()
y_test = df_test[LABEL_COLUMN].copy()

X_train = df_train.drop(
    columns=[LABEL_COLUMN] + [
        c for c in DROP_COLUMNS if c in df_train.columns
    ],
    errors="ignore"
).copy()

X_test = df_test.drop(
    columns=[LABEL_COLUMN] + [
        c for c in DROP_COLUMNS if c in df_test.columns
    ],
    errors="ignore"
).copy()

del df_train, df_test
gc.collect()


# ==========================================
# ENCODE VARIANT TYPE
# ==========================================
# Learn dummy columns from 2025 only.
# Categories found only in 2026 do not create new features.

if VARIANT_TYPE_COLUMN in X_train.columns:
    print("Encoding variant type using 2025 categories...")

    train_variant_type = (
        X_train[VARIANT_TYPE_COLUMN]
        .fillna("__MISSING__")
        .astype(str)
    )

    test_variant_type = (
        X_test[VARIANT_TYPE_COLUMN]
        .fillna("__MISSING__")
        .astype(str)
    )

    train_dummies = pd.get_dummies(
        train_variant_type,
        prefix="vartype",
        dtype=np.int8
    )

    test_dummies = pd.get_dummies(
        test_variant_type,
        prefix="vartype",
        dtype=np.int8
    )

    # Only columns learned from training data are retained.
    test_dummies = test_dummies.reindex(
        columns=train_dummies.columns,
        fill_value=0
    )

    X_train = X_train.drop(
        columns=[VARIANT_TYPE_COLUMN]
    )

    X_test = X_test.drop(
        columns=[VARIANT_TYPE_COLUMN]
    )

    X_train = pd.concat(
        [X_train.reset_index(drop=True),
         train_dummies.reset_index(drop=True)],
        axis=1
    )

    X_test = pd.concat(
        [X_test.reset_index(drop=True),
         test_dummies.reset_index(drop=True)],
        axis=1
    )

    del train_variant_type, test_variant_type
    del train_dummies, test_dummies


# ==========================================
# NUMERIC CONVERSION
# ==========================================

def clean_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value in ("", ".", "NA", "N/A", "nan", "None"):
            return np.nan

        # Preserve the existing preprocessing convention:
        # use the first value when a field contains semicolons.
        if ";" in value:
            value = value.split(";")[0].strip()

    return value


print("Converting features to numeric values...")

X_train = X_train.apply(
    lambda col: pd.to_numeric(
        col.map(clean_value), errors="coerce"
    )
)

X_test = X_test.apply(
    lambda col: pd.to_numeric(
        col.map(clean_value), errors="coerce"
    )
)

# Remove features that are entirely missing in 2025.
# This decision is based on training data only.
all_missing_train = X_train.columns[
    X_train.isna().all()
].tolist()

if all_missing_train:
    print(
        "Dropping features entirely missing in 2025:",
        all_missing_train
    )

    X_train.drop(
        columns=all_missing_train, inplace=True
    )

    X_test.drop(
        columns=all_missing_train,
        errors="ignore",
        inplace=True
    )

if X_train.shape[1] == 0:
    raise ValueError(
        "No usable features remain after preprocessing."
    )

# Use the exact 2025 feature columns and order.
# Never add features based only on the 2026 dataset.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=np.nan
)

print("Biological feature count:", X_train.shape[1])


# ==========================================
# MISSING-VALUE INDICATORS
# ==========================================
# Indicators are generated independently per row.
# The indicator feature names are based on 2025 columns.

print("Creating missing-value indicators...")

base_feature_columns = X_train.columns.tolist()


def add_missing_indicators(X, feature_columns):
    missing = X[feature_columns].isna().astype(np.int8)

    missing.columns = [
        f"{col}_missing" for col in feature_columns
    ]

    return pd.concat(
        [X[feature_columns].copy(), missing],
        axis=1
    )


X_train = add_missing_indicators(
    X_train, base_feature_columns
)

X_test = add_missing_indicators(
    X_test, base_feature_columns
)

# Enforce exact training feature schema.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# XGBoost can handle NaN values natively.
# Do not fit an imputer on the combined datasets.

X_train = X_train.replace(
    [np.inf, -np.inf], np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
)

if X_train.columns.tolist() != X_test.columns.tolist():
    raise RuntimeError(
        "Training and test feature columns do not match."
    )

if X_train.isna().all(axis=1).all():
    raise ValueError(
        "Every feature is missing for every training row."
    )

print("Final feature count:", X_train.shape[1])
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ==========================================
# CLASS-WEIGHT HELPER
# ==========================================
# Calculate weights from the labels of the current
# training partition only, including inside CV folds.

def calculate_scale_pos_weight(y):
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    if positive == 0 or negative == 0:
        return 1.0

    return negative / positive


# ==========================================
# MODEL CONFIGURATION
# ==========================================

def create_model(
    scale_pos_weight,
    n_estimators=300
):
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


# ==========================================
# STRATIFIED FIVE-FOLD CV ON 2025
# ==========================================
# Each fold calculates its class weight from its own
# fitting partition. The held-out fold is not used
# to fit that fold's model.

if y_train.value_counts().min() < N_SPLITS:
    raise ValueError(
        f"At least {N_SPLITS} samples from each class "
        "are required for stratified CV."
    )

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_mean = np.nan
cv_scores = []

if USE_CV:
    print("\nRunning five-fold CV on 2025 data...")

    for fold, (fit_idx, val_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_fold_train = X_train.iloc[fit_idx]
        y_fold_train = y_train.iloc[fit_idx]

        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_weight = calculate_scale_pos_weight(
            y_fold_train
        )

        fold_model = create_model(
            scale_pos_weight=fold_weight,
            n_estimators=CV_ESTIMATORS
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
            verbose=False
        )

        fold_prob = fold_model.predict_proba(
            X_fold_val
        )[:, 1]

        fold_auc = roc_auc_score(
            y_fold_val, fold_prob
        )

        cv_scores.append(fold_auc)

        print(
            f"Fold {fold} ROC-AUC: {fold_auc:.4f}"
        )

        del fold_model, X_fold_train, X_fold_val
        del y_fold_train, y_fold_val, fold_prob
        gc.collect()

    cv_mean = float(np.mean(cv_scores))

    print("CV scores:", np.round(cv_scores, 4))
    print("Mean CV ROC-AUC:", round(cv_mean, 4))


# ==========================================
# SELECT THRESHOLD USING 2025 OOF PREDICTIONS
# ==========================================
# Every training row gets a prediction from a model
# that did not train on that row.
# 2026 is not involved in threshold selection.

print("\nSelecting threshold using 2025 OOF predictions...")

oof_prob = np.full(len(y_train), np.nan, dtype=float)

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(X_train, y_train), start=1
):
    X_fold_train = X_train.iloc[fit_idx]
    y_fold_train = y_train.iloc[fit_idx]

    X_fold_val = X_train.iloc[val_idx]

    fold_weight = calculate_scale_pos_weight(
        y_fold_train
    )

    threshold_model = create_model(
        scale_pos_weight=fold_weight,
        n_estimators=THRESHOLD_ESTIMATORS
    )

    threshold_model.fit(
        X_fold_train,
        y_fold_train,
        verbose=False
    )

    oof_prob[val_idx] = threshold_model.predict_proba(
        X_fold_val
    )[:, 1]

    print(f"Generated OOF predictions for fold {fold}.")

    del threshold_model, X_fold_train, X_fold_val
    del y_fold_train
    gc.collect()

if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Some training rows did not receive OOF predictions."
    )

oof_auc = roc_auc_score(y_train, oof_prob)

precision, recall, thresholds = precision_recall_curve(
    y_train, oof_prob
)

# The precision and recall arrays contain one extra
# element compared with the threshold array.
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-8)
)

best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])
best_oof_f1 = float(f1_scores[best_idx])

print("Threshold source: 2025 OOF predictions")
print("2025 OOF ROC-AUC:", round(oof_auc, 4))
print("2025 OOF best F1:", round(best_oof_f1, 4))
print("Selected threshold:", round(best_threshold, 4))


# ==========================================
# TRAIN FINAL MODEL ON ALL 2025 DATA
# ==========================================
# Fixed estimator count: no 2026 early stopping,
# no 2026 evaluation set supplied during fitting.

print("\nTraining final XGBoost model on 2025...")

final_scale_pos_weight = calculate_scale_pos_weight(
    y_train
)

print(
    "Final training scale_pos_weight:",
    round(final_scale_pos_weight, 4)
)

model = create_model(
    scale_pos_weight=final_scale_pos_weight,
    n_estimators=FINAL_ESTIMATORS
)

model.fit(
    X_train,
    y_train,
    verbose=100
)

print("Final model training complete.")


# ==========================================
# SAVE MODEL AND FEATURE SCHEMA
# ==========================================
# Save the exact feature order for future predictions.

joblib.dump(
    model,
    out("CLN3_xgb_model.pkl")
)

joblib.dump(
    X_train.columns.tolist(),
    out("CLN3_feature_columns.pkl")
)

joblib.dump(
    {
        "threshold": best_threshold,
        "threshold_source": "2025 out-of-fold predictions",
        "training_year": 2025,
        "test_year": 2026,
        "base_feature_columns": base_feature_columns,
        "final_feature_columns": X_train.columns.tolist(),
        "variant_type_column": VARIANT_TYPE_COLUMN,
        "variant_type_categories": (
            [
                col.replace("vartype_", "", 1)
                for col in X_train.columns
                if col.startswith("vartype_")
            ]
        )
    },
    out("CLN3_model_metadata.pkl")
)

print("Saved model, feature schema and metadata.")


# ==========================================
# PREDICT ON 2026 TEST DATA
# ==========================================
# The 2026 labels are not used until evaluation below.

print("\nPredicting on 2026 test data...")

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)


# ==========================================
# EVALUATION METRICS
# ==========================================

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nClassification Report:")
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
print("Threshold:", round(best_threshold, 4))


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Pathogenic"]
)

disp.plot()

plt.title(
    f"CLN3 — Tested on 2026 "
    f"(2025 threshold={best_threshold:.2f})"
)

plt.savefig(
    out("CLN3_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN3_confusion_matrix.png")


# ==========================================
# ROC CURVE
# ==========================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"CLN3 XGBoost 2025 to 2026 (AUC={roc:.4f})"
)

plt.title("CLN3 ROC Curve — Trained 2025, Tested 2026")

plt.savefig(
    out("CLN3_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN3_roc_curve.png")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

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
    out("CLN3_feature_importance.csv"),
    index=False
)

print("Saved CLN3_feature_importance.csv")


# ==========================================
# SHAP
# ==========================================
# SHAP uses 2026 feature values only to explain the
# already-trained model. It does not refit the model.

if USE_SHAP:
    try:
        import shap

        print("\nRunning SHAP...")

        shap_sample = (
            X_test
            if len(X_test) <= 500
            else X_test.sample(
                500, random_state=RANDOM_STATE
            )
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)

        shap.summary_plot(
            shap_values,
            shap_sample,
            show=False
        )

        plt.title("CLN3 SHAP Summary — 2026 Test Features")
        plt.tight_layout()

        plt.savefig(
            out("CLN3_shap_summary.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved CLN3_shap_summary.png")

    except Exception as e:
        print("SHAP failed:", e)


# ==========================================
# SAVE RESULTS SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Train_year": [2025],
    "Test_year": [2026],
    "Accuracy": [round(acc, 4)],
    "ROC_AUC": [round(roc, 4)],
    "PR_AUC": [round(pr_auc, 4)],
    "CV_ROC_2025": [
        round(cv_mean, 4)
        if not np.isnan(cv_mean) else np.nan
    ],
    "OOF_ROC_AUC_2025": [round(oof_auc, 4)],
    "OOF_F1_2025": [round(best_oof_f1, 4)],
    "Threshold": [best_threshold],
    "Threshold_source": [
        "2025 out-of-fold predictions"
    ],
    "Final_estimators": [FINAL_ESTIMATORS],
    "Train_size": [len(X_train)],
    "Test_size": [len(X_test)],
    "Train_Pathogenic": [
        int((y_train == 1).sum())
    ],
    "Train_Benign": [
        int((y_train == 0).sum())
    ],
    "Test_Pathogenic": [
        int((y_test == 1).sum())
    ],
    "Test_Benign": [
        int((y_test == 0).sum())
    ],
    "Final_scale_pos_weight": [
        final_scale_pos_weight
    ]
})

summary.to_csv(
    out("CLN3_results_summary.csv"),
    index=False
)

print("Saved CLN3_results_summary.csv")

print("\n" + "=" * 60)
print("CLN3 PIPELINE FINISHED")
print("Training: 2025")
print("Testing: 2026")
print("Threshold: 2025 out-of-fold predictions")
print("2026 data was not used for model fitting or threshold selection.")
print("=" * 60)
