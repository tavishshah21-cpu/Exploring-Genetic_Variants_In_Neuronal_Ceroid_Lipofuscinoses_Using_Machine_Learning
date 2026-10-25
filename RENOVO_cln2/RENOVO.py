# ============================================================
# CLN2 RENOVO PIPELINE
# Train on 2025, Test on 2026
# Leakage-controlled temporal validation
#
# Predictor: Random Forest (RENOVO version)
# Threshold: Selected using 2025 out-of-fold predictions only
#
# Input:
#   Pipeline_1_cln2/CLN2_2025_CLN2_ML_READY.csv
#   Pipeline_1_cln2/CLN2_2026_CLN2_ML_READY.csv
#
# Output:
#   RENOVO_cln2/
# ============================================================

import os
import gc
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_cln2"

OUTPUT_DIR = r"D:\tavish_shah\PES\capstone\RENOVO_cln2"

TRAIN_PATH = os.path.join(
    SOURCE_DIR,
    "CLN2_2025_CLN2_ML_READY.csv"
)

TEST_PATH = os.path.join(
    SOURCE_DIR,
    "CLN2_2026_CLN2_ML_READY.csv"
)

GENE = "CLN2"

RANDOM_STATE = 42
N_SPLITS = 5

N_ESTIMATORS = 300
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("CLN2 RENOVO PIPELINE")
print("=" * 65)

print("\nLoading 2025 training data...")

df_train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

print("2025 dataset shape:", df_train.shape)

print("\nLoading 2026 test data...")

df_test = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

print("2026 dataset shape:", df_test.shape)


# ============================================================
# VALIDATE INPUTS
# ============================================================

for year, frame in (
    (2025, df_train),
    (2026, df_test)
):

    if frame.empty:
        raise ValueError(
            f"The {year} dataset is empty."
        )

    if "label" not in frame.columns:
        raise ValueError(
            f"The {year} dataset must contain a 'label' column."
        )

    if frame["label"].isna().any():
        raise ValueError(
            f"The {year} dataset contains missing labels."
        )

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"The {year} labels must be 0 (Benign) "
            "or 1 (Pathogenic)."
        )

    if frame["label"].nunique() != 2:
        raise ValueError(
            f"The {year} dataset must contain both classes."
        )


print("\n2025 label distribution:")
print(df_train["label"].value_counts().sort_index())

print("\n2026 label distribution:")
print(df_test["label"].value_counts().sort_index())


# ============================================================
# REMOVE OVERLAPPING VARIANTS WHEN IDENTIFIERS ARE AVAILABLE
# ============================================================

# Important:
# If both datasets contain a stable variant identifier, remove
# 2026 variants already present in 2025.
#
# This step is performed BEFORE identifier columns are removed.
# It does not use the 2026 labels to select the model or threshold.

identifier_candidates = [
    "VariationID",
    "VariationId",
    "variation_id",
    "variant_id",
    "VariantID"
]

shared_id_column = next(
    (
        col for col in identifier_candidates
        if col in df_train.columns and col in df_test.columns
    ),
    None
)

if shared_id_column is not None:

    print(
        f"\nChecking temporal overlap using {shared_id_column}..."
    )

    train_ids = set(
        df_train[shared_id_column]
        .dropna()
        .astype(str)
    )

    test_ids = df_test[shared_id_column].astype("string")

    overlap_mask = test_ids.isin(train_ids)

    overlap_count = int(overlap_mask.sum())

    print("Overlapping 2026 rows:", overlap_count)

    if overlap_count > 0:

        df_test = df_test.loc[
            ~overlap_mask
        ].reset_index(drop=True)

        print(
            "Removed overlapping variants from the 2026 test set."
        )

else:

    print(
        "\nWARNING: No shared stable variant identifier was found."
    )

    print(
        "Exact variant-level temporal overlap could not be verified."
    )

if df_test.empty:
    raise ValueError(
        "The 2026 test dataset is empty after overlap removal."
    )

if df_test["label"].nunique() != 2:
    raise ValueError(
        "The remaining 2026 test dataset must contain both classes."
    )


# ============================================================
# SEPARATE LABELS AND FEATURES
# ============================================================

y_train = (
    df_train["label"]
    .astype(int)
    .reset_index(drop=True)
)

y_test = (
    df_test["label"]
    .astype(int)
    .reset_index(drop=True)
)

X_train = df_train.drop(
    columns=["label"]
).copy()

X_test = df_test.drop(
    columns=["label"]
).copy()

del df_train, df_test
gc.collect()


# ============================================================
# REMOVE IDENTIFIERS AND NON-PREDICTIVE ANNOTATION TEXT
# ============================================================

drop_cols = [
    "VariationID",
    "VariationId",
    "variation_id",
    "variant_id",
    "VariantID",
    "Chr",
    "Chromosome",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol",
    "Gene(s)",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

X_train.drop(
    columns=[
        c for c in drop_cols
        if c in X_train.columns
    ],
    inplace=True,
    errors="ignore"
)

X_test.drop(
    columns=[
        c for c in drop_cols
        if c in X_test.columns
    ],
    inplace=True,
    errors="ignore"
)


# ============================================================
# REMOVE TARGET-LEAKAGE COLUMNS
# ============================================================

# Clinical classification fields must not be predictors.
# Independent biological prediction scores can remain features.

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
    "ReviewStatus_stars",
    "Germline classification"
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
# ALIGN FEATURES USING THE 2025 SCHEMA
# ============================================================

# 2025 defines which features are available to the model.
# 2026-only columns cannot introduce new predictors.

feature_columns_2025 = list(X_train.columns)

X_test = X_test.reindex(
    columns=feature_columns_2025
)


# ============================================================
# CLEAN AND NUMERICALLY ENCODE FEATURES
# ============================================================

print("\nCleaning and encoding features...")


def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in {
            "",
            ".",
            "NA",
            "N/A",
            "nan",
            "None"
        }:
            return np.nan

        # Preserve the existing preprocessing approach:
        # use the first value in semicolon-separated fields.
        if ";" in value:

            value = value.split(";")[0].strip()

            if value in {
                "",
                ".",
                "NA",
                "N/A",
                "nan",
                "None"
            }:
                return np.nan

    return value


def encode_features(X):

    X = X.copy()

    for column in X.columns:

        X[column] = X[column].map(clean_value)

        # Keep numeric columns numeric.
        # Convert categorical columns to numeric category codes.
        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        original_nonmissing = X[column].notna()

        conversion_failed = (
            original_nonmissing & converted.isna()
        )

        if conversion_failed.any():

            # Categorical encoding is fitted independently on
            # the 2025 training categories. Unknown 2026 values
            # will be represented as missing and imputed using
            # 2025 training statistics.
            categories = pd.Index(
                X.loc[original_nonmissing, column]
                .astype(str)
                .unique()
            )

            category_map = {
                value: index
                for index, value in enumerate(categories)
            }

            converted = X[column].map(
                lambda value: (
                    category_map.get(str(value), np.nan)
                    if pd.notna(value)
                    else np.nan
                )
            )

        X[column] = pd.to_numeric(
            converted,
            errors="coerce"
        )

    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return X


# Fit categorical encodings using the 2025 training data only.
# Do not learn category mappings from 2026.

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

category_maps = {}

for column in X_train.columns:

    train_clean = X_train[column].map(clean_value)
    test_clean = X_test[column].map(clean_value)

    train_numeric = pd.to_numeric(
        train_clean,
        errors="coerce"
    )

    # Determine whether this feature needs categorical encoding.
    nonmissing = train_clean.notna()

    conversion_failed = (
        nonmissing & train_numeric.isna()
    )

    if conversion_failed.any():

        categories = pd.Index(
            train_clean[nonmissing].astype(str).unique()
        )

        category_map = {
            value: index
            for index, value in enumerate(categories)
        }

        category_maps[column] = category_map

        X_train_encoded[column] = train_clean.map(
            lambda value: (
                category_map.get(str(value), np.nan)
                if pd.notna(value)
                else np.nan
            )
        )

        X_test_encoded[column] = test_clean.map(
            lambda value: (
                category_map.get(str(value), np.nan)
                if pd.notna(value)
                else np.nan
            )
        )

    else:

        X_train_encoded[column] = train_numeric

        X_test_encoded[column] = pd.to_numeric(
            test_clean,
            errors="coerce"
        )


X_train = X_train_encoded
X_test = X_test_encoded

del X_train_encoded, X_test_encoded
gc.collect()


# ============================================================
# REMOVE FEATURES WITH NO USABLE 2025 VALUES
# ============================================================

valid_columns = X_train.columns[
    X_train.notna().any(axis=0)
].tolist()

X_train = X_train.reindex(
    columns=valid_columns
)

X_test = X_test.reindex(
    columns=valid_columns
)

print("Usable features:", len(valid_columns))


# ============================================================
# CREATE MISSING-VALUE INDICATORS
# ============================================================

print("\nCreating missing-value indicators...")

train_missing = X_train.isna().astype(np.int8)
test_missing = X_test.isna().astype(np.int8)

train_missing.columns = [
    f"{column}_missing"
    for column in train_missing.columns
]

test_missing.columns = [
    f"{column}_missing"
    for column in test_missing.columns
]

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

X_test = X_test.reindex(
    columns=X_train.columns
)

print("Final feature count:", X_train.shape[1])


# ============================================================
# IMPUTATION — FIT ON 2025 ONLY
# ============================================================

print("\nFitting imputation using 2025 training data only...")

training_medians = X_train.median(axis=0)

X_train = X_train.fillna(
    training_medians
).fillna(0)

X_test = X_test.fillna(
    training_medians
).fillna(0)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

assert list(X_train.columns) == list(X_test.columns)

assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

print("Training matrix:", X_train.shape)
print("Testing matrix :", X_test.shape)


# ============================================================
# CLASS BALANCE — 2025 ONLY
# ============================================================

print("\nCalculating class balance from 2025...")

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())

if pos == 0 or neg == 0:
    raise ValueError(
        "Both benign and pathogenic classes must exist in 2025."
    )

print("Pathogenic:", pos)
print("Benign    :", neg)


# ============================================================
# RENOVO MODEL
# ============================================================

# Random Forest predictor.
# No 2026 data is used to fit the model.
#
# class_weight="balanced" calculates class weights from the
# labels supplied to fit(), which are the 2025 training labels
# for the final model.

def make_model():

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# FIVE-FOLD OUT-OF-FOLD PREDICTIONS ON 2025
# ============================================================

print("\nGenerating 5-fold out-of-fold predictions on 2025...")

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

    fold_model.fit(
        X_fold_train,
        y_fold_train
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

    print(
        f"Fold {fold} ROC-AUC: {fold_auc:.4f}"
    )

    del fold_model
    gc.collect()


if np.isnan(oof_prob).any():

    raise RuntimeError(
        "Some 2025 rows did not receive out-of-fold predictions."
    )


cv_mean = float(np.mean(fold_auc_scores))

oof_auc = float(
    roc_auc_score(y_train, oof_prob)
)

print("\n5-Fold ROC-AUC scores:", np.round(fold_auc_scores, 4))

print(
    "Mean 5-Fold ROC-AUC:",
    round(cv_mean, 4)
)

print(
    "Combined OOF ROC-AUC:",
    round(oof_auc, 4)
)


# ============================================================
# SELECT THRESHOLD USING 2025 ONLY
# ============================================================

print("\nSelecting classification threshold from 2025 OOF predictions...")

precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_prob
)

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
        "Could not calculate a threshold from 2025 predictions."
    )

best_idx = int(
    np.nanargmax(f1_scores)
)

best_threshold = float(
    thresholds[best_idx]
)

best_oof_f1 = float(
    f1_scores[best_idx]
)

print(
    "Selected 2025 threshold:",
    round(best_threshold, 6)
)

print(
    "2025 OOF F1-score:",
    round(best_oof_f1, 4)
)


# Save threshold selection record.

threshold_record = pd.DataFrame({
    "Gene": [GENE],
    "Training_year": [2025],
    "Method": ["5-fold out-of-fold predictions"],
    "Selection_metric": ["F1-score"],
    "Threshold": [best_threshold],
    "OOF_F1": [best_oof_f1],
    "OOF_ROC_AUC": [oof_auc]
})

threshold_record.to_csv(
    out("CLN2_RENOVO_2025_threshold_selection.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL RENOVO MODEL ON ALL 2025 DATA
# ============================================================

print("\nTraining final RENOVO model on all 2025 data...")

model = make_model()

model.fit(
    X_train,
    y_train
)

print("Final model trained on 2025 only.")


# ============================================================
# EVALUATE ON 2026
# ============================================================

print("\nEvaluating on 2026 test data...")

# Threshold is already fixed using 2025 OOF predictions.
# Do not adjust it based on 2026 results.

y_prob = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)


# ============================================================
# EVALUATION METRICS
# ============================================================

acc = accuracy_score(
    y_test,
    y_pred
)

roc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\nClassification Report — 2026")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Pathogenic"],
    zero_division=0
)

print(report)

print("Accuracy :", round(acc, 4))
print("ROC-AUC  :", round(roc, 4))
print("PR-AUC   :", round(pr_auc, 4))
print("Threshold:", round(best_threshold, 6))


# Save classification report.

with open(
    out("CLN2_RENOVO_classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("CLN2 RENOVO — 2026 Classification Report\n\n")
    f.write(report)
    f.write("\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"ROC-AUC: {roc:.6f}\n")
    f.write(f"PR-AUC: {pr_auc:.6f}\n")
    f.write(f"Threshold: {best_threshold:.6f}\n")


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
    "CLN2 RENOVO — 2026 Test Set "
    f"(2025 threshold={best_threshold:.3f})"
)

plt.savefig(
    out("CLN2_RENOVO_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved confusion matrix.")


# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"CLN2 RENOVO (AUC={roc:.4f})"
)

plt.title(
    "CLN2 RENOVO ROC Curve — Trained 2025, Tested 2026"
)

plt.savefig(
    out("CLN2_RENOVO_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved ROC curve.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(20).to_string(index=False)
)

feature_importance.to_csv(
    out("CLN2_RENOVO_feature_importance.csv"),
    index=False
)

print("Saved feature importance.")


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

model_path = out(
    "CLN2_RENOVO_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:", model_path)


# ============================================================
# SAVE MODEL METADATA
# ============================================================

metadata = {
    "gene": GENE,
    "pipeline": "RENOVO",
    "algorithm": "RandomForestClassifier",
    "training_year": 2025,
    "test_year": 2026,
    "feature_columns": list(X_train.columns),
    "training_medians": training_medians.to_dict(),
    "category_maps": category_maps,
    "threshold": best_threshold,
    "threshold_source": "2025 out-of-fold predictions",
    "threshold_metric": "F1-score",
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

joblib.dump(
    metadata,
    out("CLN2_RENOVO_metadata.pkl")
)

print("Saved model metadata.")


# ============================================================
# SAVE RESULTS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Pipeline": ["RENOVO"],
    "Algorithm": ["Random Forest"],
    "Train_year": [2025],
    "Test_year": [2026],
    "Accuracy": [round(acc, 4)],
    "ROC_AUC": [round(roc, 4)],
    "PR_AUC": [round(pr_auc, 4)],
    "CV_ROC_2025": [round(cv_mean, 4)],
    "OOF_ROC_AUC_2025": [round(oof_auc, 4)],
    "OOF_F1_2025": [round(best_oof_f1, 4)],
    "Threshold": [round(best_threshold, 6)],
    "Threshold_source": ["2025 OOF predictions"],
    "Train_size": [len(X_train)],
    "Test_size": [len(X_test)],
    "Train_Pathogenic": [int((y_train == 1).sum())],
    "Train_Benign": [int((y_train == 0).sum())],
    "Test_Pathogenic": [int((y_test == 1).sum())],
    "Test_Benign": [int((y_test == 0).sum())]
})

summary.to_csv(
    out("CLN2_RENOVO_results_summary.csv"),
    index=False
)

print("Saved results summary.")


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
    out("CLN2_RENOVO_2026_predictions.csv"),
    index=False
)

print("Saved 2026 predictions.")


# ============================================================
# SAVE 2025 OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_predictions = pd.DataFrame({
    "Actual_label": y_train,
    "OOF_pathogenic_probability": oof_prob,
    "Threshold_used": best_threshold,
    "OOF_predicted_label": (
        oof_prob >= best_threshold
    ).astype(int)
})

oof_predictions.to_csv(
    out("CLN2_RENOVO_2025_OOF_predictions.csv"),
    index=False
)

print("Saved 2025 OOF predictions.")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)
print("CLN2 RENOVO PIPELINE FINISHED")
print("=" * 65)

print("Training dataset : 2025")
print("Testing dataset  : 2026")
print("Algorithm        : Random Forest")
print("Threshold source : 2025 OOF predictions")
print("Threshold metric : F1-score")
print("Accuracy         :", round(acc, 4))
print("ROC-AUC          :", round(roc, 4))
print("PR-AUC           :", round(pr_auc, 4))
print("Output directory :", OUTPUT_DIR)

print("=" * 65)