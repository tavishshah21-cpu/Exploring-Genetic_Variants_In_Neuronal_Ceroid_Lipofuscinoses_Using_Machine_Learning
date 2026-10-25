# ============================================================
# CLN2 PIPELINE 3 — STACKING ENSEMBLE
# Train on 2025, Test on 2026
# Leakage-controlled temporal validation
#
# Base models: Random Forest, Gradient Boosting, Extra Trees
# Meta-model: Logistic Regression
# ============================================================

import os
import gc
import joblib
import warnings
import pandas as pd
import numpy as np
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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_cln2"
OUTPUT_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_3_cln2"

TRAIN_PATH = os.path.join(
    SOURCE_DIR, "CLN2_2025_CLN2_ML_READY.csv"
)

TEST_PATH = os.path.join(
    SOURCE_DIR, "CLN2_2026_CLN2_ML_READY.csv"
)

GENE = "CLN2"
RANDOM_STATE = 42
N_SPLITS = 5

RF_N_ESTIMATORS = 400
ET_N_ESTIMATORS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


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

for year, frame in ((2025, df_train), (2026, df_test)):

    if "label" not in frame.columns:
        raise ValueError(
            f"The {year} dataset must contain a 'label' column."
        )

    if frame.empty:
        raise ValueError(f"The {year} dataset is empty.")

    if frame["label"].isna().any():
        raise ValueError(
            f"Missing target labels found in {year}."
        )

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{year} labels must be 0 (Benign) or 1 (Pathogenic)."
        )

    if frame["label"].nunique() != 2:
        raise ValueError(
            f"{year} data must contain both classes."
        )


y_train = df_train["label"].astype(int).reset_index(drop=True)
y_test = df_test["label"].astype(int).reset_index(drop=True)

X_train = df_train.drop(columns=["label"]).copy()
X_test = df_test.drop(columns=["label"]).copy()

del df_train, df_test
gc.collect()


# ============================================================
# REMOVE IDENTIFIERS AND ANNOTATION TEXT
# ============================================================

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
# ALIGN FEATURES TO THE 2025 SCHEMA
# ============================================================

# The 2025 dataset determines the feature schema.
# Features exclusive to 2026 cannot introduce new model inputs.

common_columns = list(X_train.columns)

X_test = X_test.reindex(columns=common_columns)


# ============================================================
# ENCODE EXONIC VARIANT TYPE
# ============================================================

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

    # Keep only categories learned from the 2025 dataset.
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

        # Use the first value in semicolon-separated annotations.
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

    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return X


X_train = convert_numeric(X_train)
X_test = convert_numeric(X_test)


# Remove columns entirely missing in 2025.
# This decision does not depend on 2026 feature values.

valid_columns = X_train.columns[
    X_train.notna().any(axis=0)
].tolist()

X_train = X_train.reindex(columns=valid_columns)
X_test = X_test.reindex(columns=valid_columns)

print("Biological/numeric features:", X_train.shape[1])


# ============================================================
# CREATE MISSING-VALUE INDICATORS
# ============================================================

print("\nCreating missing-value indicators...")

train_missing = X_train.isna().astype(np.int8)
test_missing = X_test.isna().astype(np.int8)

train_missing.columns = [
    f"{c}_missing" for c in train_missing.columns
]

test_missing.columns = [
    f"{c}_missing" for c in test_missing.columns
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

X_test = X_test.reindex(columns=X_train.columns)

print("Final feature count:", X_train.shape[1])


# ============================================================
# IMPUTATION — MEDIANS FIT ON 2025 ONLY
# ============================================================

print("\nFitting missing-value imputation on 2025 only...")

training_medians = X_train.median(axis=0)

X_train = X_train.fillna(training_medians).fillna(0)
X_test = X_test.fillna(training_medians).fillna(0)

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
    raise ValueError(
        "Both classes must exist in 2025 training data."
    )

print("Pathogenic:", pos)
print("Benign    :", neg)
print("Benign/pathogenic ratio:", round(neg / pos, 4))


# ============================================================
# PIPELINE 3 STACKING MODEL
#
# Base models:
#   1. Random Forest
#   2. Gradient Boosting
#   3. Extra Trees
#
# Meta-model:
#   Logistic Regression
#
# StackingClassifier generates internal out-of-fold
# base-model predictions for training the meta-model.
# ============================================================

def make_model():

    base_estimators = [

        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),

        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=RANDOM_STATE
            )
        ),

        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=ET_N_ESTIMATORS,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]

    meta_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        stack_method="predict_proba",
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        ),
        passthrough=False,
        n_jobs=-1
    )


# ============================================================
# 5-FOLD OUTER OOF PREDICTIONS ON 2025
#
# These predictions determine the classification threshold.
# Each outer validation fold is excluded from that fold's fit.
# ============================================================

print("\nGenerating outer out-of-fold probabilities on 2025...")

outer_cv = StratifiedKFold(
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
    outer_cv.split(X_train, y_train),
    start=1
):

    print(f"\nTraining stacking fold {fold}/{N_SPLITS}...")

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

    print(f"Fold {fold} ROC-AUC: {fold_auc:.4f}")

    del fold_model
    gc.collect()


if np.isnan(oof_prob).any():

    raise RuntimeError(
        "Some 2025 rows did not receive OOF predictions."
    )


cv_mean = float(np.mean(fold_auc_scores))

oof_auc = float(
    roc_auc_score(y_train, oof_prob)
)

print("\n5-Fold ROC-AUC scores:", np.round(fold_auc_scores, 4))
print("Mean 5-Fold ROC-AUC:", round(cv_mean, 4))
print("Combined OOF ROC-AUC:", round(oof_auc, 4))


# ============================================================
# SELECT THRESHOLD USING 2025 OOF PREDICTIONS ONLY
# ============================================================

print("\nSelecting threshold from 2025 OOF predictions...")

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
        "Unable to calculate a threshold from 2025 predictions."
    )


best_idx = int(np.nanargmax(f1_scores))

best_threshold = float(thresholds[best_idx])

print(
    "Selected threshold from 2025:",
    round(best_threshold, 6)
)

print(
    "2025 OOF F1 at selected threshold:",
    round(float(f1_scores[best_idx]), 4)
)


# Save threshold selection record.

pd.DataFrame({
    "Training_year": [2025],
    "Method": [
        "5-fold outer OOF predictions from Pipeline 3 stacking"
    ],
    "Selection_metric": ["F1-score"],
    "Threshold": [best_threshold],
    "OOF_F1": [float(f1_scores[best_idx])],
    "OOF_ROC_AUC": [oof_auc]
}).to_csv(
    out("CLN2_Pipeline3_2025_threshold_selection.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL STACKING MODEL ON ALL 2025 DATA
# ============================================================

print("\nTraining final Pipeline 3 stacking model on all 2025 data...")

model = make_model()

model.fit(
    X_train,
    y_train
)

print("Final stacking model trained on 2025 only.")


# ============================================================
# EVALUATE ON 2026 — THRESHOLD ALREADY FIXED
# ============================================================

print("\nEvaluating on 2026 test data...")

# 2026 labels are not used to fit the model or select the threshold.

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (
    y_prob >= best_threshold
).astype(int)


# ============================================================
# EVALUATION METRICS
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
    "CLN2 Pipeline 3 Stacking — 2026 Test Set "
    f"(2025 threshold={best_threshold:.3f})"
)

plt.savefig(
    out("CLN2_Pipeline3_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN2_Pipeline3_confusion_matrix.png")


# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"CLN2 Pipeline 3 Stacking (AUC={roc:.4f})"
)

plt.title(
    "CLN2 Pipeline 3 ROC Curve — Trained 2025, Tested 2026"
)

plt.savefig(
    out("CLN2_Pipeline3_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved CLN2_Pipeline3_roc_curve.png")


# ============================================================
# FEATURE IMPORTANCE — TREE BASE MODELS
#
# Reports mean importance across available tree base models.
# These are not the Logistic Regression meta-model coefficients.
# ============================================================

print("\nFeature Importance (mean normalized tree importance)")

importance_frames = []

for estimator_name, estimator in model.named_estimators_.items():

    if hasattr(estimator, "feature_importances_"):

        importance_frames.append(
            pd.DataFrame({
                "Feature": X_train.columns,
                f"{estimator_name}_importance":
                    estimator.feature_importances_
            }).set_index("Feature")
        )


if importance_frames:

    importance_df = pd.concat(
        importance_frames,
        axis=1
    ).fillna(0)

    importance_columns = list(importance_df.columns)

    importance_df["Mean_tree_importance"] = (
        importance_df[importance_columns].mean(axis=1)
    )

    importance_df = (
        importance_df
        .reset_index()
        .rename(columns={"index": "Feature"})
        .sort_values(
            "Mean_tree_importance",
            ascending=False
        )
    )

    print(
        importance_df.head(20).to_string(index=False)
    )

    importance_df.to_csv(
        out("CLN2_Pipeline3_feature_importance.csv"),
        index=False
    )

    print("Saved CLN2_Pipeline3_feature_importance.csv")

else:

    print("No tree feature importances were available.")


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

model_path = out(
    "CLN2_Pipeline3_stacking_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:", model_path)


metadata = {
    "gene": GENE,
    "pipeline": "Pipeline 3 stacking",
    "training_year": 2025,
    "test_year": 2026,
    "base_models": [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
    "meta_model": "Logistic Regression",
    "xgboost_used": False,
    "feature_columns": list(X_train.columns),
    "training_medians": training_medians.to_dict(),
    "threshold": best_threshold,
    "threshold_source": "2025 outer out-of-fold predictions",
    "threshold_metric": "F1-score",
    "outer_cv_folds": N_SPLITS,
    "random_state": RANDOM_STATE
}

joblib.dump(
    metadata,
    out("CLN2_Pipeline3_stacking_metadata.pkl")
)

print("Saved CLN2_Pipeline3_stacking_metadata.pkl")


# ============================================================
# SAVE RESULTS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Pipeline": ["Pipeline 3 Stacking"],
    "Train_year": [2025],
    "Test_year": [2026],
    "Base_models": [
        "Random Forest; Gradient Boosting; Extra Trees"
    ],
    "Meta_model": ["Logistic Regression"],
    "Accuracy": [round(acc, 4)],
    "ROC_AUC": [round(roc, 4)],
    "PR_AUC": [round(pr_auc, 4)],
    "CV_ROC_2025": [round(cv_mean, 4)],
    "OOF_ROC_AUC_2025": [round(oof_auc, 4)],
    "Threshold": [round(best_threshold, 6)],
    "Threshold_source": ["2025 outer OOF predictions"],
    "Train_size": [len(X_train)],
    "Test_size": [len(X_test)],
    "Train_Pathogenic": [pos],
    "Train_Benign": [neg],
    "Test_Pathogenic": [int((y_test == 1).sum())],
    "Test_Benign": [int((y_test == 0).sum())]
})

summary.to_csv(
    out("CLN2_Pipeline3_results_summary.csv"),
    index=False
)

print("Saved CLN2_Pipeline3_results_summary.csv")


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
    out("CLN2_Pipeline3_2026_predictions.csv"),
    index=False
)

print("Saved CLN2_Pipeline3_2026_predictions.csv")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)

print("CLN2 PIPELINE 3 STACKING FINISHED")
print("Training dataset : 2025")
print(
    "Algorithms       : Random Forest + Gradient Boosting + Extra Trees"
)
print("Meta-model       : Logistic Regression")
print("Threshold source : 2025 outer OOF predictions")
print("Test dataset     : 2026")
print("2026 used for    : Final evaluation only")
print("Output directory :", OUTPUT_DIR)

print("=" * 65)