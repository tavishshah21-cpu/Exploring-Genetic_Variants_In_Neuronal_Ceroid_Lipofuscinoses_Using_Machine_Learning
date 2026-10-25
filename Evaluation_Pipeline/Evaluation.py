
# ============================================================
# Evaluation_Pipeline.py
#
# Compares:
#   1. Pipeline3_Stacking
#   2. RENOVO_XGBoost
#   3. ML_Graph_XGBoost
#
# Training: 2025 preprocessed dataset
# Testing:  2026 preprocessed dataset
#
# Uses the same outer 5-fold CV splits for all models.
# Selects thresholds using 2025 out-of-fold predictions only.
#
# Additional metrics:
#   - 2026_Recall_Pathogenic
#   - 2026_False_Negatives
#
# Outputs are saved in the Evaluation_Pipeline folder.
# ============================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

LABEL_COLUMN = "label"

TRAIN_FILENAME = "CLN3_2025_CLN3_ML_READY.csv"
TEST_FILENAME = "CLN3_2026_CLN3_ML_READY.csv"

# Output directory: the folder containing this script.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Threshold objective is selected using 2025 OOF predictions.
THRESHOLD_OBJECTIVE = "f1_pathogenic"

# 2025 and 2026 are expected to use labels:
# 0 = Benign
# 1 = Pathogenic


# ============================================================
# DATA AND OUTPUT PATHS
# ============================================================

def find_data_directory():
    """
    Search the evaluation folder and nearby project folders
    for the preprocessed 2025 and 2026 CSV files.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        os.path.join(
            os.path.dirname(script_dir),
            "Pipeline_1_Statistics"
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(script_dir)),
            "Pipeline_1_Statistics"
        ),
    ]

    for directory in candidates:
        train_path = os.path.join(
            directory, TRAIN_FILENAME
        )
        test_path = os.path.join(
            directory, TEST_FILENAME
        )

        if (
            os.path.isfile(train_path)
            and os.path.isfile(test_path)
        ):
            return directory

    searched = "\n".join(candidates)

    raise FileNotFoundError(
        "Could not locate both preprocessed CSV files.\n"
        f"Expected:\n  {TRAIN_FILENAME}\n"
        f"  {TEST_FILENAME}\n\n"
        "Searched these directories:\n"
        f"{searched}"
    )


DATA_DIR = find_data_directory()

TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILENAME)
TEST_PATH = os.path.join(DATA_DIR, TEST_FILENAME)


def output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


print("=" * 100)
print("CLN3 MODEL EVALUATION PIPELINE")
print("=" * 100)

print("Input directory :", DATA_DIR)
print("Output directory:", OUTPUT_DIR)
print("Training file   :", TRAIN_PATH)
print("Testing file    :", TEST_PATH)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

df_train = pd.read_csv(TRAIN_PATH, low_memory=False)
df_test = pd.read_csv(TEST_PATH, low_memory=False)

for dataset_name, frame in [
    ("2025 training", df_train),
    ("2026 testing", df_test)
]:
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(
            f"{dataset_name} dataset has no "
            f"'{LABEL_COLUMN}' column."
        )

    frame[LABEL_COLUMN] = pd.to_numeric(
        frame[LABEL_COLUMN],
        errors="coerce"
    )

    frame.dropna(subset=[LABEL_COLUMN], inplace=True)

    if not set(frame[LABEL_COLUMN].unique()).issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{dataset_name} labels must be 0 or 1."
        )

    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)

    if frame[LABEL_COLUMN].nunique() != 2:
        raise ValueError(
            f"{dataset_name} must contain both classes."
        )

print("\n2025 dataset shape:", df_train.shape)
print("2025 benign:", int((df_train[LABEL_COLUMN] == 0).sum()))
print("2025 pathogenic:", int((df_train[LABEL_COLUMN] == 1).sum()))

print("\n2026 dataset shape:", df_test.shape)
print("2026 benign:", int((df_test[LABEL_COLUMN] == 0).sum()))
print("2026 pathogenic:", int((df_test[LABEL_COLUMN] == 1).sum()))


# ============================================================
# CHECK FOR OVERLAPPING VARIANTS
# ============================================================

IDENTIFIER_GROUPS = [
    ["VariationID"],
    ["Chr", "Start", "Ref", "Alt"],
    ["Chromosome", "Position", "Ref", "Alt"],
    ["#CHROM", "POS", "REF", "ALT"],
]

overlap_columns = None

for candidate in IDENTIFIER_GROUPS:
    if all(
        col in df_train.columns and col in df_test.columns
        for col in candidate
    ):
        overlap_columns = candidate
        break

overlap_count = 0
overlap_method = "Unavailable"

if overlap_columns:
    def build_variant_keys(frame, columns):
        return (
            frame[columns]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip())
            .agg("|".join, axis=1)
        )

    train_keys = build_variant_keys(
        df_train, overlap_columns
    )
    test_keys = build_variant_keys(
        df_test, overlap_columns
    )

    overlap_set = set(train_keys).intersection(
        set(test_keys)
    )

    overlap_count = len(overlap_set)
    overlap_method = ", ".join(overlap_columns)

    print("\nVariant overlap check")
    print("Key columns:", overlap_method)
    print("Overlapping unique variant keys:", overlap_count)

    if overlap_count > 0:
        warnings.warn(
            "Some variant identifiers appear in both datasets. "
            "Review this overlap before interpreting 2026 "
            "as an independent temporal evaluation."
        )
else:
    print(
        "\nWARNING: Could not identify common variant "
        "identifier columns for the overlap check."
    )


# ============================================================
# REMOVE 2026 ROWS THAT OVERLAP WITH 2025
# ============================================================
# This removes exact identifier matches only when a suitable
# shared identifier schema is available. It does not use the
# 2026 labels to choose features, fit models, or choose thresholds.
#
# The comparison results therefore refer to non-overlapping
# 2026 rows when overlap removal is possible.

if overlap_columns:
    train_key_set = set(
        build_variant_keys(df_train, overlap_columns)
    )

    test_keys = build_variant_keys(
        df_test, overlap_columns
    )

    keep_mask = ~test_keys.isin(train_key_set)

    removed_rows = int((~keep_mask).sum())

    if removed_rows:
        print(
            f"Removing {removed_rows} overlapping 2026 rows "
            "from evaluation."
        )

    df_test = df_test.loc[keep_mask].copy()

    if df_test.empty:
        raise ValueError(
            "No 2026 rows remain after removing overlapping variants."
        )

    if df_test[LABEL_COLUMN].nunique() != 2:
        raise ValueError(
            "After overlap removal, the 2026 dataset must "
            "still contain both benign and pathogenic variants."
        )

    print("2026 rows after overlap check:", len(df_test))


# ============================================================
# PREPARE FEATURES AND EXCLUDE LABEL PROXIES
# ============================================================

# These columns are identifiers, descriptive annotations,
# or possible direct proxies for the target label.
# Keep this list consistent with the original pipeline's
# leakage prevention requirements.

DROP_COLUMNS = {
    "VariationID",
    "Chr",
    "Chromosome",
    "#CHROM",
    "CHROM",
    "Start",
    "Position",
    "POS",
    "End",
    "Ref",
    "Reference",
    "REF",
    "Alt",
    "Alternate",
    "ALT",
    "GeneSymbol",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150",

    # Direct target/clinical label proxies:
    "CLNSIG",
    "CLNSIGCONF",
    "ClinicalSignificance",
    "Clinical_Significance",
    "Clinical significance",
    "Pathogenicity_Label",
    "Target",
}

# Remove known target proxies and identifiers from both datasets.
X_train_raw = df_train.drop(
    columns=[
        col for col in DROP_COLUMNS
        if col in df_train.columns
    ] + [LABEL_COLUMN],
    errors="ignore"
).copy()

X_test_raw = df_test.drop(
    columns=[
        col for col in DROP_COLUMNS
        if col in df_test.columns
    ] + [LABEL_COLUMN],
    errors="ignore"
).copy()

y_train = df_train[LABEL_COLUMN].astype(int).reset_index(drop=True)
y_test = df_test[LABEL_COLUMN].astype(int).reset_index(drop=True)

X_train_raw.reset_index(drop=True, inplace=True)
X_test_raw.reset_index(drop=True, inplace=True)

del df_train, df_test


# ============================================================
# NORMALISE VALUES AND ALIGN FEATURES
# ============================================================

def clean_value(value):
    """
    Standardise missing values and preserve the first
    semicolon-separated annotation, matching the prior
    pipeline convention.
    """

    if isinstance(value, str):
        value = value.strip()

        if value in (
            "", ".", "NA", "N/A", "nan",
            "None", "null", "NaN"
        ):
            return np.nan

        if ";" in value:
            value = value.split(";")[0].strip()

    return value


for column in X_train_raw.columns:
    X_train_raw[column] = X_train_raw[column].map(clean_value)

# Use only columns present in 2025 to define the model schema.
# 2026-only columns cannot become new predictors.
common_columns = [
    col for col in X_train_raw.columns
    if col in X_test_raw.columns
]

X_train_raw = X_train_raw[common_columns].copy()
X_test_raw = X_test_raw[common_columns].copy()

# Infer numeric/categorical feature types from 2025 only.
numeric_columns = []
categorical_columns = []

for column in common_columns:
    converted = pd.to_numeric(
        X_train_raw[column],
        errors="coerce"
    )

    nonmissing_count = X_train_raw[column].notna().sum()

    if nonmissing_count == 0:
        # Entirely missing in 2025: do not use the feature.
        continue

    numeric_fraction = (
        converted.notna().sum() / nonmissing_count
    )

    if numeric_fraction >= 0.80:
        numeric_columns.append(column)
        X_train_raw[column] = converted
        X_test_raw[column] = pd.to_numeric(
            X_test_raw[column],
            errors="coerce"
        )
    else:
        categorical_columns.append(column)
        X_train_raw[column] = (
            X_train_raw[column]
            .fillna("__MISSING__")
            .astype(str)
        )
        X_test_raw[column] = (
            X_test_raw[column]
            .fillna("__MISSING__")
            .astype(str)
        )

# Keep the exact 2025-derived feature list.
usable_columns = numeric_columns + categorical_columns

if not usable_columns:
    raise ValueError(
        "No usable features remain after excluding "
        "identifiers and target proxies."
    )

X_train_raw = X_train_raw[usable_columns].copy()
X_test_raw = X_test_raw[usable_columns].copy()

# Replace infinities with missing values.
for column in numeric_columns:
    X_train_raw[column] = X_train_raw[column].replace(
        [np.inf, -np.inf], np.nan
    )
    X_test_raw[column] = X_test_raw[column].replace(
        [np.inf, -np.inf], np.nan
    )

print("\nFeature preparation")
print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))
print("Total input features:", len(usable_columns))
print("2025 feature matrix:", X_train_raw.shape)
print("2026 feature matrix:", X_test_raw.shape)


# ============================================================
# PREPROCESSING
# ============================================================

def make_one_hot_encoder():
    """
    Support both newer and older scikit-learn versions.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor():
    """
    Each model/fold fits its imputation and encoding only on
    that fold's training partition.
    """

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            )
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                make_one_hot_encoder()
            )
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


# ============================================================
# MODEL FACTORIES
# ============================================================

def make_base_pipeline(classifier):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier)
    ])


def make_stacking_model():
    """
    Pipeline3:
    Random Forest + Gradient Boosting + Extra Trees,
    combined by Logistic Regression.
    """

    random_forest = make_base_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    gradient_boosting = make_base_pipeline(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    )

    extra_trees = make_base_pipeline(
        ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return StackingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("gradient_boosting", gradient_boosting),
            ("extra_trees", extra_trees)
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=False,
        n_jobs=1
    )


def calculate_scale_pos_weight(y):
    """
    Calculate the positive-class weight from the labels in
    the current training partition only.
    """

    y = np.asarray(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    if positives == 0:
        return 1.0

    return negatives / positives


def make_renovo_xgb(y):
    """
    RENOVO configuration from the comparison setup.
    The supplied RENOVO script uses XGBoost.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=calculate_scale_pos_weight(y),
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", model)
    ])


def make_ml_graph_xgb(y, final_fit=False):
    """
    ML Graph XGBoost configuration.

    CV and threshold OOF predictions use 300 estimators.
    The final 2025 fit uses 1200 estimators, matching the
    original ML Graph configuration.
    """

    model = XGBClassifier(
        n_estimators=1200 if final_fit else 300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1,
        scale_pos_weight=calculate_scale_pos_weight(y),
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", model)
    ])


MODEL_NAMES = [
    "Pipeline3_Stacking",
    "RENOVO_XGBoost",
    "ML_Graph_XGBoost"
]


def make_model(model_name, y_fit, final_fit=False):
    if model_name == "Pipeline3_Stacking":
        return make_stacking_model()

    if model_name == "RENOVO_XGBoost":
        return make_renovo_xgb(y_fit)

    if model_name == "ML_Graph_XGBoost":
        return make_ml_graph_xgb(
            y_fit,
            final_fit=final_fit
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# SHARED 5-FOLD CROSS-VALIDATION SPLITS
# ============================================================

if y_train.value_counts().min() < N_SPLITS:
    raise ValueError(
        f"At least {N_SPLITS} examples of each class are "
        "required for stratified cross-validation."
    )

shared_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Materialise splits once and reuse these exact indices
# for every model.
SHARED_FOLDS = list(
    shared_cv.split(X_train_raw, y_train)
)


# ============================================================
# SELECT CLASSIFICATION THRESHOLD FROM 2025 OOF PREDICTIONS
# ============================================================

def select_f1_threshold(y_true, probabilities):
    """
    Select the threshold with the highest pathogenic F1 score.
    Only 2025 out-of-fold predictions are supplied here.
    """

    best_threshold = 0.5
    best_f1 = -1.0

    # A fixed grid keeps the selection deterministic.
    for threshold in np.arange(0.05, 0.951, 0.005):
        predictions = (
            np.asarray(probabilities) >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics using the supplied
    threshold. Positive class 1 means Pathogenic.

    Confusion matrix layout:
                      Predicted
                   Benign  Pathogenic
    Actual Benign      TN       FP
    Actual Pathogenic  FN       TP
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy": accuracy_score(
            y_true, predictions
        ),

        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),

        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),

        "Precision_Pathogenic": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "Recall_Pathogenic": recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "F1_Pathogenic": f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "Macro_F1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted_F1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Specificity_Benign": (
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ),

        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),

        "Actual_Pathogenic": int(np.sum(y_true == 1)),
        "Actual_Benign": int(np.sum(y_true == 0)),
        "Predicted_Pathogenic": int(np.sum(predictions == 1)),
        "Predicted_Benign": int(np.sum(predictions == 0)),
    }

    return metrics, predictions, cm


# ============================================================
# RUN THE COMPARISON
# ============================================================

all_oof_predictions = []
all_fold_metrics = []
all_final_predictions = []
all_classification_reports = []
all_confusion_matrices = []
all_summary_rows = []

final_models = {}
thresholds = {}
oof_probabilities_by_model = {}
final_probabilities_by_model = {}
roc_data = {}

print("\n" + "=" * 100)
print("RUNNING SHARED 5-FOLD CROSS-VALIDATION")
print("=" * 100)

for model_name in MODEL_NAMES:
    print("\n" + "-" * 90)
    print("MODEL:", model_name)
    print("-" * 90)

    oof_probabilities = np.full(
        len(y_train),
        np.nan,
        dtype=float
    )

    model_fold_aucs = []

    # --------------------------------------------------------
    # OUTER CV: identical fold indices for every model
    # --------------------------------------------------------

    for fold_number, (fit_idx, val_idx) in enumerate(
        SHARED_FOLDS,
        start=1
    ):
        X_fold_train = X_train_raw.iloc[fit_idx]
        X_fold_valid = X_train_raw.iloc[val_idx]

        y_fold_train = y_train.iloc[fit_idx]
        y_fold_valid = y_train.iloc[val_idx]

        fold_model = make_model(
            model_name,
            y_fold_train,
            final_fit=False
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train
        )

        fold_probabilities = fold_model.predict_proba(
            X_fold_valid
        )[:, 1]

        oof_probabilities[val_idx] = fold_probabilities

        fold_auc = roc_auc_score(
            y_fold_valid,
            fold_probabilities
        )

        model_fold_aucs.append(float(fold_auc))

        # Fold-level threshold-free metrics.
        all_fold_metrics.append({
            "Model": model_name,
            "Fold": fold_number,
            "Fold_ROC_AUC": float(fold_auc),
            "Fold_PR_AUC": float(
                average_precision_score(
                    y_fold_valid,
                    fold_probabilities
                )
            ),
            "Validation_Rows": len(val_idx),
        })

        print(
            f"Fold {fold_number}/{N_SPLITS} "
            f"ROC-AUC: {fold_auc:.4f}"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            f"{model_name}: some 2025 rows have no OOF predictions."
        )

    # --------------------------------------------------------
    # SELECT THRESHOLD USING 2025 OOF PREDICTIONS ONLY
    # --------------------------------------------------------

    threshold, threshold_oof_f1 = select_f1_threshold(
        y_train,
        oof_probabilities
    )

    thresholds[model_name] = threshold
    oof_probabilities_by_model[model_name] = (
        oof_probabilities.copy()
    )

    print(
        f"Threshold selected from 2025 OOF predictions: "
        f"{threshold:.4f}"
    )
    print(
        f"2025 OOF pathogenic F1 at threshold: "
        f"{threshold_oof_f1:.4f}"
    )

    # Save OOF predictions.
    for index in range(len(y_train)):
        all_oof_predictions.append({
            "Model": model_name,
            "Row_Index_2025": index,
            "Actual_Label": int(y_train.iloc[index]),
            "Actual_Class": (
                "Pathogenic"
                if int(y_train.iloc[index]) == 1
                else "Benign"
            ),
            "OOF_Pathogenic_Probability": float(
                oof_probabilities[index]
            ),
            "Threshold_From_2025": threshold,
            "OOF_Predicted_Label": int(
                oof_probabilities[index] >= threshold
            ),
        })

    # --------------------------------------------------------
    # CROSS-VALIDATION SUMMARY
    # --------------------------------------------------------

    mean_fold_auc = float(np.mean(model_fold_aucs))
    std_fold_auc = float(np.std(model_fold_aucs, ddof=1))

    print(f"Mean fold ROC-AUC: {mean_fold_auc:.4f}")
    print(f"Std fold ROC-AUC : {std_fold_auc:.4f}")

    # --------------------------------------------------------
    # FIT FINAL MODEL ON ALL 2025 DATA
    # --------------------------------------------------------

    print("\nFitting final model on all 2025 training data...")

    final_model = make_model(
        model_name,
        y_train,
        final_fit=True
    )

    final_model.fit(
        X_train_raw,
        y_train
    )

    final_models[model_name] = final_model

    # --------------------------------------------------------
    # EVALUATE ON 2026
    # --------------------------------------------------------
    # The 2026 labels are not used for training or threshold
    # selection. They are used only to calculate final metrics.

    probabilities_2026 = final_model.predict_proba(
        X_test_raw
    )[:, 1]

    final_probabilities_by_model[model_name] = (
        probabilities_2026.copy()
    )

    metrics, predictions_2026, cm = calculate_metrics(
        y_test,
        probabilities_2026,
        threshold
    )

    roc_data[model_name] = roc_curve(
        y_test,
        probabilities_2026
    )

    print("\n2026 evaluation:")
    print(f"Accuracy           : {metrics['Accuracy']:.4f}")
    print(f"ROC-AUC            : {metrics['ROC_AUC']:.4f}")
    print(f"PR-AUC             : {metrics['PR_AUC']:.4f}")
    print(
        f"Pathogenic precision: "
        f"{metrics['Precision_Pathogenic']:.4f}"
    )
    print(
        f"Pathogenic recall   : "
        f"{metrics['Recall_Pathogenic']:.4f}"
    )
    print(
        f"Pathogenic F1       : "
        f"{metrics['F1_Pathogenic']:.4f}"
    )
    print(f"Macro F1            : {metrics['Macro_F1']:.4f}")
    print(f"False negatives     : {metrics['False_Negatives']}")
    print(f"True positives      : {metrics['True_Positives']}")

    print("\nConfusion matrix:")
    print("Rows = actual; columns = predicted")
    print("Order = Benign, Pathogenic")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions_2026,
            labels=[0, 1],
            target_names=["Benign", "Pathogenic"],
            zero_division=0,
            digits=4
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY ROW
    # --------------------------------------------------------

    all_summary_rows.append({
        "Model": model_name,

        "Mean_Fold_ROC_AUC": mean_fold_auc,
        "Std_Fold_ROC_AUC": std_fold_auc,

        "Threshold_From_2025": threshold,
        "2025_OOF_F1_Pathogenic": threshold_oof_f1,

        "2026_Accuracy": metrics["Accuracy"],
        "2026_ROC_AUC": metrics["ROC_AUC"],
        "2026_PR_AUC": metrics["PR_AUC"],

        "2026_Precision_Pathogenic":
            metrics["Precision_Pathogenic"],

        # NEW METRIC 1:
        "2026_Recall_Pathogenic":
            metrics["Recall_Pathogenic"],

        "2026_F1_Pathogenic":
            metrics["F1_Pathogenic"],

        "2026_Macro_F1": metrics["Macro_F1"],
        "2026_Weighted_F1": metrics["Weighted_F1"],

        "2026_Specificity_Benign":
            metrics["Specificity_Benign"],

        "2026_True_Negatives": metrics["True_Negatives"],
        "2026_False_Positives": metrics["False_Positives"],

        # NEW METRIC 2:
        "2026_False_Negatives":
            metrics["False_Negatives"],

        "2026_True_Positives": metrics["True_Positives"],

        "2026_Actual_Pathogenic":
            metrics["Actual_Pathogenic"],

        "2026_Actual_Benign":
            metrics["Actual_Benign"],

        "2026_Predicted_Pathogenic":
            metrics["Predicted_Pathogenic"],

        "2026_Predicted_Benign":
            metrics["Predicted_Benign"],
    })

    # --------------------------------------------------------
    # SAVE INDIVIDUAL 2026 PREDICTIONS
    # --------------------------------------------------------

    for index in range(len(y_test)):
        actual = int(y_test.iloc[index])
        predicted = int(predictions_2026[index])

        all_final_predictions.append({
            "Model": model_name,
            "Row_Index_2026": index,

            "Actual_Label": actual,
            "Actual_Class": (
                "Pathogenic" if actual == 1 else "Benign"
            ),

            "Predicted_Label": predicted,
            "Predicted_Class": (
                "Pathogenic" if predicted == 1 else "Benign"
            ),

            "Pathogenic_Probability": float(
                probabilities_2026[index]
            ),

            "Threshold_From_2025": threshold,

            "Correct_Prediction": bool(
                actual == predicted
            ),

            "Is_False_Negative": bool(
                actual == 1 and predicted == 0
            ),

            "Is_False_Positive": bool(
                actual == 0 and predicted == 1
            ),
        })

    # Save class-level precision/recall/F1.
    report = classification_report(
        y_test,
        predictions_2026,
        labels=[0, 1],
        target_names=["Benign", "Pathogenic"],
        output_dict=True,
        zero_division=0
    )

    for class_name in ["Benign", "Pathogenic", "macro avg", "weighted avg"]:
        if class_name in report:
            all_classification_reports.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1_Score": report[class_name]["f1-score"],
                "Support": report[class_name]["support"],
            })

    tn, fp, fn, tp = cm.ravel()

    all_confusion_matrices.append({
        "Model": model_name,
        "TN_Benign_Correct": int(tn),
        "FP_Benign_Predicted_Pathogenic": int(fp),
        "FN_Pathogenic_Predicted_Benign": int(fn),
        "TP_Pathogenic_Correct": int(tp),
    })


# ============================================================
# SAVE ALL CSV RESULTS
# ============================================================

summary_df = pd.DataFrame(all_summary_rows)

# Sort by 2026 Macro F1, then pathogenic recall.
summary_df = summary_df.sort_values(
    by=[
        "2026_Macro_F1",
        "2026_Recall_Pathogenic",
        "2026_ROC_AUC"
    ],
    ascending=False
).reset_index(drop=True)

fold_df = pd.DataFrame(all_fold_metrics)

cv_summary_df = (
    fold_df.groupby("Model")
    .agg(
        Mean_Fold_ROC_AUC=("Fold_ROC_AUC", "mean"),
        Std_Fold_ROC_AUC=("Fold_ROC_AUC", "std"),
        Mean_Fold_PR_AUC=("Fold_PR_AUC", "mean"),
        Std_Fold_PR_AUC=("Fold_PR_AUC", "std"),
    )
    .reset_index()
)

oof_df = pd.DataFrame(all_oof_predictions)
predictions_df = pd.DataFrame(all_final_predictions)
reports_df = pd.DataFrame(all_classification_reports)
confusion_df = pd.DataFrame(all_confusion_matrices)

summary_df.to_csv(
    output_path("model_comparison_summary.csv"),
    index=False
)

fold_df.to_csv(
    output_path("cross_validation_fold_metrics.csv"),
    index=False
)

cv_summary_df.to_csv(
    output_path("cross_validation_summary.csv"),
    index=False
)

oof_df.to_csv(
    output_path("2025_out_of_fold_predictions.csv"),
    index=False
)

predictions_df.to_csv(
    output_path("2026_test_predictions.csv"),
    index=False
)

reports_df.to_csv(
    output_path("classification_reports.csv"),
    index=False
)

confusion_df.to_csv(
    output_path("confusion_matrices.csv"),
    index=False
)


# ============================================================
# FINAL MODEL COMPARISON TABLE
# ============================================================

display_columns = [
    "Model",
    "Mean_Fold_ROC_AUC",
    "Std_Fold_ROC_AUC",
    "Threshold_From_2025",
    "2026_Accuracy",
    "2026_ROC_AUC",
    "2026_PR_AUC",
    "2026_Precision_Pathogenic",

    # ADDED:
    "2026_Recall_Pathogenic",

    "2026_F1_Pathogenic",
    "2026_Macro_F1",

    # ADDED:
    "2026_False_Negatives",

    "2026_True_Positives",
    "2026_Actual_Pathogenic",
]

print("\n")
print("=" * 120)
print("FINAL MODEL COMPARISON")
print("=" * 120)

print(
    summary_df[display_columns].to_string(
        index=False,
        formatters={
            "Mean_Fold_ROC_AUC": "{:.4f}".format,
            "Std_Fold_ROC_AUC": "{:.4f}".format,
            "Threshold_From_2025": "{:.4f}".format,
            "2026_Accuracy": "{:.4f}".format,
            "2026_ROC_AUC": "{:.4f}".format,
            "2026_PR_AUC": "{:.4f}".format,
            "2026_Precision_Pathogenic": "{:.4f}".format,
            "2026_Recall_Pathogenic": "{:.4f}".format,
            "2026_F1_Pathogenic": "{:.4f}".format,
            "2026_Macro_F1": "{:.4f}".format,
        }
    )
)

print("\nMetric explanations:")
print("- Pathogenic recall = TP / (TP + FN)")
print("- False negatives = truly pathogenic variants predicted benign")
print("- Macro F1 = average F1 of benign and pathogenic classes")
print("- Thresholds are selected from 2025 OOF predictions only")


# ============================================================
# PLOT 1: COMPARISON OF CLASSIFICATION METRICS
# ============================================================

metric_plot_columns = [
    "2026_Accuracy",
    "2026_ROC_AUC",
    "2026_PR_AUC",
    "2026_Recall_Pathogenic",
    "2026_F1_Pathogenic",
    "2026_Macro_F1",
]

plot_df = summary_df.set_index("Model")[
    metric_plot_columns
]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("CLN3 Model Comparison: 2026 Evaluation")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(
    output_path("model_comparison_metrics.png"),
    dpi=300
)
plt.close()


# ============================================================
# PLOT 2: 2026 ROC CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for model_name in MODEL_NAMES:
    fpr, tpr, _ = roc_data[model_name]

    auc_value = roc_auc_score(
        y_test,
        final_probabilities_by_model[model_name]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_value:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: 2026 Temporal Evaluation")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_path("model_comparison_roc_curves.png"),
    dpi=300
)
plt.close()


# ============================================================
# SAVE TRAINED MODELS AND METADATA
# ============================================================

joblib.dump(
    final_models,
    output_path("final_comparison_models.joblib")
)

metadata = {
    "training_file": TRAIN_FILENAME,
    "testing_file": TEST_FILENAME,
    "data_directory": DATA_DIR,
    "output_directory": OUTPUT_DIR,
    "label_column": LABEL_COLUMN,
    "label_mapping": {
        "0": "Benign",
        "1": "Pathogenic"
    },
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "threshold_objective": THRESHOLD_OBJECTIVE,
    "thresholds_selected_from_2025_OOF": thresholds,
    "overlap_identifier_columns": overlap_method,
    "overlapping_variant_key_count_before_removal": int(
        overlap_count
    ),
    "numeric_feature_count": len(numeric_columns),
    "categorical_feature_count": len(categorical_columns),
    "feature_columns": usable_columns,
    "dropped_identifier_and_target_proxy_columns": sorted(
        DROP_COLUMNS
    ),
    "models": MODEL_NAMES,
    "important_note": (
        "2026 labels were not used for fitting, cross-validation, "
        "or threshold selection. They were used only for final "
        "evaluation metrics. Review variant overlap and feature "
        "provenance before interpreting generalisation."
    )
}

joblib.dump(
    metadata,
    output_path("evaluation_metadata.joblib")
)

with open(
    output_path("evaluation_metadata.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("EVALUATION COMPLETE")
print("=" * 100)

print("\nSaved files in:")
print(OUTPUT_DIR)

print("\nOutput files:")
output_files = [
    "model_comparison_summary.csv",
    "cross_validation_fold_metrics.csv",
    "cross_validation_summary.csv",
    "2025_out_of_fold_predictions.csv",
    "2026_test_predictions.csv",
    "classification_reports.csv",
    "confusion_matrices.csv",
    "model_comparison_metrics.png",
    "model_comparison_roc_curves.png",
    "final_comparison_models.joblib",
    "evaluation_metadata.joblib",
    "evaluation_metadata.json",
]

for filename in output_files:
    print(" -", filename)

print("\nNew metrics added to the final comparison:")
print(" - 2026_Recall_Pathogenic")
print(" - 2026_False_Negatives")

print("\nReminder:")
print(
    "False negatives are pathogenic variants classified as benign. "
    "For clinical interpretation, inspect these cases individually."
)
