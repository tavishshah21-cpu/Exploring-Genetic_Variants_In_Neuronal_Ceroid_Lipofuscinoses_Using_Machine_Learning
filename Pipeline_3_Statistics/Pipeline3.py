
# ============================================================
# PIPELINE 3: CLN3 PATHOGENICITY PREDICTION
#
# ALGORITHM:
#   1. Random Forest
#   2. Gradient Boosting
#   3. Extra Trees
#   4. Logistic Regression Meta-Model
#
# DATASETS:
#   Train: 2025 CLN3 ML-ready dataset
#   Test:  2026 CLN3 ML-ready dataset
#
# LEAKAGE CONTROLS:
#   1. Remove overlapping variants from the 2026 test set.
#   2. Fit imputation and encoding inside each CV fold.
#   3. Generate out-of-fold predictions for stacking.
#   4. Select threshold using nested OOF predictions on 2025.
#   5. Never use 2026 labels to select the threshold.
#
# INPUT:  Pipeline_1_Statistics
# OUTPUT: Pipeline_3_Statistics
# ============================================================

import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

BASE_DIR = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
)

OUTPUT_DIR = (
    r"D:\tavish_shah\PES\capstone\Pipeline_3_Statistics"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "CLN3_2025_CLN3_ML_READY.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

GENE = "CLN3"
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PIPELINE 3: LEAKAGE-RESISTANT CLN3 STACKING ENSEMBLE")
print("=" * 70)

print("\nInput directory :", BASE_DIR)
print("Output directory:", OUTPUT_DIR)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"2025 training dataset not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"2026 testing dataset not found:\n{TEST_PATH}"
    )

df_train = pd.read_csv(TRAIN_PATH, low_memory=False)
df_test = pd.read_csv(TEST_PATH, low_memory=False)

print("\nOriginal 2025 shape:", df_train.shape)
print("Original 2026 shape:", df_test.shape)


# ============================================================
# VALIDATE LABELS
#
# 0 = Benign
# 1 = Pathogenic
# ============================================================

if "label" not in df_train.columns:
    raise ValueError("2025 dataset must contain a 'label' column.")

if "label" not in df_test.columns:
    raise ValueError("2026 dataset must contain a 'label' column.")

df_train["label"] = pd.to_numeric(
    df_train["label"], errors="coerce"
)

df_test["label"] = pd.to_numeric(
    df_test["label"], errors="coerce"
)

df_train.dropna(subset=["label"], inplace=True)
df_test.dropna(subset=["label"], inplace=True)

valid_labels = {0, 1}

if not set(df_train["label"].unique()).issubset(valid_labels):
    raise ValueError("2025 labels must be 0 (Benign) or 1 (Pathogenic).")

if not set(df_test["label"].unique()).issubset(valid_labels):
    raise ValueError("2026 labels must be 0 (Benign) or 1 (Pathogenic).")

df_train["label"] = df_train["label"].astype(int)
df_test["label"] = df_test["label"].astype(int)

if df_train["label"].nunique() != 2:
    raise ValueError("2025 must contain both classes.")

if df_test["label"].nunique() != 2:
    raise ValueError("2026 must contain both classes for ROC-AUC.")

print("\n2025 Pathogenic:", int((df_train["label"] == 1).sum()))
print("2025 Benign    :", int((df_train["label"] == 0).sum()))

print("\n2026 Pathogenic:", int((df_test["label"] == 1).sum()))
print("2026 Benign    :", int((df_test["label"] == 0).sum()))


# ============================================================
# REMOVE OVERLAPPING VARIANTS
#
# VariationID is preferred when available in both datasets.
# Otherwise, use chromosome + position + ref + alt when
# those columns are available in both datasets.
#
# Identifiers are used for matching only, not as model features.
# ============================================================

print("\nChecking for overlapping variants...")

train_id = None
test_id = None
overlap_method = None

# Preferred identifier.
if "VariationID" in df_train.columns and "VariationID" in df_test.columns:

    train_ids = df_train["VariationID"].astype("string").str.strip()
    test_ids = df_test["VariationID"].astype("string").str.strip()

    valid_train_ids = (
        train_ids.notna()
        & train_ids.ne("")
        & train_ids.ne(".")
        & train_ids.ne("nan")
    )

    valid_test_ids = (
        test_ids.notna()
        & test_ids.ne("")
        & test_ids.ne(".")
        & test_ids.ne("nan")
    )

    known_ids = set(train_ids[valid_train_ids])

    overlap_mask = (
        test_ids.isin(known_ids)
        & valid_test_ids
    )

    overlap_method = "VariationID"

else:

    # Support common alternative column names.
    column_aliases = {
        "chrom": ["Chr", "Chromosome", "CHROM", "#CHROM"],
        "pos": ["Start", "Position", "POS"],
        "ref": ["Ref", "Reference", "REF"],
        "alt": ["Alt", "Alternate", "ALT"]
    }

    resolved = {}

    for key, aliases in column_aliases.items():
        train_col = next(
            (c for c in aliases if c in df_train.columns),
            None
        )
        test_col = next(
            (c for c in aliases if c in df_test.columns),
            None
        )

        if train_col is not None and test_col is not None:
            resolved[key] = (train_col, test_col)

    if len(resolved) == 4:

        def variant_keys(df, side):
            columns = [
                resolved["chrom"][0 if side == "train" else 1],
                resolved["pos"][0 if side == "train" else 1],
                resolved["ref"][0 if side == "train" else 1],
                resolved["alt"][0 if side == "train" else 1]
            ]

            parts = []

            for col in columns:
                values = df[col].astype("string").str.strip()
                parts.append(values.fillna("").str.upper())

            keys = (
                parts[0] + ":" +
                parts[1] + ":" +
                parts[2] + ":" +
                parts[3]
            )

            valid = (
                parts[0].ne("")
                & parts[1].ne("")
                & parts[2].ne("")
                & parts[3].ne("")
                & parts[0].ne(".")
                & parts[1].ne(".")
                & parts[2].ne(".")
                & parts[3].ne(".")
            )

            return keys, valid

        train_keys, valid_train_keys = variant_keys(
            df_train, "train"
        )

        test_keys, valid_test_keys = variant_keys(
            df_test, "test"
        )

        known_keys = set(train_keys[valid_train_keys])

        overlap_mask = (
            test_keys.isin(known_keys)
            & valid_test_keys
        )

        overlap_method = "Chromosome:Position:Ref:Alt"

    else:
        overlap_mask = pd.Series(
            False, index=df_test.index
        )

        print(
            "WARNING: No shared VariationID or complete variant "
            "coordinate key was found."
        )
        print(
            "Cross-year duplicate variants cannot be verified."
        )

overlap_count = int(overlap_mask.sum())

print("Overlap identification method:", overlap_method or "Unavailable")
print("Overlapping 2026 rows removed:", overlap_count)

df_test = df_test.loc[~overlap_mask].copy()

if df_test.empty:
    raise ValueError(
        "No 2026 test rows remain after removing overlapping variants."
    )

# Recheck class availability after duplicate removal.
if df_test["label"].nunique() != 2:
    raise ValueError(
        "The 2026 test set must contain both classes after "
        "overlap removal."
    )

print("2026 shape after overlap removal:", df_test.shape)

print("\nRemaining 2026 Pathogenic:",
      int((df_test["label"] == 1).sum()))
print("Remaining 2026 Benign    :",
      int((df_test["label"] == 0).sum()))


# ============================================================
# SAVE VARIANT IDENTIFIERS FOR OUTPUT TRACEABILITY
# ============================================================

identifier_candidates = [
    "VariationID",
    "Chr",
    "Chromosome",
    "Start",
    "Position",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol"
]

available_identifiers = [
    c for c in identifier_candidates
    if c in df_test.columns
]

test_identifiers = df_test[
    available_identifiers
].copy()

# Reset indexes after filtering.
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
test_identifiers = test_identifiers.reset_index(drop=True)


# ============================================================
# REMOVE IDENTIFIER AND ANNOTATION COLUMNS FROM FEATURES
# ============================================================

print("\nPreparing model features...")

identifier_columns = [
    "VariationID",
    "Chr",
    "Chromosome",
    "#CHROM",
    "Start",
    "Position",
    "End",
    "POS",
    "Ref",
    "Reference",
    "REF",
    "Alt",
    "Alternate",
    "ALT",
    "GeneSymbol"
]

annotation_columns = [
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

columns_to_drop = (
    identifier_columns + annotation_columns + ["label"]
)

X_train = df_train.drop(
    columns=[
        c for c in columns_to_drop
        if c in df_train.columns
    ]
).copy()

X_test = df_test.drop(
    columns=[
        c for c in columns_to_drop
        if c in df_test.columns
    ]
).copy()

# Remove duplicate column names.
X_train = X_train.loc[
    :, ~X_train.columns.duplicated()
].copy()

X_test = X_test.loc[
    :, ~X_test.columns.duplicated()
].copy()

# The training feature schema defines the allowed features.
X_test = X_test.reindex(
    columns=X_train.columns
)

y_train = df_train["label"].copy()
y_test = df_test["label"].copy()

if X_train.shape[1] == 0:
    raise ValueError("No model features remain after cleaning.")

print("Training features before type cleaning:", X_train.shape)
print("Testing features before type cleaning :", X_test.shape)


# ============================================================
# CLEAN FEATURE VALUES
#
# This step only normalizes raw representations.
# Learned imputation and categorical encoding happen inside
# each fold's sklearn pipeline below.
# ============================================================

def normalize_raw_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value in ("", ".", "NA", "N/A", "nan", "None"):
            return np.nan

        if ";" in value:
            value = value.split(";")[0].strip()

            if value in ("", ".", "NA", "N/A", "nan", "None"):
                return np.nan

    return value


for col in X_train.columns:

    train_clean = X_train[col].map(normalize_raw_value)
    test_clean = X_test[col].map(normalize_raw_value)

    # Determine numeric versus categorical from 2025 only.
    train_numeric = pd.to_numeric(
        train_clean, errors="coerce"
    )

    non_missing_count = train_clean.notna().sum()

    if non_missing_count == 0:
        # Retain the column as categorical; fold-safe imputation
        # will handle missing values.
        X_train[col] = train_clean.astype("object")
        X_test[col] = test_clean.astype("object")
        continue

    numeric_ratio = (
        train_numeric.notna().sum() / non_missing_count
    )

    if pd.api.types.is_numeric_dtype(X_train[col]) or numeric_ratio >= 0.80:

        X_train[col] = train_numeric

        X_test[col] = pd.to_numeric(
            test_clean, errors="coerce"
        )

    else:
        X_train[col] = train_clean.astype("object")
        X_test[col] = test_clean.astype("object")

# Remove columns that have no observed training values.
valid_features = [
    col for col in X_train.columns
    if X_train[col].notna().any()
]

X_train = X_train[valid_features].copy()
X_test = X_test.reindex(columns=valid_features)

if X_train.shape[1] == 0:
    raise ValueError(
        "All features are empty. Check the ML-ready datasets."
    )

# Identify data types using the 2025 training data only.
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    c for c in X_train.columns
    if c not in numeric_features
]

print("\nNumeric features    :", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# FOLD-SAFE PREPROCESSOR
#
# Each base estimator has its own preprocessing pipeline.
# During cross-validation, imputers and encoders are fitted
# only on that fold's training rows.
# ============================================================

def make_preprocessor():

    transformers = []

    if numeric_features:
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
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )


# ============================================================
# BASE MODELS
# ============================================================

def make_base_pipeline(classifier):

    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", classifier)
    ])


def make_stacking_model():

    rf = make_base_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    gb = make_base_pipeline(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    )

    et = make_base_pipeline(
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
            ("random_forest", rf),
            ("gradient_boosting", gb),
            ("extra_trees", et)
        ],
        final_estimator=LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=False,
        n_jobs=1
    )


# ============================================================
# THRESHOLD SELECTION USING 2025 ONLY
#
# Outer CV:
#   Each held-out 2025 row is predicted by a complete stacking
#   model that was fitted without that row.
#
# Inner CV:
#   Each stacking model generates out-of-fold base-model
#   predictions for training its Logistic Regression meta-model.
#
# All learned preprocessing is refitted within each fold.
# ============================================================

print("\n" + "=" * 70)
print("GENERATING 2025 OUT-OF-FOLD PROBABILITIES")
print("=" * 70)

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

threshold_model = make_stacking_model()

oof_prob_2025 = cross_val_predict(
    threshold_model,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

if not np.isfinite(oof_prob_2025).all():
    raise ValueError(
        "Invalid OOF probabilities generated on 2025 data."
    )

precision_2025, recall_2025, thresholds_2025 = (
    precision_recall_curve(
        y_train,
        oof_prob_2025
    )
)

if len(thresholds_2025) == 0:
    best_threshold = 0.5
else:
    f1_scores_2025 = (
        2 * precision_2025[:-1] * recall_2025[:-1]
        / (
            precision_2025[:-1]
            + recall_2025[:-1]
            + 1e-8
        )
    )

    best_index = int(np.argmax(f1_scores_2025))

    best_threshold = float(
        thresholds_2025[best_index]
    )

print("\nThreshold selected using 2025 OOF predictions:",
      round(best_threshold, 4))

# Save OOF threshold-selection results for auditability.
threshold_results = pd.DataFrame({
    "Actual_Label_2025": y_train.to_numpy(),
    "OOF_Pathogenicity_Probability_2025": oof_prob_2025
})

threshold_results.to_csv(
    out("Pipeline3_CLN3_2025_OOF_threshold_data.csv"),
    index=False
)


# ============================================================
# FIT FINAL STACKING MODEL ON ALL 2025 DATA
#
# This final model is used only to predict 2026.
# The threshold remains fixed from 2025 OOF predictions.
# ============================================================

print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON 2025")
print("=" * 70)

final_model = make_stacking_model()

final_model.fit(
    X_train,
    y_train
)

print("\nPredicting 2026 test probabilities...")

y_prob = final_model.predict_proba(
    X_test
)[:, 1]

# Apply the already-selected 2025 threshold.
y_pred = (
    y_prob >= best_threshold
).astype(int)


# ============================================================
# EVALUATION ON 2026
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE 3 EVALUATION: TRAIN 2025, TEST 2026")
print("=" * 70)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\nThreshold selected from 2025:", round(best_threshold, 4))

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

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
    f"Pipeline 3 Stacking — 2025 to 2026 "
    f"(Threshold={best_threshold:.2f})"
)

plt.savefig(
    out("Pipeline3_CLN3_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"Stacking Ensemble (AUC={roc:.4f})"
)

plt.title(
    "Pipeline 3 CLN3 ROC Curve — 2025 to 2026"
)

plt.savefig(
    out("Pipeline3_CLN3_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# RANDOM FOREST + EXTRA TREES
#
# Importance is calculated from the final 2025-fitted models.
# ============================================================

rf_pipeline = final_model.named_estimators_["random_forest"]
et_pipeline = final_model.named_estimators_["extra_trees"]

rf_classifier = rf_pipeline.named_steps["classifier"]
et_classifier = et_pipeline.named_steps["classifier"]

rf_feature_names = (
    rf_pipeline.named_steps["preprocess"].get_feature_names_out()
)

et_feature_names = (
    et_pipeline.named_steps["preprocess"].get_feature_names_out()
)

if not np.array_equal(rf_feature_names, et_feature_names):
    raise ValueError(
        "Random Forest and Extra Trees transformed feature "
        "columns do not match."
    )

feature_importance = (
    rf_classifier.feature_importances_
    + et_classifier.feature_importances_
) / 2.0

feat = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": feature_importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features:")

print(
    feat.head(20).to_string(index=False)
)

feat.to_csv(
    out("Pipeline3_CLN3_feature_importance.csv"),
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test_identifiers.copy()

predictions["Actual_Label"] = y_test.to_numpy()
predictions["Predicted_Label"] = y_pred

predictions["Predicted_Class"] = np.where(
    y_pred == 1,
    "Pathogenic",
    "Benign"
)

predictions["Predicted_Pathogenicity_Probability"] = y_prob

predictions["Correct"] = (
    y_test.to_numpy() == y_pred
)

predictions.to_csv(
    out("Pipeline3_CLN3_predictions.csv"),
    index=False
)


# ============================================================
# SAVE FINAL MODEL
#
# The saved stacking model contains its fitted preprocessing
# pipelines, base estimators, and meta-model.
# ============================================================

model_bundle = {
    "gene": GENE,
    "model": final_model,
    "threshold": best_threshold,
    "raw_feature_columns": list(X_train.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "overlap_method": overlap_method,
    "overlapping_2026_rows_removed": overlap_count,
    "algorithm": (
        "Random Forest + Gradient Boosting + Extra Trees "
        "with Logistic Regression meta-model"
    ),
    "threshold_method": (
        "F1 maximization using nested 5-fold OOF predictions "
        "from 2025 training data"
    )
}

joblib.dump(
    model_bundle,
    out("Pipeline3_CLN3_stacking_model.pkl")
)


# ============================================================
# SAVE RESULTS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Pipeline": ["Pipeline 3"],
    "Gene": [GENE],
    "Algorithm": ["Stacking Ensemble"],
    "Base_Models": [
        "Random Forest; Gradient Boosting; Extra Trees"
    ],
    "Meta_Model": ["Logistic Regression"],
    "Train_year": [2025],
    "Test_year": [2026],
    "Accuracy": [round(acc, 4)],
    "ROC_AUC": [round(roc, 4)],
    "PR_AUC": [round(pr_auc, 4)],
    "Threshold": [round(best_threshold, 4)],
    "Threshold_Source": ["2025 OOF predictions"],
    "Train_size": [len(X_train)],
    "Test_size": [len(X_test)],
    "Train_Pathogenic": [int((y_train == 1).sum())],
    "Train_Benign": [int((y_train == 0).sum())],
    "Test_Pathogenic": [int((y_test == 1).sum())],
    "Test_Benign": [int((y_test == 0).sum())],
    "Feature_count_raw": [X_train.shape[1]],
    "Feature_count_transformed": [len(rf_feature_names)],
    "Overlapping_2026_rows_removed": [overlap_count],
    "Overlap_Method": [overlap_method or "Unavailable"]
})

summary.to_csv(
    out("Pipeline3_CLN3_results_summary.csv"),
    index=False
)


# ============================================================
# SAVE TRANSFORMED FEATURE COLUMNS
# ============================================================

pd.Series(
    rf_feature_names,
    name="Feature"
).to_csv(
    out("Pipeline3_CLN3_feature_columns.txt"),
    index=False,
    header=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE 3 COMPLETED")
print("=" * 70)

print("\nInput datasets:")
print("Training:", TRAIN_PATH)
print("Testing :", TEST_PATH)

print("\nLeakage controls:")
print("- Overlapping variants removed where identifiers permit")
print("- Imputation fitted inside model folds")
print("- Categorical encoding fitted inside model folds")
print("- Stacking uses out-of-fold base-model predictions")
print("- Threshold selected using 2025 OOF predictions")
print("- 2026 labels used only for final evaluation")

print("\nFinal threshold:", round(best_threshold, 4))

print("\nFinal metrics on 2026:")
print("Accuracy :", round(acc, 4))
print("ROC-AUC  :", round(roc, 4))
print("PR-AUC   :", round(pr_auc, 4))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")
print("- Pipeline3_CLN3_stacking_model.pkl")
print("- Pipeline3_CLN3_predictions.csv")
print("- Pipeline3_CLN3_results_summary.csv")
print("- Pipeline3_CLN3_feature_importance.csv")
print("- Pipeline3_CLN3_confusion_matrix.png")
print("- Pipeline3_CLN3_roc_curve.png")
print("- Pipeline3_CLN3_2025_OOF_threshold_data.csv")
print("- Pipeline3_CLN3_feature_columns.txt")

print("\n" + "=" * 70)

gc.collect()
