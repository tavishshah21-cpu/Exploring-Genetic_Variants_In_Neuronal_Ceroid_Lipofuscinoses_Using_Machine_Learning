# ============================================================
# CLN3 RENOVO — 2026 INPUT PREDICTOR
#
# Algorithm: Random Forest (RENOVO)
# Training: CLN3 2026 dataset only
# Threshold: Selected using 5-fold OOF predictions from 2026
#
# Input:
#   Pipeline_1_Statistics/CLN3_2026_CLN3_ML_READY.csv
#
# Output:
#   RENOVO/
#
# Prediction:
#   Pathogenic / Benign / INSUFFICIENT INPUT
# ============================================================

import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = (
    r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
)

OUTPUT_DIR = (
    r"D:\tavish_shah\PES\capstone\RENOVO"
)

TRAIN_PATH = os.path.join(
    SOURCE_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

GENE = "CLN3"

RANDOM_STATE = 42
N_SPLITS = 5

N_ESTIMATORS = 300
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================
# COLUMNS THAT MUST NOT BE USED AS PREDICTORS
# ============================================================

DROP_COLUMNS = [
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

# Clinical labels and classification fields can leak the answer.
LEAKAGE_COLUMNS = [
    "CLNSIG",
    "CLNSIGCONF",
    "ClinicalSignificance",
    "clinical_significance",
    "Pathogenicity",
    "Pathogenic",
    "label_source",
    "Label",
    "classification",
    "Classification",
    "ReviewStatus",
    "ReviewStatus_stars",
    "Germline classification"
]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_value(value):
    """Normalize missing values and semicolon-separated annotations."""

    if isinstance(value, str):
        value = value.strip()

        if value in {
            "", ".", "NA", "N/A", "nan", "None", "NaN"
        }:
            return np.nan

        # Keep the first annotation when multiple values are present.
        if ";" in value:
            value = value.split(";")[0].strip()

            if value in {
                "", ".", "NA", "N/A", "nan", "None", "NaN"
            }:
                return np.nan

    return value


# ============================================================
# LOAD 2026 DATA
# ============================================================

print("\n" + "=" * 70)
print("CLN3 RENOVO — 2026 INPUT PREDICTOR")
print("=" * 70)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_PATH}"
    )

df = pd.read_csv(TRAIN_PATH, low_memory=False)

print("\nDataset shape:", df.shape)

if df.empty:
    raise ValueError("The 2026 training dataset is empty.")

if "label" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'label' column "
        "with 0 = Benign and 1 = Pathogenic."
    )

df = df.loc[df["label"].notna()].copy()

df["label"] = pd.to_numeric(
    df["label"], errors="coerce"
)

df = df.loc[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

if df["label"].nunique() != 2:
    raise ValueError(
        "The 2026 dataset must contain both benign and "
        "pathogenic variants."
    )

if df["label"].value_counts().min() < N_SPLITS:
    raise ValueError(
        f"Each class must contain at least {N_SPLITS} rows "
        "for stratified cross-validation."
    )

print("\n2026 label distribution:")
print(df["label"].value_counts().sort_index().rename(
    index={0: "Benign", 1: "Pathogenic"}
))

y = df["label"].reset_index(drop=True)

# Exclude label and columns that should not be model features.
X_raw = df.drop(
    columns=["label"] + LEAKAGE_COLUMNS + DROP_COLUMNS,
    errors="ignore"
).copy()

X_raw = X_raw.reset_index(drop=True)

del df
gc.collect()

if X_raw.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")


# ============================================================
# LEARN FEATURE ENCODING FROM 2026 TRAINING DATA
# ============================================================

print("\nPreparing model features...")

feature_kinds = {}
category_maps = {}
numeric_columns = []
categorical_columns = []

X_encoded = pd.DataFrame(index=X_raw.index)

for column in X_raw.columns:

    cleaned = X_raw[column].map(clean_value)

    numeric = pd.to_numeric(
        cleaned, errors="coerce"
    )

    nonmissing = cleaned.notna()
    conversion_failed = nonmissing & numeric.isna()

    # If nonmissing values fail numeric conversion, treat the
    # feature as categorical and learn its mapping from 2026.
    if conversion_failed.any():

        categories = list(
            cleaned[nonmissing].astype(str).unique()
        )

        category_map = {
            category: index
            for index, category in enumerate(categories)
        }

        category_maps[column] = category_map
        feature_kinds[column] = "categorical"
        categorical_columns.append(column)

        X_encoded[column] = cleaned.map(
            lambda value: (
                category_map.get(str(value), np.nan)
                if pd.notna(value)
                else np.nan
            )
        )

    else:
        feature_kinds[column] = "numeric"
        numeric_columns.append(column)

        X_encoded[column] = numeric

X_encoded = X_encoded.replace(
    [np.inf, -np.inf], np.nan
)

# Remove features with no usable values in the training dataset.
valid_columns = [
    column
    for column in X_encoded.columns
    if X_encoded[column].notna().any()
]

X_encoded = X_encoded[valid_columns].copy()

# Store raw feature names that the predictor will request.
input_columns = list(valid_columns)

if not input_columns:
    raise ValueError(
        "No features contain usable training values."
    )

print("Usable biological/annotation features:", len(input_columns))
print("Numeric features:", sum(
    feature_kinds[c] == "numeric" for c in input_columns
))
print("Categorical features:", sum(
    feature_kinds[c] == "categorical" for c in input_columns
))


# ============================================================
# CREATE MISSING-VALUE INDICATORS
# ============================================================

X_missing = X_encoded.isna().astype(np.int8)

X_missing.columns = [
    f"{column}_missing"
    for column in X_missing.columns
]

X = pd.concat(
    [
        X_encoded.reset_index(drop=True),
        X_missing.reset_index(drop=True)
    ],
    axis=1
)

# Fit median imputation values on the 2026 training dataset.
training_medians = X.median(axis=0).fillna(0)

X = X.fillna(training_medians).fillna(0)
X = X.astype(np.float32)

model_feature_columns = list(X.columns)

print("Final model feature count:", X.shape[1])
print("Training matrix:", X.shape)


# ============================================================
# RENOVO MODEL
# ============================================================

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
# FIVE-FOLD OOF PREDICTIONS ON 2026
# Used to choose the threshold
# ============================================================

print("\nGenerating 5-fold out-of-fold predictions on 2026...")

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_prob = np.full(len(y), np.nan, dtype=float)
fold_auc_scores = []

for fold, (fit_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):
    print(f"\nTraining fold {fold}/{N_SPLITS}...")

    fold_model = make_model()

    fold_model.fit(
        X.iloc[fit_idx],
        y.iloc[fit_idx]
    )

    probabilities = fold_model.predict_proba(
        X.iloc[valid_idx]
    )[:, 1]

    oof_prob[valid_idx] = probabilities

    fold_auc = roc_auc_score(
        y.iloc[valid_idx],
        probabilities
    )

    fold_auc_scores.append(fold_auc)

    print(f"Fold {fold} ROC-AUC: {fold_auc:.4f}")

    del fold_model
    gc.collect()

if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Some training rows did not receive OOF predictions."
    )

mean_cv_auc = float(np.mean(fold_auc_scores))
oof_auc = float(roc_auc_score(y, oof_prob))

print("\nMean 5-fold ROC-AUC:", round(mean_cv_auc, 4))
print("Combined OOF ROC-AUC:", round(oof_auc, 4))


# ============================================================
# SELECT THRESHOLD USING OOF PREDICTIONS
# ============================================================

print("\nSelecting threshold using 2026 OOF predictions...")

precision, recall, thresholds = precision_recall_curve(
    y, oof_prob
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

if len(thresholds) == 0:
    raise RuntimeError(
        "Could not calculate a classification threshold."
    )

best_index = int(np.nanargmax(f1_values))
best_threshold = float(thresholds[best_index])
best_oof_f1 = float(f1_values[best_index])

print("Selected threshold:", round(best_threshold, 6))
print("OOF F1-score:", round(best_oof_f1, 4))

threshold_record = pd.DataFrame({
    "Gene": [GENE],
    "Training_year": [2026],
    "Algorithm": ["RandomForestClassifier"],
    "Threshold_method": [
        "5-fold out-of-fold predictions"
    ],
    "Threshold_metric": ["F1-score"],
    "Threshold": [best_threshold],
    "OOF_F1": [best_oof_f1],
    "OOF_ROC_AUC": [oof_auc],
    "Mean_fold_ROC_AUC": [mean_cv_auc]
})

threshold_record.to_csv(
    out("CLN3_RENOVO_2026_threshold_selection.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL MODEL ON ALL LABELLED 2026 DATA
# ============================================================

print("\nTraining final RENOVO model on all 2026 data...")

model = make_model()
model.fit(X, y)

print("Final model trained on all labelled 2026 data.")


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

model_path = out("CLN3_RENOVO_2026_model.pkl")
joblib.dump(model, model_path)

metadata = {
    "gene": GENE,
    "pipeline": "RENOVO",
    "algorithm": "RandomForestClassifier",
    "training_year": 2026,
    "input_columns": input_columns,
    "model_feature_columns": model_feature_columns,
    "feature_kinds": feature_kinds,
    "category_maps": category_maps,
    "training_medians": training_medians.to_dict(),
    "threshold": best_threshold,
    "threshold_source": "2026 five-fold OOF predictions",
    "threshold_metric": "F1-score",
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

joblib.dump(
    metadata,
    out("CLN3_RENOVO_2026_metadata.pkl")
)

print("Saved model:", model_path)


# ============================================================
# SAVE TRAINING OOF PREDICTIONS
# ============================================================

oof_results = pd.DataFrame({
    "Actual_label": y,
    "OOF_pathogenic_probability": oof_prob,
    "Threshold_used": best_threshold,
    "OOF_predicted_label": (
        oof_prob >= best_threshold
    ).astype(int)
})

oof_results["Actual_class"] = oof_results[
    "Actual_label"
].map({0: "Benign", 1: "Pathogenic"})

oof_results["OOF_predicted_class"] = oof_results[
    "OOF_predicted_label"
].map({0: "Benign", 1: "Pathogenic"})

oof_results.to_csv(
    out("CLN3_RENOVO_2026_OOF_predictions.csv"),
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": model_feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    out("CLN3_RENOVO_2026_feature_importance.csv"),
    index=False
)

print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))


# ============================================================
# INTERACTIVE INPUT AND PREDICTION
# ============================================================

def prepare_single_input(user_values):
    """
    Convert the user's raw feature values into the exact feature
    matrix expected by the trained model.
    """

    row = {}

    # Convert raw inputs using mappings learned from 2026.
    for column in input_columns:

        value = clean_value(user_values.get(column, np.nan))

        if pd.isna(value):
            row[column] = np.nan
            continue

        if feature_kinds[column] == "categorical":

            category_map = category_maps.get(column, {})

            # Unknown categories are treated as missing.
            row[column] = category_map.get(str(value), np.nan)

        else:
            try:
                numeric_value = pd.to_numeric(
                    value, errors="coerce"
                )

                row[column] = (
                    float(numeric_value)
                    if pd.notna(numeric_value)
                    else np.nan
                )

            except (TypeError, ValueError):
                row[column] = np.nan

    raw_row = pd.DataFrame([row], columns=input_columns)

    raw_row = raw_row.replace(
        [np.inf, -np.inf], np.nan
    )

    # Missingness indicators must match the training schema.
    missing_row = raw_row.isna().astype(np.int8)

    missing_row.columns = [
        f"{column}_missing"
        for column in missing_row.columns
    ]

    final_row = pd.concat(
        [raw_row, missing_row],
        axis=1
    )

    final_row = final_row.reindex(
        columns=model_feature_columns
    )

    final_row = final_row.fillna(training_medians).fillna(0)
    final_row = final_row.astype(np.float32)

    return final_row


def predict_variant(user_values):
    """
    Return a prediction, or INSUFFICIENT INPUT if every feature
    is blank or no recognized usable value was supplied.
    """

    supplied_count = 0

    for column in input_columns:

        value = clean_value(user_values.get(column, np.nan))

        if pd.isna(value):
            continue

        if feature_kinds[column] == "categorical":

            # Do not count unrecognized category values as valid.
            if str(value) not in category_maps.get(column, {}):
                continue

            supplied_count += 1

        else:

            numeric_value = pd.to_numeric(
                value, errors="coerce"
            )

            if pd.notna(numeric_value) and np.isfinite(
                float(numeric_value)
            ):
                supplied_count += 1

    if supplied_count == 0:
        return {
            "status": "INSUFFICIENT INPUT",
            "message": (
                "No usable feature values were provided. "
                "Enter at least one valid feature value."
            )
        }

    prepared = prepare_single_input(user_values)

    probability = float(
        model.predict_proba(prepared)[0, 1]
    )

    predicted_label = int(
        probability >= best_threshold
    )

    predicted_class = (
        "Pathogenic" if predicted_label == 1 else "Benign"
    )

    return {
        "status": "OK",
        "gene": GENE,
        "algorithm": "RENOVO Random Forest",
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "pathogenic_probability": probability,
        "benign_probability": 1.0 - probability,
        "threshold": best_threshold,
        "features_provided": supplied_count
    }


# ============================================================
# INTERACTIVE LOOP
# ============================================================

print("\n" + "=" * 70)
print("CLN3 RENOVO — VARIANT PREDICTOR")
print("=" * 70)

print("\nModel feature inputs:")
print("- Enter a numeric value for numeric features.")
print("- For categorical features, enter one of the listed values.")
print("- Press Enter to leave a feature blank.")
print("- If every feature is blank or invalid, prediction is refused.")
print("- Type 'exit' at the variant prompt to stop.\n")

prediction_log = []

while True:

    print("\n" + "-" * 70)

    variant_name = input(
        "Variant identifier/name (or 'exit'): "
    ).strip()

    if variant_name.lower() in {"exit", "quit", "q"}:
        break

    user_values = {}

    print("\nEnter feature values. Blank = missing.\n")

    for index, column in enumerate(input_columns, start=1):

        if feature_kinds[column] == "categorical":

            allowed_values = list(
                category_maps.get(column, {}).keys()
            )

            # Keep the prompt manageable for high-cardinality fields.
            shown_values = allowed_values[:15]

            prompt = (
                f"{index}/{len(input_columns)} {column} "
                f"[categorical; examples: {shown_values}]: "
            )

        else:
            prompt = (
                f"{index}/{len(input_columns)} {column} [numeric]: "
            )

        value = input(prompt).strip()

        user_values[column] = (
            np.nan if value == "" else value
        )

    result = predict_variant(user_values)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print("Variant:", variant_name)

    if result["status"] != "OK":

        print("Classification:", result["status"])
        print(result["message"])

        prediction_log.append({
            "Variant": variant_name,
            "Status": result["status"],
            "Predicted_class": None,
            "Pathogenic_probability": np.nan,
            "Benign_probability": np.nan,
            "Threshold": best_threshold,
            "Features_provided": 0
        })

    else:

        print("Gene:", result["gene"])
        print("Algorithm:", result["algorithm"])
        print("Classification:", result["predicted_class"])
        print(
            "Pathogenic probability: "
            f"{result['pathogenic_probability']:.4f} "
            f"({result['pathogenic_probability'] * 100:.2f}%)"
        )
        print(
            "Benign probability: "
            f"{result['benign_probability']:.4f} "
            f"({result['benign_probability'] * 100:.2f}%)"
        )
        print("Threshold:", f"{result['threshold']:.6f}")
        print("Features provided:", result["features_provided"])

        prediction_log.append({
            "Variant": variant_name,
            "Status": result["status"],
            "Predicted_class": result["predicted_class"],
            "Predicted_label": result["predicted_label"],
            "Pathogenic_probability": result[
                "pathogenic_probability"
            ],
            "Benign_probability": result["benign_probability"],
            "Threshold": result["threshold"],
            "Features_provided": result["features_provided"]
        })

    pd.DataFrame(prediction_log).to_csv(
        out("CLN3_RENOVO_2026_prediction_log.csv"),
        index=False
    )

    print("\nPrediction log updated.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLN3 RENOVO PREDICTOR FINISHED")
print("=" * 70)

print("Training dataset:", TRAIN_PATH)
print("Training year: 2026")
print("Algorithm: Random Forest")
print("Threshold:", round(best_threshold, 6))
print("Mean 5-fold ROC-AUC:", round(mean_cv_auc, 4))
print("Combined OOF ROC-AUC:", round(oof_auc, 4))
print("OOF F1-score:", round(best_oof_f1, 4))
print("Output directory:", OUTPUT_DIR)

print("=" * 70)
