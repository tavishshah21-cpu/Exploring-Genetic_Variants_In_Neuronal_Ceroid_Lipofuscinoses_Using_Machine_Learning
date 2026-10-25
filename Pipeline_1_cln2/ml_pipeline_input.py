# ============================================================
# CLN2 VARIANT PATHOGENICITY PREDICTOR
# Train on the full 2026 CLN2 dataset
# Predict an input CLN2 variant as pathogenic or benign
# ============================================================

import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "CLN2_2026_CLN2_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "CLN2_2026_trained_model.pkl"
)

GENE = "CLN2"
RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS_FOR_FINAL_MODEL = 1200
N_ESTIMATORS_FOR_THRESHOLD = 300


def out(filename):
    return os.path.join(BASE_DIR, filename)


# ============================================================
# DATA LOADING
# ============================================================

print("\n" + "=" * 70)
print("CLN2 VARIANT PATHOGENICITY PREDICTOR")
print("Training year: 2026")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nCould not find the 2026 CLN2 dataset:\n{DATA_PATH}\n"
        "Place CLN2_2026_CLN2_ML_READY.csv in the same folder as this script, "
        "or update DATA_PATH."
    )

print("\nLoading 2026 CLN2 dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

if "label" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'label' column where "
        "0 = benign and 1 = pathogenic."
    )

# Keep only rows with known binary labels.
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

if df.empty:
    raise ValueError("No rows with valid labels 0 or 1 were found.")

y = df["label"].reset_index(drop=True)

print("2026 labelled dataset:", df.shape)
print("Pathogenic (1):", int((y == 1).sum()))
print("Benign (0):", int((y == 0).sum()))

if y.nunique() != 2:
    raise ValueError(
        "The 2026 training dataset must contain both benign (0) "
        "and pathogenic (1) examples."
    )

if min(y.value_counts()) < N_SPLITS:
    raise ValueError(
        f"Each class needs at least {N_SPLITS} rows for "
        f"{N_SPLITS}-fold stratified cross-validation."
    )


# ============================================================
# REMOVE IDENTIFIERS AND LABEL-LEAKAGE COLUMNS
# ============================================================

drop_cols = [
    "Chr",
    "Chromosome",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol",
    "Gene",
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

target_leakage_columns = [
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
    "ReviewStatus_stars"
]

X = df.drop(columns=["label"], errors="ignore").copy()

X.drop(
    columns=[
        c for c in drop_cols + target_leakage_columns
        if c in X.columns
    ],
    inplace=True,
    errors="ignore"
)

del df
gc.collect()


# ============================================================
# CLEAN AND ENCODE FEATURES
# ============================================================

print("\nPreparing features...")


def clean_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value in {"", ".", "NA", "N/A", "nan", "None"}:
            return np.nan

        # Use the first annotation if multiple values are present.
        if ";" in value:
            value = value.split(";")[0].strip()

            if value in {"", ".", "NA", "N/A", "nan", "None"}:
                return np.nan

    return value


# Encode categorical variant type if present.
if "ExonicFunc.refGene" in X.columns:

    category = (
        X["ExonicFunc.refGene"]
        .fillna("__MISSING__")
        .astype(str)
    )

    dummies = pd.get_dummies(
        category,
        prefix="vartype",
        dtype=np.int8
    )

    X.drop(
        columns=["ExonicFunc.refGene"],
        inplace=True
    )

    X = pd.concat(
        [
            X.reset_index(drop=True),
            dummies.reset_index(drop=True)
        ],
        axis=1
    )


# Convert remaining features to numeric.
for col in X.columns:
    X[col] = X[col].map(clean_value)
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# Remove features with no usable values.
valid_columns = X.columns[
    X.notna().any(axis=0)
].tolist()

X = X.reindex(columns=valid_columns)

if X.shape[1] == 0:
    raise ValueError(
        "No usable numeric predictor features remain. "
        "Check the columns in the ML-ready CSV."
    )


# ============================================================
# MISSING-VALUE INDICATORS
# ============================================================

print("\nCreating missing-value indicators...")

missing = X.isna().astype(np.int8)

missing.columns = [
    f"{c}_missing"
    for c in missing.columns
]

X = pd.concat(
    [
        X.reset_index(drop=True),
        missing.reset_index(drop=True)
    ],
    axis=1
)

FINAL_FEATURES = X.columns.tolist()


# ============================================================
# IMPUTATION USING 2026 TRAINING DATA ONLY
# ============================================================

training_medians = X.median(axis=0)

X = X.fillna(training_medians)
X = X.fillna(0)
X = X.astype(np.float32)

if X.isna().any().any():
    raise RuntimeError(
        "Missing values remain after imputation."
    )

print("Biological/numeric source features:", len(valid_columns))
print("Final model feature count:", X.shape[1])
print("Training matrix:", X.shape)


# ============================================================
# CLASS IMBALANCE
# ============================================================

positive_count = int((y == 1).sum())
negative_count = int((y == 0).sum())

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance:")
print("Pathogenic:", positive_count)
print("Benign    :", negative_count)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(n_estimators):

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
# SELECT THRESHOLD USING 2026 OOF PREDICTIONS
# ============================================================

print(
    "\nSelecting threshold using five-fold "
    "out-of-fold predictions..."
)

print("Threshold selection uses only 2026 data.")

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

threshold_model = make_model(
    N_ESTIMATORS_FOR_THRESHOLD
)

oof_prob = cross_val_predict(
    threshold_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

oof_auc = roc_auc_score(
    y,
    oof_prob
)

precision, recall, thresholds = precision_recall_curve(
    y,
    oof_prob
)

if len(thresholds) == 0:
    raise RuntimeError(
        "Could not calculate a threshold from OOF predictions."
    )

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)

best_idx = int(
    np.nanargmax(f1_scores)
)

best_threshold = float(
    thresholds[best_idx]
)

oof_f1 = float(
    f1_scores[best_idx]
)

print("2026 OOF ROC-AUC:", round(oof_auc, 4))
print("2026 OOF F1:", round(oof_f1, 4))
print("Selected threshold:", round(best_threshold, 6))


# ============================================================
# TRAIN FINAL MODEL ON ALL 2026 DATA
# ============================================================

print(
    "\nTraining final XGBoost model on all "
    "labelled 2026 CLN2 data..."
)

model = make_model(
    N_ESTIMATORS_FOR_FINAL_MODEL
)

model.fit(
    X,
    y,
    verbose=100
)

print("\nFinal model training complete.")
print("Rows used:", len(X))
print("Features used:", X.shape[1])


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": FINAL_FEATURES,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

feature_importance.to_csv(
    out("CLN2_2026_feature_importance.csv"),
    index=False
)

print("\nTop 20 features:")
print(
    feature_importance.head(20).to_string(index=False)
)

print("Saved CLN2_2026_feature_importance.csv")


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

model_bundle = {
    "model": model,
    "features": FINAL_FEATURES,
    "source_features": valid_columns,
    "training_medians": training_medians.to_dict(),
    "threshold": best_threshold,
    "gene": GENE,
    "training_year": 2026,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nSaved trained model:", MODEL_PATH)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Train_year": [2026],
    "Training_rows": [len(X)],
    "Feature_count": [len(FINAL_FEATURES)],
    "Train_Pathogenic": [positive_count],
    "Train_Benign": [negative_count],
    "Threshold": [best_threshold],
    "Threshold_source": [
        "5-fold out-of-fold predictions on the full 2026 dataset"
    ],
    "OOF_ROC_AUC": [oof_auc],
    "OOF_F1_at_threshold": [oof_f1]
})

summary.to_csv(
    out("CLN2_2026_results_summary.csv"),
    index=False
)

print("Saved CLN2_2026_results_summary.csv")


# ============================================================
# INTERACTIVE INPUT HELPERS
# ============================================================

def get_numeric_input(feature_name):
    """
    Request a numeric feature value.
    Press Enter if unknown; the training median will be used.
    """

    while True:

        raw = input(
            f"  {feature_name} (Enter if unknown): "
        ).strip()

        if raw == "":
            return np.nan

        try:
            value = float(raw)

            if np.isfinite(value):
                return value

            print(
                "  Please enter a finite number or press Enter."
            )

        except ValueError:
            print(
                "  Please enter a number or press Enter to skip."
            )


def build_input_row():
    """
    Collect the same source features used for training.
    Missing values use medians learned from the 2026 dataset.
    """

    raw_values = {}

    print(
        "\nEnter biological annotation scores for this CLN2 variant."
    )

    print(
        "Use scores from the same annotation pipeline "
        "as the training CSV."
    )

    print(
        "Press Enter for any value you do not know.\n"
    )

    for feature in valid_columns:
        raw_values[feature] = get_numeric_input(feature)

    source_row = pd.DataFrame(
        [raw_values],
        columns=valid_columns
    )

    # Apply the same numeric cleaning used during training.
    for col in source_row.columns:

        source_row[col] = source_row[col].map(
            clean_value
        )

        source_row[col] = pd.to_numeric(
            source_row[col],
            errors="coerce"
        )

    source_row.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Create missing indicators in the training feature order.
    missing_row = source_row.isna().astype(np.int8)

    missing_row.columns = [
        f"{c}_missing"
        for c in source_row.columns
    ]

    input_row = pd.concat(
        [source_row, missing_row],
        axis=1
    )

    input_row = input_row.reindex(
        columns=FINAL_FEATURES
    )

    # Apply training medians only.
    input_row = input_row.fillna(
        training_medians
    )

    input_row = input_row.fillna(0)
    input_row = input_row.astype(np.float32)

    return input_row


# ============================================================
# INTERACTIVE CLN2 VARIANT PREDICTION
# ============================================================

def predict_variant():

    print("\n" + "=" * 70)
    print("CLN2 VARIANT PATHOGENICITY PREDICTOR")
    print("Input: biological features for a variant in CLN2")
    print("=" * 70)

    variant_name = input(
        "\nOptional variant identifier (e.g. chr12:position "
        "or HGVS; press Enter to skip): "
    ).strip()

    input_row = build_input_row()

    prob_pathogenic = float(
        model.predict_proba(input_row)[0, 1]
    )

    prob_benign = 1.0 - prob_pathogenic

    predicted_value = int(
        prob_pathogenic >= best_threshold
    )

    prediction = (
        "PATHOGENIC"
        if predicted_value == 1
        else "BENIGN"
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    if variant_name:
        print("Variant identifier       :", variant_name)

    print("Gene                     :", GENE)
    print("Predicted value          :", predicted_value)
    print("Predicted class          :", prediction)

    print(
        f"Pathogenicity probability: "
        f"{prob_pathogenic * 100:.2f}%"
    )

    print(
        f"Benign probability       : "
        f"{prob_benign * 100:.2f}%"
    )

    print(
        f"Decision threshold       : {best_threshold:.6f}"
    )

    print("\nClass mapping: 0 = BENIGN, 1 = PATHOGENIC")

    print(
        "\nImportant: this is a computational variant-level estimate,"
    )

    print(
        "not a clinical diagnosis. Confirm findings with a qualified"
    )

    print(
        "clinical genetics professional and established clinical evidence."
    )

    print("=" * 70)

    # Save each interactive prediction.
    prediction_record = {
        "Gene": GENE,
        "Variant_identifier": variant_name,
        "Predicted_value": predicted_value,
        "Predicted_class": prediction,
        "Pathogenicity_probability": prob_pathogenic,
        "Benign_probability": prob_benign,
        "Threshold": best_threshold
    }

    results_path = out(
        "CLN2_interactive_predictions.csv"
    )

    pd.DataFrame(
        [prediction_record]
    ).to_csv(
        results_path,
        mode="a",
        header=not os.path.exists(results_path),
        index=False
    )

    print(
        "Prediction record appended to:",
        results_path
    )


# ============================================================
# RUN PREDICTOR
# ============================================================

print(
    "\nTraining is complete. Starting interactive prediction."
)

print(
    "The model was trained only on labelled rows from 2026."
)

print(
    "The 2026 OOF results are internal estimates, "
    "not independent test results."
)

while True:

    predict_variant()

    again = input(
        "\nPredict another CLN2 variant? (yes/no): "
    ).strip().lower()

    if again not in {"yes", "y"}:

        print("\nExiting CLN2 predictor.")
        break