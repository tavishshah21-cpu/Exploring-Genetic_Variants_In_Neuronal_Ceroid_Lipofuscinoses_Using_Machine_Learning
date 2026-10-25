
# ============================================================
# CLN3 PIPELINE 3 — INTERACTIVE VARIANT PREDICTOR
#
# Training dataset: 2026 CLN3 ML-ready dataset only
# Models:
#   Base models: Random Forest, Gradient Boosting, Extra Trees
#   Meta-model: Logistic Regression
#
# Threshold: Selected from 5-fold OOF predictions on 2026
# Input: CLN3 biological annotation features
# Output: BENIGN or PATHOGENIC, with probabilities
# ============================================================

import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve

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

SOURCE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"
OUTPUT_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_3_Statistics"

DATA_PATH = os.path.join(
    SOURCE_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "Pipeline3_CLN3_2026_model.pkl"
)

GENE = "CLN3"
RANDOM_STATE = 42
N_SPLITS = 5

RF_N_ESTIMATORS = 400
ET_N_ESTIMATORS = 400
GB_N_ESTIMATORS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================
# LOAD 2026 DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("CLN3 PIPELINE 3 — INTERACTIVE VARIANT PREDICTOR")
print("=" * 70)

print("\nLoading 2026 CLN3 training dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Check SOURCE_DIR and the dataset filename."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

if "label" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'label' column: "
        "0 = benign, 1 = pathogenic."
    )

df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

if df.empty:
    raise ValueError("No labelled rows with labels 0 or 1 were found.")

y = df["label"].reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Benign rows:", int((y == 0).sum()))
print("Pathogenic rows:", int((y == 1).sum()))

if y.nunique() != 2:
    raise ValueError(
        "The 2026 dataset must contain both benign and pathogenic rows."
    )

if int(y.value_counts().min()) < N_SPLITS:
    raise ValueError(
        f"At least {N_SPLITS} examples from each class are required."
    )


# ============================================================
# REMOVE IDENTIFIERS AND TARGET-LEAKAGE COLUMNS
# ============================================================

identifier_columns = [
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
        c for c in identifier_columns + target_leakage_columns
        if c in X.columns
    ],
    inplace=True,
    errors="ignore"
)

del df
gc.collect()


# ============================================================
# CLEAN FEATURE VALUES
# ============================================================

def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in {"", ".", "NA", "N/A", "nan", "None"}:
            return np.nan

        # Use first value if annotations are semicolon-separated.
        if ";" in value:
            value = value.split(";")[0].strip()

            if value in {"", ".", "NA", "N/A", "nan", "None"}:
                return np.nan

    return value


# ============================================================
# HANDLE EXONIC VARIANT TYPE
# Learn categories from 2026 training data only.
# ============================================================

EXONIC_COLUMN = "ExonicFunc.refGene"
EXONIC_CATEGORIES = []

if EXONIC_COLUMN in X.columns:

    print("\nEncoding exonic variant categories...")

    exonic_values = (
        X[EXONIC_COLUMN]
        .fillna("__MISSING__")
        .astype(str)
        .str.strip()
    )

    EXONIC_CATEGORIES = sorted(exonic_values.unique().tolist())

    exonic_dummies = pd.get_dummies(
        pd.Categorical(
            exonic_values,
            categories=EXONIC_CATEGORIES
        ),
        prefix="vartype",
        dtype=np.int8
    )

    exonic_dummies.index = X.index

    X.drop(
        columns=[EXONIC_COLUMN],
        inplace=True
    )

    X = pd.concat(
        [X, exonic_dummies],
        axis=1
    )

    # Save dummy names and category order for interactive prediction.
    EXONIC_DUMMY_COLUMNS = exonic_dummies.columns.tolist()

else:
    EXONIC_DUMMY_COLUMNS = []


# ============================================================
# CONVERT REMAINING FEATURES TO NUMERIC
# ============================================================

print("\nConverting biological features to numeric values...")

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


# ============================================================
# REMOVE FEATURES THAT ARE ENTIRELY MISSING
# Decision uses 2026 training features only.
# ============================================================

valid_columns = X.columns[
    X.notna().any(axis=0)
].tolist()

X = X.reindex(columns=valid_columns)

if X.shape[1] == 0:
    raise ValueError(
        "No usable numeric features remain. "
        "Check the 2026 ML-ready dataset."
    )

print("Usable source features:", len(valid_columns))


# ============================================================
# CREATE MISSING-VALUE INDICATORS
# ============================================================

print("\nCreating missing-value indicators...")

missing = X.isna().astype(np.int8)

missing.columns = [
    f"{column}_missing"
    for column in missing.columns
]

X = pd.concat(
    [
        X.reset_index(drop=True),
        missing.reset_index(drop=True)
    ],
    axis=1
)

FINAL_FEATURES = X.columns.tolist()

X = X.astype(np.float32)

print("Final model feature count:", X.shape[1])
print("Training matrix:", X.shape)


# ============================================================
# PIPELINE 3 STACKING MODEL
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
                n_estimators=GB_N_ESTIMATORS,
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

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        stack_method="predict_proba",
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        ),
        passthrough=False,
        n_jobs=1
    )

    # Imputation is fitted separately inside each training fold.
    # This avoids calculating imputation medians from validation folds.
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        ("stacking", stacking)
    ])


# ============================================================
# SELECT THRESHOLD USING 2026 OOF PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

print("Generating five-fold out-of-fold predictions on 2026...")
print("Only 2026 data is used in this predictor.")

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

threshold_model = make_model()

oof_prob = cross_val_predict(
    threshold_model,
    X,
    y,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

oof_auc = float(
    roc_auc_score(y, oof_prob)
)

precision, recall, thresholds = precision_recall_curve(
    y,
    oof_prob
)

if len(thresholds) == 0:
    raise RuntimeError(
        "Could not calculate a classification threshold."
    )

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)

best_idx = int(np.nanargmax(f1_scores))

best_threshold = float(thresholds[best_idx])
oof_f1 = float(f1_scores[best_idx])

print("2026 OOF ROC-AUC:", round(oof_auc, 4))
print("2026 OOF F1:", round(oof_f1, 4))
print("Selected threshold:", round(best_threshold, 6))

pd.DataFrame({
    "Training_year": [2026],
    "Method": ["Five-fold out-of-fold predictions"],
    "Threshold_metric": ["F1-score"],
    "Threshold": [best_threshold],
    "OOF_F1": [oof_f1],
    "OOF_ROC_AUC": [oof_auc]
}).to_csv(
    out("Pipeline3_CLN3_2026_threshold_selection.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL MODEL ON ALL LABELLED 2026 DATA
# ============================================================

print("\nTraining final Pipeline 3 stacking model on all 2026 data...")

model = make_model()

model.fit(
    X,
    y
)

print("Final model training complete.")
print("Training rows:", len(X))
print("Final features:", X.shape[1])


# ============================================================
# FEATURE IMPORTANCE
# Tree-model importance, not meta-model coefficients.
# ============================================================

print("\nCalculating tree-model feature importance...")

stacking_model = model.named_steps["stacking"]

importance_frames = []

for estimator_name, estimator in stacking_model.named_estimators_.items():

    if hasattr(estimator, "feature_importances_"):

        importance_frames.append(
            pd.DataFrame({
                "Feature": FINAL_FEATURES,
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
        .sort_values(
            "Mean_tree_importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        out("Pipeline3_CLN3_2026_feature_importance.csv"),
        index=False
    )

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

else:
    importance_df = pd.DataFrame()
    print("No tree-model feature importance was available.")


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

model_bundle = {
    "model": model,
    "features": FINAL_FEATURES,
    "source_features": valid_columns,
    "exonic_column": EXONIC_COLUMN,
    "exonic_categories": EXONIC_CATEGORIES,
    "exonic_dummy_columns": EXONIC_DUMMY_COLUMNS,
    "threshold": best_threshold,
    "gene": GENE,
    "training_year": 2026,
    "base_models": [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
    "meta_model": "Logistic Regression",
    "threshold_source": "Five-fold OOF predictions on 2026",
    "threshold_metric": "F1-score",
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nSaved model:", MODEL_PATH)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Pipeline": ["Pipeline 3 Stacking"],
    "Training_year": [2026],
    "Training_rows": [len(X)],
    "Feature_count": [len(FINAL_FEATURES)],
    "Benign_rows": [int((y == 0).sum())],
    "Pathogenic_rows": [int((y == 1).sum())],
    "Base_models": [
        "Random Forest; Gradient Boosting; Extra Trees"
    ],
    "Meta_model": ["Logistic Regression"],
    "Threshold": [best_threshold],
    "Threshold_source": [
        "Five-fold OOF predictions on 2026"
    ],
    "Threshold_metric": ["F1-score"],
    "OOF_ROC_AUC": [oof_auc],
    "OOF_F1": [oof_f1]
})

summary.to_csv(
    out("Pipeline3_CLN3_2026_results_summary.csv"),
    index=False
)

print("Saved training summary.")


# ============================================================
# INTERACTIVE INPUT HELPERS
# ============================================================

def get_numeric_input(feature_name):

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

            print("Enter a finite number or press Enter.")

        except ValueError:
            print("Enter a number or press Enter.")


def build_interactive_row():

    raw_values = {}

    print("\nEnter the biological annotation features.")
    print("Use values from the same annotation pipeline as the dataset.")
    print("Press Enter for any unknown numeric feature.\n")

    # Collect numeric source features, excluding generated dummies.
    numeric_source_features = [
        column for column in valid_columns
        if column not in EXONIC_DUMMY_COLUMNS
    ]

    for feature in numeric_source_features:
        raw_values[feature] = get_numeric_input(feature)

    # Ask for the categorical exonic variant type, if applicable.
    exonic_value = None

    if EXONIC_COLUMN in X.columns or EXONIC_CATEGORIES:

        print("\nKnown exonic categories from the training dataset:")
        for category in EXONIC_CATEGORIES:
            if category != "__MISSING__":
                print("  -", category)

        exonic_value = input(
            "ExonicFunc.refGene (Enter if unknown): "
        ).strip()

    supplied_numeric = [
        value for value in raw_values.values()
        if pd.notna(value) and np.isfinite(float(value))
    ]

    category_supplied = (
        exonic_value is not None
        and exonic_value != ""
        and exonic_value in EXONIC_CATEGORIES
        and exonic_value != "__MISSING__"
    )

    # Do not predict from an entirely empty feature input.
    if len(supplied_numeric) == 0 and not category_supplied:
        return None, 0

    source_row = pd.DataFrame(
        [{column: np.nan for column in valid_columns}],
        columns=valid_columns
    )

    for column, value in raw_values.items():
        if column in source_row.columns:
            source_row.loc[0, column] = value

    # Recreate the exonic one-hot columns learned during training.
    if EXONIC_DUMMY_COLUMNS:

        for dummy_column in EXONIC_DUMMY_COLUMNS:
            source_row.loc[0, dummy_column] = 0

        if exonic_value in EXONIC_CATEGORIES:

            category_index = EXONIC_CATEGORIES.index(exonic_value)

            # Match the get_dummies prefix format.
            dummy_names = pd.get_dummies(
                pd.Categorical(
                    [exonic_value],
                    categories=EXONIC_CATEGORIES
                ),
                prefix="vartype",
                dtype=np.int8
            ).columns.tolist()

            selected_dummy = (
                dummy_names[category_index]
                if category_index < len(dummy_names)
                else None
            )

            if selected_dummy in source_row.columns:
                source_row.loc[0, selected_dummy] = 1

    for column in source_row.columns:

        source_row[column] = source_row[column].map(clean_value)

        source_row[column] = pd.to_numeric(
            source_row[column],
            errors="coerce"
        )

    source_row.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    missing_row = source_row.isna().astype(np.int8)

    missing_row.columns = [
        f"{column}_missing"
        for column in missing_row.columns
    ]

    input_row = pd.concat(
        [source_row, missing_row],
        axis=1
    )

    input_row = input_row.reindex(
        columns=FINAL_FEATURES
    )

    input_row = input_row.astype(np.float32)

    return input_row, len(supplied_numeric) + int(category_supplied)


# ============================================================
# PREDICT A CLN3 VARIANT
# ============================================================

def predict_variant():

    print("\n" + "=" * 70)
    print("CLN3 PIPELINE 3 STACKING — PREDICTION")
    print("=" * 70)

    variant_name = input(
        "Optional variant identifier (HGVS/coordinates; "
        "Enter to skip): "
    ).strip()

    input_row, features_supplied = build_interactive_row()

    if input_row is None:

        print("\n" + "=" * 70)
        print("INSUFFICIENT INPUT")
        print("=" * 70)
        print("No biological feature values were supplied.")
        print("No benign/pathogenic classification has been made.")
        print("Please enter at least one known annotation feature.")
        print("=" * 70)
        return

    prob_pathogenic = float(
        model.predict_proba(input_row)[0, 1]
    )

    prob_benign = 1.0 - prob_pathogenic

    predicted_value = int(
        prob_pathogenic >= best_threshold
    )

    predicted_class = (
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
    print("Predicted class          :", predicted_class)

    print(
        f"Pathogenicity probability: "
        f"{prob_pathogenic * 100:.2f}%"
    )

    print(
        f"Benign probability       : "
        f"{prob_benign * 100:.2f}%"
    )

    print(
        f"Decision threshold       : "
        f"{best_threshold:.6f}"
    )

    print("Features supplied        :", features_supplied)
    print("Class mapping            : 0 = BENIGN, 1 = PATHOGENIC")

    print("\nThis is a computational variant-level estimate,")
    print("not a clinical diagnosis.")
    print("=" * 70)

    # Save prediction record.
    prediction_record = {
        "Gene": GENE,
        "Variant_identifier": variant_name,
        "Predicted_value": predicted_value,
        "Predicted_class": predicted_class,
        "Pathogenicity_probability": prob_pathogenic,
        "Benign_probability": prob_benign,
        "Threshold": best_threshold,
        "Features_supplied": features_supplied,
        "Training_year": 2026
    }

    prediction_path = out(
        "Pipeline3_CLN3_2026_interactive_predictions.csv"
    )

    pd.DataFrame([prediction_record]).to_csv(
        prediction_path,
        mode="a",
        header=not os.path.exists(prediction_path),
        index=False
    )

    print("Prediction saved to:", prediction_path)


# ============================================================
# START INTERACTIVE PREDICTOR
# ============================================================

print("\n" + "=" * 70)
print("CLN3 PIPELINE 3 TRAINING COMPLETE")
print("Training dataset: 2026 only")
print("Base models: Random Forest + Gradient Boosting + Extra Trees")
print("Meta-model: Logistic Regression")
print("Threshold source: 2026 five-fold OOF predictions")
print("=" * 70)

while True:

    predict_variant()

    again = input(
        "\nPredict another CLN3 variant? (yes/no): "
    ).strip().lower()

    if again not in {"yes", "y"}:
        print("\nExiting CLN3 Pipeline 3 predictor.")
        break
