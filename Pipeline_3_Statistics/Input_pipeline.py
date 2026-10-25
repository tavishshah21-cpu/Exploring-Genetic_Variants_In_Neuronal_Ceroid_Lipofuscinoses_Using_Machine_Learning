# ============================================================
# PIPELINE 3: CLN3 PATHOGENICITY PREDICTOR
#
# TRAINING DATA: CLN3 2026 ONLY
#
# ALGORITHM:
#   1. Random Forest
#   2. Gradient Boosting
#   3. Extra Trees
#   4. Logistic Regression Meta-Model
#
# THRESHOLD:
#   Selected using 5-fold out-of-fold predictions on 2026.
#
# INPUT:
#   Interactive biological feature scores for a CLN3 variant.
#
# OUTPUT:
#   Pathogenic / Benign classification
#   Pathogenicity probability
#   Benign probability
#
# IMPORTANT:
#   This is a variant-level predictor, not a gene-level
#   diagnostic tool. Enter the scores for a specific variant.
# ============================================================

import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics"

OUTPUT_DIR = r"D:\tavish_shah\PES\capstone\Pipeline_3_Statistics"

DATA_PATH = os.path.join(
    BASE_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "Pipeline3_CLN3_2026_model.pkl"
)

GENE = "CLN3"
RANDOM_STATE = 42
N_SPLITS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================
# BIOLOGICAL FEATURES
# Based on ml_pipeline_input(6).py
# ============================================================

FEATURES = [
    "LINSIGHT",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SiPhy_29way_logOdds",
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "MutationTaster_score",
    "MetaRNN_score",
    "VEST4_score",
    "REVEL_score",
    "CADD_indel_raw",
    "CADD_indel_phred",
    "var_length",
    "am_pathogenicity",
    "am_likely_pathogenic",
    "am_ambiguous",
    "pLDDT",
    "RSA",
    "ss_helix",
    "ss_sheet",
    "ss_coil",
    "is_buried",
    "is_exposed",
    "is_intermediate"
]


# ============================================================
# LOAD 2026 DATASET ONLY
# ============================================================

print("=" * 65)
print("PIPELINE 3: CLN3 2026 STACKING PREDICTOR")
print("=" * 65)

print("\nLoading 2026 dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n2026 dataset not found:\n{DATA_PATH}\n"
        "Update DATA_PATH to the correct CSV location."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

if "label" not in df.columns:
    raise ValueError(
        "The 2026 dataset must contain a 'label' column."
    )

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

if df.empty:
    raise ValueError(
        "No valid benign/pathogenic labels were found."
    )

if df["label"].nunique() != 2:
    raise ValueError(
        "The 2026 dataset must contain both benign (0) "
        "and pathogenic (1) variants."
    )

print("Dataset shape:", df.shape)
print("Pathogenic:", int((df["label"] == 1).sum()))
print("Benign:", int((df["label"] == 0).sum()))


# ============================================================
# PREPARE FEATURES
# ============================================================

# Use the curated biological features from the input pipeline.
# Do not include identifiers, annotations, or the target label.

available_features = [
    col for col in FEATURES
    if col in df.columns
]

if not available_features:
    raise ValueError(
        "None of the expected biological features were found "
        "in the 2026 dataset. Check the CSV column names."
    )

print("\nAvailable biological features:")
for feature in available_features:
    print(" -", feature)

print("\nNumber of biological features:", len(available_features))

X = df[available_features].copy()
y = df["label"].copy()


# ============================================================
# CLEAN VALUES
# ============================================================

def clean_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value in ("", ".", "NA", "N/A", "nan", "None"):
            return np.nan

        if ";" in value:
            value = value.split(";")[0].strip()

            if value in ("", ".", "NA", "N/A", "nan", "None"):
                return np.nan

    return value


for col in X.columns:
    X[col] = X[col].map(clean_value)
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove features with no usable values in the 2026 dataset.
valid_features = [
    col for col in X.columns
    if X[col].notna().any()
]

X = X[valid_features].copy()
available_features = valid_features

if X.shape[1] == 0:
    raise ValueError(
        "All selected biological features are missing."
    )

# Remove rows that have no biological feature information.
usable_rows = X.notna().any(axis=1)

X = X.loc[usable_rows].reset_index(drop=True)
y = y.loc[usable_rows].reset_index(drop=True)

if y.nunique() != 2:
    raise ValueError(
        "Both classes must remain after feature cleaning."
    )

print("\nRows retained:", len(X))
print("Features retained:", len(available_features))


# ============================================================
# ADD MISSING-VALUE INDICATORS
# ============================================================

def add_missing_indicators(data):
    missing = data.isna().astype(np.int8)

    missing.columns = [
        str(col) + "_missing"
        for col in data.columns
    ]

    return pd.concat(
        [data, missing],
        axis=1
    )


X = add_missing_indicators(X)

FINAL_FEATURES = X.columns.tolist()

print("Final model feature count:", len(FINAL_FEATURES))


# ============================================================
# VALIDATE DATA FOR CROSS-VALIDATION
# ============================================================

class_counts = y.value_counts()

if class_counts.min() < N_SPLITS:
    raise ValueError(
        f"Each class needs at least {N_SPLITS} rows "
        "for stratified cross-validation."
    )


# ============================================================
# BASE MODELS
# ============================================================

def make_stacking_model():

    # Every model has its own imputer.
    # Imputation is fitted only on the training fold.

    rf = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    gb = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False
            )
        ),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        )
    ])

    et = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False
            )
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

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
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=False,
        n_jobs=1
    )


# ============================================================
# SELECT THRESHOLD USING 2026 OOF PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("GENERATING 2026 OUT-OF-FOLD PREDICTIONS")
print("=" * 65)

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

threshold_model = make_stacking_model()

oof_prob = cross_val_predict(
    threshold_model,
    X,
    y,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

if not np.isfinite(oof_prob).all():
    raise ValueError(
        "Invalid probabilities generated during cross-validation."
    )

oof_roc_auc = roc_auc_score(y, oof_prob)

precision, recall, thresholds = precision_recall_curve(
    y,
    oof_prob
)

if len(thresholds) == 0:
    best_threshold = 0.5
else:
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-8
        )
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

oof_predictions = (
    oof_prob >= best_threshold
).astype(int)

oof_f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0
)

print("\n2026 OOF ROC-AUC:", round(oof_roc_auc, 4))
print("2026 OOF F1:", round(oof_f1, 4))
print("Selected threshold:", round(best_threshold, 4))

pd.DataFrame({
    "Actual_Label": y,
    "OOF_Pathogenicity_Probability": oof_prob,
    "OOF_Predicted_Label": oof_predictions
}).to_csv(
    out("Pipeline3_CLN3_2026_OOF_predictions.csv"),
    index=False
)


# ============================================================
# TRAIN FINAL MODEL ON ALL USABLE 2026 DATA
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL ON ALL 2026 DATA")
print("=" * 65)

model = make_stacking_model()

model.fit(X, y)

print("\nModel training completed.")


# ============================================================
# SAVE MODEL AND PREDICTOR CONFIGURATION
# ============================================================

model_bundle = {
    "gene": GENE,
    "model": model,
    "threshold": best_threshold,
    "biological_features": available_features,
    "final_features": FINAL_FEATURES,
    "training_year": 2026,
    "algorithm": (
        "Random Forest + Gradient Boosting + Extra Trees "
        "with Logistic Regression meta-model"
    ),
    "threshold_method": (
        "F1 maximization using 5-fold OOF predictions "
        "from the 2026 dataset"
    )
}

joblib.dump(model_bundle, MODEL_PATH)

print("\nSaved model:", MODEL_PATH)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Training_Year": [2026],
    "Algorithm": ["Pipeline 3 Stacking"],
    "Base_Models": [
        "Random Forest; Gradient Boosting; Extra Trees"
    ],
    "Meta_Model": ["Logistic Regression"],
    "Training_Rows": [len(X)],
    "Pathogenic_Rows": [int((y == 1).sum())],
    "Benign_Rows": [int((y == 0).sum())],
    "Biological_Features": [len(available_features)],
    "Final_Feature_Count": [len(FINAL_FEATURES)],
    "OOF_ROC_AUC": [round(oof_roc_auc, 4)],
    "OOF_F1": [round(oof_f1, 4)],
    "Threshold": [round(best_threshold, 4)],
    "Threshold_Source": ["2026 OOF predictions"]
})

summary.to_csv(
    out("Pipeline3_CLN3_2026_results_summary.csv"),
    index=False
)


# ============================================================
# INTERACTIVE INPUT HELPERS
# ============================================================

def get_float_input(prompt):

    while True:
        value = input(
            f"{prompt} (Enter if unknown): "
        ).strip()

        if value == "":
            return np.nan

        try:
            result = float(value)

            if not np.isfinite(result):
                print("Please enter a finite number.")
                continue

            return result

        except ValueError:
            print("Please enter a valid number or press Enter.")


def get_category_input(prompt, allowed_values):

    while True:
        value = input(
            f"{prompt} ({'/'.join(allowed_values)}, "
            "Enter if unknown): "
        ).strip().upper()

        if value == "":
            return None

        if value in allowed_values:
            return value

        print(
            "Invalid choice. Use "
            + ", ".join(allowed_values)
            + " or press Enter."
        )


# ============================================================
# INTERACTIVE VARIANT PREDICTOR
# ============================================================

def predict_variant():

    print("\n" + "=" * 65)
    print("CLN3 VARIANT PATHOGENICITY PREDICTOR")
    print("=" * 65)

    variant_id = input(
        "\nEnter a variant identifier or description "
        "(e.g. VariationID or chr:position:ref:alt): "
    ).strip()

    if not variant_id:
        print(
            "\nNo variant entered. Prediction cancelled."
        )
        return

    print(
        "\nEnter the biological scores for THIS variant."
    )
    print(
        "Use Enter for genuinely unknown values."
    )
    print(
        "Do not enter the gene name alone as a variant."
    )

    raw = {}

    # Conservation features
    print("\n--- Conservation scores ---")

    for feature in [
        "LINSIGHT",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "SiPhy_29way_logOdds"
    ]:
        if feature in available_features:
            raw[feature] = get_float_input(feature)

    # Functional prediction features
    print("\n--- Functional prediction scores ---")

    for feature in [
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "Polyphen2_HVAR_score",
        "MutationTaster_score",
        "MetaRNN_score",
        "VEST4_score",
        "REVEL_score",
        "CADD_indel_raw",
        "CADD_indel_phred"
    ]:
        if feature in available_features:
            raw[feature] = get_float_input(feature)

    # Variant information
    print("\n--- Variant information ---")

    if "var_length" in available_features:
        raw["var_length"] = get_float_input(
            "Variant length (ALT length - REF length; 0 for SNP)"
        )

    # AlphaMissense features
    print("\n--- AlphaMissense scores ---")

    if "am_pathogenicity" in available_features:
        raw["am_pathogenicity"] = get_float_input(
            "AlphaMissense pathogenicity score"
        )

    if (
        "am_likely_pathogenic" in available_features
        or "am_ambiguous" in available_features
    ):
        am_class = get_category_input(
            "AlphaMissense class: 1=likely pathogenic, "
            "2=ambiguous, 3=likely benign",
            ["1", "2", "3"]
        )

        if "am_likely_pathogenic" in available_features:
            raw["am_likely_pathogenic"] = (
                np.nan if am_class is None
                else float(am_class == "1")
            )

        if "am_ambiguous" in available_features:
            raw["am_ambiguous"] = (
                np.nan if am_class is None
                else float(am_class == "2")
            )

    # AlphaFold structural features
    print("\n--- AlphaFold structural features ---")

    for feature in ["pLDDT", "RSA"]:
        if feature in available_features:
            raw[feature] = get_float_input(feature)

    if any(
        feature in available_features
        for feature in ["ss_helix", "ss_sheet", "ss_coil"]
    ):
        ss = get_category_input(
            "Secondary structure: H=helix, E=sheet, C=coil",
            ["H", "E", "C"]
        )

        for feature, category in [
            ("ss_helix", "H"),
            ("ss_sheet", "E"),
            ("ss_coil", "C")
        ]:
            if feature in available_features:
                raw[feature] = (
                    np.nan if ss is None
                    else float(ss == category)
                )

    if any(
        feature in available_features
        for feature in [
            "is_buried",
            "is_exposed",
            "is_intermediate"
        ]
    ):
        burial = get_category_input(
            "Burial: 1=buried, 2=intermediate, 3=exposed",
            ["1", "2", "3"]
        )

        for feature, category in [
            ("is_buried", "1"),
            ("is_intermediate", "2"),
            ("is_exposed", "3")
        ]:
            if feature in available_features:
                raw[feature] = (
                    np.nan if burial is None
                    else float(burial == category)
                )

    # Prevent accidental classification with no information.
    known_count = sum(
        pd.notna(raw.get(feature, np.nan))
        for feature in available_features
    )

    if known_count == 0:
        print(
            "\nPrediction cancelled: all biological features "
            "are missing. Enter at least one valid score."
        )
        return

    # Build input in the exact training feature order.
    input_df = pd.DataFrame(
        [[raw.get(feature, np.nan)
          for feature in available_features]],
        columns=available_features
    )

    # Missing indicators must match the training schema.
    missing_df = input_df.isna().astype(np.int8)
    missing_df.columns = [
        feature + "_missing"
        for feature in available_features
    ]

    input_df = pd.concat(
        [input_df, missing_df],
        axis=1
    )

    input_df = input_df.reindex(
        columns=FINAL_FEATURES
    )

    # Predict probabilities.
    prob_pathogenic = float(
        model.predict_proba(input_df)[0][1]
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

    # Display result.
    print("\n" + "=" * 65)
    print("PREDICTION RESULT")
    print("=" * 65)

    print("Gene                    :", GENE)
    print("Variant identifier      :", variant_id)
    print("Known input features    :", known_count)
    print("Predicted class         :", prediction)
    print("Predicted value         :", predicted_value)
    print(
        "Pathogenicity probability:",
        f"{prob_pathogenic * 100:.2f}%"
    )
    print(
        "Benign probability      :",
        f"{prob_benign * 100:.2f}%"
    )
    print(
        "Decision threshold      :",
        f"{best_threshold * 100:.2f}%"
    )

    print("\nImportant interpretation:")
    print(
        "This result is a computational prediction for the "
        "entered feature values. It is not a clinical diagnosis."
    )
    print(
        "Missing values are imputed using the trained model; "
        "many missing scores can make the result unreliable."
    )
    print(
        "The variant identifier is recorded for reference; "
        "it does not automatically retrieve or verify variant scores."
    )
    print("=" * 65)


# ============================================================
# RUN INTERACTIVE PREDICTOR
# ============================================================

print("\n" + "=" * 65)
print("2026 TRAINING COMPLETE")
print("Starting interactive variant prediction.")
print("=" * 65)

while True:

    predict_variant()

    again = input(
        "\nPredict another variant? (yes/no): "
    ).strip().lower()

    if again not in ("yes", "y"):
        print("\nExiting CLN3 predictor.")
        break

gc.collect()