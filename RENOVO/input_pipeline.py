
# ============================================================
# RENOVO: CLN3 PATHOGENICITY PREDICTOR
#
# Training dataset: 2026 ONLY
# Algorithm: XGBoost (RENOVO)
# Threshold: 5-fold OOF predictions from 2026
# Input: Specific variant + biological feature scores
# Output: Pathogenic / Benign + probabilities
# ============================================================

import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "Pipeline_1_Statistics"
)

DATA_PATH = os.path.join(
    DATA_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "RENOVO_CLN3_2026_model.pkl"
)

GENE = "CLN3"
RANDOM_STATE = 42
N_SPLITS = 5


def out(filename):
    return os.path.join(BASE_DIR, filename)


os.makedirs(BASE_DIR, exist_ok=True)


# ============================================================
# 2. BIOLOGICAL FEATURES
# Based on ml_pipeline_input for CLN3
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
# 3. LOAD 2026 DATA ONLY
# ============================================================

print("=" * 65)
print("RENOVO - CLN3 2026 PATHOGENICITY PREDICTOR")
print("Training: 2026 dataset only")
print("=" * 65)

print("Input dataset :", DATA_PATH)
print("Output folder :", BASE_DIR)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n2026 dataset not found:\n{DATA_PATH}\n"
        "Please update DATA_PATH to the correct location."
    )

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

if "label" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'label' column."
    )

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[
    df["label"].isin([0, 1])
].copy()

df["label"] = df["label"].astype(int)

if df.empty:
    raise ValueError(
        "No rows with valid binary labels were found."
    )

if df["label"].nunique() != 2:
    raise ValueError(
        "The 2026 dataset must contain both classes: "
        "Benign (0) and Pathogenic (1)."
    )

print("\n2026 dataset shape:", df.shape)
print("Pathogenic:", int((df["label"] == 1).sum()))
print("Benign    :", int((df["label"] == 0).sum()))


# ============================================================
# 4. SELECT AVAILABLE BIOLOGICAL FEATURES
# ============================================================

available_features = [
    col for col in FEATURES
    if col in df.columns
]

if not available_features:
    raise ValueError(
        "None of the expected biological features were found. "
        "Check the feature names in your 2026 CSV."
    )

print("\nAvailable biological features:")

for feature in available_features:
    print(" -", feature)

print(
    "\nBiological feature count:",
    len(available_features)
)


# ============================================================
# 5. CLEAN FEATURE VALUES
# ============================================================

def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in (
            "",
            ".",
            "NA",
            "N/A",
            "nan",
            "None"
        ):
            return np.nan

        # Some annotation files store multiple scores
        # separated by semicolons.
        if ";" in value:
            value = value.split(";")[0].strip()

            if value in (
                "",
                ".",
                "NA",
                "N/A",
                "nan",
                "None"
            ):
                return np.nan

    return value


X = df[available_features].copy()

y = df["label"].copy()

for col in available_features:

    X[col] = X[col].map(clean_value)

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

# Drop features that have no usable training values.
available_features = [
    col for col in X.columns
    if X[col].notna().any()
]

X = X[available_features].copy()

if X.shape[1] == 0:
    raise ValueError(
        "All selected biological features are missing."
    )

# Drop rows without any usable biological feature.
usable_rows = X.notna().any(axis=1)

X = X.loc[usable_rows].reset_index(drop=True)

y = y.loc[usable_rows].reset_index(drop=True)

if y.nunique() != 2:
    raise ValueError(
        "Both classes must remain after cleaning."
    )

if y.value_counts().min() < N_SPLITS:
    raise ValueError(
        f"Each class needs at least {N_SPLITS} rows "
        "for stratified cross-validation."
    )

print("\nRows retained:", len(X))
print("Features retained:", len(available_features))
print("Pathogenic:", int((y == 1).sum()))
print("Benign:", int((y == 0).sum()))


# ============================================================
# 6. ADD MISSING-VALUE INDICATORS
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

print(
    "Final model feature count:",
    len(FINAL_FEATURES)
)


# ============================================================
# 7. XGBOOST MODEL
# ============================================================

def make_model(scale_pos_weight):

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )


def calculate_class_weight(labels):

    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())

    if positive == 0 or negative == 0:
        raise ValueError(
            "Both classes are required for model training."
        )

    return negative / positive


# ============================================================
# 8. FIVE-FOLD OOF PREDICTIONS ON 2026
# ============================================================

print("\n" + "=" * 65)
print("GENERATING 2026 OUT-OF-FOLD PREDICTIONS")
print("=" * 65)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_prob = np.full(
    len(y),
    np.nan,
    dtype=float
)

fold_auc = []

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    print(f"\nTraining fold {fold}/{N_SPLITS}...")

    X_fit = X.iloc[fit_idx]
    y_fit = y.iloc[fit_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # Class weight comes only from the current training fold.
    fold_weight = calculate_class_weight(y_fit)

    fold_model = make_model(fold_weight)

    fold_model.fit(
        X_fit,
        y_fit
    )

    probabilities = fold_model.predict_proba(
        X_val
    )[:, 1]

    oof_prob[val_idx] = probabilities

    fold_auc_value = roc_auc_score(
        y_val,
        probabilities
    )

    fold_auc.append(fold_auc_value)

    print(
        f"Fold {fold} ROC-AUC:",
        round(fold_auc_value, 4)
    )

if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Some rows did not receive OOF predictions."
    )

mean_cv_auc = float(np.mean(fold_auc))

combined_oof_auc = float(
    roc_auc_score(y, oof_prob)
)

print("\nMean fold ROC-AUC:", round(mean_cv_auc, 4))

print(
    "Combined OOF ROC-AUC:",
    round(combined_oof_auc, 4)
)


# ============================================================
# 9. SELECT THRESHOLD USING 2026 OOF PREDICTIONS
# ============================================================

print("\nSelecting classification threshold...")

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
            + 1e-12
        )
    )

    best_idx = int(np.nanargmax(f1_scores))

    best_threshold = float(
        thresholds[best_idx]
    )

oof_predictions = (
    oof_prob >= best_threshold
).astype(int)

oof_f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0
)

print(
    "Selected threshold:",
    round(best_threshold, 6)
)

print(
    "2026 OOF F1:",
    round(oof_f1, 4)
)

pd.DataFrame({
    "Actual_Label": y,
    "OOF_Pathogenic_Probability": oof_prob,
    "OOF_Predicted_Label": oof_predictions,
    "Threshold": best_threshold
}).to_csv(
    out("RENOVO_CLN3_2026_OOF_predictions.csv"),
    index=False
)


# ============================================================
# 10. TRAIN FINAL MODEL ON ALL 2026 DATA
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL XGBOOST MODEL ON ALL 2026 DATA")
print("=" * 65)

final_weight = calculate_class_weight(y)

model = make_model(final_weight)

model.fit(
    X,
    y
)

print("Final model training completed.")


# ============================================================
# 11. SAVE MODEL AND METADATA
# ============================================================

model_bundle = {
    "gene": GENE,
    "model": model,
    "threshold": best_threshold,
    "biological_features": available_features,
    "final_features": FINAL_FEATURES,
    "training_year": 2026,
    "algorithm": "RENOVO XGBoost",
    "class_weight": final_weight,
    "threshold_method": (
        "F1 maximization using 5-fold OOF predictions "
        "from the 2026 dataset"
    )
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nSaved trained model:")
print(MODEL_PATH)

summary = pd.DataFrame({
    "Gene": [GENE],
    "Algorithm": ["RENOVO XGBoost"],
    "Training_Year": [2026],
    "Training_Rows": [len(X)],
    "Pathogenic_Rows": [int((y == 1).sum())],
    "Benign_Rows": [int((y == 0).sum())],
    "Biological_Features": [len(available_features)],
    "Final_Feature_Count": [len(FINAL_FEATURES)],
    "Mean_Fold_ROC_AUC": [mean_cv_auc],
    "Combined_OOF_ROC_AUC": [combined_oof_auc],
    "OOF_F1": [oof_f1],
    "Threshold": [best_threshold],
    "Threshold_Source": ["2026 OOF predictions"]
})

summary.to_csv(
    out("RENOVO_CLN3_2026_results_summary.csv"),
    index=False
)


# ============================================================
# 12. INTERACTIVE INPUT FUNCTIONS
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
                print("Enter a finite number.")
                continue

            return result

        except ValueError:

            print(
                "Invalid input. Enter a number "
                "or press Enter."
            )


def get_category_input(prompt, allowed_values):

    while True:

        value = input(
            f"{prompt} "
            f"({'/'.join(allowed_values)}, "
            "Enter if unknown): "
        ).strip().upper()

        if value == "":
            return None

        if value in allowed_values:
            return value

        print(
            "Invalid choice. Allowed values: "
            + ", ".join(allowed_values)
        )


# ============================================================
# 13. PREDICT A SPECIFIC CLN3 VARIANT
# ============================================================

def predict_variant():

    print("\n" + "=" * 65)
    print("RENOVO - CLN3 VARIANT PREDICTION")
    print("=" * 65)

    variant_id = input(
        "\nEnter variant identifier "
        "(e.g. VariationID or chr:position:ref:alt): "
    ).strip()

    if not variant_id:

        print(
            "\nPrediction cancelled: no variant was entered."
        )
        return

    print(
        "\nEnter the biological scores for this specific variant."
    )

    print(
        "Use Enter for unknown scores. "
        "Do not guess missing values."
    )

    raw = {}

    # --------------------------------------------------------
    # Conservation scores
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Functional prediction scores
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Variant length
    # --------------------------------------------------------

    print("\n--- Variant information ---")

    if "var_length" in available_features:

        raw["var_length"] = get_float_input(
            "Variant length (ALT length - REF length; SNP = 0)"
        )

    # --------------------------------------------------------
    # AlphaMissense
    # --------------------------------------------------------

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
            "AlphaMissense class: "
            "1=likely pathogenic, "
            "2=ambiguous, 3=likely benign",
            ["1", "2", "3"]
        )

        if "am_likely_pathogenic" in available_features:

            raw["am_likely_pathogenic"] = (
                np.nan
                if am_class is None
                else float(am_class == "1")
            )

        if "am_ambiguous" in available_features:

            raw["am_ambiguous"] = (
                np.nan
                if am_class is None
                else float(am_class == "2")
            )

    # --------------------------------------------------------
    # AlphaFold structural features
    # --------------------------------------------------------

    print("\n--- AlphaFold structural features ---")

    for feature in ["pLDDT", "RSA"]:

        if feature in available_features:

            raw[feature] = get_float_input(feature)

    if any(
        feature in available_features
        for feature in [
            "ss_helix",
            "ss_sheet",
            "ss_coil"
        ]
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
                    np.nan
                    if ss is None
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
            "Burial: 1=buried, "
            "2=intermediate, 3=exposed",
            ["1", "2", "3"]
        )

        for feature, category in [
            ("is_buried", "1"),
            ("is_intermediate", "2"),
            ("is_exposed", "3")
        ]:

            if feature in available_features:

                raw[feature] = (
                    np.nan
                    if burial is None
                    else float(burial == category)
                )

    # --------------------------------------------------------
    # Prevent empty-feature predictions
    # --------------------------------------------------------

    known_count = sum(
        pd.notna(raw.get(feature, np.nan))
        for feature in available_features
    )

    if known_count == 0:

        print(
            "\nPrediction cancelled: all biological features "
            "are missing."
        )

        print(
            "Enter at least one valid feature score."
        )

        return

    # --------------------------------------------------------
    # Construct model input
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [[
            raw.get(feature, np.nan)
            for feature in available_features
        ]],
        columns=available_features
    )

    # Match training-time missing indicators.
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

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_df
    )[0]

    class_order = list(
        model.named_steps["classifier"].classes_
    )

    pathogenic_index = class_order.index(1)
    benign_index = class_order.index(0)

    prob_pathogenic = float(
        probabilities[pathogenic_index]
    )

    prob_benign = float(
        probabilities[benign_index]
    )

    predicted_label = int(
        prob_pathogenic >= best_threshold
    )

    prediction = (
        "PATHOGENIC"
        if predicted_label == 1
        else "BENIGN"
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PREDICTION RESULT")
    print("=" * 65)

    print("Gene:", GENE)

    print("Variant identifier:", variant_id)

    print("Known input features:", known_count)

    print("Algorithm: RENOVO XGBoost")

    print("Predicted class:", prediction)

    print("Predicted label:", predicted_label)

    print(
        "Pathogenicity probability:",
        f"{prob_pathogenic * 100:.2f}%"
    )

    print(
        "Benign probability:",
        f"{prob_benign * 100:.2f}%"
    )

    print(
        "Decision threshold:",
        f"{best_threshold * 100:.2f}%"
    )

    print("\nInterpretation:")

    print(
        "This is a computational prediction based on "
        "the supplied feature values."
    )

    print(
        "The variant identifier is recorded for reference; "
        "it does not automatically retrieve annotations."
    )

    print(
        "Unknown features are imputed by the trained pipeline. "
        "Predictions with many missing features may be unreliable."
    )

    print(
        "This prediction is not a clinical diagnosis."
    )

    print("=" * 65)

    # Save each prediction to a CSV for later review.
    result = pd.DataFrame([{
        "Gene": GENE,
        "Variant_ID": variant_id,
        "Algorithm": "RENOVO XGBoost",
        "Training_Year": 2026,
        "Known_Features": known_count,
        "Predicted_Label": predicted_label,
        "Predicted_Class": prediction,
        "Pathogenic_Probability": prob_pathogenic,
        "Benign_Probability": prob_benign,
        "Threshold": best_threshold
    }])

    result_path = out(
        "RENOVO_CLN3_user_predictions.csv"
    )

    result.to_csv(
        result_path,
        mode="a",
        header=not os.path.exists(result_path),
        index=False
    )

    print("\nPrediction saved to:", result_path)


# ============================================================
# 14. INTERACTIVE PREDICTION LOOP
# ============================================================

print("\n" + "=" * 65)
print("2026 MODEL TRAINING COMPLETE")
print("Starting interactive prediction.")
print("=" * 65)

while True:

    predict_variant()

    again = input(
        "\nPredict another variant? (yes/no): "
    ).strip().lower()

    if again not in ("yes", "y"):

        print("\nExiting RENOVO CLN3 predictor.")

        break

gc.collect()
