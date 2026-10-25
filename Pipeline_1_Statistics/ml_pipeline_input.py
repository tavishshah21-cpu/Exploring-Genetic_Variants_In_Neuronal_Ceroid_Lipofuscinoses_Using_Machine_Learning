# ==========================================
# CLN3 VARIANT PATHOGENICITY PREDICTOR
# Train on the full 2026 dataset
# Predict user-input variants
# ==========================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# ==========================================
# CONFIG
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "CLN3_2026_trained_model.pkl"
)

GENE = "CLN3"
RANDOM_STATE = 42

USE_SHAP = True


def out(filename):
    return os.path.join(BASE_DIR, filename)


# ==========================================
# FEATURE LIST
# ==========================================

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
    "is_intermediate",
]


# ==========================================
# LOAD 2026 DATA
# ==========================================

print("\nLoading 2026 dataset for training...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

if "label" not in df.columns:
    raise ValueError(
        "The training dataset must contain a 'label' column."
    )

# Keep only rows with valid benign/pathogenic labels.
df = df[df["label"].isin([0, 1])].copy()

print("2026 dataset:", df.shape)
print("Pathogenic:", (df["label"] == 1).sum())
print("Benign    :", (df["label"] == 0).sum())

if df.empty:
    raise ValueError(
        "No rows with valid labels 0 or 1 were found."
    )


# ==========================================
# CLEAN — DROP IDENTIFIERS
# ==========================================

print("\nCleaning dataset...")

drop_cols = [
    "Chr",
    "Start",
    "End",
    "Ref",
    "Alt",
    "GeneSymbol"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(
            columns=[col],
            inplace=True
        )

drop_anno = [
    "Func.refGene",
    "AAChange.refGene",
    "avsnp150"
]

for col in drop_anno:
    if col in df.columns:
        df.drop(
            columns=[col],
            inplace=True
        )


# ==========================================
# SEPARATE LABELS AND FEATURES
# ==========================================

y = df["label"].astype(int)

X = df.drop(
    columns=["label"]
).copy()


# ==========================================
# CLEAN VALUES
# ==========================================

print("\nConverting features to numeric values...")


def clean_value(value):

    if isinstance(value, str):
        if ";" in value:
            return value.split(";")[0]

    return value


X = X.replace(".", np.nan)

X = X.map(clean_value).apply(
    pd.to_numeric,
    errors="coerce"
)

print("Biological features:", X.shape[1])


# ==========================================
# MISSING INDICATORS
# ==========================================

print("\nCreating missing-value indicators...")


def add_missing_indicators(data):

    missing = data.isna().astype(np.int8)

    missing.columns = [
        col + "_missing"
        for col in data.columns
    ]

    return pd.concat(
        [data, missing],
        axis=1
    )


X = add_missing_indicators(X)

print("Final feature count:", X.shape[1])

# These are the exact columns expected by the trained model.
FINAL_FEATURES = X.columns.tolist()


# ==========================================
# CLASS BALANCE — FULL 2026 DATASET
# ==========================================

print("\nHandling class imbalance...")

pos = int((y == 1).sum())
neg = int((y == 0).sum())

if pos == 0 or neg == 0:
    raise ValueError(
        "The training dataset must contain both benign (0) "
        "and pathogenic (1) variants."
    )

scale_pos_weight = neg / pos

if scale_pos_weight < 1:
    print(
        "Warning: Pathogenic variants are the majority class."
    )

    scale_pos_weight = pos / neg

print("Pathogenic:", pos)
print("Benign    :", neg)

print(
    "scale_pos_weight:",
    round(scale_pos_weight, 2)
)


# ==========================================
# SELECT THRESHOLD USING OOF PREDICTIONS
# ==========================================
#
# Five-fold out-of-fold predictions are used only
# to select the classification threshold.
#
# No permanent validation split is created.
# The final model will train on all labelled 2026 rows.
# ==========================================

print(
    "\nSelecting threshold using five-fold "
    "out-of-fold predictions..."
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

threshold_model = XGBClassifier(
    n_estimators=300,
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

oof_prob = cross_val_predict(
    threshold_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]


# ==========================================
# FIND THE BEST F1 THRESHOLD
# ==========================================

precision_train, recall_train, thresholds_train = (
    precision_recall_curve(
        y,
        oof_prob
    )
)

f1_scores_train = (
    2 * precision_train * recall_train
    / (
        precision_train
        + recall_train
        + 1e-8
    )
)

# precision_recall_curve returns one more precision/recall
# value than threshold values.
best_idx = int(
    np.argmax(f1_scores_train[:-1])
)

best_threshold = float(
    thresholds_train[best_idx]
)

oof_roc_auc = roc_auc_score(
    y,
    oof_prob
)

oof_f1 = float(
    f1_scores_train[best_idx]
)

print(
    "Threshold source: five-fold out-of-fold "
    "predictions on the full 2026 dataset"
)

print(
    "OOF ROC-AUC:",
    round(oof_roc_auc, 4)
)

print(
    "OOF F1 at selected threshold:",
    round(oof_f1, 4)
)

print(
    "Selected threshold:",
    round(best_threshold, 4)
)


# ==========================================
# TRAIN FINAL MODEL ON ALL 2026 DATA
# ==========================================

print(
    "\nTraining final XGBoost model on the "
    "full 2026 CLN3 dataset..."
)

model = XGBClassifier(
    n_estimators=1200,
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

# Train on every labelled row.
# No holdout validation set or early stopping is used.
model.fit(
    X,
    y,
    verbose=100
)

print("\nFinal model training complete.")
print("Rows used for final training:", len(X))
print("Features used:", X.shape[1])


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\nFeature importance:")

feat = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feat = feat.sort_values(
    "Importance",
    ascending=False
)

print(feat.head(20))

feat.to_csv(
    out("CLN3_2026_feature_importance.csv"),
    index=False
)

print(
    "Saved CLN3_2026_feature_importance.csv"
)


# ==========================================
# SHAP
# ==========================================

if USE_SHAP:

    try:

        import shap

        print("\nRunning SHAP...")

        shap_sample = (
            X
            if len(X) <= 500
            else X.sample(
                500,
                random_state=RANDOM_STATE
            )
        )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            shap_sample
        )

        shap.summary_plot(
            shap_values,
            shap_sample,
            show=False
        )

        plt.title(
            "CLN3 SHAP Summary — 2026 Model"
        )

        plt.tight_layout()

        plt.savefig(
            out("CLN3_2026_shap_summary.png"),
            dpi=300
        )

        plt.close()

        print(
            "Saved CLN3_2026_shap_summary.png"
        )

    except Exception as e:

        print("SHAP failed:", e)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": FINAL_FEATURES,
        "threshold": best_threshold,
        "scale_pos_weight": scale_pos_weight
    },
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)


# ==========================================
# SAVE PREDICTOR SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Gene": [GENE],
    "Train_year": [2026],
    "Training_rows": [len(X)],
    "Feature_count": [len(FINAL_FEATURES)],
    "Train_Pathogenic": [pos],
    "Train_Benign": [neg],
    "Threshold": [round(best_threshold, 4)],
    "Threshold_source": [
        "5-fold out-of-fold predictions on full 2026 dataset"
    ],
    "OOF_ROC_AUC": [round(oof_roc_auc, 4)],
    "OOF_F1_at_threshold": [round(oof_f1, 4)]
})

summary.to_csv(
    out("CLN3_2026_results_summary.csv"),
    index=False
)

print(
    "Saved CLN3_2026_results_summary.csv"
)

print("\nCLN3 predictor training complete.")

print(
    "The saved model is ready for interactive "
    "variant prediction."
)


# ==========================================
# VARIANT INPUT HELPERS
# ==========================================

def get_float_input(prompt, feature_name):

    """
    Get a numeric input.
    Press Enter to leave a score unknown.
    """

    while True:

        val = input(
            f"  {prompt} (press Enter if unknown): "
        ).strip()

        if val == "":
            return np.nan

        try:
            return float(val)

        except ValueError:
            print(
                "  Please enter a number or press Enter to skip."
            )


# ==========================================
# INTERACTIVE VARIANT PREDICTION
# ==========================================

def predict_variant():

    print("\n" + "=" * 60)
    print("   CLN3 VARIANT PATHOGENICITY PREDICTOR")
    print("   Batten Disease (NCL3) Risk Assessment")
    print("=" * 60)

    print(
        "\nEnter the biological scores for your CLN3 variant."
    )

    print(
        "Press Enter to skip any score you don't have."
    )

    print(
        "Scores can be obtained from ClinVar, "
        "ANNOVAR annotations, or dbNSFP.\n"
    )

    # ------------------------------------------
    # CONSERVATION SCORES
    # ------------------------------------------

    print("── Conservation Scores ─────────────────────")

    LINSIGHT = get_float_input(
        "LINSIGHT score (0-1)",
        "LINSIGHT"
    )

    GERP = get_float_input(
        "GERP++ RS score (typically -12 to 6)",
        "GERP++_RS"
    )

    phyloP = get_float_input(
        "phyloP100way vertebrate score",
        "phyloP100way_vertebrate"
    )

    phastCons = get_float_input(
        "phastCons100way vertebrate score (0-1)",
        "phastCons100way_vertebrate"
    )

    SiPhy = get_float_input(
        "SiPhy 29way logOdds score",
        "SiPhy_29way_logOdds"
    )

    # ------------------------------------------
    # FUNCTIONAL PREDICTION SCORES
    # ------------------------------------------

    print("\n── Functional Prediction Scores ────────────")

    SIFT = get_float_input(
        "SIFT score (0-1, lower = more damaging)",
        "SIFT_score"
    )

    Poly_HDIV = get_float_input(
        "Polyphen2 HDIV score (0-1)",
        "Polyphen2_HDIV_score"
    )

    Poly_HVAR = get_float_input(
        "Polyphen2 HVAR score (0-1)",
        "Polyphen2_HVAR_score"
    )

    MutTaster = get_float_input(
        "MutationTaster score (0-1)",
        "MutationTaster_score"
    )

    MetaRNN = get_float_input(
        "MetaRNN score (0-1)",
        "MetaRNN_score"
    )

    VEST4 = get_float_input(
        "VEST4 score (0-1)",
        "VEST4_score"
    )

    REVEL = get_float_input(
        "REVEL score (0-1)",
        "REVEL_score"
    )

    # ------------------------------------------
    # CADD-INDEL SCORES
    # ------------------------------------------

    print(
        "\n── CADD-Indel Scores "
        "(only for insertions/deletions) ─────────"
    )

    CADD_raw = get_float_input(
        "CADD indel raw score",
        "CADD_indel_raw"
    )

    CADD_phred = get_float_input(
        "CADD indel phred score",
        "CADD_indel_phred"
    )

    # ------------------------------------------
    # VARIANT INFORMATION
    # ------------------------------------------

    print("\n── Variant Information ─────────────────────")

    var_length = get_float_input(
        "Variant length (alt length - ref length, 0 for SNP)",
        "var_length"
    )

    # ------------------------------------------
    # ALPHAMISSENSE SCORES
    # ------------------------------------------

    print(
        "\n── AlphaMissense Scores "
        "(missense variants only) ─────────────────"
    )

    am_path = get_float_input(
        "AlphaMissense pathogenicity score (0-1)",
        "am_pathogenicity"
    )

    print("  AlphaMissense class:")
    print("    1 = likely pathogenic")
    print("    2 = ambiguous")
    print("    3 = likely benign")
    print("    Press Enter if unknown")

    am_class_input = input(
        "  Your choice: "
    ).strip()

    am_likely_path = (
        1 if am_class_input == "1" else 0
    )

    am_ambiguous = (
        1 if am_class_input == "2" else 0
    )

    # ------------------------------------------
    # ALPHAFOLD STRUCTURAL FEATURES
    # ------------------------------------------

    print("\n── AlphaFold Structural Features ───────────")

    pLDDT = get_float_input(
        "pLDDT score (0-100, AlphaFold confidence)",
        "pLDDT"
    )

    RSA = get_float_input(
        "RSA — Relative Solvent Accessibility (0-1)",
        "RSA"
    )

    print("  Secondary structure:")
    print("    H = Helix")
    print("    E = Sheet")
    print("    C = Coil/Loop")
    print("    Press Enter if unknown")

    ss = input(
        "  Your choice (H/E/C): "
    ).strip().upper()

    ss_helix = 1 if ss == "H" else 0
    ss_sheet = 1 if ss == "E" else 0
    ss_coil = 1 if ss == "C" else 0

    print("  Burial status:")
    print("    1 = buried (RSA < 0.2)")
    print("    2 = intermediate (0.2-0.5)")
    print("    3 = exposed (RSA > 0.5)")
    print("    Press Enter if unknown")

    burial_input = input(
        "  Your choice: "
    ).strip()

    is_buried = 1 if burial_input == "1" else 0
    is_intermediate = 1 if burial_input == "2" else 0
    is_exposed = 1 if burial_input == "3" else 0

    # ------------------------------------------
    # BUILD INPUT FEATURE VECTOR
    # ------------------------------------------

    raw_features = {
        "LINSIGHT": LINSIGHT,
        "GERP++_RS": GERP,
        "phyloP100way_vertebrate": phyloP,
        "phastCons100way_vertebrate": phastCons,
        "SiPhy_29way_logOdds": SiPhy,
        "SIFT_score": SIFT,
        "Polyphen2_HDIV_score": Poly_HDIV,
        "Polyphen2_HVAR_score": Poly_HVAR,
        "MutationTaster_score": MutTaster,
        "MetaRNN_score": MetaRNN,
        "VEST4_score": VEST4,
        "REVEL_score": REVEL,
        "CADD_indel_raw": CADD_raw,
        "CADD_indel_phred": CADD_phred,
        "var_length": var_length,
        "am_pathogenicity": am_path,
        "am_likely_pathogenic": float(am_likely_path),
        "am_ambiguous": float(am_ambiguous),
        "pLDDT": pLDDT,
        "RSA": RSA,
        "ss_helix": float(ss_helix),
        "ss_sheet": float(ss_sheet),
        "ss_coil": float(ss_coil),
        "is_buried": float(is_buried),
        "is_exposed": float(is_exposed),
        "is_intermediate": float(is_intermediate)
    }

    input_df = pd.DataFrame([raw_features])

    # Keep the original input values missing where unknown.
    missing_df = input_df.isna().astype(np.int8)

    missing_df.columns = [
        col + "_missing"
        for col in input_df.columns
    ]

    input_df = pd.concat(
        [input_df, missing_df],
        axis=1
    )

    # Add any expected model features not supplied by the
    # interactive input and align to the training columns.
    for col in FINAL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FINAL_FEATURES]

    # ------------------------------------------
    # PREDICT
    # ------------------------------------------

    prob_pathogenic = float(
        model.predict_proba(input_df)[0][1]
    )

    prob_benign = 1 - prob_pathogenic

    # 1 = pathogenic, 0 = benign
    predicted_value = (
        1 if prob_pathogenic >= best_threshold else 0
    )

    prediction = (
        "PATHOGENIC"
        if predicted_value == 1
        else "BENIGN"
    )

    # ------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("   PREDICTION RESULT")
    print("=" * 60)

    print(
        f"\n  Predicted value        : {predicted_value}"
    )

    print(
        f"  Predicted class        : {prediction}"
    )

    print(
        f"  Pathogenicity score    : "
        f"{round(prob_pathogenic * 100, 2)}%"
    )

    print(
        f"  Benign score           : "
        f"{round(prob_benign * 100, 2)}%"
    )

    print(
        f"  Decision threshold     : "
        f"{round(best_threshold * 100, 2)}%"
    )

    print("\n  Class mapping:")
    print("    0 = BENIGN / NON-PATHOGENIC")
    print("    1 = PATHOGENIC")

    # ------------------------------------------
    # INTERPRETATION
    # ------------------------------------------

    print("\n── Interpretation ──────────────────────────")

    if prob_pathogenic >= 0.85:

        print(
            "  HIGH risk — strongly suggests a "
            "pathogenic variant."
        )

        print(
            "  Further clinical assessment is recommended."
        )

    elif prob_pathogenic >= best_threshold:

        print(
            "  MODERATE-HIGH risk — classified as pathogenic "
            "by the model's selected threshold."
        )

        print(
            "  Consider further clinical evaluation."
        )

    elif prob_pathogenic >= 0.3:

        print(
            "  UNCERTAIN — borderline prediction."
        )

        print(
            "  The model's prediction should be interpreted "
            "cautiously."
        )

    else:

        print(
            "  LOW risk — classified as benign by the model."
        )

        print(
            "  This is a computational prediction, not a "
            "clinical diagnosis."
        )

    # ------------------------------------------
    # DISCLAIMER
    # ------------------------------------------

    print("\n── Important Disclaimer ───────────────────")

    print(
        "  This prediction is a computational estimate only."
    )

    print(
        "  It should NOT replace clinical genetic counselling."
    )

    print(
        "  Consult a qualified clinical geneticist for "
        "variant interpretation and disease diagnosis."
    )

    print("=" * 60)

    return (
        predicted_value,
        prob_pathogenic,
        prediction
    )


# ==========================================
# RUN PREDICTOR
# ==========================================

print("\n" + "=" * 60)

print(
    "  Model training complete. Starting prediction tool."
)

print("=" * 60)

while True:

    predicted_value, prob, pred = predict_variant()

    print("\n")

    again = input(
        "Would you like to predict another variant? (yes/no): "
    ).strip().lower()

    if again not in ["yes", "y"]:

        print(
            "\nThank you. Exiting CLN3 predictor."
        )

        break