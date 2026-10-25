
# ============================================================
# CLN2 VUS RANKING USING XGBOOST
# Train on labelled 2026 ClinVar data.
# Rank VUS from the 2026 ClinVar snapshot.
#
# Primary ranking criterion:
#   XGBoost predicted pathogenicity probability (descending)
#
# No XGBRanker and no temporal validation.
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = r"D:\tavish_shah\PES\capstone"
DATA_DIR = os.path.join(ROOT_DIR, "Pipeline_1_cln2")
RANKING_DIR = os.path.join(ROOT_DIR, "cln2_Ranking")

TRAIN_PATH = os.path.join(
    DATA_DIR, "CLN2_2026_CLN2_ML_READY.csv"
)
VUS_PATH = os.path.join(
    RANKING_DIR, "CLN2_VUS_CLN2_ML_READY.csv"
)

os.makedirs(RANKING_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    RANKING_DIR, "CLN2_XGBClassifier_2026.pkl"
)
RESULTS_CSV = os.path.join(
    RANKING_DIR, "CLN2_VUS_RANKED_RESULTS.csv"
)
RESULTS_XLSX = os.path.join(
    RANKING_DIR, "CLN2_VUS_RANKED_RESULTS.xlsx"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.50

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
    "contact_count",
]

IDENTIFIERS = [
    "VariationID",
    "GeneSymbol",
    "ClinicalSignificance",
    "Chromosome",
    "Start",
    "ReferenceAllele",
    "AlternateAllele",
    "Name",
    "Assembly",
    "residue_position",
    "alt_aa",
]


# ============================================================
# HELPERS
# ============================================================

def clean_features(frame):
    """Convert annotation features to numeric values."""
    frame = frame.copy()

    for column in frame.columns:
        frame[column] = frame[column].replace(
            [".", "NA", "na", "None", "nan", ""],
            np.nan,
        )

        frame[column] = frame[column].map(
            lambda value: (
                value.split(";")[0]
                if isinstance(value, str) and ";" in value
                else value
            )
        )

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame.replace([np.inf, -np.inf], np.nan)


def add_missing_indicators(frame):
    """Add one indicator per feature before imputation."""
    missing = frame.isna().astype(np.int8)
    missing.columns = [
        f"{column}_missing" for column in frame.columns
    ]
    return pd.concat([frame, missing], axis=1)


def priority(probability):
    if probability >= 0.90:
        return "Very High"
    if probability >= 0.75:
        return "High"
    if probability >= 0.50:
        return "Moderate"
    return "Low"


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 75)
print("CLN2 XGBOOST VUS RANKING")
print("=" * 75)
print("Training data:", TRAIN_PATH)
print("VUS data:", VUS_PATH)
print("Output directory:", RANKING_DIR)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_PATH}\n"
        "Run cln2_VUS_Preprocessing.py first."
    )

if not os.path.exists(VUS_PATH):
    raise FileNotFoundError(
        f"VUS dataset not found:\n{VUS_PATH}\n"
        "Run cln2_VUS_Preprocessing.py first."
    )


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH, low_memory=False)
vus = pd.read_csv(VUS_PATH, low_memory=False)

if "label" not in train.columns:
    raise ValueError(
        "The labelled training CSV must contain a 'label' column."
    )

train["label"] = pd.to_numeric(
    train["label"], errors="coerce"
)
train = train[train["label"].isin([0, 1])].copy()
train["label"] = train["label"].astype(int)

if train.empty:
    raise ValueError("The labelled training dataset is empty.")

if train["label"].nunique() != 2:
    raise ValueError(
        "Training data must contain both classes: "
        "benign (0) and pathogenic (1)."
    )

if vus.empty:
    raise ValueError("The VUS dataset is empty.")

print("\nTraining rows:", len(train))
print("Benign training variants:", int((train["label"] == 0).sum()))
print("Pathogenic training variants:", int((train["label"] == 1).sum()))
print("VUS candidates:", len(vus))


# ============================================================
# SELECT FEATURES AVAILABLE IN BOTH DATASETS
# ============================================================

available_features = [
    column for column in FEATURES
    if column in train.columns and column in vus.columns
]

if not available_features:
    raise ValueError(
        "No expected ML features are shared by training and VUS files."
    )

missing_train_columns = [
    column for column in FEATURES if column not in train.columns
]
missing_vus_columns = [
    column for column in FEATURES if column not in vus.columns
]

if missing_train_columns:
    print(
        "\nFeatures absent from training CSV:",
        missing_train_columns,
    )

if missing_vus_columns:
    print(
        "\nFeatures absent from VUS CSV:",
        missing_vus_columns,
    )

X_train_raw = clean_features(train[available_features])
X_vus_raw = clean_features(vus[available_features])

# Drop features that have no numeric observations in training.
# This decision is made from training data only.
usable_features = [
    column for column in available_features
    if X_train_raw[column].notna().any()
]

if not usable_features:
    raise ValueError(
        "All candidate features are missing in the training data."
    )

X_train_raw = X_train_raw[usable_features].copy()
X_vus_raw = X_vus_raw[usable_features].copy()

# Ensure the same column order in both datasets.
X_vus_raw = X_vus_raw.reindex(columns=X_train_raw.columns)


# ============================================================
# MISSINGNESS INDICATORS
# ============================================================
# Construct indicators before imputation so that missing values
# in the original annotation features remain identifiable.

X_train = add_missing_indicators(X_train_raw)
X_vus = add_missing_indicators(X_vus_raw)

X_vus = X_vus.reindex(columns=X_train.columns)


# ============================================================
# IMPUTATION — FIT USING TRAINING DATA ONLY
# ============================================================

training_medians = X_train_raw.median()

# If a feature is all missing in training it should already have
# been removed above. This fallback protects against unexpected data.
training_medians = training_medians.fillna(0)

for column in usable_features:
    X_train[column] = X_train[column].fillna(
        training_medians[column]
    )
    X_vus[column] = X_vus[column].fillna(
        training_medians[column]
    )

# Missing-indicator columns are binary and should be fully populated.
X_train = X_train.fillna(0)
X_vus = X_vus.fillna(0)

# Make feature order identical.
X_vus = X_vus.reindex(columns=X_train.columns)

y_train = train["label"].astype(int)

print("\nBiological features used:", len(usable_features))
print("Total model columns including missing indicators:", X_train.shape[1])


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / max(positive_count, 1)

print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# TRAIN XGBOOST CLASSIFIER
# ============================================================
# The model is trained on labelled 2026 variants.
# VUS candidates are never used as labelled training examples.

model = XGBClassifier(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.90,
    colsample_bytree=0.90,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining XGBoost classifier...")
model.fit(X_train, y_train, verbose=False)

# Save model and the exact preprocessing information needed to reuse it.
joblib.dump(
    {
        "model": model,
        "features": usable_features,
        "model_columns": list(X_train.columns),
        "training_medians": training_medians.to_dict(),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    },
    MODEL_PATH,
)

print("Saved model bundle:", MODEL_PATH)


# ============================================================
# PREDICT VUS PATHOGENICITY PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_vus)[:, 1]

# This class label is descriptive only. Ranking is based on probability,
# not on the threshold or class label.
predicted_classes = np.where(
    probabilities >= CLASSIFICATION_THRESHOLD,
    "Pathogenic",
    "Non-Pathogenic",
)


# ============================================================
# CREATE RANKED OUTPUT
# ============================================================

identifier_columns = [
    column for column in IDENTIFIERS
    if column in vus.columns
]

results = vus[identifier_columns].copy()

results.insert(
    0,
    "Original_Row",
    np.arange(1, len(results) + 1),
)

results["XGB_Pathogenic_Probability"] = probabilities
results["XGB_Pathogenic_Percentage"] = probabilities * 100
results["Predicted_Class"] = predicted_classes
results["Pathogenicity_Priority"] = [
    priority(probability) for probability in probabilities
]

# Sort solely by predicted pathogenicity probability, highest first.
results = results.sort_values(
    by="XGB_Pathogenic_Probability",
    ascending=False,
    kind="mergesort",
).reset_index(drop=True)

results.insert(
    0,
    "Rank",
    np.arange(1, len(results) + 1),
)

results["XGB_Pathogenic_Probability"] = (
    results["XGB_Pathogenic_Probability"].round(6)
)
results["XGB_Pathogenic_Percentage"] = (
    results["XGB_Pathogenic_Percentage"].round(2)
)


# ============================================================
# SAVE CSV AND EXCEL
# ============================================================

results.to_csv(RESULTS_CSV, index=False)

summary = pd.DataFrame({
    "Metric": [
        "Training dataset",
        "VUS dataset",
        "Training variants",
        "Benign training variants",
        "Pathogenic training variants",
        "VUS variants ranked",
        "Biological features",
        "Total model columns including missing indicators",
        "Primary ranking criterion",
        "Classification threshold (descriptive only)",
    ],
    "Value": [
        os.path.basename(TRAIN_PATH),
        os.path.basename(VUS_PATH),
        len(train),
        negative_count,
        positive_count,
        len(vus),
        len(usable_features),
        X_train.shape[1],
        "XGBoost pathogenicity probability, descending",
        CLASSIFICATION_THRESHOLD,
    ],
})

feature_sheet = pd.DataFrame({
    "Feature": usable_features,
    "Training_Median_Used_For_Imputation": [
        training_medians[column] for column in usable_features
    ],
})

model_feature_sheet = pd.DataFrame({
    "Model_Column": list(X_train.columns)
})

with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
    results.to_excel(
        writer, sheet_name="Ranked_VUS", index=False
    )
    summary.to_excel(
        writer, sheet_name="Summary", index=False
    )
    feature_sheet.to_excel(
        writer, sheet_name="Biological_Features", index=False
    )
    model_feature_sheet.to_excel(
        writer, sheet_name="Model_Features", index=False
    )

print("\n" + "=" * 75)
print("CLN2 VUS RANKING COMPLETE")
print("=" * 75)
print("Ranked CSV:", RESULTS_CSV)
print("Ranked Excel:", RESULTS_XLSX)
print("\nTop 20 ranked variants:")
print(results.head(20).to_string(index=False))

print(
    "\nImportant: These scores are research prioritisation estimates, "
    "not clinical diagnoses or calibrated clinical probabilities."
)
