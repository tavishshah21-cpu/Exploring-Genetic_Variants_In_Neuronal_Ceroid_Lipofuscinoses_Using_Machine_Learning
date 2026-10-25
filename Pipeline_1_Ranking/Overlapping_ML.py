
# ============================================================
# CLN3 VUS CONSENSUS RANKING
# Pipeline3 Stacking + RENOVO XGBoost + ML Graph XGBoost
# ============================================================
#
# TRAINING:
#   CLN3_2026_CLN3_ML_READY.csv
#
# VUS INPUT:
#   CLN3_VUS_CLN3_ML_READY.csv
#
# ALGORITHMS:
#   1. Pipeline3 Stacking:
#      Random Forest + Gradient Boosting + Extra Trees
#      Logistic Regression meta-model
#
#   2. RENOVO_XGBoost:
#      XGBoost with RENOVO-style configuration
#
#   3. ML_Graph_XGBoost:
#      XGBoost with ML Graph configuration
#
# RANKING:
#   - Rank every VUS variant separately using each model.
#   - Identify variants in the top 10% for each model.
#   - Identify variants in the top 10% for all three models.
#   - Also identify variants in the top 10% for two models.
#   - Rank unanimous overlaps first, followed by 2-model
#     overlaps, then other variants.
#
# SAFEGUARDS:
#   - Train on labelled 2026 data only.
#   - Use an explicit ML feature whitelist.
#   - Exclude labels and identifiers from model features.
#   - Check exact genomic overlap between training and VUS.
#   - Fit median imputation using training data only.
#   - Never fit any model on VUS data.
#
# OUTPUT:
#   CLN3_VUS_CONSENSUS_RESULTS.csv
#   CLN3_VUS_CONSENSUS_RESULTS.xlsx
#   CLN3_VUS_CONSENSUS_MODELS.pkl
#
# IMPORTANT:
#   The top 10% is a ranking choice, not a clinical threshold.
#   The models' probability scores may not be calibrated equally.
#   Predictions are for research prioritisation, not diagnosis.
# ============================================================

import os
import gc
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

GENE = "CLN3"
RANDOM_STATE = 42

# Used only for the optional per-model binary predictions.
# The consensus ranking itself is based on the top-N rankings.
CLASSIFICATION_THRESHOLD = 0.50

# Top proportion of VUS ranked by each model.
TOP_FRACTION = 0.10

# Number of folds used to generate out-of-fold predictions
# for the Stacking meta-model.
STACKING_CV_FOLDS = 5


# ============================================================
# DIRECTORY
# ============================================================

RANKING_DIR = (
    r"D:\tavish_shah\PES\capstone"
    r"\Pipeline_1_Ranking"
)

os.makedirs(RANKING_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_PATH = os.path.join(
    RANKING_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

VUS_PATH = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_CLN3_ML_READY.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_CONSENSUS_MODELS.pkl"
)

RESULTS_CSV = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_CONSENSUS_RESULTS.csv"
)

RESULTS_EXCEL = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_CONSENSUS_RESULTS.xlsx"
)


# ============================================================
# EXPLICIT MODEL FEATURES
# ============================================================
#
# These are the candidate predictors retained from the original
# ML_Training_VUS code.
#
# Clinical labels and identifiers must not be predictors.
# Verify feature provenance before interpreting results:
# a predictor that directly encodes the target classification
# would create target leakage.
# ============================================================

FEATURES = [
    # Conservation
    "LINSIGHT",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SiPhy_29way_logOdds",

    # Variant-effect predictions
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "MutationTaster_score",
    "MetaRNN_score",
    "VEST4_score",
    "REVEL_score",

    # CADD
    "CADD_indel_raw",
    "CADD_indel_phred",

    # Variant property
    "var_length",

    # AlphaMissense
    "am_pathogenicity",
    "am_likely_pathogenic",
    "am_ambiguous",

    # AlphaFold structural features
    "pLDDT",
    "RSA",

    # Secondary structure
    "ss_helix",
    "ss_sheet",
    "ss_coil",

    # Burial
    "is_buried",
    "is_exposed",
    "is_intermediate",

    # Structural contacts
    "contact_count"
]


# ============================================================
# IDENTIFIER COLUMNS
# ============================================================

IDENTIFIER_COLUMNS = [
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
    "alt_aa"
]


# ============================================================
# FEATURE CLEANING
# ============================================================

def clean_features(df):
    """
    Convert selected feature columns to numeric values.
    Invalid and missing values become NaN.
    """

    df = df.copy()

    for column in df.columns:
        df[column] = df[column].replace(
            [".", "NA", "N/A", ""],
            np.nan
        )

        # Preserve the original script's handling of
        # semicolon-separated annotation values.
        df[column] = df[column].map(
            lambda value:
                value.split(";")[0]
                if isinstance(value, str)
                and ";" in value
                else value
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# NORMALIZE GENOMIC VARIANT KEYS
# ============================================================

def normalize_variant_keys(df):
    """
    Build exact genomic keys:
    chromosome + position + reference + alternate allele.

    This checks exact representation matches. It does not
    normalize equivalent indel representations.
    """

    required = [
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Cannot check genomic overlap. Missing columns: "
            + str(missing)
        )

    keys = pd.DataFrame(index=df.index)

    keys["chrom"] = (
        df["Chromosome"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"^chr", "", regex=True)
    )

    keys["pos"] = pd.to_numeric(
        df["Start"],
        errors="coerce"
    )

    keys["ref"] = (
        df["ReferenceAllele"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    keys["alt"] = (
        df["AlternateAllele"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid = (
        keys["chrom"].notna()
        & keys["pos"].notna()
        & keys["ref"].notna()
        & keys["alt"].notna()
        & (keys["chrom"] != "")
        & (keys["pos"] > 0)
        & (keys["ref"] != "")
        & (keys["alt"] != "")
        & (~keys["ref"].isin([".", "NA", "N/A"]))
        & (~keys["alt"].isin([".", "NA", "N/A"]))
    )

    keys["pos"] = keys["pos"].astype("Int64")
    keys.loc[~valid, :] = pd.NA

    return keys


def key_tuples(keys):
    """Return valid genomic keys as a set of tuples."""

    valid = keys.dropna()

    return set(
        valid.itertuples(
            index=False,
            name=None
        )
    )


# ============================================================
# PRIORITY LABEL
# ============================================================

def get_priority(probability):
    if probability >= 0.90:
        return "Very High"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.50:
        return "Moderate"
    return "Low"


# ============================================================
# TOP-RANKING HELPER
# ============================================================

def get_top_count(number_of_variants):
    """
    Return the number of rows included in the top fraction.
    At least one row is selected when the input is non-empty.
    """

    if number_of_variants <= 0:
        return 0

    return max(
        1,
        int(np.ceil(number_of_variants * TOP_FRACTION))
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("CLN3 VUS CONSENSUS RANKING")
print("=" * 80)

print("\nTraining file:")
print(TRAIN_PATH)

print("\nVUS file:")
print(VUS_PATH)

print("\nModels:")
print("1. Pipeline3 Stacking")
print("2. RENOVO XGBoost")
print("3. ML Graph XGBoost")

print(
    "\nTop-ranking fraction per model:",
    f"{TOP_FRACTION:.0%}"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        "2026 training file not found:\n" + TRAIN_PATH
    )

if not os.path.exists(VUS_PATH):
    raise FileNotFoundError(
        "VUS file not found:\n" + VUS_PATH
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("=" * 80)
print("LOADING 2026 LABELLED TRAINING DATA")
print("=" * 80)

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

print("Original training shape:", train.shape)

if train.empty:
    raise ValueError("Training dataset is empty.")

if "label" not in train.columns:
    raise ValueError(
        "Training data does not contain the label column."
    )


# ============================================================
# VALIDATE LABELS
# ============================================================

train["label"] = pd.to_numeric(
    train["label"],
    errors="coerce"
)

train = train[
    train["label"].isin([0, 1])
].copy()

train["label"] = train["label"].astype(int)

if train.empty:
    raise ValueError(
        "No valid binary training labels were found."
    )

y_train = train["label"].copy()

pathogenic_count = int((y_train == 1).sum())
benign_count = int((y_train == 0).sum())

print("\nTraining labels:")
print("Pathogenic (1):", pathogenic_count)
print("Benign (0):", benign_count)

if pathogenic_count == 0 or benign_count == 0:
    raise ValueError(
        "Both benign (0) and pathogenic (1) classes are required."
    )


# ============================================================
# LOAD VUS DATA
# ============================================================

print()
print("=" * 80)
print("LOADING VUS DATA")
print("=" * 80)

vus = pd.read_csv(
    VUS_PATH,
    low_memory=False
)

print("Original VUS shape:", vus.shape)

if vus.empty:
    raise ValueError("VUS dataset is empty.")

vus_original_count = len(vus)

if "label" in vus.columns:
    print(
        "VUS contains a label column. It will not be used "
        "for training, feature selection, or prediction."
    )


# ============================================================
# CHECK EXACT TRAINING / VUS GENOMIC OVERLAP
# ============================================================

print()
print("=" * 80)
print("CHECKING TRAINING / VUS GENOMIC OVERLAP")
print("=" * 80)

train_keys_df = normalize_variant_keys(train)
vus_keys_df = normalize_variant_keys(vus)

train_keys = key_tuples(train_keys_df)
vus_keys = key_tuples(vus_keys_df)

overlap_keys = train_keys.intersection(vus_keys)

print("Unique valid training variant keys:", len(train_keys))
print("Unique valid VUS variant keys:", len(vus_keys))
print("Exact genomic overlaps:", len(overlap_keys))

if overlap_keys:
    print(
        "\nWARNING: Exact genomic overlaps detected."
    )

    print(
        "Overlapping VUS rows will be excluded from "
        "the prediction results."
    )

    valid_vus_rows = vus_keys_df.notna().all(axis=1)

    vus_key_rows = pd.Series(
        list(
            vus_keys_df.itertuples(
                index=False,
                name=None
            )
        ),
        index=vus_keys_df.index,
        dtype=object
    )

    overlap_mask = (
        valid_vus_rows
        & vus_key_rows.isin(overlap_keys)
    )

    excluded_count = int(overlap_mask.sum())

    print("VUS rows excluded:", excluded_count)

    vus = vus.loc[~overlap_mask].copy()

    if vus.empty:
        raise ValueError(
            "No VUS variants remain after removing exact "
            "training overlaps."
        )
else:
    excluded_count = 0
    print("No exact genomic overlaps detected.")

print("VUS rows retained:", len(vus))

if vus.empty:
    raise ValueError(
        "No VUS rows remain for prediction."
    )


# ============================================================
# CHECK TRAINING LABEL CONSISTENCY
# ============================================================

print()
print("=" * 80)
print("CHECKING TRAINING VARIANT LABEL CONSISTENCY")
print("=" * 80)

train_keys_for_check = train_keys_df.copy()
train_keys_for_check["label"] = train["label"].values

valid_training_keys = train_keys_for_check.dropna(
    subset=["chrom", "pos", "ref", "alt"]
)

label_counts = (
    valid_training_keys
    .groupby(["chrom", "pos", "ref", "alt"])["label"]
    .nunique()
)

conflicting_keys = label_counts[label_counts > 1]

print(
    "Variants with conflicting labels:",
    len(conflicting_keys)
)

if len(conflicting_keys) > 0:
    raise ValueError(
        "The training set contains identical genomic variants "
        "with conflicting labels. Resolve these conflicts "
        "before training."
    )


# ============================================================
# SELECT FEATURES
# ============================================================

forbidden_features = {
    "label",
    "ClinicalSignificance",
    "VariationID",
    "GeneSymbol",
    "Chromosome",
    "Start",
    "ReferenceAllele",
    "AlternateAllele",
    "Name",
    "Assembly",
    "residue_position",
    "alt_aa"
}

unsafe_requested = set(FEATURES).intersection(
    forbidden_features
)

if unsafe_requested:
    raise ValueError(
        "Identifiers or labels were included in FEATURES: "
        + str(sorted(unsafe_requested))
    )

missing_training_features = [
    feature for feature in FEATURES
    if feature not in train.columns
]

if missing_training_features:
    print("\nFeatures absent from training data:")
    for feature in missing_training_features:
        print("  -", feature)

available_features = [
    feature for feature in FEATURES
    if feature in train.columns
]

if not available_features:
    raise ValueError(
        "No requested ML features are present in the training file."
    )

missing_vus_features = [
    feature for feature in available_features
    if feature not in vus.columns
]

if missing_vus_features:
    raise ValueError(
        "The VUS dataset is missing required features: "
        + str(missing_vus_features)
    )

print(
    "\nInitial available feature count:",
    len(available_features)
)


# ============================================================
# PREPARE TRAINING MATRIX
# ============================================================

X_train = clean_features(
    train[available_features]
)

# Remove features that have no usable training values.
usable_features = [
    column for column in X_train.columns
    if X_train[column].notna().any()
]

X_train = X_train[usable_features].copy()
available_features = usable_features

if not available_features:
    raise ValueError(
        "All selected training features are missing."
    )


# ============================================================
# FIT IMPUTATION ON TRAINING DATA ONLY
# ============================================================

train_medians = X_train.median()

valid_medians = train_medians[
    train_medians.notna()
].index.tolist()

X_train = X_train[valid_medians].copy()
train_medians = train_medians[valid_medians]
available_features = valid_medians

if not available_features:
    raise ValueError(
        "No usable numeric features remain after cleaning."
    )

X_train = X_train.fillna(train_medians)

if X_train.isna().any().any():
    raise ValueError(
        "Missing values remain in the training matrix."
    )

print("\nTraining matrix:", X_train.shape)
print("Final feature count:", len(available_features))


# ============================================================
# PREPARE VUS MATRIX
# ============================================================

print()
print("=" * 80)
print("PREPARING VUS FEATURE MATRIX")
print("=" * 80)

X_vus = clean_features(
    vus[available_features]
)

# Reuse medians learned from labelled training data only.
X_vus = X_vus.fillna(train_medians)

if X_vus.isna().any().any():
    raise ValueError(
        "Missing values remain in the VUS prediction matrix."
    )

print("VUS matrix:", X_vus.shape)


# ============================================================
# CLASS IMBALANCE
# ============================================================

scale_pos_weight = benign_count / pathogenic_count

print(
    "\nXGBoost scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# MODEL 1: PIPELINE3 STACKING
# ============================================================

print()
print("=" * 80)
print("TRAINING PIPELINE3 STACKING")
print("=" * 80)

stacking_estimators = [
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    ),
    (
        "gradient_boosting",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    ),
    (
        "extra_trees",
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
]

stacking_model = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    stack_method="predict_proba",
    cv=STACKING_CV_FOLDS,
    n_jobs=-1,
    passthrough=False
)

stacking_model.fit(
    X_train,
    y_train
)

print("Pipeline3 Stacking training complete.")


# ============================================================
# MODEL 2: RENOVO XGBOOST
# ============================================================

print()
print("=" * 80)
print("TRAINING RENOVO XGBOOST")
print("=" * 80)

renovo_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

renovo_model.fit(
    X_train,
    y_train,
    verbose=False
)

print("RENOVO XGBoost training complete.")


# ============================================================
# MODEL 3: ML GRAPH XGBOOST
# ============================================================

print()
print("=" * 80)
print("TRAINING ML GRAPH XGBOOST")
print("=" * 80)

ml_graph_model = XGBClassifier(
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
    n_jobs=-1
)

ml_graph_model.fit(
    X_train,
    y_train,
    verbose=False
)

print("ML Graph XGBoost training complete.")


# ============================================================
# SAVE MODELS AND PREPROCESSING METADATA
# ============================================================

model_bundle = {
    "models": {
        "Pipeline3_Stacking": stacking_model,
        "RENOVO_XGBoost": renovo_model,
        "ML_Graph_XGBoost": ml_graph_model
    },
    "features": available_features,
    "training_medians": train_medians.to_dict(),
    "classification_threshold": CLASSIFICATION_THRESHOLD,
    "top_fraction": TOP_FRACTION,
    "training_file": TRAIN_PATH,
    "gene": GENE,
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nAll three models saved:")
print(MODEL_PATH)


# ============================================================
# PREDICT PATHOGENICITY WITH ALL THREE MODELS
# ============================================================

print()
print("=" * 80)
print("PREDICTING VUS WITH ALL THREE MODELS")
print("=" * 80)

stacking_probability = stacking_model.predict_proba(
    X_vus
)[:, 1]

renovo_probability = renovo_model.predict_proba(
    X_vus
)[:, 1]

ml_graph_probability = ml_graph_model.predict_proba(
    X_vus
)[:, 1]

# Binary predictions are provided for reference.
# The consensus ranking does not depend on this threshold.
stacking_class = (
    stacking_probability >= CLASSIFICATION_THRESHOLD
)

renovo_class = (
    renovo_probability >= CLASSIFICATION_THRESHOLD
)

ml_graph_class = (
    ml_graph_probability >= CLASSIFICATION_THRESHOLD
)

print("VUS variants scored:", len(vus))

print(
    "Stacking predicted pathogenic:",
    int(stacking_class.sum())
)

print(
    "RENOVO XGBoost predicted pathogenic:",
    int(renovo_class.sum())
)

print(
    "ML Graph XGBoost predicted pathogenic:",
    int(ml_graph_class.sum())
)


# ============================================================
# BUILD RESULTS TABLE
# ============================================================

identifier_columns = [
    column for column in IDENTIFIER_COLUMNS
    if column in vus.columns
]

results = vus[identifier_columns].copy()

results["Pipeline3_Stacking_Probability"] = (
    stacking_probability
)

results["RENOVO_XGBoost_Probability"] = (
    renovo_probability
)

results["ML_Graph_XGBoost_Probability"] = (
    ml_graph_probability
)

results["Pipeline3_Stacking_Percentage"] = (
    stacking_probability * 100
)

results["RENOVO_XGBoost_Percentage"] = (
    renovo_probability * 100
)

results["ML_Graph_XGBoost_Percentage"] = (
    ml_graph_probability * 100
)

results["Pipeline3_Stacking_Class"] = np.where(
    stacking_class,
    "Pathogenic",
    "Non-Pathogenic"
)

results["RENOVO_XGBoost_Class"] = np.where(
    renovo_class,
    "Pathogenic",
    "Non-Pathogenic"
)

results["ML_Graph_XGBoost_Class"] = np.where(
    ml_graph_class,
    "Pathogenic",
    "Non-Pathogenic"
)


# ============================================================
# RANK VUS SEPARATELY FOR EACH MODEL
# ============================================================

results["Pipeline3_Stacking_Rank"] = (
    results["Pipeline3_Stacking_Probability"]
    .rank(method="first", ascending=False)
    .astype(int)
)

results["RENOVO_XGBoost_Rank"] = (
    results["RENOVO_XGBoost_Probability"]
    .rank(method="first", ascending=False)
    .astype(int)
)

results["ML_Graph_XGBoost_Rank"] = (
    results["ML_Graph_XGBoost_Probability"]
    .rank(method="first", ascending=False)
    .astype(int)
)


# ============================================================
# IDENTIFY TOP 10% FOR EACH MODEL
# ============================================================

number_of_vus = len(results)
top_count = get_top_count(number_of_vus)

print()
print("=" * 80)
print("IDENTIFYING TOP-RANKED VUS")
print("=" * 80)

print("Total VUS variants:", number_of_vus)
print("Top fraction per model:", f"{TOP_FRACTION:.0%}")
print("Top rows per model:", top_count)

results["Top10_Pipeline3_Stacking"] = (
    results["Pipeline3_Stacking_Rank"] <= top_count
)

results["Top10_RENOVO_XGBoost"] = (
    results["RENOVO_XGBoost_Rank"] <= top_count
)

results["Top10_ML_Graph_XGBoost"] = (
    results["ML_Graph_XGBoost_Rank"] <= top_count
)


# ============================================================
# CALCULATE MODEL OVERLAP
# ============================================================

top_columns = [
    "Top10_Pipeline3_Stacking",
    "Top10_RENOVO_XGBoost",
    "Top10_ML_Graph_XGBoost"
]

results["Top_Model_Votes"] = (
    results[top_columns]
    .astype(int)
    .sum(axis=1)
)

results["Top10_Overlap_All_Three"] = (
    results["Top_Model_Votes"] == 3
)

results["Top10_Overlap_At_Least_Two"] = (
    results["Top_Model_Votes"] >= 2
)

results["Consensus_Category"] = np.select(
    [
        results["Top_Model_Votes"] == 3,
        results["Top_Model_Votes"] == 2,
        results["Top_Model_Votes"] == 1
    ],
    [
        "Top 10% in all 3 models",
        "Top 10% in 2 models",
        "Top 10% in 1 model"
    ],
    default="Not in any model's top 10%"
)


# ============================================================
# CONSENSUS SCORE AND RANK
# ============================================================
#
# The mean probability is used as a tie-breaker, not as a
# calibrated ensemble probability.
#
# The models can have different probability calibration.
# Consensus status and number of top-list votes take priority.
# ============================================================

probability_columns = [
    "Pipeline3_Stacking_Probability",
    "RENOVO_XGBoost_Probability",
    "ML_Graph_XGBoost_Probability"
]

rank_columns = [
    "Pipeline3_Stacking_Rank",
    "RENOVO_XGBoost_Rank",
    "ML_Graph_XGBoost_Rank"
]

results["Mean_Model_Probability"] = (
    results[probability_columns].mean(axis=1)
)

results["Mean_Model_Rank"] = (
    results[rank_columns].mean(axis=1)
)

results["Consensus_Priority"] = np.select(
    [
        results["Top_Model_Votes"] == 3,
        results["Top_Model_Votes"] == 2,
        results["Top_Model_Votes"] == 1
    ],
    [
        "Very High - 3-model overlap",
        "High - 2-model overlap",
        "Moderate - 1-model overlap"
    ],
    default="Other"
)


# ============================================================
# SORT RESULTS
# ============================================================
#
# Ordering:
#   1. Variants in the top 10% of all three models.
#   2. Variants in the top 10% of two models.
#   3. Variants in the top 10% of one model.
#   4. Remaining variants.
#
# Within the same group, use mean model rank, followed by mean
# model probability, to order the rows.
# ============================================================

results = results.sort_values(
    by=[
        "Top_Model_Votes",
        "Mean_Model_Rank",
        "Mean_Model_Probability"
    ],
    ascending=[False, True, False],
    kind="mergesort"
).reset_index(drop=True)

results.insert(
    0,
    "Consensus_Rank",
    np.arange(1, len(results) + 1)
)


# ============================================================
# ROUND DISPLAY COLUMNS
# ============================================================

for column in probability_columns:
    results[column] = results[column].round(6)

for column in [
    "Pipeline3_Stacking_Percentage",
    "RENOVO_XGBoost_Percentage",
    "ML_Graph_XGBoost_Percentage"
]:
    results[column] = results[column].round(2)

results["Mean_Model_Probability"] = (
    results["Mean_Model_Probability"].round(6)
)

results["Mean_Model_Rank"] = (
    results["Mean_Model_Rank"].round(2)
)


# ============================================================
# CONSENSUS SUMMARY
# ============================================================

all_three_count = int(
    results["Top10_Overlap_All_Three"].sum()
)

at_least_two_count = int(
    results["Top10_Overlap_At_Least_Two"].sum()
)

exactly_two_count = int(
    (results["Top_Model_Votes"] == 2).sum()
)

exactly_one_count = int(
    (results["Top_Model_Votes"] == 1).sum()
)

no_model_top_count = int(
    (results["Top_Model_Votes"] == 0).sum()
)

print()
print("=" * 80)
print("CONSENSUS RESULTS")
print("=" * 80)

print(
    "Top 10% overlap in all three models:",
    all_three_count
)

print(
    "Top 10% overlap in at least two models:",
    at_least_two_count
)

print(
    "Top 10% in exactly two models:",
    exactly_two_count
)

print(
    "Top 10% in exactly one model:",
    exactly_one_count
)

print(
    "Not in the top 10% of any model:",
    no_model_top_count
)


# ============================================================
# SAVE CSV
# ============================================================

results.to_csv(
    RESULTS_CSV,
    index=False
)

print("\nCSV saved:")
print(RESULTS_CSV)


# ============================================================
# SAVE EXCEL WORKBOOK
# ============================================================

print()
print("=" * 80)
print("CREATING EXCEL WORKBOOK")
print("=" * 80)

summary = pd.DataFrame({
    "Metric": [
        "Gene",
        "Training Dataset",
        "VUS Dataset",
        "Training Variants",
        "Pathogenic Training Variants",
        "Benign Training Variants",
        "VUS Variants Before Overlap Filtering",
        "VUS Variants After Overlap Filtering",
        "Exact Genomic Overlap Keys Excluded",
        "Number of Features",
        "Models Used",
        "Top Fraction Per Model",
        "Top Rows Per Model",
        "Top 10% Overlap in All Three Models",
        "Top 10% Overlap in At Least Two Models",
        "Top 10% in Exactly Two Models",
        "Top 10% in Exactly One Model",
        "Not in Any Model's Top 10%",
        "Binary Classification Threshold",
        "Ranking Criterion"
    ],
    "Value": [
        GENE,
        os.path.basename(TRAIN_PATH),
        os.path.basename(VUS_PATH),
        len(train),
        pathogenic_count,
        benign_count,
        vus_original_count,
        len(vus),
        excluded_count,
        len(available_features),
        "Pipeline3 Stacking; RENOVO XGBoost; ML Graph XGBoost",
        TOP_FRACTION,
        top_count,
        all_three_count,
        at_least_two_count,
        exactly_two_count,
        exactly_one_count,
        no_model_top_count,
        CLASSIFICATION_THRESHOLD,
        "Model top-10% overlap, then mean rank"
    ]
})

feature_table = pd.DataFrame({
    "ML_Feature": available_features
})

with pd.ExcelWriter(
    RESULTS_EXCEL,
    engine="openpyxl"
) as writer:

    results.to_excel(
        writer,
        sheet_name="Consensus_Ranking",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    feature_table.to_excel(
        writer,
        sheet_name="ML_Features",
        index=False
    )

    # Format each worksheet for easier inspection.
    for sheet_name in [
        "Consensus_Ranking",
        "Summary",
        "ML_Features"
    ]:
        ws = writer.sheets[sheet_name]

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for column_cells in ws.columns:
            column_letter = column_cells[0].column_letter

            max_length = max(
                (
                    len(str(cell.value))
                    for cell in column_cells
                    if cell.value is not None
                ),
                default=10
            )

            ws.column_dimensions[column_letter].width = min(
                max(max_length + 2, 12),
                40
            )

print("\nExcel saved:")
print(RESULTS_EXCEL)


# ============================================================
# DISPLAY TOP 20 CONSENSUS RESULTS
# ============================================================

print()
print("=" * 80)
print("TOP 20 CONSENSUS-RANKED VUS")
print("=" * 80)

display_columns = [
    "Consensus_Rank",
    "Consensus_Category",
    "Top_Model_Votes",
    "Top10_Overlap_All_Three",
    "Pipeline3_Stacking_Probability",
    "RENOVO_XGBoost_Probability",
    "ML_Graph_XGBoost_Probability",
    "Mean_Model_Rank",
    "Consensus_Priority"
]

for column in [
    "VariationID",
    "Name",
    "Chromosome",
    "Start",
    "ReferenceAllele",
    "AlternateAllele"
]:
    if column in results.columns:
        display_columns.append(column)

print(
    results[display_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# DISPLAY ONLY UNANIMOUS TOP-10% OVERLAPS
# ============================================================

print()
print("=" * 80)
print("VUS IN THE TOP 10% OF ALL THREE MODELS")
print("=" * 80)

unanimous = results[
    results["Top10_Overlap_All_Three"]
].copy()

if unanimous.empty:
    print(
        "No variants appeared in the top 10% of all three "
        "models. Inspect the two-model overlap results."
    )
else:
    print("Unanimous overlap count:", len(unanimous))

    print(
        unanimous[display_columns]
        .to_string(index=False)
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("CLN3 VUS CONSENSUS RANKING COMPLETE")
print("=" * 80)

print("\nTraining source: labelled 2026 dataset")
print("Models:", MODEL_PATH)
print("Ranked CSV:", RESULTS_CSV)
print("Ranked Excel:", RESULTS_EXCEL)

print(
    "\nConsensus_Rank 1 is the highest-priority row under "
    "the consensus ranking rule."
)

print(
    "\nTop 10% overlap in all three models is the strictest "
    "ranking group. It does not establish clinical pathogenicity."
)

print(
    "\nImportant: the configurations above reproduce the "
    "described model families and XGBoost settings. For exact "
    "numerical equivalence to your existing pipelines, match "
    "their complete estimator parameters and preprocessing."
)

print(
    "\nThese predictions are for research prioritisation "
    "and are not clinical diagnoses."
)

gc.collect()
