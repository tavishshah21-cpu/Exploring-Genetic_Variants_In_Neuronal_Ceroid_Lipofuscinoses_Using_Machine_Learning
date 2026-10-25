
# ============================================================
# CLN3 VUS XGBOOST PATHOGENICITY PREDICTION
# ============================================================
#
# TRAINING:
#   CLN3_2026_CLN3_ML_READY.csv
#
# VUS INPUT:
#   CLN3_VUS_CLN3_ML_READY.csv
#
# SAFEGUARDS:
#   - Train on labelled 2026 data only
#   - Explicit ML feature whitelist
#   - Exclude labels and identifiers from features
#   - Check exact genomic overlap between training and VUS
#   - Fit imputation values using training data only
#   - Do not fit the model on VUS
#   - Sort by XGBoost pathogenicity probability
#
# OUTPUT:
#   CLN3_VUS_RANKED_RESULTS.csv
#   CLN3_VUS_RANKED_RESULTS.xlsx
#   CLN3_XGBClassifier_VUS.pkl
#
# ============================================================

import os
import gc
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

GENE = "CLN3"
RANDOM_STATE = 42
THRESHOLD = 0.50


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

CLASSIFIER_MODEL = os.path.join(
    RANKING_DIR,
    "CLN3_XGBClassifier_VUS.pkl"
)

RESULTS_CSV = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_RANKED_RESULTS.csv"
)

RESULTS_EXCEL = os.path.join(
    RANKING_DIR,
    "CLN3_VUS_RANKED_RESULTS.xlsx"
)


# ============================================================
# EXPLICIT MODEL FEATURES
# ============================================================
#
# Only these columns may be model inputs.
# Clinical labels, identifiers and genomic coordinates are
# deliberately excluded from the model feature matrix.
#
# IMPORTANT:
# Verify the provenance of every predictor. A predictor that
# directly encodes ClinVar target classifications must not
# be used for predicting those classifications.
#
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

    df = df.copy()

    for column in df.columns:

        df[column] = df[column].replace(
            [".", "NA", "N/A", ""],
            np.nan
        )

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
#
# Exact key:
#   chromosome + position + reference allele + alternate allele
#
# This identifies exact representation matches only. It does
# not normalize equivalent indel representations or prove that
# two different variants are biologically independent.
#
# ============================================================

def normalize_variant_keys(df):

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
# START
# ============================================================

print()
print("=" * 80)
print("CLN3 VUS XGBOOST PATHOGENICITY PREDICTION")
print("=" * 80)

print("\nTraining file:")
print(TRAIN_PATH)

print("\nVUS file:")
print(VUS_PATH)


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

pathogenic_count = int(
    (y_train == 1).sum()
)

benign_count = int(
    (y_train == 0).sum()
)

print("\nTraining labels:")
print("Pathogenic:", pathogenic_count)
print("Benign:", benign_count)

if pathogenic_count == 0 or benign_count == 0:
    raise ValueError(
        "Both benign (0) and pathogenic (1) classes are required."
    )


# ============================================================
# LOAD VUS
# ============================================================

print()
print("=" * 80)
print("LOADING VUS DATA")
print("=" * 80)

vus = pd.read_csv(
    VUS_PATH,
    low_memory=False
)

print("VUS shape:", vus.shape)

if vus.empty:
    raise ValueError("VUS dataset is empty.")


# ============================================================
# CHECK LABEL SEPARATION
# ============================================================

# The VUS set is an inference dataset, not a labelled training
# dataset. Do not use its clinical significance as an input.
#
# If the VUS file contains a label column, it must not be used.

if "label" in vus.columns:
    print(
        "\nVUS file contains a label column. "
        "It will not be used for model training or prediction."
    )


# ============================================================
# CHECK EXACT VARIANT OVERLAP
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
        "\nWARNING: Exact genomic variant overlaps were detected."
    )

    print(
        "Overlapping variants will be excluded from the VUS "
        "prediction output."
    )

    # Remove VUS rows whose exact variant key is present in
    # the labelled training dataset.
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

    print("VUS rows excluded:", int(overlap_mask.sum()))

    vus = vus.loc[~overlap_mask].copy()

    if vus.empty:
        raise ValueError(
            "No VUS variants remain after removing exact "
            "training overlaps."
        )

else:

    print("No exact genomic overlaps detected.")

print("VUS rows retained:", len(vus))


# ============================================================
# CHECK FOR DUPLICATE LABEL CONFLICTS IN TRAINING
# ============================================================

print()
print("=" * 80)
print("CHECKING TRAINING VARIANT LABEL CONSISTENCY")
print("=" * 80)

train_keys_for_check = train_keys_df.copy()
train_keys_for_check["label"] = train["label"].values

valid_training_keys = (
    train_keys_for_check
    .dropna(subset=["chrom", "pos", "ref", "alt"])
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
        "with conflicting labels. Resolve these conflicts in "
        "preprocessing before training."
    )


# ============================================================
# SELECT FEATURES
# ============================================================

# Explicitly exclude target and identifiers even if the
# FEATURES list is accidentally edited in the future.

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

unsafe_requested = (
    set(FEATURES).intersection(forbidden_features)
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

AVAILABLE_FEATURES = [
    feature for feature in FEATURES
    if feature in train.columns
]

if not AVAILABLE_FEATURES:
    raise ValueError(
        "No requested ML features are present in the training file."
    )

missing_vus_features = [
    feature for feature in AVAILABLE_FEATURES
    if feature not in vus.columns
]

if missing_vus_features:
    raise ValueError(
        "The VUS dataset is missing required model features: "
        + str(missing_vus_features)
    )

print()
print("Initial available feature count:", len(AVAILABLE_FEATURES))


# ============================================================
# PREPARE TRAINING MATRIX
# ============================================================

X_train = clean_features(
    train[AVAILABLE_FEATURES]
)

# Remove features that contain no usable training values.
usable_features = [
    column for column in X_train.columns
    if X_train[column].notna().any()
]

X_train = X_train[usable_features].copy()

AVAILABLE_FEATURES = usable_features

if not AVAILABLE_FEATURES:
    raise ValueError(
        "All selected training features are missing."
    )


# ============================================================
# FIT IMPUTATION USING TRAINING DATA ONLY
# ============================================================

TRAIN_MEDIANS = X_train.median()

valid_medians = TRAIN_MEDIANS[
    TRAIN_MEDIANS.notna()
].index.tolist()

X_train = X_train[valid_medians].copy()

TRAIN_MEDIANS = TRAIN_MEDIANS[valid_medians]

AVAILABLE_FEATURES = valid_medians

if not AVAILABLE_FEATURES:
    raise ValueError(
        "No usable numeric features remain after cleaning."
    )

X_train = X_train.fillna(TRAIN_MEDIANS)

if X_train.isna().any().any():
    raise ValueError(
        "Missing values remain in the training matrix."
    )

print("\nTraining matrix:", X_train.shape)
print("Final feature count:", len(AVAILABLE_FEATURES))


# ============================================================
# CLASS WEIGHT
# ============================================================

scale_pos_weight = (
    benign_count / pathogenic_count
)

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print()
print("=" * 80)
print("TRAINING XGBOOST ON 2026 DATA")
print("=" * 80)

classifier = XGBClassifier(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.90,
    colsample_bytree=0.90,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

classifier.fit(
    X_train,
    y_train,
    verbose=False
)

print("\nXGBoost classifier trained on labelled 2026 data.")


# ============================================================
# SAVE MODEL AND PREPROCESSING METADATA
# ============================================================

# Save the medians and exact feature order together with the
# model so future predictions can use the same preprocessing.

model_bundle = {
    "model": classifier,
    "features": AVAILABLE_FEATURES,
    "training_medians": TRAIN_MEDIANS.to_dict(),
    "threshold": THRESHOLD,
    "training_file": TRAIN_PATH,
    "gene": GENE,
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_bundle,
    CLASSIFIER_MODEL
)

print("\nModel bundle saved:")
print(CLASSIFIER_MODEL)


# ============================================================
# PREPARE VUS MATRIX
# ============================================================

print()
print("=" * 80)
print("PREPARING VUS PREDICTION MATRIX")
print("=" * 80)

X_vus = clean_features(
    vus[AVAILABLE_FEATURES]
)

# Use medians from the labelled training data only.
X_vus = X_vus.fillna(TRAIN_MEDIANS)

if X_vus.isna().any().any():
    raise ValueError(
        "Missing values remain in the VUS prediction matrix."
    )

print("VUS prediction matrix:", X_vus.shape)


# ============================================================
# PREDICT PATHOGENICITY
# ============================================================

print()
print("=" * 80)
print("PREDICTING VUS PATHOGENICITY")
print("=" * 80)

pathogenic_probability = (
    classifier.predict_proba(X_vus)[:, 1]
)

predicted_class = (
    pathogenic_probability >= THRESHOLD
)

predicted_class_text = np.where(
    predicted_class,
    "Pathogenic",
    "Non-Pathogenic"
)

print("\nPrediction threshold:", THRESHOLD)

print(
    "Predicted Pathogenic:",
    int(predicted_class.sum())
)

print(
    "Predicted Non-Pathogenic:",
    int((~predicted_class).sum())
)


# ============================================================
# BUILD RESULTS
# ============================================================

identifier_columns = [
    column for column in IDENTIFIER_COLUMNS
    if column in vus.columns
]

results = vus[identifier_columns].copy()

results["XGB_Pathogenic_Probability"] = (
    pathogenic_probability
)

results["XGB_Pathogenic_Percentage"] = (
    pathogenic_probability * 100
)

results["Predicted_Class"] = predicted_class_text

results["Pathogenicity_Priority"] = (
    results["XGB_Pathogenic_Probability"]
    .apply(get_priority)
)


# ============================================================
# SORT BY PATHOGENICITY PROBABILITY
# ============================================================

results = results.sort_values(
    by="XGB_Pathogenic_Probability",
    ascending=False,
    kind="mergesort"
).reset_index(drop=True)

results.insert(
    0,
    "Rank",
    np.arange(1, len(results) + 1)
)

results["XGB_Pathogenic_Probability"] = (
    results["XGB_Pathogenic_Probability"].round(6)
)

results["XGB_Pathogenic_Percentage"] = (
    results["XGB_Pathogenic_Percentage"].round(2)
)

if not results[
    "XGB_Pathogenic_Probability"
].is_monotonic_decreasing:

    raise RuntimeError(
        "Results failed descending probability verification."
    )

print("\nSorted results verified.")


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
# SAVE EXCEL
# ============================================================

print()
print("=" * 80)
print("CREATING EXCEL WORKBOOK")
print("=" * 80)

with pd.ExcelWriter(
    RESULTS_EXCEL,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # RANKED VUS
    # --------------------------------------------------------

    results.to_excel(
        writer,
        sheet_name="Ranked_VUS",
        index=False
    )

    ws = writer.sheets["Ranked_VUS"]

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
            35
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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
            "Exact Genomic Overlaps Excluded",
            "Predicted Pathogenic VUS",
            "Predicted Non-Pathogenic VUS",
            "Prediction Threshold",
            "Number of Features",
            "Sorting Criterion"
        ],

        "Value": [
            GENE,
            os.path.basename(TRAIN_PATH),
            os.path.basename(VUS_PATH),
            len(train),
            pathogenic_count,
            benign_count,
            len(pd.read_csv(VUS_PATH, low_memory=False)),
            len(vus),
            len(overlap_keys),
            int(predicted_class.sum()),
            int((~predicted_class).sum()),
            THRESHOLD,
            len(AVAILABLE_FEATURES),
            "XGBoost pathogenicity probability, descending"
        ]
    })

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_table = pd.DataFrame({
        "ML_Feature": AVAILABLE_FEATURES
    })

    feature_table.to_excel(
        writer,
        sheet_name="ML_Features",
        index=False
    )


print("\nExcel saved:")
print(RESULTS_EXCEL)


# ============================================================
# DISPLAY TOP 20
# ============================================================

print()
print("=" * 80)
print("TOP 20 VUS")
print("=" * 80)

display_columns = [
    "Rank",
    "XGB_Pathogenic_Probability",
    "XGB_Pathogenic_Percentage",
    "Predicted_Class",
    "Pathogenicity_Priority"
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
# FINAL
# ============================================================

print()
print("=" * 80)
print("CLN3 VUS PREDICTION COMPLETE")
print("=" * 80)

print("\nTraining source: 2026 labelled dataset")
print("Model:", CLASSIFIER_MODEL)
print("Ranked CSV:", RESULTS_CSV)
print("Ranked Excel:", RESULTS_EXCEL)

print(
    "\nRank 1 represents the highest XGBoost-predicted "
    "pathogenicity probability."
)

print(
    "\nThese predictions are for research prioritization "
    "and are not clinical diagnoses."
)

gc.collect()
