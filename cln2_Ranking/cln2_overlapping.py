# ============================================================
# CLN2 VUS CONSENSUS RANKING
#
# Models:
#   1. Pipeline3_Stacking
#   2. RENOVO_XGBoost
#   3. ML_Graph_XGBoost
#
# Training:
#   CLN2_2026_CLN2_ML_READY.csv
#
# Prediction:
#   CLN2_VUS_CLN2_ML_READY.csv
#
# Preprocessing:
#   Matches the 27 biological features and numeric cleaning
#   approach used in cln2_Ranker.py.
#
# Important:
#   Genomic overlap cannot be verified when identifiers are
#   absent from the labelled training CSV. The script reports
#   this explicitly rather than claiming overlap was checked.
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = r"D:\tavish_shah\PES\capstone"

DATA_DIR = os.path.join(
    ROOT_DIR,
    "Pipeline_1_cln2"
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "cln2_Ranking"
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "CLN2_2026_CLN2_ML_READY.csv"
)

VUS_PATH = os.path.join(
    OUTPUT_DIR,
    "CLN2_VUS_CLN2_ML_READY.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.50
TOP_PERCENT = 10
N_SPLITS = 5

LABEL_COLUMN = "label"


# ============================================================
# OUTPUT PATHS
# ============================================================

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "CLN2_VUS_CONSENSUS_RESULTS.csv"
)

RESULTS_XLSX = os.path.join(
    OUTPUT_DIR,
    "CLN2_VUS_CONSENSUS_RESULTS.xlsx"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "CLN2_VUS_CONSENSUS_MODELS.pkl"
)


# ============================================================
# BIOLOGICAL FEATURES
# Keep these aligned with cln2_Ranker.py.
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
    "is_intermediate",
    "contact_count",
]


# ============================================================
# IDENTIFIER COLUMNS
# These are retained in the results but never used as features.
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
    "alt_aa",
]


# ============================================================
# FEATURE CLEANING
# Based on cln2_Ranker.py.
# ============================================================

def clean_features(frame):
    """
    Convert biological features to numeric values.

    Missing values and non-numeric entries become NaN.
    Semicolon-separated values use their first entry.
    """

    frame = frame.copy()

    for column in frame.columns:

        frame[column] = frame[column].replace(
            [
                ".",
                "NA",
                "N/A",
                "na",
                "None",
                "null",
                "nan",
                "NaN",
                ""
            ],
            np.nan
        )

        frame[column] = frame[column].map(
            lambda value: (
                value.split(";")[0].strip()
                if isinstance(value, str)
                and ";" in value
                else value
            )
        )

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return frame


def add_missing_indicators(frame):
    """
    Add one binary column for each feature, indicating whether
    its original value was missing.
    """

    missing = frame.isna().astype(np.int8)

    missing.columns = [
        str(column) + "_missing"
        for column in frame.columns
    ]

    return pd.concat(
        [frame, missing],
        axis=1
    )


def get_priority(probability):
    if probability >= 0.90:
        return "Very High"

    if probability >= 0.75:
        return "High"

    if probability >= 0.50:
        return "Moderate"

    return "Low"


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_training_labels(train):
    if LABEL_COLUMN not in train.columns:
        raise ValueError(
            f"Training data must contain '{LABEL_COLUMN}'."
        )

    train = train.copy()

    train[LABEL_COLUMN] = pd.to_numeric(
        train[LABEL_COLUMN],
        errors="coerce"
    )

    train = train[
        train[LABEL_COLUMN].isin([0, 1])
    ].copy()

    train[LABEL_COLUMN] = (
        train[LABEL_COLUMN].astype(int)
    )

    if train.empty:
        raise ValueError(
            "No valid binary training labels were found."
        )

    if train[LABEL_COLUMN].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes: "
            "0 = benign and 1 = pathogenic."
        )

    return train


# ============================================================
# OVERLAP CHECK
# ============================================================

def normalize_variant_keys(frame):
    """
    Normalize exact genomic keys:
        chromosome + position + reference + alternate

    This function is called only when all required columns
    are available.
    """

    required = [
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Cannot check genomic overlap. Missing columns: "
            + str(missing)
        )

    keys = pd.DataFrame(index=frame.index)

    keys["chrom"] = (
        frame["Chromosome"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"^chr", "", regex=True)
    )

    keys["pos"] = pd.to_numeric(
        frame["Start"],
        errors="coerce"
    )

    keys["ref"] = (
        frame["ReferenceAllele"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    keys["alt"] = (
        frame["AlternateAllele"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # FIXED: each .any(axis=1) is called on a DataFrame,
    # not on a Series returned by another .any(axis=1).
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


def remove_training_overlaps(train, vus):
    """
    Remove VUS rows that exactly match labelled training
    variants, when identifiers are available in both datasets.

    If genomic identifiers are missing, preserve all VUS rows
    and return 'not_verified'. Do not claim that overlap was
    ruled out.
    """

    print("\n" + "=" * 80)
    print("CHECKING EXACT GENOMIC OVERLAP")
    print("=" * 80)

    required = [
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
    ]

    missing_train = [
        column
        for column in required
        if column not in train.columns
    ]

    missing_vus = [
        column
        for column in required
        if column not in vus.columns
    ]

    if missing_train or missing_vus:

        print(
            "\nWARNING: Exact genomic overlap cannot be verified."
        )

        if missing_train:
            print(
                "Missing from training dataset:",
                missing_train
            )

        if missing_vus:
            print(
                "Missing from VUS dataset:",
                missing_vus
            )

        print(
            "No VUS rows will be removed based on an "
            "unverified overlap check."
        )

        return train, vus, "not_verified"

    train_keys_df = normalize_variant_keys(train)
    vus_keys_df = normalize_variant_keys(vus)

    train_keys = set(
        train_keys_df.dropna().itertuples(
            index=False,
            name=None
        )
    )

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

    valid_vus_rows = (
        vus_keys_df.notna()
        .all(axis=1)
        .fillna(False)
    )

    overlap_mask = (
        valid_vus_rows
        & vus_key_rows.isin(train_keys)
    )

    overlap_count = int(overlap_mask.sum())

    print("Exact overlapping VUS rows:", overlap_count)

    if overlap_count > 0:
        vus = vus.loc[~overlap_mask].copy()

        print(
            "Removed overlapping VUS rows:",
            overlap_count
        )

    if vus.empty:
        raise ValueError(
            "No VUS rows remain after the overlap check."
        )

    print("VUS rows retained:", len(vus))

    return train, vus, "verified"


# ============================================================
# FEATURE MATRIX PREPARATION
# ============================================================

def prepare_feature_matrices(train, vus):
    """
    Use the biological features shared by training and VUS.

    Feature selection and imputation are based on training data.
    The VUS labels, if present, are never model features.
    """

    forbidden_features = {
        LABEL_COLUMN,
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
        "alt_aa",
    }

    requested_features = [
        feature
        for feature in FEATURES
        if feature not in forbidden_features
        and feature in train.columns
        and feature in vus.columns
    ]

    if not requested_features:
        raise ValueError(
            "No biological features from cln2_Ranker.py "
            "are present in both training and VUS data."
        )

    print("\nBiological features present in both datasets:")
    print(len(requested_features))

    X_train_base = clean_features(
        train[requested_features]
    )

    X_vus_base = clean_features(
        vus[requested_features]
    )

    # Do not use features that are entirely missing in training.
    usable_features = [
        feature
        for feature in requested_features
        if X_train_base[feature].notna().any()
    ]

    removed_features = [
        feature
        for feature in requested_features
        if feature not in usable_features
    ]

    if removed_features:
        print(
            "\nFeatures removed because all training values "
            "are missing:"
        )

        for feature in removed_features:
            print(" -", feature)

    if not usable_features:
        raise ValueError(
            "No usable biological features remain."
        )

    X_train_base = X_train_base[
        usable_features
    ].copy()

    X_vus_base = X_vus_base[
        usable_features
    ].copy()

    # Generate indicators before imputation, so missingness
    # reflects the original data rather than imputed values.
    X_train = add_missing_indicators(
        X_train_base
    )

    X_vus = add_missing_indicators(
        X_vus_base
    )

    # Fit imputation using labelled training data only.
    imputer = SimpleImputer(
        strategy="median"
    )

    train_array = imputer.fit_transform(
        X_train
    )

    vus_array = imputer.transform(
        X_vus
    )

    # SimpleImputer may drop empty columns in some sklearn
    # versions. Rebuild a consistent DataFrame explicitly.
    model_columns = list(X_train.columns)

    X_train = pd.DataFrame(
        train_array,
        columns=model_columns,
        index=train.index
    )

    X_vus = pd.DataFrame(
        vus_array,
        columns=model_columns,
        index=vus.index
    )

    # Any residual non-finite values are replaced safely.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_vus = X_vus.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # The imputer should have handled feature missingness.
    # Fail explicitly rather than passing NaNs to a model.
    if X_train.isna().any().any():
        raise ValueError(
            "Missing values remain in the training matrix "
            "after imputation."
        )

    if X_vus.isna().any().any():
        raise ValueError(
            "Missing values remain in the VUS matrix "
            "after imputation."
        )

    print("\nFinal biological feature count:", len(usable_features))
    print("Final model column count:", X_train.shape[1])
    print("Training matrix shape:", X_train.shape)
    print("VUS matrix shape:", X_vus.shape)

    return (
        X_train,
        X_vus,
        usable_features,
        model_columns,
        imputer
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

def calculate_scale_pos_weight(y):
    y = np.asarray(y)

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    if positives == 0:
        return 1.0

    return negatives / positives


# ============================================================
# MODEL 1: PIPELINE 3 STACKING
# Based on the stacking configuration in Evaluation.py.
# ============================================================

def make_stacking_model():

    if N_SPLITS > 5:
        raise ValueError(
            "Stacking configuration expects five-fold CV."
        )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return StackingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("gradient_boosting", gradient_boosting),
            ("extra_trees", extra_trees),
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
# MODEL 2: RENOVO XGBOOST
# Based on the RENOVO configuration in Evaluation.py.
# ============================================================

def make_renovo_xgb(y):

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=calculate_scale_pos_weight(y),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# MODEL 3: ML GRAPH XGBOOST
# Based on the ML Graph configuration in Evaluation.py.
# ============================================================

def make_ml_graph_xgb(y):

    return XGBClassifier(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1,
        scale_pos_weight=calculate_scale_pos_weight(y),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# TRAIN MODELS AND PREDICT VUS
# ============================================================

def train_and_predict_models(X_train, y_train, X_vus):

    minimum_class_count = int(
        pd.Series(y_train).value_counts().min()
    )

    if minimum_class_count < N_SPLITS:
        raise ValueError(
            f"Five-fold stacking requires at least "
            f"{N_SPLITS} examples of each class. "
            f"Smallest class has {minimum_class_count}."
        )

    models = {
        "Pipeline3_Stacking": make_stacking_model(),
        "RENOVO_XGBoost": make_renovo_xgb(y_train),
        "ML_Graph_XGBoost": make_ml_graph_xgb(y_train),
    }

    probabilities = {}
    fitted_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 80)
        print("TRAINING:", model_name)
        print("=" * 80)

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train)

        probability = model.predict_proba(
            X_vus
        )[:, 1]

        probability = np.asarray(
            probability,
            dtype=float
        )

        if (
            not np.isfinite(probability).all()
            or (probability < 0).any()
            or (probability > 1).any()
        ):
            raise ValueError(
                f"{model_name} produced invalid probabilities."
            )

        probabilities[model_name] = probability
        fitted_models[model_name] = model

        print("Model training complete.")
        print("VUS predictions:", len(probability))
        print(
            "Predicted pathogenic at threshold "
            f"{CLASSIFICATION_THRESHOLD}:",
            int(
                (
                    probability
                    >= CLASSIFICATION_THRESHOLD
                ).sum()
            )
        )

    return probabilities, fitted_models


# ============================================================
# CREATE CONSENSUS RANKING
# ============================================================

def build_consensus_results(vus, probabilities):

    results = vus.copy()

    # Preserve input row identity for traceability.
    results.insert(
        0,
        "Original_Row",
        np.arange(1, len(results) + 1)
    )

    model_names = list(probabilities.keys())

    for model_name in model_names:

        probability = probabilities[model_name]

        probability_column = (
            model_name + "_Pathogenic_Probability"
        )

        rank_column = model_name + "_Rank"

        top_column = model_name + "_Top10Percent"

        results[probability_column] = probability

        # Rank 1 is the highest pathogenicity probability.
        results[rank_column] = (
            pd.Series(probability, index=results.index)
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        # Use ceiling to define the top 10% count.
        top_n = max(
            1,
            int(np.ceil(len(results) * TOP_PERCENT / 100))
        )

        top_indices = (
            results[probability_column]
            .nlargest(top_n)
            .index
        )

        results[top_column] = (
            results.index.isin(top_indices)
        )

        results[
            model_name + "_Predicted_Class"
        ] = np.where(
            probability >= CLASSIFICATION_THRESHOLD,
            "Pathogenic",
            "Non-Pathogenic"
        )

        results[
            model_name + "_Priority"
        ] = [
            get_priority(value)
            for value in probability
        ]

    top_columns = [
        model_name + "_Top10Percent"
        for model_name in model_names
    ]

    rank_columns = [
        model_name + "_Rank"
        for model_name in model_names
    ]

    probability_columns = [
        model_name + "_Pathogenic_Probability"
        for model_name in model_names
    ]

    # Number of models that place a variant in their top 10%.
    results["Consensus_Top10_Votes"] = (
        results[top_columns]
        .astype(int)
        .sum(axis=1)
    )

    # Average rank across all three models.
    results["Consensus_Mean_Rank"] = (
        results[rank_columns]
        .mean(axis=1)
    )

    # Average probability across all three models.
    results["Consensus_Mean_Probability"] = (
        results[probability_columns]
        .mean(axis=1)
    )

    results["Consensus_Mean_Percentage"] = (
        results["Consensus_Mean_Probability"] * 100
    )

    results["All_Three_Top10"] = (
        results["Consensus_Top10_Votes"] == 3
    )

    results["At_Least_Two_Top10"] = (
        results["Consensus_Top10_Votes"] >= 2
    )

    def consensus_category(row):

        votes = row["Consensus_Top10_Votes"]
        probability = row["Consensus_Mean_Probability"]

        if votes == 3:
            return "Strong consensus: top 10% in all models"

        if votes == 2:
            return "High consensus: top 10% in two models"

        if votes == 1:
            return "Partial consensus: top 10% in one model"

        if probability >= 0.75:
            return "High mean probability; outside model top 10%"

        return "Lower consensus priority"

    results["Consensus_Category"] = results.apply(
        consensus_category,
        axis=1
    )

    # Primary sorting:
    #   1. More top-10% votes
    #   2. Better average rank
    #   3. Higher average pathogenicity probability
    results = results.sort_values(
        by=[
            "Consensus_Top10_Votes",
            "Consensus_Mean_Rank",
            "Consensus_Mean_Probability",
        ],
        ascending=[
            False,
            True,
            False,
        ],
        kind="mergesort"
    ).reset_index(drop=True)

    results.insert(
        0,
        "Consensus_Rank",
        np.arange(1, len(results) + 1)
    )

    # Round display values.
    for column in probability_columns:
        results[column] = results[column].round(6)

    results["Consensus_Mean_Probability"] = (
        results["Consensus_Mean_Probability"].round(6)
    )

    results["Consensus_Mean_Percentage"] = (
        results["Consensus_Mean_Percentage"].round(2)
    )

    results["Consensus_Mean_Rank"] = (
        results["Consensus_Mean_Rank"].round(2)
    )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    train,
    original_vus_count,
    usable_features,
    model_columns,
    overlap_status,
    fitted_models,
    imputer
):

    results.to_csv(
        RESULTS_CSV,
        index=False
    )

    model_names = list(fitted_models.keys())

    model_summary_rows = []

    for model_name in model_names:

        probability_column = (
            model_name + "_Pathogenic_Probability"
        )

        predicted_column = (
            model_name + "_Predicted_Class"
        )

        top_column = (
            model_name + "_Top10Percent"
        )

        model_summary_rows.append({
            "Model": model_name,
            "VUS_Predicted_Pathogenic": int(
                (
                    results[predicted_column]
                    == "Pathogenic"
                ).sum()
            ),
            "VUS_Predicted_Non_Pathogenic": int(
                (
                    results[predicted_column]
                    == "Non-Pathogenic"
                ).sum()
            ),
            "Top10_Count": int(
                results[top_column].sum()
            ),
            "Mean_Predicted_Probability": float(
                results[probability_column].mean()
            ),
            "Maximum_Predicted_Probability": float(
                results[probability_column].max()
            ),
        })

    model_summary = pd.DataFrame(
        model_summary_rows
    )

    consensus_counts = (
        results["Consensus_Top10_Votes"]
        .value_counts()
        .reindex([3, 2, 1, 0], fill_value=0)
        .rename_axis("Number_of_Top10_Model_Votes")
        .reset_index(name="VUS_Count")
    )

    summary = pd.DataFrame({
        "Metric": [
            "Training dataset",
            "VUS dataset",
            "Training rows used",
            "Benign training variants",
            "Pathogenic training variants",
            "Original VUS rows",
            "VUS rows ranked",
            "Exact genomic overlap status",
            "Biological features used",
            "Model columns including missing indicators",
            "Classification threshold",
            "Top percentage per model",
            "Consensus ranking method",
        ],
        "Value": [
            os.path.basename(TRAIN_PATH),
            os.path.basename(VUS_PATH),
            len(train),
            int((train[LABEL_COLUMN] == 0).sum()),
            int((train[LABEL_COLUMN] == 1).sum()),
            original_vus_count,
            len(results),
            overlap_status,
            len(usable_features),
            len(model_columns),
            CLASSIFICATION_THRESHOLD,
            TOP_PERCENT,
            (
                "Top-10% votes descending, mean rank ascending, "
                "mean pathogenicity probability descending"
            ),
        ],
    })

    feature_summary = pd.DataFrame({
        "Biological_Feature": usable_features
    })

    model_feature_summary = pd.DataFrame({
        "Model_Column": model_columns
    })

    with pd.ExcelWriter(
        RESULTS_XLSX,
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

        model_summary.to_excel(
            writer,
            sheet_name="Model_Summary",
            index=False
        )

        consensus_counts.to_excel(
            writer,
            sheet_name="Consensus_Vote_Counts",
            index=False
        )

        feature_summary.to_excel(
            writer,
            sheet_name="Biological_Features",
            index=False
        )

        model_feature_summary.to_excel(
            writer,
            sheet_name="Model_Features",
            index=False
        )

    joblib.dump(
        {
            "models": fitted_models,
            "imputer": imputer,
            "biological_features": usable_features,
            "model_columns": model_columns,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "top_percent": TOP_PERCENT,
            "overlap_status": overlap_status,
            "model_names": model_names,
        },
        MODEL_PATH
    )

    print("\n" + "=" * 80)
    print("OUTPUT FILES SAVED")
    print("=" * 80)

    print("CSV:", RESULTS_CSV)
    print("Excel:", RESULTS_XLSX)
    print("Models:", MODEL_PATH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CLN2 VUS CONSENSUS RANKING")
    print("=" * 80)

    print("Training dataset:", TRAIN_PATH)
    print("VUS dataset:", VUS_PATH)
    print("Output directory:", OUTPUT_DIR)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.isfile(TRAIN_PATH):
        raise FileNotFoundError(
            "Training CSV not found:\n" + TRAIN_PATH
        )

    if not os.path.isfile(VUS_PATH):
        raise FileNotFoundError(
            "VUS CSV not found:\n" + VUS_PATH
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading labelled 2026 CLN2 data...")

    train = pd.read_csv(
        TRAIN_PATH,
        low_memory=False
    )

    if train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    train = validate_training_labels(train)

    y_train = train[LABEL_COLUMN].astype(int)

    print("\nTraining rows:", len(train))
    print(
        "Benign variants:",
        int((y_train == 0).sum())
    )
    print(
        "Pathogenic variants:",
        int((y_train == 1).sum())
    )

    # --------------------------------------------------------
    # Load VUS data
    # --------------------------------------------------------

    print("\nLoading CLN2 VUS candidates...")

    vus = pd.read_csv(
        VUS_PATH,
        low_memory=False
    )

    if vus.empty:
        raise ValueError(
            "VUS dataset is empty."
        )

    original_vus_count = len(vus)

    print("Original VUS rows:", original_vus_count)

    if LABEL_COLUMN in vus.columns:
        print(
            "VUS label column detected. It will not be used "
            "as a model feature."
        )

    # --------------------------------------------------------
    # Remove exact overlaps when possible
    # --------------------------------------------------------

    train, vus, overlap_status = remove_training_overlaps(
        train,
        vus
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X_train,
        X_vus,
        usable_features,
        model_columns,
        imputer
    ) = prepare_feature_matrices(
        train,
        vus
    )

    y_train = train[LABEL_COLUMN].astype(int)

    # --------------------------------------------------------
    # Train all models and predict VUS
    # --------------------------------------------------------

    probabilities, fitted_models = train_and_predict_models(
        X_train,
        y_train,
        X_vus
    )

    # --------------------------------------------------------
    # Build consensus ranking
    # --------------------------------------------------------

    print("\nBuilding consensus ranking...")

    results = build_consensus_results(
        vus,
        probabilities
    )

    # --------------------------------------------------------
    # Save output files
    # --------------------------------------------------------

    save_results(
        results=results,
        train=train,
        original_vus_count=original_vus_count,
        usable_features=usable_features,
        model_columns=model_columns,
        overlap_status=overlap_status,
        fitted_models=fitted_models,
        imputer=imputer
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLN2 VUS CONSENSUS RANKING COMPLETE")
    print("=" * 80)

    print("\nConsensus vote counts:")

    print(
        results["Consensus_Top10_Votes"]
        .value_counts()
        .sort_index(ascending=False)
        .to_string()
    )

    display_columns = [
        "Consensus_Rank",
        "Original_Row",
        "Consensus_Top10_Votes",
        "Consensus_Mean_Rank",
        "Consensus_Mean_Probability",
        "Consensus_Category",
    ]

    for column in [
        "VariationID",
        "Name",
        "Chromosome",
        "Start",
        "ReferenceAllele",
        "AlternateAllele",
    ]:
        if column in results.columns:
            display_columns.append(column)

    print("\nTop 20 consensus-ranked VUS:")

    print(
        results[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nOverlap status:", overlap_status)

    if overlap_status == "not_verified":
        print(
            "WARNING: Exact genomic overlap was not verified. "
            "The training CSV lacks required genomic identifiers."
        )

    print(
        "\nThese model scores are for research prioritisation, "
        "not clinical diagnosis."
    )


if __name__ == "__main__":
    main()