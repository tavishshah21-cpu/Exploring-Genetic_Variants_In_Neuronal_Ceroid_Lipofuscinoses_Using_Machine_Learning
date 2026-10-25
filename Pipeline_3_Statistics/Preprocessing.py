# ==========================================
# PREPROCESSING
# CLN3 TEMPORAL VALIDATION
#
# INPUT:
# Pipeline_1_Statistics
#
# OUTPUT:
# Pipeline_3_Statistics
# ==========================================

import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_PATH = r"D:\tavish_shah\PES\capstone"

INPUT_DIR = os.path.join(
    BASE_PATH,
    "Pipeline_1_Statistics"
)

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "Pipeline_3_Statistics"
)


TRAIN_FILE = os.path.join(
    INPUT_DIR,
    "variant_summary_2025.txt"
)

TEST_FILE = os.path.join(
    INPUT_DIR,
    "variant_summary_2026.txt"
)


TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "CLN3_2025_preprocessed.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "CLN3_2026_preprocessed.csv"
)


# Create Pipeline 3 directory if needed
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    print("==========================================")
    print("LOADING DATA")
    print("==========================================")

    print("\nLoading 2025:")
    print(TRAIN_FILE)

    train_df = pd.read_csv(
        TRAIN_FILE,
        sep="\t",
        low_memory=False
    )

    print("2025 original shape:", train_df.shape)

    print("\nLoading 2026:")
    print(TEST_FILE)

    test_df = pd.read_csv(
        TEST_FILE,
        sep="\t",
        low_memory=False
    )

    print("2026 original shape:", test_df.shape)

    return train_df, test_df


# ==========================================
# 3. FILTER CLN3
# ==========================================

def filter_cln3(train_df, test_df):

    train_df = train_df[
        train_df["GeneSymbol"] == "CLN3"
    ].copy()

    test_df = test_df[
        test_df["GeneSymbol"] == "CLN3"
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No CLN3 variants found in 2025"
        )

    if test_df.empty:
        raise ValueError(
            "No CLN3 variants found in 2026"
        )

    print("\nAfter CLN3 filtering:")
    print("2025:", train_df.shape)
    print("2026:", test_df.shape)

    return train_df, test_df


# ==========================================
# 4. NORMALIZE LABELS
# ==========================================

def normalize_labels(train_df, test_df):

    for df in [train_df, test_df]:

        df["ClinicalSignificance"] = (
            df["ClinicalSignificance"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return train_df, test_df


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

def remove_duplicates(train_df, test_df):

    if "VariationID" in train_df.columns:

        before = len(train_df)

        train_df = train_df.drop_duplicates(
            subset="VariationID"
        )

        print(
            "\n2025 duplicates removed:",
            before - len(train_df)
        )

    if "VariationID" in test_df.columns:

        before = len(test_df)

        test_df = test_df.drop_duplicates(
            subset="VariationID"
        )

        print(
            "2026 duplicates removed:",
            before - len(test_df)
        )

    return train_df, test_df


# ==========================================
# 6. REMOVE TEMPORAL OVERLAP
# ==========================================

def remove_overlap(train_df, test_df):

    if (
        "VariationID" in train_df.columns
        and "VariationID" in test_df.columns
    ):

        train_ids = set(
            train_df["VariationID"]
        )

        before = len(test_df)

        test_df = test_df[
            ~test_df["VariationID"].isin(train_ids)
        ].copy()

        print(
            "2026 overlapping variants removed:",
            before - len(test_df)
        )

    return train_df, test_df


# ==========================================
# 7. SELECT LABELED VARIANTS
# ==========================================

def select_labeled(train_df, test_df):

    pathogenic_terms = [
        "pathogenic",
        "likely pathogenic"
    ]

    benign_terms = [
        "benign",
        "likely benign"
    ]

    valid_labels = (
        pathogenic_terms +
        benign_terms
    )

    train_labeled = train_df[
        train_df["ClinicalSignificance"]
        .isin(valid_labels)
    ].copy()

    test_labeled = test_df[
        test_df["ClinicalSignificance"]
        .isin(valid_labels)
    ].copy()

    print("\nLabeled variants:")
    print("2025:", train_labeled.shape)
    print("2026:", test_labeled.shape)

    return train_labeled, test_labeled


# ==========================================
# 8. ENCODE TARGET LABEL
# ==========================================

def encode_target(train_labeled, test_labeled):

    label_map = {

        "pathogenic": 1,
        "likely pathogenic": 1,

        "benign": 0,
        "likely benign": 0
    }

    train_labeled["label"] = (
        train_labeled["ClinicalSignificance"]
        .map(label_map)
    )

    test_labeled["label"] = (
        test_labeled["ClinicalSignificance"]
        .map(label_map)
    )

    return train_labeled, test_labeled


# ==========================================
# 9. SELECT FEATURES
# ==========================================

def prepare_features(
    train_labeled,
    test_labeled
):

    features = [
        "Type",
        "ReviewStatus",
        "Chromosome"
    ]

    features = [
        f for f in features
        if (
            f in train_labeled.columns
            and f in test_labeled.columns
        )
    ]

    if not features:
        raise ValueError(
            "No common features found"
        )

    print("\nFeatures used:")
    print(features)

    X_train = train_labeled[
        features
    ].copy()

    X_test = test_labeled[
        features
    ].copy()


    # ======================================
    # ENCODING
    # ======================================
    #
    # Encoder is fitted ONLY on 2025.
    # 2026 is transformed using the
    # same encoder.
    #

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train_encoded = encoder.fit_transform(
        X_train.astype(str)
    )

    X_test_encoded = encoder.transform(
        X_test.astype(str)
    )


    # ======================================
    # CREATE MODEL-READY DATAFRAMES
    # ======================================

    train_processed = pd.DataFrame(
        X_train_encoded,
        columns=features,
        index=train_labeled.index
    )

    test_processed = pd.DataFrame(
        X_test_encoded,
        columns=features,
        index=test_labeled.index
    )


    # Add target label
    train_processed["label"] = (
        train_labeled["label"].values
    )

    test_processed["label"] = (
        test_labeled["label"].values
    )


    # Keep VariationID if available
    if "VariationID" in train_labeled.columns:

        train_processed.insert(
            0,
            "VariationID",
            train_labeled["VariationID"].values
        )

    if "VariationID" in test_labeled.columns:

        test_processed.insert(
            0,
            "VariationID",
            test_labeled["VariationID"].values
        )


    return train_processed, test_processed


# ==========================================
# 10. SAVE PREPROCESSED DATA
# ==========================================

def save_processed_data(
    train_processed,
    test_processed
):

    train_processed.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_processed.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print("\n==========================================")
    print("PREPROCESSED DATA SAVED")
    print("==========================================")

    print("\n2025:")
    print(TRAIN_OUTPUT)

    print("\n2026:")
    print(TEST_OUTPUT)


# ==========================================
# 11. MAIN
# ==========================================

def preprocess():

    train_df, test_df = load_data()

    train_df, test_df = filter_cln3(
        train_df,
        test_df
    )

    train_df, test_df = normalize_labels(
        train_df,
        test_df
    )

    train_df, test_df = remove_duplicates(
        train_df,
        test_df
    )

    train_df, test_df = remove_overlap(
        train_df,
        test_df
    )

    train_labeled, test_labeled = (
        select_labeled(
            train_df,
            test_df
        )
    )

    train_labeled, test_labeled = (
        encode_target(
            train_labeled,
            test_labeled
        )
    )

    train_processed, test_processed = (
        prepare_features(
            train_labeled,
            test_labeled
        )
    )

    save_processed_data(
        train_processed,
        test_processed
    )

    return (
        train_processed,
        test_processed
    )


# ==========================================
# RUN PREPROCESSING
# ==========================================

if __name__ == "__main__":

    train_processed, test_processed = (
        preprocess()
    )

    print("\n==========================================")
    print("PREPROCESSING COMPLETE")
    print("==========================================")

    print(
        "\n2025 processed shape:",
        train_processed.shape
    )

    print(
        "2026 processed shape:",
        test_processed.shape
    )