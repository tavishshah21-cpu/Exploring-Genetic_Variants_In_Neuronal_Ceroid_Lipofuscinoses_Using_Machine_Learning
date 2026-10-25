
# ============================================================
# CLN3 ALPHAFOLD ABLATION STUDY
# Leakage-aware temporal validation: 2025 -> 2026
#
# EXPERIMENTS:
#   1. WITHOUT_AF
#   2. WITH_AF
#
# TRAINING LABELS: 2025 ONLY
# THRESHOLD: 2025 OUT-OF-FOLD PREDICTIONS ONLY
# FINAL EVALUATION: 2026 ONLY
# ============================================================

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "AlphaFold_Comparison_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_WITHOUT_AF = (
    BASE_DIR / "CLN3_2025_WITHOUT_AF_CLN3_ML_READY.csv"
)
TRAIN_WITH_AF = (
    BASE_DIR / "CLN3_2025_WITH_AF_CLN3_ML_READY.csv"
)
TEST_WITHOUT_AF = (
    BASE_DIR / "CLN3_2026_WITHOUT_AF_CLN3_ML_READY.csv"
)
TEST_WITH_AF = (
    BASE_DIR / "CLN3_2026_WITH_AF_CLN3_ML_READY.csv"
)

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Metadata must never be used as model features.
IDENTIFIER_COLUMNS = {
    "variationid",
    "name",
    "genesymbol",
    "gene",
    "chromosome",
    "chr",
    "assembly",
    "referenceallele",
    "alternateallele",
    "ref",
    "alt",
    "start",
    "end",
    "position",
    "residueposition",
    "residue_position",
    "clinicalsignificance",
    "label",
    "target",
    "secondary_structure",
    "amino_acid",
    "alt_aa",
    "am_class",
    "burial",
    "reviewstatus",
}

# Names matching these patterns are automatically excluded.
# Review this list if a legitimate predictor has a similar name.
LEAKAGE_PATTERNS = [
    r"^label$",
    r"^target$",
    r"clinical.?significance",
    r"^review.?status$",
    r"^variation.?id$",
    r"^ground.?truth$",
    r"^true.?label$",
    r"^target.?label$",
    r"^is.?pathogenic$",
    r"^is.?benign$",
    r"^prediction.?label$",
]

# ============================================================
# GENERAL UTILITIES
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalise_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_forbidden_feature(column):
    normalised = normalise_name(column)

    if normalised in {
        normalise_name(c) for c in IDENTIFIER_COLUMNS
    }:
        return True

    return any(
        re.search(pattern, str(column), flags=re.IGNORECASE)
        for pattern in LEAKAGE_PATTERNS
    )


def check_files():
    files = [
        TRAIN_WITHOUT_AF,
        TRAIN_WITH_AF,
        TEST_WITHOUT_AF,
        TEST_WITH_AF,
    ]

    section("CHECKING INPUT FILES")

    for path in files:
        print(f"{'OK' if path.exists() else 'MISSING'}: {path.name}")

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )


# ============================================================
# VARIANT IDENTIFICATION
# Used to detect train/test variant overlap.
# Never used as model input.
# ============================================================

def find_column(df, candidates):
    lookup = {
        normalise_name(col): col for col in df.columns
    }

    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]

    return None


def make_variant_keys(df):
    """
    Prefer genomic coordinates plus reference and alternate alleles.
    VariationID is a fallback if genomic keys are unavailable.

    Returns None if reliable variant identifiers are unavailable.
    """

    chromosome = find_column(
        df, ["Chromosome", "Chr", "#Chrom"]
    )
    position = find_column(
        df, ["Start", "Position", "POS"]
    )
    ref = find_column(
        df, ["ReferenceAllele", "Ref", "Reference"]
    )
    alt = find_column(
        df, ["AlternateAllele", "Alt", "Alternate"]
    )

    if all([chromosome, position, ref, alt]):
        keys = (
            df[chromosome].astype("string").str.strip().str.lower()
            + ":"
            + df[position].astype("string").str.strip()
            + ":"
            + df[ref].astype("string").str.strip().str.upper()
            + ":"
            + df[alt].astype("string").str.strip().str.upper()
        )

        valid = (
            df[chromosome].notna()
            & df[position].notna()
            & df[ref].notna()
            & df[alt].notna()
        )

        keys = keys.where(valid, np.nan)
        return keys

    variation_id = find_column(df, ["VariationID"])

    if variation_id is not None:
        return df[variation_id].astype("string").where(
            df[variation_id].notna(), np.nan
        )

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):
    print(f"\nLoading {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    if "label" not in df.columns:
        raise ValueError(
            f"'label' column missing from {file_path.name}"
        )

    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )

    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    if df.empty:
        raise ValueError(
            f"No valid binary labels in {file_path.name}"
        )

    print(f"Valid rows: {len(df)}")
    print(f"Benign: {(df['label'] == 0).sum()}")
    print(f"Pathogenic: {(df['label'] == 1).sum()}")

    # Preserve identifiers for leakage auditing only.
    variant_keys = make_variant_keys(df)

    forbidden = [
        col for col in df.columns
        if is_forbidden_feature(col)
    ]

    candidate_features = [
        col for col in df.columns
        if col != "label" and col not in forbidden
    ]

    # Convert candidate predictors to numeric.
    numeric_features = []

    for col in candidate_features:
        converted = pd.to_numeric(df[col], errors="coerce")

        # Drop columns that contain no numeric information.
        if converted.notna().any():
            df[col] = converted.replace(
                [np.inf, -np.inf], np.nan
            )
            numeric_features.append(col)

    print(f"Excluded metadata/target columns: {forbidden}")
    print(f"Numeric candidate features: {len(numeric_features)}")

    if not numeric_features:
        raise ValueError(
            f"No usable numeric features in {file_path.name}"
        )

    return df, numeric_features, variant_keys


# ============================================================
# REMOVE OVERLAPPING VARIANTS FROM TEMPORAL TEST SET
# ============================================================

def remove_overlapping_variants(
    train_df, test_df, train_keys, test_keys
):
    section("TRAIN / TEST VARIANT OVERLAP AUDIT")

    if train_keys is None or test_keys is None:
        print(
            "WARNING: Reliable variant keys unavailable in at "
            "least one dataset."
        )
        print(
            "Cannot guarantee that the same variant is absent "
            "from both years. Check the raw files manually."
        )
        return test_df, False

    train_valid_keys = set(train_keys.dropna().astype(str))
    test_valid_keys = test_keys.astype("string")

    overlap_mask = test_valid_keys.isin(train_valid_keys)
    overlap_mask &= test_valid_keys.notna()

    overlap_count = int(overlap_mask.sum())

    print(f"2025 unique identifiable variants: {len(train_valid_keys)}")
    print(f"2026 rows overlapping 2025: {overlap_count}")

    # Save overlap details for auditing.
    overlap_rows = test_df.loc[overlap_mask].copy()

    if not overlap_rows.empty:
        overlap_rows["audit_variant_key"] = (
            test_valid_keys.loc[overlap_mask].values
        )
        overlap_rows.to_csv(
            OUTPUT_DIR / "overlapping_2025_2026_variants.csv",
            index=False,
        )

    # Primary evaluation excludes variants seen in 2025.
    clean_test = test_df.loc[~overlap_mask].copy()

    print(f"2026 rows before filtering: {len(test_df)}")
    print(f"2026 rows after filtering: {len(clean_test)}")

    if clean_test.empty:
        raise ValueError(
            "No 2026 test rows remain after overlap filtering."
        )

    return clean_test, True


# ============================================================
# THRESHOLD SELECTION
# Threshold is selected using 2025 out-of-fold predictions only.
# ============================================================

def select_threshold(y_true, probabilities):
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# OUT-OF-FOLD PREDICTIONS
#
# Each fold fits its own imputer and model.
# Each fold's class weight is calculated only from that fold's
# training labels.
# ============================================================

def generate_oof_predictions(X, y, experiment_name):
    section(f"{experiment_name}: 2025 OOF THRESHOLD SELECTION")

    class_counts = y.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "Both benign (0) and pathogenic (1) labels are required."
        )

    smallest_class = int(class_counts.min())
    n_splits = min(N_SPLITS, smallest_class)

    if n_splits < 2:
        raise ValueError(
            "Not enough samples in each class for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        # Imputer is fitted ONLY on this fold's training data.
        imputer = SimpleImputer(strategy="median")

        X_fold_train_imp = imputer.fit_transform(X_fold_train)
        X_fold_valid_imp = imputer.transform(X_fold_valid)

        benign_count = int((y_fold_train == 0).sum())
        pathogenic_count = int((y_fold_train == 1).sum())

        scale_pos_weight = (
            benign_count / pathogenic_count
            if pathogenic_count > 0 else 1.0
        )

        model = XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(X_fold_train_imp, y_fold_train)

        fold_probabilities = model.predict_proba(
            X_fold_valid_imp
        )[:, 1]

        oof_probabilities[valid_idx] = fold_probabilities

        fold_auc = roc_auc_score(
            y_fold_valid, fold_probabilities
        )

        print(f"Fold {fold}: ROC-AUC = {fold_auc:.4f}")

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Some OOF predictions were not generated.")

    threshold, oof_f1 = select_threshold(
        y.to_numpy(), oof_probabilities
    )

    oof_auc = roc_auc_score(y, oof_probabilities)

    print(f"2025 OOF ROC-AUC: {oof_auc:.4f}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"2025 OOF F1 at selected threshold: {oof_f1:.4f}")

    pd.DataFrame({
        "actual_label_2025": y.to_numpy(),
        "oof_pathogenic_probability": oof_probabilities,
    }).to_csv(
        OUTPUT_DIR / f"{experiment_name}_2025_oof_predictions.csv",
        index=False,
    )

    return threshold, oof_auc, oof_f1


# ============================================================
# TRAIN FINAL MODEL ON ALL 2025 DATA AND TEST ON 2026
# ============================================================

def run_experiment(experiment_name, train_file, test_file):
    section(f"EXPERIMENT: {experiment_name}")

    train_df, train_features, train_keys = load_data(train_file)
    test_df, test_features, test_keys = load_data(test_file)

    # Audit and remove identical variants from the 2026 test set.
    test_df, overlap_check_completed = remove_overlapping_variants(
        train_df, test_df, train_keys, test_keys
    )

    # Feature schema is based on available training predictors.
    # 2026 labels are never used to choose predictors.
    features = [
        col for col in train_features
        if col in test_features
    ]

    if not features:
        raise ValueError(
            f"No common features for {experiment_name}."
        )

    # Explicitly check the final feature list for forbidden columns.
    forbidden_final = [
        col for col in features if is_forbidden_feature(col)
    ]

    if forbidden_final:
        raise RuntimeError(
            f"Potential target/metadata leakage detected: "
            f"{forbidden_final}"
        )

    print(f"\nFeatures used ({len(features)}):")
    for feature in features:
        print(f"  - {feature}")

    # Save the actual feature list.
    pd.Series(features, name="feature").to_csv(
        OUTPUT_DIR / f"{experiment_name}_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2025 TRAINING DATA ONLY
    # --------------------------------------------------------

    X_train = train_df[features].copy()
    y_train = train_df["label"].astype(int).copy()

    # --------------------------------------------------------
    # 2026 TEMPORAL TEST DATA
    # Labels are held back until final evaluation.
    # --------------------------------------------------------

    X_test = test_df[features].copy()
    y_test = test_df["label"].astype(int).copy()

    # Drop predictors that are completely missing in 2025.
    all_missing = X_train.columns[X_train.isna().all()].tolist()

    if all_missing:
        print(
            "\nDropping features entirely missing in 2025:",
            all_missing,
        )
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)
        features = X_train.columns.tolist()

    if X_train.shape[1] == 0:
        raise ValueError("No usable features remain.")

    # --------------------------------------------------------
    # SELECT THRESHOLD FROM 2025 OOF PREDICTIONS
    # --------------------------------------------------------

    threshold, cv_auc, oof_f1 = generate_oof_predictions(
        X_train, y_train, experiment_name
    )

    # --------------------------------------------------------
    # FINAL FIT: IMPUTER FIT ON ALL 2025 TRAINING FEATURES
    # --------------------------------------------------------

    final_imputer = SimpleImputer(strategy="median")

    X_train_imp = final_imputer.fit_transform(X_train)
    X_test_imp = final_imputer.transform(X_test)

    benign_count = int((y_train == 0).sum())
    pathogenic_count = int((y_train == 1).sum())

    scale_pos_weight = (
        benign_count / pathogenic_count
        if pathogenic_count > 0 else 1.0
    )

    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    section(f"{experiment_name}: FINAL TRAINING")

    print(f"Final training rows (2025): {len(X_train)}")
    print(f"Final test rows (2026): {len(X_test)}")
    print("Fitting final model on all eligible 2025 rows...")

    # Do not pass 2026 labels or features as an eval_set.
    model.fit(X_train_imp, y_train)

    # --------------------------------------------------------
    # FINAL 2026 EVALUATION
    # --------------------------------------------------------

    test_probabilities = model.predict_proba(X_test_imp)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, test_predictions)
    roc_auc = roc_auc_score(y_test, test_probabilities)
    pr_auc = average_precision_score(y_test, test_probabilities)
    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )

    section(f"{experiment_name}: 2026 TEMPORAL TEST RESULTS")

    print(f"Threshold (2025 OOF): {threshold:.2f}")
    print(f"Accuracy:             {accuracy:.6f}")
    print(f"ROC-AUC:              {roc_auc:.6f}")
    print(f"PR-AUC:               {pr_auc:.6f}")
    print(f"Precision:            {precision:.6f}")
    print(f"Recall:               {recall:.6f}")
    print(f"F1-score:             {f1:.6f}")
    print(f"2025 OOF ROC-AUC:      {cv_auc:.6f}")
    print(f"2025 OOF F1:           {oof_f1:.6f}")

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Benign", "Pathogenic"],
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {
        "experiment": experiment_name,
        "train_year": 2025,
        "test_year": 2026,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(features),
        "overlap_audit_completed": overlap_check_completed,
        "threshold_from_2025_oof": threshold,
        "2025_oof_roc_auc": cv_auc,
        "2025_oof_f1": oof_f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    pd.DataFrame([results]).to_csv(
        OUTPUT_DIR / f"{experiment_name}_metrics.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / f"{experiment_name}_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Experiment: {experiment_name}\n")
        file.write("Training data: 2025\n")
        file.write("Temporal test data: 2026\n")
        file.write(f"Threshold from 2025 OOF: {threshold:.4f}\n")
        file.write(f"2025 OOF ROC-AUC: {cv_auc:.6f}\n")
        file.write(f"2026 ROC-AUC: {roc_auc:.6f}\n\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(cm))

    # Confusion matrix plot.
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Pathogenic"],
    ).plot(ax=ax, values_format="d")

    ax.set_title(f"{experiment_name}: 2026 Test")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{experiment_name}_confusion_matrix.png",
        dpi=300,
    )
    plt.close(fig)

    # Feature importance.
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / f"{experiment_name}_feature_importance.csv",
        index=False,
    )

    # Save test predictions for auditing.
    predictions_df = pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "pathogenic_probability": test_probabilities,
        "predicted_label": test_predictions,
        "threshold": threshold,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / f"{experiment_name}_2026_predictions.csv",
        index=False,
    )

    return results


# ============================================================
# COMPARE ALPHAFOLD ABLATION RESULTS
# ============================================================

def compare_results(without_af, with_af):
    section("ALPHAFOLD ABLATION COMPARISON")

    metric_names = [
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "2025_oof_roc_auc",
    ]

    rows = []

    for metric in metric_names:
        without_value = without_af[metric]
        with_value = with_af[metric]

        rows.append({
            "metric": metric,
            "without_alphafold": without_value,
            "with_alphafold": with_value,
            "absolute_change": with_value - without_value,
        })

    comparison = pd.DataFrame(rows)

    print(comparison.to_string(index=False, float_format="%.6f"))

    comparison.to_csv(
        OUTPUT_DIR / "AlphaFold_Ablation_Comparison.csv",
        index=False,
    )

    plot_metrics = [
        "accuracy", "roc_auc", "pr_auc",
        "precision", "recall", "f1",
    ]

    x = np.arange(len(plot_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        [without_af[m] for m in plot_metrics],
        width,
        label="Without AlphaFold",
    )
    ax.bar(
        x + width / 2,
        [with_af[m] for m in plot_metrics],
        width,
        label="With AlphaFold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics, rotation=20)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("CLN3 AlphaFold Ablation: 2026 Test")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "AlphaFold_Metric_Comparison.png",
        dpi=300,
    )
    plt.close(fig)

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():
    section("CLN3 ALPHAFOLD ABLATION STUDY")
    print("Training labels: ClinVar 2025 ONLY")
    print("Threshold: 2025 out-of-fold predictions ONLY")
    print("Final evaluation: ClinVar 2026")
    print(f"Working directory: {BASE_DIR}")
    print(f"Results directory: {OUTPUT_DIR}")

    check_files()

    # Experiment 1: no AlphaFold features.
    without_af = run_experiment(
        "WITHOUT_AF",
        TRAIN_WITHOUT_AF,
        TEST_WITHOUT_AF,
    )

    # Experiment 2: AlphaFold features included.
    with_af = run_experiment(
        "WITH_AF",
        TRAIN_WITH_AF,
        TEST_WITH_AF,
    )

    compare_results(without_af, with_af)

    section("COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
