
# ============================================================
# CLN3 GNOMAD EXTERNAL VALIDATION MODEL
# Train on Pipeline_1_Statistics
# Predict and validate Pipeline_1_Validation candidates
# ============================================================
#
# Training:
#   Pipeline_1_Statistics/CLN3_2026_CLN3_ML_READY.csv
#
# External validation:
#   Pipeline_1_Validation/
#       CLN3_gnomAD_external_candidates_CLN3_ML_READY.csv
#
# Outputs:
#   CLN3_gnomAD_validation_predictions.csv
#   CLN3_gnomAD_validation_results_summary.csv
#   CLN3_gnomAD_validation_classification_report.txt
#   CLN3_gnomAD_validation_confusion_matrix.png
#   CLN3_gnomAD_validation_roc_curve.png
#   CLN3_gnomAD_validation_feature_importance.csv
#   CLN3_gnomAD_validation_model.pkl
#
# IMPORTANT:
# External candidates without known labels cannot be
# evaluated for accuracy. Their predictions are still saved.
# ============================================================

import os
import gc
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

STATISTICS_DIR = os.path.join(
    PROJECT_DIR,
    "Pipeline_1_Statistics"
)

TRAIN_FILE = os.path.join(
    STATISTICS_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

EXTERNAL_FILE = os.path.join(
    BASE_DIR,
    "CLN3_gnomAD_external_candidates_CLN3_ML_READY.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "CLN3_gnomAD_validation_model.pkl"
)

RANDOM_STATE = 42

USE_CV = True

# Set to False if SHAP is not installed or you do not
# need SHAP explanations.
USE_SHAP = False


def out(filename):
    return os.path.join(BASE_DIR, filename)


# ============================================================
# FEATURES
# Based on ml_pipeline_input.py
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
]


# ============================================================
# HELPER: CLEAN VALUES
# ============================================================

def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in ["", ".", "NA", "NaN", "nan", "None"]:
            return np.nan

        # Match the uploaded model's behavior for
        # semicolon-separated annotation values.
        if ";" in value:
            value = value.split(";")[0]

    return value


# ============================================================
# HELPER: LOAD AND CLEAN DATA
# ============================================================

def load_dataset(path, require_labels=False):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    if require_labels and "label" not in df.columns:
        raise ValueError(
            "The training dataset must contain a 'label' column."
        )

    print(f"\nLoaded: {path}")
    print("Original shape:", df.shape)

    return df


def prepare_features(df, feature_columns):

    X = pd.DataFrame(index=df.index)

    for col in feature_columns:

        if col in df.columns:
            X[col] = df[col]
        else:
            # Consistent feature layout for both datasets.
            X[col] = np.nan

    X = X.apply(
        lambda column: column.map(clean_value)
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Same missing-indicator approach as the uploaded model.
    missing = X.isna().astype(np.int8)

    missing.columns = [
        col + "_missing"
        for col in X.columns
    ]

    X = pd.concat(
        [X, missing],
        axis=1
    )

    return X


def clean_labels(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Only binary labels are valid for this classifier.
    numeric = numeric.where(
        numeric.isin([0, 1])
    )

    return numeric


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("CLN3 GNOMAD EXTERNAL VALIDATION")
print("=" * 70)

print("\nTRAINING DATA")
print("-" * 70)

train_df = load_dataset(
    TRAIN_FILE,
    require_labels=True
)

train_df["label"] = clean_labels(
    train_df["label"]
)

train_df = train_df.loc[
    train_df["label"].notna()
].copy()

train_df["label"] = train_df["label"].astype(int)

print("Labelled training rows:", len(train_df))
print("Pathogenic (1):", (train_df["label"] == 1).sum())
print("Benign (0):", (train_df["label"] == 0).sum())

if train_df.empty:
    raise ValueError(
        "No valid labelled rows remain in the training dataset."
    )

if train_df["label"].nunique() != 2:
    raise ValueError(
        "Training requires both benign (0) and pathogenic (1) "
        "examples."
    )


# ============================================================
# SELECT FEATURES AVAILABLE IN TRAINING DATA
# ============================================================

available_features = [
    feature
    for feature in FEATURES
    if feature in train_df.columns
]

if len(available_features) == 0:
    raise ValueError(
        "None of the expected biological features were found "
        "in the training dataset.\n"
        "Check the column names in CLN3_2026_CLN3_ML_READY.csv."
    )

print("\nExpected biological features:", len(FEATURES))
print("Features present in training data:", len(available_features))

print("Using features:")
for feature in available_features:
    print("  -", feature)

X = prepare_features(
    train_df,
    available_features
)

y = train_df["label"].copy()

FINAL_FEATURES = X.columns.tolist()

print("\nTraining matrix:", X.shape)
print("Final feature count including missing indicators:",
      len(FINAL_FEATURES))


# ============================================================
# INTERNAL TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("INTERNAL TRAINING VALIDATION")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Internal validation rows:", len(X_val))


# ============================================================
# CLASS IMBALANCE
# ============================================================

print("\nHandling class imbalance...")

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())

if pos == 0 or neg == 0:
    raise ValueError(
        "Both classes must be present in the training split."
    )

scale_pos_weight = neg / pos

# Preserve the uploaded model's class-weight logic.
if scale_pos_weight < 1:
    print(
        "Pathogenic is the majority class. "
        "Using the inverse class ratio."
    )
    scale_pos_weight = pos / neg

print("Pathogenic:", pos)
print("Benign:", neg)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

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
    early_stopping_rounds=50,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

print("\nModel training completed.")


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

cv_mean = np.nan
cv_scores = []

if USE_CV:

    print("\nRunning 5-fold cross-validation...")

    cv_model = XGBClassifier(
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
        n_jobs=1
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        cv_model,
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    cv_mean = float(np.mean(cv_scores))

    print("CV scores:", np.round(cv_scores, 4))
    print("Mean CV ROC-AUC:", round(cv_mean, 4))


# ============================================================
# INTERNAL VALIDATION METRICS
# ============================================================

print("\n" + "=" * 70)
print("INTERNAL VALIDATION METRICS")
print("=" * 70)

y_prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    y_prob
)

f1_scores = (
    2 * precision * recall
    / (precision + recall + 1e-8)
)

if len(thresholds) > 0:
    best_index = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_index])
else:
    best_threshold = 0.5

print("Selected threshold:", round(best_threshold, 4))

y_pred = (
    y_prob >= best_threshold
).astype(int)

internal_accuracy = accuracy_score(
    y_val,
    y_pred
)

internal_roc_auc = roc_auc_score(
    y_val,
    y_prob
)

internal_pr_auc = average_precision_score(
    y_val,
    y_prob
)

print("\nClassification report:")

internal_report = classification_report(
    y_val,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Pathogenic"],
    zero_division=0
)

print(internal_report)

print("Accuracy:", round(internal_accuracy, 4))
print("ROC-AUC:", round(internal_roc_auc, 4))
print("PR-AUC:", round(internal_pr_auc, 4))

internal_cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1]
)

print("\nInternal confusion matrix:")
print(internal_cm)


# ============================================================
# LOAD EXTERNAL GNOMAD CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL GNOMAD VALIDATION")
print("=" * 70)

external_df = load_dataset(
    EXTERNAL_FILE,
    require_labels=False
)

if external_df.empty:
    raise ValueError(
        "The external validation dataset contains no rows."
    )

# Keep original columns for output.
external_original = external_df.copy()

# Ensure external data has the exact same feature columns.
X_external = prepare_features(
    external_df,
    available_features
)

# Align to the training model's exact feature order.
X_external = X_external.reindex(
    columns=FINAL_FEATURES
)

print("\nExternal candidates:", len(X_external))
print("External feature matrix:", X_external.shape)


# ============================================================
# PREDICT EXTERNAL CANDIDATES
# ============================================================

print("\nGenerating external predictions...")

external_prob = model.predict_proba(
    X_external
)[:, 1]

external_pred = (
    external_prob >= best_threshold
).astype(int)

results = external_original.copy()

results["predicted_value"] = external_pred

results["predicted_class"] = np.where(
    external_pred == 1,
    "PATHOGENIC",
    "BENIGN"
)

results["pathogenicity_probability"] = external_prob

results["benign_probability"] = 1 - external_prob

results["decision_threshold"] = best_threshold

print("\nPrediction counts:")
print(
    results["predicted_class"].value_counts(
        dropna=False
    ).to_string()
)

print("\nTotal candidates:", len(results))
print("Predicted pathogenic:", int((external_pred == 1).sum()))
print("Predicted benign:", int((external_pred == 0).sum()))


# ============================================================
# EXTERNAL EVALUATION
# Only calculate correctness when real labels exist.
# ============================================================

external_accuracy = np.nan
external_roc_auc = np.nan
external_pr_auc = np.nan
external_evaluated = 0
external_report = "Ground-truth labels unavailable."
external_cm = None

if "label" in external_df.columns:

    external_labels = clean_labels(
        external_df["label"]
    )

    labelled_mask = external_labels.notna()

    external_evaluated = int(labelled_mask.sum())

    print("\nCandidates with known ground-truth labels:",
          external_evaluated)

    if external_evaluated > 0:

        y_true_external = external_labels[
            labelled_mask
        ].astype(int)

        y_pred_external = external_pred[
            labelled_mask.to_numpy()
        ]

        y_prob_external = external_prob[
            labelled_mask.to_numpy()
        ]

        correct = (
            y_true_external.to_numpy()
            == y_pred_external
        )

        correct_count = int(correct.sum())
        incorrect_count = int((~correct).sum())

        external_accuracy = accuracy_score(
            y_true_external,
            y_pred_external
        )

        external_report = classification_report(
            y_true_external,
            y_pred_external,
            labels=[0, 1],
            target_names=["Benign", "Pathogenic"],
            zero_division=0
        )

        external_cm = confusion_matrix(
            y_true_external,
            y_pred_external,
            labels=[0, 1]
        )

        print("\nEXTERNAL CLASSIFICATION REPORT")
        print(external_report)

        print("Correct:", correct_count)
        print("Incorrect:", incorrect_count)

        print(
            "External accuracy:",
            round(external_accuracy, 4)
        )

        if y_true_external.nunique() == 2:

            external_roc_auc = roc_auc_score(
                y_true_external,
                y_prob_external
            )

            external_pr_auc = average_precision_score(
                y_true_external,
                y_prob_external
            )

            print(
                "External ROC-AUC:",
                round(external_roc_auc, 4)
            )

            print(
                "External PR-AUC:",
                round(external_pr_auc, 4)
            )

        else:

            print(
                "External ROC-AUC and PR-AUC cannot be "
                "computed reliably because only one "
                "ground-truth class is present."
            )

        results["ground_truth_label"] = (
            external_labels.to_numpy()
        )

        results["correct_prediction"] = np.nan

        results.loc[
            labelled_mask,
            "correct_prediction"
        ] = (
            y_true_external.to_numpy()
            == y_pred_external
        ).astype(int)

    else:

        print(
            "\nNO EXTERNAL ACCURACY AVAILABLE."
        )

        print(
            "The external dataset has no known binary labels."
        )

        print(
            "Predictions are generated, but their correctness "
            "cannot be verified without independent labels."
        )

        results["ground_truth_label"] = np.nan
        results["correct_prediction"] = np.nan

else:

    print(
        "\nThe external dataset has no 'label' column."
    )

    print(
        "External accuracy cannot be calculated."
    )

    results["ground_truth_label"] = np.nan
    results["correct_prediction"] = np.nan


# ============================================================
# SAVE EXTERNAL PREDICTIONS
# ============================================================

predictions_path = out(
    "CLN3_gnomAD_validation_predictions.csv"
)

results.to_csv(
    predictions_path,
    index=False
)

print("\nPredictions saved:")
print(predictions_path)


# ============================================================
# EXTERNAL CONFUSION MATRIX
# Only available with known labels.
# ============================================================

if external_cm is not None:

    disp = ConfusionMatrixDisplay(
        confusion_matrix=external_cm,
        display_labels=["Benign", "Pathogenic"]
    )

    disp.plot()

    plt.title(
        "CLN3 gnomAD External Validation Confusion Matrix"
    )

    plt.savefig(
        out("CLN3_gnomAD_validation_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    with open(
        out("CLN3_gnomAD_validation_classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as handle:
        handle.write(external_report)

    print("External confusion matrix saved.")


# ============================================================
# INTERNAL CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=internal_cm,
    display_labels=["Benign", "Pathogenic"]
)

disp.plot()

plt.title(
    f"CLN3 Internal Validation Confusion Matrix "
    f"(threshold={best_threshold:.3f})"
)

plt.savefig(
    out("CLN3_gnomAD_internal_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# INTERNAL ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_val,
    y_prob,
    name="CLN3 XGBoost"
)

plt.title("CLN3 Internal Validation ROC Curve")

plt.savefig(
    out("CLN3_gnomAD_internal_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

importance_df = pd.DataFrame({
    "Feature": FINAL_FEATURES,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print(importance_df.head(20).to_string(index=False))

importance_df.to_csv(
    out("CLN3_gnomAD_validation_feature_importance.csv"),
    index=False
)


# ============================================================
# OPTIONAL SHAP
# ============================================================

if USE_SHAP:

    try:

        import shap

        print("\nRunning SHAP...")

        shap_sample = (
            X_val
            if len(X_val) <= 500
            else X_val.sample(
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
            "CLN3 gnomAD Validation Model SHAP Summary"
        )

        plt.tight_layout()

        plt.savefig(
            out("CLN3_gnomAD_validation_shap_summary.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("SHAP plot saved.")

    except Exception as exc:

        print("SHAP failed:", exc)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": FINAL_FEATURES,
        "biological_features": available_features,
        "threshold": best_threshold,
        "scale_pos_weight": scale_pos_weight,
        "training_file": TRAIN_FILE,
        "external_file": EXTERNAL_FILE,
        "label_mapping": {
            0: "BENIGN",
            1: "PATHOGENIC"
        }
    },
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)


# ============================================================
# SAVE METRICS SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Gene": "CLN3",
    "Training_dataset": TRAIN_FILE,
    "External_dataset": EXTERNAL_FILE,

    "Internal_Accuracy": internal_accuracy,
    "Internal_ROC_AUC": internal_roc_auc,
    "Internal_PR_AUC": internal_pr_auc,
    "CV_ROC_AUC": cv_mean,

    "External_Accuracy": external_accuracy,
    "External_ROC_AUC": external_roc_auc,
    "External_PR_AUC": external_pr_auc,

    "Decision_Threshold": best_threshold,

    "Training_Rows": len(X_train),
    "Internal_Validation_Rows": len(X_val),
    "External_Candidate_Rows": len(results),

    "External_Labelled_Rows": external_evaluated,

    "External_Predicted_Pathogenic": int(
        (external_pred == 1).sum()
    ),

    "External_Predicted_Benign": int(
        (external_pred == 0).sum()
    ),

    "External_Accuracy_Available": bool(
        external_evaluated > 0
    ),

    "Biological_Feature_Count": len(available_features),
    "Final_Feature_Count": len(FINAL_FEATURES)
}])

summary_path = out(
    "CLN3_gnomAD_validation_results_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nResults summary saved:", summary_path)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLN3 GNOMAD VALIDATION COMPLETED")
print("=" * 70)

print("\nINTERNAL VALIDATION")
print("Accuracy:", round(internal_accuracy, 4))
print("ROC-AUC:", round(internal_roc_auc, 4))
print("PR-AUC:", round(internal_pr_auc, 4))
print("Mean CV ROC-AUC:", round(cv_mean, 4))

print("\nEXTERNAL CANDIDATES")
print("Total:", len(results))
print("Predicted benign:", int((external_pred == 0).sum()))
print("Predicted pathogenic:", int((external_pred == 1).sum()))

if external_evaluated > 0:
    print("Externally evaluated:", external_evaluated)
    print("External accuracy:", round(external_accuracy, 4))
else:
    print(
        "External accuracy: NOT AVAILABLE "
        "(no known ground-truth labels)"
    )

print("\nOutput directory:", BASE_DIR)
print("Model:", MODEL_PATH)
print("Predictions:", predictions_path)
print("Summary:", summary_path)

print("\nValidation process finished.")

gc.collect()
