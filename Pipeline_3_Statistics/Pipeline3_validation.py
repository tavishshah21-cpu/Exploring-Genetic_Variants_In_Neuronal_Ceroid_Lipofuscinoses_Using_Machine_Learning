# ============================================================
# PIPELINE 3: CLN3 GNOMAD EXTERNAL VALIDATION
#
# TRAINING:
#   Pipeline_1_Statistics/
#       CLN3_2026_CLN3_ML_READY.csv
#
# EXTERNAL CANDIDATES:
#   Pipeline_1_Validation/
#       CLN3_gnomAD_external_candidates_CLN3_ML_READY.csv
#
# OUTPUTS:
#   Pipeline_3_Statistics/
#
# MODEL:
#   Random Forest
#   Gradient Boosting
#   Extra Trees
#   Logistic Regression meta-model
#
# METRICS (matching ml_pipeline_validation.py):
#   Internal Accuracy
#   Internal ROC-AUC
#   Internal PR-AUC
#   Mean 5-fold CV ROC-AUC
#   Classification report
#   Confusion matrix
#   External prediction counts
#   External metrics if verified labels exist
#
# LABELS:
#   0 = Benign
#   1 = Pathogenic
# ============================================================

import os
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression

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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

STATISTICS_DIR = os.path.join(
    PROJECT_DIR,
    "Pipeline_1_Statistics"
)

VALIDATION_DIR = os.path.join(
    PROJECT_DIR,
    "Pipeline_1_Validation"
)

TRAIN_FILE = os.path.join(
    STATISTICS_DIR,
    "CLN3_2026_CLN3_ML_READY.csv"
)

EXTERNAL_FILE = os.path.join(
    VALIDATION_DIR,
    "CLN3_gnomAD_external_candidates_CLN3_ML_READY.csv"
)

RANDOM_STATE = 42
USE_CV = True
TEST_SIZE = 0.20

# The threshold is selected from internal validation by
# maximising F1, following ml_pipeline_validation.py.
PREDICTION_THRESHOLD = 0.50

os.makedirs(BASE_DIR, exist_ok=True)


def out(filename):
    return os.path.join(BASE_DIR, filename)


# ============================================================
# BIOLOGICAL FEATURES
# From ml_pipeline_validation.py
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
# CLEAN VALUES
# ============================================================

def clean_value(value):

    if isinstance(value, str):

        value = value.strip()

        if value in [
            "",
            ".",
            "NA",
            "NaN",
            "nan",
            "None",
            "null"
        ]:
            return np.nan

        if ";" in value:
            value = value.split(";")[0].strip()

            if value in [
                "",
                ".",
                "NA",
                "NaN",
                "nan",
                "None",
                "null"
            ]:
                return np.nan

    return value


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, require_labels=False):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n"
            "Please check the file name and folder."
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    print("\nLoaded:", path)
    print("Shape:", df.shape)

    if require_labels and "label" not in df.columns:
        raise ValueError(
            "The training dataset must contain a 'label' column."
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def clean_labels(series):

    labels = pd.to_numeric(
        series,
        errors="coerce"
    )

    return labels.where(labels.isin([0, 1]))


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, feature_columns):

    X = pd.DataFrame(index=df.index)

    for feature in feature_columns:

        if feature in df.columns:
            X[feature] = df[feature]
        else:
            X[feature] = np.nan

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

    # Missing-value indicators.
    missing = X.isna().astype(np.int8)

    missing.columns = [
        column + "_missing"
        for column in X.columns
    ]

    X = pd.concat(
        [X, missing],
        axis=1
    )

    return X


# ============================================================
# CREATE ORIGINAL STACKING MODEL
# ============================================================

def create_stacking_model():

    estimators = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False
    )

    # Fit imputation only on the training portion during each
    # validation/CV fit, avoiding imputation leakage.
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "stacking",
            stacking
        )
    ])

    return model


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE 3: CLN3 GNOMAD EXTERNAL VALIDATION")
print("=" * 70)

print("\nTraining dataset:")
print(TRAIN_FILE)

print("\nPreprocessed external dataset:")
print(EXTERNAL_FILE)

train_df = load_dataset(
    TRAIN_FILE,
    require_labels=True
)

external_df = load_dataset(
    EXTERNAL_FILE,
    require_labels=False
)

# Preserve the external candidate columns.
external_original = external_df.copy()


# ============================================================
# CLEAN TRAINING LABELS
# ============================================================

train_df["label"] = clean_labels(
    train_df["label"]
)

train_df = train_df.loc[
    train_df["label"].notna()
].copy()

train_df["label"] = train_df["label"].astype(int)

if train_df.empty:
    raise ValueError("No valid labelled training rows remain.")

if train_df["label"].nunique() != 2:
    raise ValueError(
        "Training data must contain both labels: "
        "0 = Benign, 1 = Pathogenic."
    )

y = train_df["label"].reset_index(drop=True)

print("\nTraining rows:", len(train_df))
print("Benign:", int((y == 0).sum()))
print("Pathogenic:", int((y == 1).sum()))


# ============================================================
# SELECT FEATURES
# ============================================================

available_features = [
    feature
    for feature in FEATURES
    if feature in train_df.columns
]

if not available_features:
    raise ValueError(
        "None of the expected biological features were found "
        "in the training dataset."
    )

print("\nExpected biological features:", len(FEATURES))
print("Features present in training:", len(available_features))

print("\nUsing features:")

for feature in available_features:
    print("  -", feature)

X = prepare_features(
    train_df.reset_index(drop=True),
    available_features
)

X_external = prepare_features(
    external_original.reset_index(drop=True),
    available_features
)

# Drop columns entirely missing in training.
# Apply exactly the same columns to external candidates.
usable_features = [
    column
    for column in X.columns
    if X[column].notna().any()
]

X = X[usable_features].copy()

X_external = X_external.reindex(
    columns=usable_features
)

FINAL_FEATURES = usable_features

print("\nTraining matrix:", X.shape)
print("External matrix:", X_external.shape)
print("Final feature count:", len(FINAL_FEATURES))


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
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Internal validation rows:", len(X_val))


# ============================================================
# TRAIN STACKING MODEL FOR INTERNAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING STACKING ENSEMBLE")
print("=" * 70)

validation_model = create_stacking_model()

validation_model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

cv_mean = np.nan
cv_scores = []

if USE_CV:

    print("\nRunning 5-fold cross-validation...")

    cv_model = create_stacking_model()

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
        n_jobs=1
    )

    cv_mean = float(np.mean(cv_scores))

    print("CV scores:", np.round(cv_scores, 4))
    print("Mean CV ROC-AUC:", round(cv_mean, 4))


# ============================================================
# INTERNAL VALIDATION METRICS
# Same metric style as ml_pipeline_validation.py
# ============================================================

print("\n" + "=" * 70)
print("INTERNAL VALIDATION METRICS")
print("=" * 70)

y_prob = validation_model.predict_proba(
    X_val
)[:, 1]

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

PREDICTION_THRESHOLD = best_threshold

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

internal_report = classification_report(
    y_val,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Pathogenic"],
    zero_division=0
)

print("\nClassification report:")
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

with open(
    out("Pipeline3_internal_classification_report.txt"),
    "w",
    encoding="utf-8"
) as handle:
    handle.write(internal_report)

pd.DataFrame(
    internal_cm,
    index=["Actual_Benign", "Actual_Pathogenic"],
    columns=["Predicted_Benign", "Predicted_Pathogenic"]
).to_csv(
    out("Pipeline3_internal_confusion_matrix.csv")
)


# ============================================================
# INTERNAL CONFUSION MATRIX PLOT
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=internal_cm,
    display_labels=["Benign", "Pathogenic"]
)

disp.plot()

plt.title(
    "Pipeline 3 Internal Validation Confusion Matrix "
    f"(threshold={best_threshold:.3f})"
)

plt.savefig(
    out("Pipeline3_internal_confusion_matrix.png"),
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
    name="Pipeline 3 Stacking Ensemble"
)

plt.title("Pipeline 3 Internal Validation ROC Curve")

plt.savefig(
    out("Pipeline3_internal_roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TRAIN FINAL STACKING MODEL ON ALL LABELLED 2026 DATA
# ============================================================

print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON ALL 2026 TRAINING DATA")
print("=" * 70)

final_model = create_stacking_model()

final_model.fit(
    X,
    y
)

print("Final model fitted.")


# ============================================================
# EXTERNAL GNOMAD VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL GNOMAD VALIDATION")
print("=" * 70)

print("\nExternal candidates:", len(X_external))
print("External feature matrix:", X_external.shape)

print("\nGenerating external predictions...")

external_prob = final_model.predict_proba(
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


# ============================================================
# EXTERNAL PREDICTION COUNTS
# ============================================================

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
# EXTERNAL METRICS
#
# These metrics are only calculated if verified labels exist.
# ============================================================

external_accuracy = np.nan
external_roc_auc = np.nan
external_pr_auc = np.nan
external_evaluated = 0
external_cm = None
external_report = "Ground-truth labels unavailable."
correct_count = np.nan
incorrect_count = np.nan

if "label" in external_original.columns:

    external_labels = clean_labels(
        external_original["label"]
    )

    labelled_mask = external_labels.notna().to_numpy()
    external_evaluated = int(labelled_mask.sum())

    print(
        "\nCandidates with known ground-truth labels:",
        external_evaluated
    )

    if external_evaluated > 0:

        y_true_external = external_labels.to_numpy()[
            labelled_mask
        ].astype(int)

        y_pred_external = external_pred[labelled_mask]
        y_prob_external = external_prob[labelled_mask]

        correct = y_true_external == y_pred_external

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

        if len(np.unique(y_true_external)) == 2:

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
                "computed because only one ground-truth "
                "class is present."
            )

        results["ground_truth_label"] = external_labels.to_numpy()

        results["correct_prediction"] = np.nan

        results.loc[
            labelled_mask,
            "correct_prediction"
        ] = correct.astype(int)

    else:

        print("\nNO EXTERNAL ACCURACY AVAILABLE.")
        print("No valid binary ground-truth labels were found.")

        results["ground_truth_label"] = np.nan
        results["correct_prediction"] = np.nan

else:

    print("\nThe external dataset has no 'label' column.")
    print("External accuracy cannot be calculated.")

    results["ground_truth_label"] = np.nan
    results["correct_prediction"] = np.nan


# ============================================================
# SAVE EXTERNAL PREDICTIONS
# Preserve input order; do not rank by probability.
# ============================================================

predictions_path = out(
    "Pipeline3_CLN3_gnomAD_validation_predictions.csv"
)

results.to_csv(
    predictions_path,
    index=False
)

print("\nPredictions saved:")
print(predictions_path)


# ============================================================
# EXTERNAL CONFUSION MATRIX AND REPORT
# Only when ground-truth labels exist.
# ============================================================

if external_cm is not None:

    disp = ConfusionMatrixDisplay(
        confusion_matrix=external_cm,
        display_labels=["Benign", "Pathogenic"]
    )

    disp.plot()

    plt.title("Pipeline 3 External Validation Confusion Matrix")

    plt.savefig(
        out("Pipeline3_external_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    with open(
        out("Pipeline3_external_classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as handle:
        handle.write(external_report)


# ============================================================
# SAVE METRICS SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "Gene": "CLN3",
    "Model": "Random Forest + Gradient Boosting + Extra Trees "
             "+ Logistic Regression",

    "Training_Dataset": TRAIN_FILE,
    "External_Dataset": EXTERNAL_FILE,

    "Internal_Accuracy": internal_accuracy,
    "Internal_ROC_AUC": internal_roc_auc,
    "Internal_PR_AUC": internal_pr_auc,
    "CV_ROC_AUC": cv_mean,

    "External_Accuracy": external_accuracy,
    "External_ROC_AUC": external_roc_auc,
    "External_PR_AUC": external_pr_auc,

    "Decision_Threshold": best_threshold,

    "Training_Rows": len(X),
    "Internal_Validation_Rows": len(X_val),
    "External_Candidate_Rows": len(results),

    "External_Labelled_Rows": external_evaluated,
    "External_Correct": correct_count,
    "External_Incorrect": incorrect_count,

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
    "Pipeline3_CLN3_gnomAD_validation_results_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nResults summary saved:")
print(summary_path)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

stacking_fitted = final_model.named_steps["stacking"]

importance_rows = []

for model_name, model in stacking_fitted.named_estimators_.items():

    if hasattr(model, "feature_importances_"):

        for feature, importance in zip(
            FINAL_FEATURES,
            model.feature_importances_
        ):

            importance_rows.append({
                "Model": model_name,
                "Feature": feature,
                "Importance": float(importance)
            })

importance_df = pd.DataFrame(importance_rows)

if not importance_df.empty:

    importance_df = importance_df.sort_values(
        ["Model", "Importance"],
        ascending=[True, False]
    )

    importance_df.to_csv(
        out("Pipeline3_feature_importance.csv"),
        index=False
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

joblib.dump(
    {
        "model": final_model,
        "features": FINAL_FEATURES,
        "biological_features": available_features,
        "threshold": best_threshold,
        "training_file": TRAIN_FILE,
        "external_file": EXTERNAL_FILE,
        "label_mapping": {
            0: "BENIGN",
            1: "PATHOGENIC"
        }
    },
    out("Pipeline3_CLN3_gnomAD_stacking_model.pkl")
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE 3 GNOMAD VALIDATION COMPLETED")
print("=" * 70)

print("\nINTERNAL VALIDATION")
print("Accuracy:", round(internal_accuracy, 4))
print("ROC-AUC:", round(internal_roc_auc, 4))
print("PR-AUC:", round(internal_pr_auc, 4))
print("Mean CV ROC-AUC:", round(cv_mean, 4))
print("Selected threshold:", round(best_threshold, 4))

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
print("Predictions:", predictions_path)
print("Summary:", summary_path)

print("\nValidation process finished.")

gc.collect()