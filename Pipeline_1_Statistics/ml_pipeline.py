import os
import gc
import pandas as pd
import numpy as np

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
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


# ==========================================
# CONFIG
# ==========================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "ML_READY.csv"
)

SAMPLE_SIZE = 400000
TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_CV = True
USE_SHAP = True

THRESHOLD = 0.40


# ==========================================
# LOAD
# ==========================================
print("\n📊 Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

if SAMPLE_SIZE:

    df = df.sample(
        SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

print(
    "Dataset:",
    df.shape
)


# ==========================================
# CLEAN
# REMOVE END
# ==========================================
print("\n🧹 Cleaning...")

drop_cols = [

    "Chr",

    "Start",

    "End",

    "Ref",

    "Alt"
]

drop_cols = [

    c

    for c in drop_cols

    if c in df.columns
]

df.drop(

    columns=drop_cols,

    inplace=True,

    errors="ignore"
)


# ==========================================
# LABEL
# ==========================================
y = df["label"]

X = df.drop(
    columns=["label"]
)


# ==========================================
# CLEAN VALUES
# ==========================================
print("\n⚙️ Numeric conversion...")

X = X.replace(
    ".",
    np.nan
)


def clean_value(v):

    if isinstance(v, str):

        if ";" in v:

            return v.split(";")[0]

    return v


X = X.map(
    clean_value
)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    "\n🧬 Biological features:",
    X.shape[1]
)


# ==========================================
# MISSING FEATURES
# ==========================================
print(
    "\n⚙️ Creating missing indicators..."
)

missing = X.isna().astype(
    np.int8
)

missing.columns = [

    c + "_missing"

    for c in X.columns
]

X = pd.concat(

    [

        X,

        missing

    ],

    axis=1
)

print(
    "Final feature count:",
    X.shape[1]
)


# ==========================================
# CLASS BALANCE
# ==========================================
print(
    "\n⚖️ Handling imbalance..."
)

pos = (
    y == 1
).sum()

neg = (
    y == 0
).sum()

scale_pos_weight = neg / pos

print(
    "scale_pos_weight:",
    round(
        scale_pos_weight,
        2
    )
)


# ==========================================
# SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    stratify=y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE
)

del df
gc.collect()


# ==========================================
# TRAIN
# ==========================================
print("\n🤖 Training XGBoost...")

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

    random_state=42,

    n_jobs=-1
)

model.fit(

    X_train,

    y_train,

    eval_set=[

        (
            X_test,
            y_test
        )

    ],

    verbose=100
)


# ==========================================
# CV
# ==========================================
cv_mean = np.nan

if USE_CV:

    print("\n🔁 Running 5-Fold CV...")

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

        random_state=42,

        n_jobs=-1
    )

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42
    )

    cv_scores = cross_val_score(

        cv_model,

        X,

        y,

        scoring="roc_auc",

        cv=cv,

        n_jobs=-1
    )

    cv_mean = cv_scores.mean()

    print(
        "\nMean CV:",
        round(
            cv_mean,
            4
        )
    )


# ==========================================
# EVALUATION
# ==========================================
print("\n📊 Evaluation")

y_prob = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_prob >
    THRESHOLD
).astype(int)

acc = accuracy_score(
    y_test,
    y_pred
)

roc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "\nAccuracy:",
    round(acc,4)
)

print(
    "ROC-AUC:",
    round(roc,4)
)

print(
    "PR-AUC:",
    round(pr_auc,4)
)


# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n📉 Confusion Matrix")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "confusion_matrix_no_end.png",
    dpi=300
)

plt.close()


# ==========================================
# FEATURE IMPORTANCE
# ==========================================
feat = pd.DataFrame({

    "Feature":

        X.columns,

    "Importance":

        model.feature_importances_
})

feat = feat.sort_values(

    "Importance",

    ascending=False
)

feat.to_csv(
    "feature_importance_no_end.csv",
    index=False
)

print(
    feat.head(30)
)


# ==========================================
# SHAP
# ==========================================
if USE_SHAP:

    try:

        import shap

        print("\n🔍 Running SHAP...")

        sample = X_test.sample(
            1000,
            random_state=42
        )

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            sample
        )

        shap.summary_plot(

            shap_values,

            sample,

            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "shap_summary_no_end.png",
            dpi=300
        )

        plt.close()

        print(
            "📊 SHAP saved"
        )

    except Exception as e:

        print(
            e
        )


summary = pd.DataFrame({

    "Accuracy":[acc],

    "ROC_AUC":[roc],

    "PR_AUC":[pr_auc],

    "CV_ROC":[cv_mean]
})

summary.to_csv(
    "xgb_results_no_end.csv",
    index=False
)

print("\n✅ Finished")