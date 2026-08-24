"""
train_model.py
----------------
Loads the synthetic loan book, trains a baseline Logistic Regression and a
Random Forest / XGBoost classifier to flag likely defaulters, evaluates both
on precision / recall / F1 (with a focus on the minority "default" class),
and saves the best model + evaluation artifacts (confusion matrix,
precision-recall curve, ROC curve, threshold-tuning chart) to reports/figures.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_SEED = 42
FIG_DIR = "reports/figures"
MODEL_DIR = "models"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_length_years",
    "credit_score",
    "credit_utilization",
    "delinquencies_2yrs",
    "months_since_last_delinquency",
    "open_credit_lines",
    "existing_loans_count",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "debt_to_income_ratio",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
TARGET = "default"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate(name, y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, target_names=["Non-default", "Default"], output_dict=True)
    metrics = {
        "model": name,
        "precision_default": precision_score(y_true, y_pred),
        "recall_default": recall_score(y_true, y_pred),
        "f1_default": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "full_report": report,
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, target_names=["Non-default", "Default"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    return metrics


def main():
    df = pd.read_csv("data/loan_data.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor()

    # ---- Baseline: Logistic Regression (class_weight balanced for imbalance)
    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]
    logreg_pred = (logreg_proba >= 0.5).astype(int)
    logreg_metrics = evaluate("Logistic Regression", y_test, logreg_pred, logreg_proba)

    # ---- Main model: Random Forest (or XGBoost if available) --------------
    if HAS_XGB:
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        model_name = "XGBoost"
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
    else:
        model_name = "Random Forest"
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    main_model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    main_model.fit(X_train, y_train)
    main_proba = main_model.predict_proba(X_test)[:, 1]
    main_pred_default_thresh = (main_proba >= 0.5).astype(int)
    main_metrics_default_thresh = evaluate(
        f"{model_name} (threshold=0.50)", y_test, main_pred_default_thresh, main_proba
    )

    # ---- Threshold tuning: maximize F1 on the default class ---------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, main_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
    best_threshold = thresholds[best_idx]

    main_pred_best = (main_proba >= best_threshold).astype(int)
    main_metrics_best = evaluate(
        f"{model_name} (tuned threshold={best_threshold:.2f})",
        y_test,
        main_pred_best,
        main_proba,
    )

    # ---- Save figures ------------------------------------------------------
    import os

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Confusion matrix (tuned threshold)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(y_test, main_pred_best)
    ConfusionMatrixDisplay(cm, display_labels=["Non-default", "Default"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"{model_name} — Confusion Matrix (threshold={best_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, logreg_proba, name="Logistic Regression", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, main_proba, name=model_name, ax=ax)
    ax.scatter(
        recalls[best_idx],
        precisions[best_idx],
        color="red",
        zorder=5,
        label=f"Tuned threshold ({best_threshold:.2f})",
    )
    ax.set_title("Precision-Recall Curve — Default Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # ROC curve comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, logreg_proba, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, main_proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
    plt.close(fig)

    # Precision / Recall / F1 vs threshold
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precisions[:-1], label="Precision")
    ax.plot(thresholds, recalls[:-1], label="Recall")
    ax.plot(thresholds, f1_scores[:-1], label="F1")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"Best F1 threshold = {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Precision / Recall / F1 vs Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/threshold_tuning.png", dpi=150)
    plt.close(fig)

    # ---- Persist model + metrics -------------------------------------------
    joblib.dump(main_model, f"{MODEL_DIR}/best_model.joblib")

    all_metrics = {
        "model_name": model_name,
        "best_threshold": float(best_threshold),
        "logistic_regression": {k: v for k, v in logreg_metrics.items() if k != "full_report"},
        f"{model_name.lower().replace(' ', '_')}_default_threshold": {
            k: v for k, v in main_metrics_default_thresh.items() if k != "full_report"
        },
        f"{model_name.lower().replace(' ', '_')}_tuned_threshold": {
            k: v for k, v in main_metrics_best.items() if k != "full_report"
        },
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    print("\nSaved model to models/best_model.joblib")
    print("Saved figures to reports/figures/")
    print("Saved metrics to reports/metrics.json")

    return main_model, X_train, X_test, y_train, y_test, model_name, best_threshold


if __name__ == "__main__":
    main()
