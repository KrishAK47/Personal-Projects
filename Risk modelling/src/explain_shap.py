"""
explain_shap.py
-----------------
Loads the trained pipeline (preprocessor + tree model) and produces SHAP
explainability artifacts:
  1. Global summary (beeswarm) plot -- which features drive default risk
  2. Global mean |SHAP value| bar plot -- ranked feature importance
  3. Dependence plots for the top 3 drivers
  4. A local force/waterfall-style explanation for a single flagged defaulter

All figures are saved to reports/figures/ for use in the README / notebook.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def get_feature_names(preprocessor):
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


def main():
    df = pd.read_csv("data/loan_data.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pipeline = joblib.load("models/best_model.joblib")
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    # Use a sample for speed
    sample = X.sample(n=2000, random_state=42)
    X_transformed = preprocessor.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = get_feature_names(preprocessor)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=sample.index)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed_df)

    # XGBoost/sklearn binary classifiers can return either a single array
    # or a list [class0, class1] depending on version -- normalize to the
    # "positive class" (default) SHAP values.
    if isinstance(shap_values, list):
        shap_values_default = shap_values[1]
    else:
        shap_values_default = shap_values

    # ---- 1. Global summary (beeswarm) plot ---------------------------------
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values_default, X_transformed_df, show=False, max_display=15)
    plt.title("SHAP Summary — Drivers of Predicted Default Risk")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 2. Global mean |SHAP| bar plot ------------------------------------
    plt.figure(figsize=(8, 7))
    shap.summary_plot(
        shap_values_default, X_transformed_df, plot_type="bar", show=False, max_display=15
    )
    plt.title("Mean |SHAP value| — Feature Importance Ranking")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 3. Dependence plots for top 3 drivers -----------------------------
    mean_abs_shap = np.abs(shap_values_default).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False).head(3)
    print("Top SHAP drivers:", list(top_features.index))

    for feat in top_features.index:
        plt.figure(figsize=(6.5, 5))
        shap.dependence_plot(
            feat, shap_values_default, X_transformed_df, show=False, interaction_index=None
        )
        plt.title(f"SHAP Dependence — {feat}")
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_")
        plt.savefig(f"{FIG_DIR}/shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- 4. Local explanation for a single high-risk, flagged defaulter ----
    proba = pipeline.predict_proba(sample)[:, 1]
    sample_with_proba = sample.copy()
    sample_with_proba["proba"] = proba
    sample_with_proba["actual_default"] = df.loc[sample.index, TARGET]

    # Pick a true defaulter that the model scored as high risk
    candidates = sample_with_proba[
        (sample_with_proba["actual_default"] == 1) & (sample_with_proba["proba"] > 0.7)
    ]
    if len(candidates) == 0:
        candidates = sample_with_proba.sort_values("proba", ascending=False)
    example_idx = candidates.index[0]
    example_pos = sample.index.get_loc(example_idx)

    plt.figure(figsize=(9, 5))
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.shape(expected_value)) else expected_value
    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values_default[example_pos],
        feature_names=feature_names,
        max_display=12,
        show=False,
    )
    plt.title(
        f"Local Explanation — Customer {df.loc[example_idx, 'customer_id']} "
        f"(predicted risk = {proba[example_pos]:.1%})"
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_local_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved SHAP artifacts to reports/figures/")
    print(f"Example explained customer: {df.loc[example_idx, 'customer_id']} "
          f"(predicted risk={proba[example_pos]:.1%}, actual_default={df.loc[example_idx, TARGET]})")


if __name__ == "__main__":
    main()
