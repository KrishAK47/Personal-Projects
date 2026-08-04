"""
03_upgrade_propensity_model.py

Trains a free->premium upgrade-propensity model on customer-level features
(tenure, engagement, support contacts, acquisition channel, segment), the
same shape of model used to prioritize lifecycle/CRM outreach.

Run:
    python 03_upgrade_propensity_model.py
Outputs:
    output/propensity_model.joblib
    output/model_metrics.txt
    output/top_upgrade_candidates.csv
    output/feature_importance.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "customers.csv")

NUM_FEATS = ["tenure_days", "sessions_last_30d", "support_contacts_90d"]
CAT_FEATS = ["channel_acquired", "segment"]
TARGET = "converted_upgrade"

X = df[NUM_FEATS + CAT_FEATS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
], remainder="passthrough")

clf = Pipeline([
    ("prep", preprocess),
    ("model", GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3)),
])

clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
auc = roc_auc_score(y_test, proba)
report = classification_report(y_test, pred)

print("=" * 70)
print("UPGRADE PROPENSITY MODEL")
print("=" * 70)
print(f"Test AUC: {auc:.3f}\n")
print(report)

with open(OUT_DIR / "model_metrics.txt", "w") as f:
    f.write(f"Test AUC: {auc:.3f}\n\n")
    f.write(report)

# Feature importance (post-encoding names)
feat_names = clf.named_steps["prep"].get_feature_names_out()
importances = clf.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
    "importance", ascending=False
)
print("\nTop features:")
print(imp_df.head(10).to_string(index=False))

imp_df.head(10).plot(kind="barh", x="feature", y="importance", legend=False, figsize=(7, 5))
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Upgrade propensity: top drivers")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
plt.close()

# Score the full population and export a ranked outreach list
df["upgrade_probability"] = clf.predict_proba(X)[:, 1]
top_candidates = (
    df.sort_values("upgrade_probability", ascending=False)
    .head(500)[["customer_id", "segment", "channel_acquired", "tenure_days",
                "sessions_last_30d", "support_contacts_90d", "upgrade_probability"]]
)
top_candidates.to_csv(OUT_DIR / "top_upgrade_candidates.csv", index=False)

joblib.dump(clf, OUT_DIR / "propensity_model.joblib")

print(f"\nSaved: {OUT_DIR / 'propensity_model.joblib'}")
print(f"Saved: {OUT_DIR / 'top_upgrade_candidates.csv'} (top 500 customers to target)")
print(f"Saved: {OUT_DIR / 'feature_importance.png'}")
