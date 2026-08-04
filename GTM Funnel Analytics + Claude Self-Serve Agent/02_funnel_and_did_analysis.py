"""
02_funnel_and_did_analysis.py

Two analyses, mirroring a typical GTM data science workflow:

  A) Top-of-funnel channel analysis: traffic -> signup -> activation ("start")
     conversion rates by paid channel and customer segment, to identify
     where to shift budget.

  B) Difference-in-differences (DiD) evaluation of a redesigned upsell flow:
     did the treatment group's free->premium conversion rate move more than
     the control group's, after launch, beyond what secular trend explains?

Run:
    python 02_funnel_and_did_analysis.py
Outputs:
    output/channel_funnel_summary.csv
    output/channel_funnel_chart.png
    output/did_regression_summary.txt
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# A) Channel funnel analysis
# ---------------------------------------------------------------------------
funnel = pd.read_csv(DATA_DIR / "gtm_funnel.csv")

summary = (
    funnel.groupby(["channel", "segment"], as_index=False)
    .agg(traffic=("traffic", "sum"), signups=("signups", "sum"), starts=("starts", "sum"))
)
summary["traffic_to_signup"] = (summary["signups"] / summary["traffic"]).round(4)
summary["signup_to_start"] = (summary["starts"] / summary["signups"]).round(4)
summary["traffic_to_start"] = (summary["starts"] / summary["traffic"]).round(4)
summary = summary.sort_values("traffic_to_start", ascending=False)
summary.to_csv(OUT_DIR / "channel_funnel_summary.csv", index=False)

print("=" * 70)
print("A) CHANNEL FUNNEL SUMMARY (traffic -> signup -> start)")
print("=" * 70)
print(summary.to_string(index=False))

best = summary.iloc[0]
worst = summary.iloc[-1]
print(f"\nBest end-to-end channel: {best['channel']} / {best['segment']} "
      f"({best['traffic_to_start']:.1%} traffic->start)")
print(f"Weakest end-to-end channel: {worst['channel']} / {worst['segment']} "
      f"({worst['traffic_to_start']:.1%} traffic->start)")

# quick chart
pivot = summary.pivot_table(index="channel", columns="segment", values="traffic_to_start")
pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Traffic -> Start conversion rate")
plt.title("Top-of-funnel conversion by channel and segment")
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_funnel_chart.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# B) Difference-in-differences on the upsell flow redesign
# ---------------------------------------------------------------------------
exp = pd.read_csv(DATA_DIR / "upsell_experiment.csv")
exp["treated"] = (exp["group"] == "treatment").astype(int)
exp["post"] = (exp["period"] == "post").astype(int)
exp["treated_x_post"] = exp["treated"] * exp["post"]

model = smf.ols(
    "converted_to_premium ~ treated + post + treated_x_post + C(segment)",
    data=exp,
).fit(cov_type="HC1")  # robust SEs

print("\n" + "=" * 70)
print("B) DIFFERENCE-IN-DIFFERENCES: upsell flow redesign")
print("=" * 70)
print(model.summary())

did_coef = model.params["treated_x_post"]
did_p = model.pvalues["treated_x_post"]
verdict = "statistically significant" if did_p < 0.05 else "not statistically significant"
interpretation = (
    f"\nDiD estimate: the redesigned upsell flow is associated with a "
    f"{did_coef:+.1%} change in free->premium conversion for the treatment "
    f"group, beyond the control group's trend ({verdict}, p={did_p:.4f}).\n"
    "Read alongside the raw group means below to sanity-check the regression."
)
print(interpretation)

group_means = exp.groupby(["group", "period"])["converted_to_premium"].mean().unstack()
print(group_means)

with open(OUT_DIR / "did_regression_summary.txt", "w") as f:
    f.write(str(model.summary()))
    f.write("\n\n" + interpretation)
    f.write("\n\nRaw group means:\n")
    f.write(group_means.to_string())

print(f"\nSaved: {OUT_DIR / 'channel_funnel_summary.csv'}")
print(f"Saved: {OUT_DIR / 'channel_funnel_chart.png'}")
print(f"Saved: {OUT_DIR / 'did_regression_summary.txt'}")
