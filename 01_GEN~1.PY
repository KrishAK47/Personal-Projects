"""
01_generate_data.py

Generates a synthetic, publicly-shareable dataset that mirrors the shape of a
freemium SaaS growth funnel (top-of-funnel traffic -> signup -> activation,
plus a free-to-premium upsell experiment and a customer-level upgrade-propensity
table). No proprietary or employer data is used anywhere in this project --
every number below is generated from a seeded random process.

Run:
    python 01_generate_data.py
Outputs:
    data/gtm_funnel.csv         (channel x segment x week top-of-funnel metrics)
    data/upsell_experiment.csv  (customer-level DiD experiment table)
    data/customers.csv          (customer-level features for propensity model)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Top-of-funnel table: Traffic -> Signups -> Activations ("Starts")
#    by paid channel, customer segment, and week.
# ---------------------------------------------------------------------------
CHANNELS = ["paid_search", "paid_social", "display", "video", "affiliate", "organic"]
SEGMENTS = ["net_new", "retained"]
WEEKS = pd.date_range("2025-11-03", periods=16, freq="W-MON")

# Baseline conversion rates per channel (traffic->signup, signup->start),
# loosely calibrated to public digital-marketing benchmark ranges rather
# than any single company's real numbers.
channel_profile = {
    "paid_search": dict(base_traffic=9000, sig_rate=0.052, start_rate=0.61),
    "paid_social": dict(base_traffic=7000, sig_rate=0.035, start_rate=0.55),
    "display":     dict(base_traffic=12000, sig_rate=0.018, start_rate=0.48),
    "video":       dict(base_traffic=5000, sig_rate=0.028, start_rate=0.52),
    "affiliate":   dict(base_traffic=3000, sig_rate=0.061, start_rate=0.66),
    "organic":     dict(base_traffic=6000, sig_rate=0.071, start_rate=0.69),
}

rows = []
for wk in WEEKS:
    seasonal = 1 + 0.15 * np.sin(2 * np.pi * (wk.week / 52))
    for ch, prof in channel_profile.items():
        for seg in SEGMENTS:
            seg_mult = 1.0 if seg == "net_new" else 0.55
            traffic = int(prof["base_traffic"] * seg_mult * seasonal * RNG.uniform(0.9, 1.1))
            sig_rate = np.clip(prof["sig_rate"] * RNG.uniform(0.85, 1.15), 0, 1)
            signups = int(traffic * sig_rate)
            start_rate = np.clip(prof["start_rate"] * RNG.uniform(0.9, 1.1), 0, 1)
            starts = int(signups * start_rate)
            rows.append([wk.date(), ch, seg, traffic, signups, starts])

funnel = pd.DataFrame(rows, columns=["week", "channel", "segment", "traffic", "signups", "starts"])
funnel.to_csv(DATA_DIR / "gtm_funnel.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Upsell experiment table: free -> premium conversion, treatment
#    (redesigned upsell flow) vs. control, pre- vs post-launch.
#    Designed for a difference-in-differences read.
# ---------------------------------------------------------------------------
N_CUSTOMERS = 25000
LAUNCH_DATE = pd.Timestamp("2025-12-15")

cust_ids = np.arange(100000, 100000 + N_CUSTOMERS)
segment = RNG.choice(SEGMENTS, size=N_CUSTOMERS, p=[0.7, 0.3])
group = RNG.choice(["treatment", "control"], size=N_CUSTOMERS, p=[0.5, 0.5])
period = RNG.choice(["pre", "post"], size=N_CUSTOMERS, p=[0.5, 0.5])

# baseline conversion + true treatment effect only applied post-launch to treatment group
base_rate = 0.085
seg_bump = np.where(segment == "net_new", 0.01, -0.005)
true_lift = 0.043  # true DiD effect we're trying to recover
noise = RNG.normal(0, 0.01, N_CUSTOMERS)

p_convert = base_rate + seg_bump + noise
is_treated_post = (group == "treatment") & (period == "post")
p_convert = np.where(is_treated_post, p_convert + true_lift, p_convert)
p_convert = np.clip(p_convert, 0.01, 0.6)

converted = RNG.binomial(1, p_convert)

upsell = pd.DataFrame({
    "customer_id": cust_ids,
    "segment": segment,
    "group": group,
    "period": period,
    "converted_to_premium": converted,
})
upsell.to_csv(DATA_DIR / "upsell_experiment.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Customer-level table for an upgrade-propensity model.
# ---------------------------------------------------------------------------
N = 15000
tenure_days = RNG.integers(1, 900, N)
sessions_30d = RNG.poisson(4, N)
support_contacts_90d = RNG.poisson(0.4, N)
channel_acquired = RNG.choice(CHANNELS, N)
seg = RNG.choice(SEGMENTS, N, p=[0.6, 0.4])
plan = RNG.choice(["free"], N)

logit = (
    -2.1
    + 0.0022 * tenure_days
    + 0.18 * sessions_30d
    - 0.35 * support_contacts_90d
    + np.where(seg == "net_new", 0.25, -0.1)
    + np.where(np.isin(channel_acquired, ["organic", "affiliate"]), 0.3, 0.0)
    + RNG.normal(0, 0.6, N)
)
prob_upgrade = 1 / (1 + np.exp(-logit))
upgraded = RNG.binomial(1, prob_upgrade)

customers = pd.DataFrame({
    "customer_id": np.arange(200000, 200000 + N),
    "tenure_days": tenure_days,
    "sessions_last_30d": sessions_30d,
    "support_contacts_90d": support_contacts_90d,
    "channel_acquired": channel_acquired,
    "segment": seg,
    "plan": plan,
    "converted_upgrade": upgraded,
})
customers.to_csv(DATA_DIR / "customers.csv", index=False)

print("Generated:")
print(f"  data/gtm_funnel.csv         ({len(funnel):,} rows)")
print(f"  data/upsell_experiment.csv  ({len(upsell):,} rows)")
print(f"  data/customers.csv          ({len(customers):,} rows)")
