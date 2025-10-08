"""
AUDJPY carry-crash prototype
----------------------------
Build weekly AUDJPY spot & carry returns using:
  - Spot: AUDUSD, USDJPY  ->  AUDJPY_spot = AUDUSD * USDJPY
  - 1M Forward POINTS for AUDJPY (in JPY pips): convert to outright forward
  - 1M 25Δ Risk Reversal (RR) for AUDJPY: build "fear" signal from changes in RR

Then:
  - Create next-4-week (≈monthly) forward-looking return target
  - Event study on big negative moves in RR (steepening toward puts)
  - Predictive regressions with Newey–West SEs for overlap

Dependencies:
  pandas, numpy, statsmodels
"""

# ====== Imports ======
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from statsmodels.stats.sandwich_covariance import cov_hac, se_cov

# ====== USER INPUTS (file paths & column names) ===============================
# Your CSVs (already on disk)
PATH_SPOT    = "Data/Daily_Spot_Prices_G10_FX_Pairs_Daily_2000_2025.csv"
PATH_FWDPTS  = "Data/forwards.csv"                 # contains AUDJPY 1M forward POINTS (JPY pips)
PATH_RR      = "Data/RR_Data.csv"                  # contains AUDJPY 1M 25Δ RR

# Column names (as they appear in your files)
COL_AUDUSD_DATE   = "AUDUSD - Date"
COL_AUDUSD_LAST   = "AUDUSD - Last Price"
COL_USDJPY_DATE   = "USDJPY - Date"
COL_USDJPY_LAST   = "USDJPY - Last Price"

COL_AUDJPY_FWDPTS_DATE = "Date"
COL_AUDJPY_FWDPTS_VAL  = "AUDJPY1M BGN Curncy  (R1)"    # points (JPY pips), not outright forward

COL_AUDJPY_RR_DATE = "Date"
COL_AUDJPY_RR_VAL  = "AUDJPY25R1M BGN Curncy  (R2)"     # 1M 25Δ risk reversal

# Analysis controls
WEEKLY_FREQ         = "W-FRI"   # resample to Friday close
HORIZON_WEEKS       = 4         # next-4-week (≈ month) return target
WARNING_QUANTILE    = 0.10      # bottom 10% dfear as "warning"
MIN_EXPANDING_WEEKS = 52        # start z-scores after ~1 year

# ====== 1) LOAD DATA =========================================================
spot_df   = pd.read_csv(PATH_SPOT)
fwdpts_df = pd.read_csv(PATH_FWDPTS)
rr_df     = pd.read_csv(PATH_RR)

# ====== 2) BUILD AUDJPY SPOT FROM AUDUSD * USDJPY ============================
# Rename for clarity and keep only what's needed
spot_df = spot_df.rename(columns={
    COL_AUDUSD_DATE: "AUDUSD_Date",
    COL_AUDUSD_LAST: "AUDUSD",
    COL_USDJPY_DATE: "USDJPY_Date",
    COL_USDJPY_LAST: "USDJPY",
})

spot_df = spot_df[["AUDUSD_Date", "AUDUSD", "USDJPY_Date", "USDJPY"]].copy()
spot_df["Date"] = pd.to_datetime(spot_df["AUDUSD_Date"], errors="coerce")

# Basic cleaning
spot_df = spot_df.drop(columns=["AUDUSD_Date", "USDJPY_Date"])
spot_df = spot_df.dropna(subset=["Date", "AUDUSD", "USDJPY"])

# AUDJPY (JPY per AUD) = (USD per AUD) * (JPY per USD)
spot_df["AUDJPY_spot"] = spot_df["AUDUSD"].astype(float) * spot_df["USDJPY"].astype(float)

# Keep tidy weekly index
audjpy_spot_w = (
    spot_df[["Date", "AUDJPY_spot"]]
    .sort_values("Date")
    .drop_duplicates("Date")
    .set_index("Date")
    .resample(WEEKLY_FREQ)
    .last()
)

# ====== 3) CONVERT AUDJPY 1M FORWARD POINTS -> OUTRIGHT FORWARD ==============
# Your series is in JPY *pips* (e.g., -54.31 means -0.5431 JPY)
fwdpts_df = fwdpts_df.rename(columns={
    COL_AUDJPY_FWDPTS_DATE: "Date",
    COL_AUDJPY_FWDPTS_VAL:  "AUDJPY_1M_points_pips",
})
fwdpts_df["Date"] = pd.to_datetime(fwdpts_df["Date"], errors="coerce")

audjpy_pts_w = (
    fwdpts_df[["Date", "AUDJPY_1M_points_pips"]]
    .sort_values("Date")
    .drop_duplicates("Date")
    .set_index("Date")
    .resample(WEEKLY_FREQ)
    .last()
)

# Replace zeros (likely “missing after export”) with NaN BEFORE math
audjpy_pts_w["AUDJPY_1M_points_pips"] = (
    audjpy_pts_w["AUDJPY_1M_points_pips"].astype(float).replace(0.0, np.nan)
)

# Join spot & points on same weeks
audjpy_df = audjpy_spot_w.join(audjpy_pts_w, how="inner")

# Convert pips -> price units: 1 pip = 0.01 JPY (for JPY crosses)
audjpy_df["AUDJPY_1M_points_price"] = audjpy_df["AUDJPY_1M_points_pips"] / 100.0

# Outright 1M forward = spot + points_in_price_units
audjpy_df["AUDJPY_1M_forward"] = (
    audjpy_df["AUDJPY_spot"] + audjpy_df["AUDJPY_1M_points_price"]
)

# ====== 4) LOAD RR AND BUILD FEAR SIGNAL ====================================
rr_df = rr_df.rename(columns={
    COL_AUDJPY_RR_DATE: "Date",
    COL_AUDJPY_RR_VAL:  "AUDJPY_RR25D",
})
rr_df["Date"] = pd.to_datetime(rr_df["Date"], errors="coerce")

audjpy_rr_w = (
    rr_df[["Date", "AUDJPY_RR25D"]]
    .dropna()
    .sort_values("Date")
    .drop_duplicates("Date")
    .set_index("Date")
    .resample(WEEKLY_FREQ)
    .last()
)

# Merge everything & drop any remaining NaNs in critical fields
audjpy_df = (
    audjpy_df
    .join(audjpy_rr_w, how="inner")
    .dropna(subset=["AUDJPY_spot", "AUDJPY_1M_forward", "AUDJPY_RR25D"])
)

# Higher = more fear. For JPY crosses, more negative RR = puts expensive => multiply by -1
audjpy_df["fear_level"] = -audjpy_df["AUDJPY_RR25D"].astype(float)

# Expanding z-score (avoids look-ahead): start after MIN_EXPANDING_WEEKS
exp_mean = audjpy_df["fear_level"].expanding(min_periods=MIN_EXPANDING_WEEKS).mean()
exp_std  = audjpy_df["fear_level"].expanding(min_periods=MIN_EXPANDING_WEEKS).std()
audjpy_df["fear_z"] = (audjpy_df["fear_level"] - exp_mean) / exp_std

# Weekly change in fear (steepening); this is the predictive variable that usually matters
audjpy_df["dfear"] = audjpy_df["fear_z"].diff()

# ====== 5) COMPUTE WEEKLY RETURNS ============================================
# Spot weekly return (percentage)
audjpy_df["ret_spot_w"] = audjpy_df["AUDJPY_spot"].pct_change()

# Annualized carry yield from forward vs spot:
# carry_yield_annual ~ ((F - S) / S) * 12  (12 months in a year)
audjpy_df["carry_yield_annual"] = (
    (audjpy_df["AUDJPY_1M_forward"] - audjpy_df["AUDJPY_spot"])
    / audjpy_df["AUDJPY_spot"]
) * 12.0

# Weekly carry approx = annual carry / 52
audjpy_df["ret_carry_w"] = audjpy_df["carry_yield_annual"] / 52.0

# Total weekly return ≈ spot move + carry
audjpy_df["ret_total_w"] = (audjpy_df["ret_spot_w"] + audjpy_df["ret_carry_w"])

# Optional sanity check (uncomment to inspect)
# print(audjpy_df[["AUDJPY_spot","AUDJPY_1M_forward","ret_total_w"]].describe(percentiles=[.01,.05,.5,.95,.99]))

# ====== 6) TARGET = NEXT 4-WEEK (≈ MONTHLY) RETURN ===========================
# Rolling sum of next HORIZON_WEEKS returns, shifted so that predictors at t map to returns t+1..t+H
h = HORIZON_WEEKS
audjpy_df["ret_next_h"] = audjpy_df["ret_total_w"].rolling(h).sum().shift(-h+1)

# Build aligned dataset and drop NaNs from early expanding windows / diffs
aligned = audjpy_df[["ret_total_w", "fear_z", "dfear", "ret_next_h"]].dropna().copy()
aligned = aligned.rename(columns={
    "ret_total_w": "ret_w",
    "ret_next_h":  "ret_next_h",
})

# ====== 7) EVENT STUDY =======================================================
# Define "warning" as bottom WARNING_QUANTILE of dfear (largest *negative* change = steepening toward puts)
q = aligned["dfear"].quantile(WARNING_QUANTILE)
warning_mask = aligned["dfear"] <= q

evt_avg_warning   = aligned.loc[warning_mask,  "ret_next_h"].mean()
evt_hit_warning   = (aligned.loc[warning_mask,  "ret_next_h"] < 0).mean()
evt_avg_nowarning = aligned.loc[~warning_mask, "ret_next_h"].mean()

print("Event study (AUDJPY):")
print(f"  Avg next-{h}w return | warning weeks: {evt_avg_warning:.4%}")
print(f"  Hit rate (<0)        | warning weeks: {evt_hit_warning:.1%}")
print(f"  Avg next-{h}w return | non-warning  : {evt_avg_nowarning:.4%}")

# ====== 8) PREDICTIVE REGRESSIONS (Newey–West SEs for overlap) ===============
y  = aligned["ret_next_h"]

# (a) Using dfear only
X1 = add_constant(aligned[["dfear"]])
ols1 = OLS(y, X1, missing="drop").fit()
nw_cov1 = cov_hac(ols1, nlags=h-1)         # lag = horizon-1 for overlapping sums
nw_se1  = se_cov(nw_cov1)
tstats1 = ols1.params / nw_se1

print(f"\nRegression: ret_next_{h}w ~ dfear (AUDJPY)")
print("Params:\n", ols1.params)
print("Newey–West t-stats:\n", tstats1)

# (b) Level of fear + change in fear
X2 = add_constant(aligned[["fear_z", "dfear"]])
ols2 = OLS(y, X2, missing="drop").fit()
nw_cov2 = cov_hac(ols2, nlags=h-1)
nw_se2  = se_cov(nw_cov2)
tstats2 = ols2.params / nw_se2

print(f"\nRegression: ret_next_{h}w ~ fear_z + dfear (AUDJPY)")
print("Params:\n", ols2.params)
print("Newey–West t-stats:\n", tstats2)

# ====== 9) OPTIONAL: DIAGNOSTIC PRINTS =======================================
# Uncomment to quickly eyeball units & scales
# print("\nSanity check — first rows (spot vs forward):")
# print(audjpy_df[["AUDJPY_spot","AUDJPY_1M_points_pips","AUDJPY_1M_points_price","AUDJPY_1M_forward"]].head(10))
# print("\nWeekly return distribution:")
# print(aligned["ret_w"].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))