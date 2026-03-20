"""
prepare_dataset.py  — tailored to crop_yield.csv (1997–2020, 19689 rows)

What this does:
  1. Cleans the dataset (strips whitespace, drops 2020 incomplete year)
  2. Extrapolates each (Crop x State x Season) group's yield + features
     up to 2026 using linear regression on its own historical trend
  3. Saves  ->  crop_yield_extended.csv

Run BEFORE train.py:
    python prepare_dataset.py
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# 1. Load & clean
df = pd.read_csv("crop_yield.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates().dropna()

# Strip trailing spaces (dataset has e.g. "Kharif     ")
df["crop"]   = df["crop"].str.strip()
df["season"] = df["season"].str.strip()
df["state"]  = df["state"].str.strip()

# Drop 2020 -- only 37 rows, very incomplete; hurts trend fitting
df = df[df["crop_year"] <= 2019]

print(f"Loaded {len(df):,} rows  |  years: {df['crop_year'].min()}-{df['crop_year'].max()}")
print(f"Crops: {df['crop'].nunique()}  |  States: {df['state'].nunique()}  |  Seasons: {df['season'].nunique()}")

FEATURE_COLS = ["area", "annual_rainfall", "fertilizer", "pesticide"]
TARGET_YEARS = list(range(2020, 2027))   # fill 2020 to 2026

# 2. Extrapolate per (crop, state, season) group
synthetic_rows = []
skipped = 0

for (crop, state, season), grp in df.groupby(["crop", "state", "season"]):
    grp = grp.sort_values("crop_year")

    if len(grp) < 4:
        skipped += 1
        continue

    X_hist = grp["crop_year"].values.reshape(-1, 1)

    lr = {"yield": LinearRegression().fit(X_hist, grp["yield"].values)}
    for col in FEATURE_COLS:
        lr[col] = LinearRegression().fit(X_hist, grp[col].values)

    for yr in TARGET_YEARS:
        row = {
            "crop":      crop,
            "crop_year": yr,
            "season":    season,
            "state":     state,
            "yield":     max(0.0, lr["yield"].predict([[yr]])[0]),
        }
        for col in FEATURE_COLS:
            row[col] = max(0.0, lr[col].predict([[yr]])[0])
        synthetic_rows.append(row)

df_synth = pd.DataFrame(synthetic_rows)
print(f"\nSynthetic rows generated: {len(df_synth):,}  (skipped {skipped} tiny groups)")
print(f"Covers years: {TARGET_YEARS[0]}-{TARGET_YEARS[-1]}")

# 3. Combine, reorder columns, save
col_order = ["crop", "crop_year", "season", "state"] + FEATURE_COLS + ["yield"]
df_extended = (
    pd.concat([df[col_order], df_synth[col_order]], ignore_index=True)
    .sort_values(["crop", "state", "crop_year"])
    .reset_index(drop=True)
)

df_extended.to_csv("crop_yield_extended.csv", index=False)

print(f"\nSaved -> crop_yield_extended.csv")
print(f"Total rows : {len(df_extended):,}")
print(f"Year range : {df_extended['crop_year'].min()}-{df_extended['crop_year'].max()}")
print(f"\nDone! Now run:  python train.py")