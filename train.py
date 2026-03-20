# # train.py

# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_absolute_error

# # 📥 Load dataset
# df = pd.read_csv("crop_yield.csv")

# # 🧹 -------- CLEAN COLUMN NAMES (VERY IMPORTANT) --------
# df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# print("Columns:", df.columns.tolist())

# # 🧹 -------- DATA CLEANING --------
# df = df.drop_duplicates()
# df = df.dropna()

# # Clean text values (remove extra spaces)
# df['crop'] = df['crop'].str.strip()
# df['season'] = df['season'].str.strip()
# df['state'] = df['state'].str.strip()

# # ❌ Remove production if exists (safe)
# if "production" in df.columns:
#     df = df.drop("production", axis=1)

# # 🏷️ -------- ENCODING --------
# le_crop = LabelEncoder()
# le_season = LabelEncoder()
# le_state = LabelEncoder()

# df['crop'] = le_crop.fit_transform(df['crop'])
# df['season'] = le_season.fit_transform(df['season'])
# df['state'] = le_state.fit_transform(df['state'])

# # Save encoders
# joblib.dump(le_crop, "le_crop.pkl")
# joblib.dump(le_season, "le_season.pkl")
# joblib.dump(le_state, "le_state.pkl")

# # 🎯 Features & Target
# X = df.drop("yield", axis=1)
# y = df["yield"]

# # ✂️ Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 📏 Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Save scaler
# joblib.dump(scaler, "scaler.pkl")

# # 🤖 -------- MODEL --------
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 📊 Evaluation
# y_pred = model.predict(X_test)

# print("\n📊 Model Performance:")
# print("R2 Score:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))

# # 💾 Save model
# joblib.dump(model, "crop_yield_model.pkl")

# print("\n✅ Model training complete and saved!")









"""
train.py  — updated for crop_yield_extended.csv
Run AFTER prepare_dataset.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load extended dataset
df = pd.read_csv("crop_yield_extended.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["crop"]   = df["crop"].str.strip()
df["season"] = df["season"].str.strip()
df["state"]  = df["state"].str.strip()
df = df.drop_duplicates().dropna()

print(f"Dataset: {len(df):,} rows | years {df['crop_year'].min()}-{df['crop_year'].max()}")

# 2. Encode categoricals
le_crop   = LabelEncoder()
le_season = LabelEncoder()
le_state  = LabelEncoder()

df["crop"]   = le_crop.fit_transform(df["crop"])
df["season"] = le_season.fit_transform(df["season"])
df["state"]  = le_state.fit_transform(df["state"])

joblib.dump(le_crop,   "le_crop.pkl")
joblib.dump(le_season, "le_season.pkl")
joblib.dump(le_state,  "le_state.pkl")

# 3. Features & target
X = df.drop("yield", axis=1)
y = df["yield"]
print("Features:", X.columns.tolist())

# 4. Temporal train/test split
#    Train on everything up to 2017, test on 2018-2019 (real data only)
#    This gives an honest measure of how well the model predicts forward
real_test_mask = (df.index.isin(X[X["crop_year"].between(2018, 2019)].index))

X_train = X[~real_test_mask]
y_train = y[~real_test_mask]
X_test  = X[real_test_mask]
y_test  = y[real_test_mask]

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows (2018-2019 real data)")

# 5. Scale
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# 6. Train
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,   # prevents overfitting on individual year noise
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_s, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_s)
print(f"\nModel Performance (tested on 2018-2019 real data):")
print(f"  R2  Score : {r2_score(y_test, y_pred):.4f}")
print(f"  MAE       : {mean_absolute_error(y_test, y_pred):.4f}")

# 8. Save model
joblib.dump(model, "crop_yield_model.pkl")
print("\nModel saved -> crop_yield_model.pkl")

# 9. Quick 2026 sanity check
print("\n--- Sample 2026 predictions ---")
df_raw = pd.read_csv("crop_yield_extended.csv")
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(" ", "_")
df_raw["crop"]   = df_raw["crop"].str.strip()
df_raw["season"] = df_raw["season"].str.strip()
df_raw["state"]  = df_raw["state"].str.strip()

sample = df_raw[df_raw["crop_year"] == 2026].head(8).copy()
sample["crop"]   = le_crop.transform(sample["crop"])
sample["season"] = le_season.transform(sample["season"])
sample["state"]  = le_state.transform(sample["state"])

X_sample = sample.drop("yield", axis=1)
X_sample_s = scaler.transform(X_sample)
preds = model.predict(X_sample_s)

df_show = df_raw[df_raw["crop_year"] == 2026].head(8)[["crop","state","season"]].copy()
df_show["predicted_yield"] = preds.round(4)
print(df_show.to_string(index=False))
print("\nAll done!")