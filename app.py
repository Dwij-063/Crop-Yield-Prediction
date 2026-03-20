from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")
le_state = joblib.load("le_state.pkl")

ACRES_TO_HA = 0.404686


@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    try:
        # 🔹 Encode categorical
        crop   = le_crop.transform([data["crop"]])[0]
        season = le_season.transform([data["season"]])[0]
        state  = le_state.transform([data["state"]])[0]

        # 🔹 Convert numeric inputs
        area_acres = float(data["area"])
        rainfall   = float(data["rainfall"])
        fertilizer = float(data["fertilizer"])
        pesticide  = float(data["pesticide"])

        crop_year = 2026   # FIXED (same as training)

        # 🔥 Convert acres → hectares
        area_ha = area_acres * ACRES_TO_HA

        # 🔥 CORRECT feature order (VERY IMPORTANT)
        features = pd.DataFrame([[
            crop, crop_year, season, state,
            area_ha, rainfall, fertilizer, pesticide
        ]], columns=[
            "crop", "crop_year", "season", "state",
            "area", "annual_rainfall", "fertilizer", "pesticide"
        ])

        # Scale
        scaled = scaler.transform(features)

        # Predict
        yield_per_hectare = model.predict(scaled)[0]

        # 🔥 SAME AS predict.py
        yield_per_acre = yield_per_hectare / 2.47105
        total_yield = yield_per_hectare * area_ha

        return {
            "crop": data["crop"],
            "season": data["season"],
            "state": data["state"],
            "year": crop_year,
            "yield_per_acre_tonnes": round(float(yield_per_acre), 4),
            "yield_per_hectare_tonnes": round(float(yield_per_hectare), 4),
            "total_yield_tonnes": round(float(total_yield), 2)
        }

    except Exception as e:
        return {"error": str(e)}