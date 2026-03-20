from fastapi import FastAPI
import joblib
import numpy as np
from datetime import datetime

app = FastAPI()

# Load model + encoders + scaler
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")
le_state = joblib.load("le_state.pkl")


@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Expected input JSON
        # {
        #   "crop": "Cotton(lint)",
        #   "season": "Kharif",
        #   "state": "Maharashtra",
        #   "area": 20,
        #   "rainfall": 1200,
        #   "fertilizer": 200000,
        #   "pesticide": 1500
        # }

        # Encode categorical
        crop = le_crop.transform([data["crop"]])[0]
        season = le_season.transform([data["season"]])[0]
        state = le_state.transform([data["state"]])[0]

        # Numeric inputs
        area = data["area"]
        rainfall = data["rainfall"]
        fertilizer = data["fertilizer"]
        pesticide = data["pesticide"]

        year = datetime.now().year  # auto current year

        # Feature order MUST match training
        features = np.array([[year, crop, season, state, area, rainfall, fertilizer, pesticide]])

        # Scale
        scaled = scaler.transform(features)

        # Predict
        yield_per_hectare = model.predict(scaled)[0]

        # Convert like your CLI output
        yield_per_acre = yield_per_hectare / 2.471
        total_yield = yield_per_acre * area

        return {
            "crop": data["crop"],
            "season": data["season"],
            "state": data["state"],
            "year": year,

            "yield_per_acre_tonnes": round(yield_per_acre, 4),
            "yield_per_hectare_tonnes": round(yield_per_hectare, 4),
            "total_yield_tonnes": round(total_yield, 2)
        }

    except Exception as e:
        return {"error": str(e)}