"""
predict.py  —  Interactive input script to test the trained crop yield model.
               Area input is in ACRES (converted to hectares internally).

Usage:
    python predict.py
"""

import joblib
import numpy as np
import pandas as pd
import sys

# ── Load saved artefacts ───────────────────────────────────────────────────────
try:
    model     = joblib.load("crop_yield_model.pkl")
    le_crop   = joblib.load("le_crop.pkl")
    le_season = joblib.load("le_season.pkl")
    le_state  = joblib.load("le_state.pkl")
    scaler    = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Make sure you have run prepare_dataset.py and train.py first.")
    sys.exit(1)

CROPS   = sorted(le_crop.classes_.tolist())
SEASONS = sorted(le_season.classes_.tolist())
STATES  = sorted(le_state.classes_.tolist())

ACRES_TO_HA = 0.404686   # 1 acre = 0.404686 hectares

# ── Helpers ────────────────────────────────────────────────────────────────────
def print_options(label, options):
    print(f"\n  Available {label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i:>2}. {opt}")

def pick_from_list(label, options):
    """Let user type a number OR the name directly."""
    print_options(label, options)
    while True:
        raw = input(f"\n  Enter {label} name or number: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"  Please enter a number between 1 and {len(options)}.")
            continue
        match = next((o for o in options if o.lower() == raw.lower()), None)
        if match:
            return match
        print(f"  '{raw}' not found. Please choose from the list.")

def get_float(prompt, hint=""):
    full_prompt = f"  {prompt}"
    if hint:
        full_prompt += f"  [e.g. {hint}]"
    full_prompt += " : "
    while True:
        raw = input(full_prompt).strip()
        try:
            val = float(raw)
            if val < 0:
                print("  Value must be 0 or greater.")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")

def predict(crop, crop_year, season, state, area_ha, rainfall, fertilizer, pesticide):
    enc_crop   = le_crop.transform([crop])[0]
    enc_season = le_season.transform([season])[0]
    enc_state  = le_state.transform([state])[0]

    feature_names = ["crop", "crop_year", "season", "state",
                     "area", "annual_rainfall", "fertilizer", "pesticide"]
    features = pd.DataFrame(
        [[enc_crop, crop_year, enc_season, enc_state,
          area_ha, rainfall, fertilizer, pesticide]],
        columns=feature_names
    )
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]

# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("       CROP YIELD PREDICTION — India (2026)")
    print("=" * 60)
    print("  Model : Random Forest")
    print("  Crops : 55   |   States : 30   |   Seasons : 6")
    print("  Area input is in ACRES")

    while True:
        print("\n" + "-" * 60)

        crop   = pick_from_list("Crop",   CROPS)
        season = pick_from_list("Season", SEASONS)
        state  = pick_from_list("State",  STATES)
        year   = 2026   # fixed prediction year

        print("\n  Enter agricultural inputs:")
        area_acres  = get_float("Area                (acres)", hint="17,000")
        rainfall    = get_float("Annual Rainfall     (mm)",    hint="1200")
        fertilizer  = get_float("Fertilizer used     (kg)",    hint="500000")
        pesticide   = get_float("Pesticide used      (kg)",    hint="2000")

        # Convert acres -> hectares for the model
        area_ha = area_acres * ACRES_TO_HA

        yield_per_ha    = predict(crop, year, season, state,
                                  area_ha, rainfall, fertilizer, pesticide)

        # Also express yield in acres for the user
        yield_per_acre  = yield_per_ha / 2.47105   # 1 ha = 2.47105 acres
        total_yield     = yield_per_ha * area_ha    # total tonnes

        print("\n" + "=" * 60)
        print("  PREDICTION RESULT")
        print("=" * 60)
        print(f"  Crop             : {crop}")
        print(f"  Year             : {year}")
        print(f"  Season           : {season}")
        print(f"  State            : {state}")
        print(f"  Area             : {area_acres:>12,.1f} acres")
        print(f"  Annual Rainfall  : {rainfall:>12,.1f} mm")
        print(f"  Fertilizer       : {fertilizer:>12,.1f} kg")
        print(f"  Pesticide        : {pesticide:>12,.1f} kg")
        yield_kg_per_acre = yield_per_acre * 1000
        yield_kg_per_ha   = yield_per_ha   * 1000
        total_yield_kg    = total_yield    * 1000

        print("-" * 60)
        print(f"  Yield per Acre   : {yield_per_acre:>12.4f}  tonnes / acre"
              f"   ({yield_kg_per_acre:>10,.1f} kg / acre)")
        print(f"  Yield per Hectare: {yield_per_ha:>12.4f}  tonnes / hectare"
              f"   ({yield_kg_per_ha:>10,.1f} kg / hectare)")
        print(f"  Total Yield      : {total_yield:>12.2f}  tonnes"
              f"            ({total_yield_kg:>10,.1f} kg)")
        print("=" * 60)

        again = input("\n  Run another prediction? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break

if __name__ == "__main__":
    main()