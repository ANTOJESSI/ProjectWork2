import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load trained model
# ----------------------------
print("📦 Loading trained XGBoost model...")
model = joblib.load("output/rail_stress_model.pkl")

# ----------------------------
# Load station data
# ----------------------------
stations = pd.read_csv("data/stations_data.csv")
stations = stations.dropna(subset=["lat", "lng"])

# ----------------------------
# Synthetic weather (demo-safe)
# ----------------------------
stations["temp"] = np.random.normal(38, 2, len(stations))
stations["humidity"] = np.random.normal(60, 5, len(stations))
stations["solar"] = np.random.normal(800, 50, len(stations))

stations.rename(
    columns={"track_age_years": "track_age"},
    inplace=True
)

features = stations[["temp", "humidity", "solar", "track_age"]]

# ----------------------------
# Predict TMSI
# ----------------------------
stations["predicted_tmsi"] = model.predict(features)

print("\n📍 Sample Predictions:\n")
print(stations[["station_name", "predicted_tmsi"]].head())
