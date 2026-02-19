import pandas as pd
import random

# -----------------------------
# Configuration
# -----------------------------
INPUT_PATH = "data/Train_stations.csv"
OUTPUT_PATH = "data/stations_data.csv"

MIN_TRACK_AGE = 8     # years
MAX_TRACK_AGE = 35    # years

# -----------------------------
# Load raw station data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# Keep only required columns
# -----------------------------
df = df[
    [
        "station_name",
        "station_code",
        "state_name",
        "lat",
        "lng"
    ]
]

# -----------------------------
# Handle missing state names
# -----------------------------
df["state_name"] = df["state_name"].fillna("Unknown")

# -----------------------------
# Generate synthetic track age
# -----------------------------
def generate_track_age():
    return random.randint(MIN_TRACK_AGE, MAX_TRACK_AGE)

df["track_age_years"] = df.apply(lambda _: generate_track_age(), axis=1)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Station preprocessing completed")
print(f"📁 Saved clean data to: {OUTPUT_PATH}")
print(df.head())
