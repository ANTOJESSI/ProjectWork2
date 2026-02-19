import pandas as pd
import numpy as np
import folium

# -----------------------------
# CONFIGURATION
# -----------------------------
DEMO_MODE = True      # ← SET TO False only after everything works
API_CALL_LIMIT = 50   # safety limit if API is ON

# -----------------------------
# Load station data
# -----------------------------
df = pd.read_csv("data/stations_data.csv")

df = df.dropna(subset=["lat", "lng"])

# Synthetic track age (realistic)
df["track_age"] = np.random.randint(8, 35, size=len(df))

# -----------------------------
# Risk calculation
# -----------------------------
def calculate_risk(temp, humidity, solar, track_age):

    temp_score = min(temp / 60, 1)
    humidity_score = min(humidity / 100, 1)
    solar_score = min(solar / 1000, 1)
    age_score = min(track_age / 40, 1)

    return round(
        0.35 * temp_score +
        0.25 * solar_score +
        0.20 * age_score +
        0.20 * humidity_score,
        2
    )

# -----------------------------
# Create map
# -----------------------------
m = folium.Map(
    location=[df["lat"].mean(), df["lng"].mean()],
    zoom_start=7
)

# -----------------------------
# Plot stations
# -----------------------------
for i, row in df.iterrows():

    lat, lon = row["lat"], row["lng"]

    # -----------------------------
    # WEATHER (DEMO vs LIVE)
    # -----------------------------
    if DEMO_MODE:
        # Simulated PEAK SUMMER NOON conditions
        temp = np.random.uniform(42, 52)
        humidity = np.random.uniform(40, 70)
        solar = np.random.uniform(900, 1150)
        weather_source = "Simulated Heatwave"

    else:
        if i >= API_CALL_LIMIT:
            break  # prevent freezing

        from weather_service import get_live_weather
        weather = get_live_weather(lat, lon)

        if weather:
            temp = weather["temp"]
            humidity = weather["humidity"]
            solar = weather["solar"]
            weather_source = "Live API"
        else:
            temp = 38
            humidity = 60
            solar = 800
            weather_source = "Fallback"

    track_age = row["track_age"]

    risk = calculate_risk(temp, humidity, solar, track_age)

    if risk < 0.4:
        color, label = "green", "LOW"
    elif risk < 0.7:
        color, label = "orange", "MEDIUM"
    else:
        color, label = "red", "HIGH"

    popup = f"""
    <b>{row['station_name']}</b><br>
    Temp: {temp:.1f} °C<br>
    Humidity: {humidity:.0f}%<br>
    Solar: {solar:.0f} W/m²<br>
    Track Age: {track_age} yrs<br>
    Weather: {weather_source}<br>
    <b>Risk: {label}</b>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=popup
    ).add_to(m)

# -----------------------------
# Save output
# -----------------------------
m.save("rail_risk_map.html")
print("✅ SUCCESS: rail_risk_map.html generated")
