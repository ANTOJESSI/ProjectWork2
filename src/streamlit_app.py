import streamlit as st
import pandas as pd
import numpy as np
import folium
import tempfile
from weather_service import get_live_weather

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Rail Thermal Buckling Risk Map",
    layout="wide"
)

st.title("🚆 Rail Thermal Buckling Risk Map")

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("⚙️ Settings")

use_live_weather = st.sidebar.checkbox(
    "Use Live Weather API (slow)",
    value=False
)

st.sidebar.markdown(
    """
    **Demo Mode (Recommended):**
    - Fast
    - Stable
    - Synthetic but realistic data

    **Live Mode:**
    - Real weather
    - Slower
    - API dependent
    """
)

# ---------------------------------
# Load data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/stations_data.csv")
    df = df.dropna(subset=["lat", "lng"])

    # Add synthetic track age ONCE
    if "track_age_years" not in df.columns:
        df["track_age_years"] = np.random.randint(5, 35, size=len(df))

    return df

df = load_data()

# ---------------------------------
# Risk calculation
# ---------------------------------
def calculate_risk(temp, humidity, solar, track_age):
    temp_score = min(temp / 60, 1)
    humidity_score = min(humidity / 100, 1)
    solar_score = min(solar / 1000, 1)
    age_score = min(track_age / 40, 1)

    risk = (
        0.35 * temp_score +
        0.25 * solar_score +
        0.20 * age_score +
        0.20 * humidity_score
    )

    return round(risk, 2)

# ---------------------------------
# Build map (NOT cached)
# ---------------------------------
def build_map(df, use_live_weather):

    center_lat = df["lat"].mean()
    center_lon = df["lng"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    for _, row in df.iterrows():

        lat = row["lat"]
        lon = row["lng"]
        track_age = row["track_age_years"]

        # Weather logic
        if use_live_weather:
            weather = get_live_weather(lat, lon)

            if weather:
                temp = weather["temp"]
                humidity = weather["humidity"]
                solar = weather["solar"]
                source = "Live API"
            else:
                temp = np.random.normal(38, 3)
                humidity = np.random.normal(60, 8)
                solar = np.random.normal(850, 120)
                source = "Fallback"
        else:
            temp = np.random.normal(38, 3)
            humidity = np.random.normal(60, 8)
            solar = np.random.normal(850, 120)
            source = "Demo"

        risk = calculate_risk(temp, humidity, solar, track_age)

        if risk < 0.4:
            color = "green"
            level = "LOW"
        elif risk < 0.7:
            color = "orange"
            level = "MEDIUM"
        else:
            color = "red"
            level = "HIGH"

        popup = f"""
        <b>{row['station_name']}</b><br>
        Temperature: {temp:.1f} °C<br>
        Humidity: {humidity:.0f}%<br>
        Solar: {solar:.0f} W/m²<br>
        Track Age: {track_age} years<br>
        Weather Source: {source}<br>
        <b>Risk Level: {level}</b>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup
        ).add_to(m)

    return m

# ---------------------------------
# Render map (NO infinite refresh)
# ---------------------------------
st.subheader("🗺️ Risk Visualization")

if use_live_weather:
    st.warning("Live weather enabled — click button to fetch data")

    if st.button("🔄 Fetch Live Weather & Build Map"):
        with st.spinner("Fetching live weather..."):
            map_obj = build_map(df, True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                map_obj.save(f.name)
                st.components.v1.html(
                    open(f.name).read(),
                    height=650,
                    scrolling=False
                )

else:
    # Demo mode → instant & stable
    map_obj = build_map(df, False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        map_obj.save(f.name)
        st.components.v1.html(
            open(f.name).read(),
            height=650,
            scrolling=False
        )

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("Thermal Buckling Risk System • Demo-safe • Live API optional")
