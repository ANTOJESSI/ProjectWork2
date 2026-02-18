import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="RailSafe AI Dashboard", layout="wide")

st.title("🚆 AI-Driven Railway Track Buckling Prediction")
st.markdown("### Real-time Risk Monitoring: Chennai Suburban Line")

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('output/rail_stress_model.pkl')

model = load_model()

# 3. Sidebar for Live Simulation
st.sidebar.header("Simulate Environment")
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 20, 50, 38)
humidity = st.sidebar.slider("Humidity (%)", 10, 90, 60)
solar = st.sidebar.slider("Solar Radiation (W/m²)", 0, 1000, 800)

# 4. Locations and Live Prediction Logic
locations = [
    {"name": "Chennai Beach", "lat": 13.0913, "lon": 80.2837, "age": 35},
    {"name": "Chennai Egmore", "lat": 13.0822, "lon": 80.2599, "age": 10},
    {"name": "Mambalam", "lat": 13.0383, "lon": 80.2337, "age": 40},
    {"name": "Guindy", "lat": 13.0067, "lon": 80.2206, "age": 25},
    {"name": "Tambaram", "lat": 12.9229, "lon": 80.1273, "age": 5},
]

# 5. Create Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    m = folium.Map(location=[13.04, 80.23], zoom_start=11, tiles="cartodbpositron")
    
    table_data = []
    for loc in locations:
        # Prepare data for AI model
        input_data = pd.DataFrame({
            'temp_c': [ambient_temp],
            'humidity': [humidity],
            'solarradiation': [solar],
            'track_age': [loc['age']]
        })
        
        risk_score = model.predict(input_data)[0]
        
        # Color Logic
        if risk_score > 0.7:
            color, status = 'red', "CRITICAL"
        elif risk_score > 0.4:
            color, status = 'orange', "WARNING"
        else:
            color, status = 'green', "SAFE"
            
        folium.Marker(
            location=[loc['lat'], loc['lon']],
            popup=f"{loc['name']}: {status}",
            icon=folium.Icon(color=color)
        ).add_to(m)
        
        table_data.append({"Location": loc['name'], "Risk Index": round(risk_score, 2), "Status": status})

    # Display Map
    st_folium(m, width=800, height=500)

with col2:
    st.write("### Risk Summary Table")
    st.table(pd.DataFrame(table_data))
    
    # Simple KPI Metrics
    st.metric("Max Risk Detected", f"{max([d['Risk Index'] for d in table_data])}")





    # Streamlit = web app framework
# Turns Python into a dashboard
# Folium = map library (Leaflet.js)