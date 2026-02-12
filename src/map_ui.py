import folium
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained AI model
model = joblib.load('output/rail_stress_model.pkl')

# 2. Simulate Chennai Railway Coordinates (Points along a track)
# These represent specific GPS locations on the Chennai suburban line
locations = [
    {"name": "Chennai Beach", "lat": 13.0913, "lon": 80.2837},
    {"name": "Chennai Egmore", "lat": 13.0822, "lon": 80.2599},
    {"name": "Mambalam", "lat": 13.0383, "lon": 80.2337},
    {"name": "Guindy", "lat": 13.0067, "lon": 80.2206},
    {"name": "Tambaram", "lat": 12.9229, "lon": 80.1273},
]

# 3. Simulate "Current" Weather Data for these points
# In a real system, this would come from a live API
current_weather = pd.DataFrame({
    'temp_c': [38, 37, 40, 39, 36], # High summer temps
    'humidity': [60, 62, 55, 58, 65],
    'solarradiation': [800, 750, 900, 850, 700],
    'track_age': [35, 10, 40, 25, 5] # Years
})

# 4. Use AI to Predict Risk (TMSI)
predictions = model.predict(current_weather)

# 5. Create the Interactive Map
m = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="cartodbpositron")

for i, loc in enumerate(locations):
    risk_score = predictions[i]
    
    # Define color based on AI risk prediction
    if risk_score > 0.7:
        color = 'red'
        status = "CRITICAL: High Risk of Buckling"
    elif risk_score > 0.4:
        color = 'orange'
        status = "WARNING: Moderate Thermal Stress"
    else:
        color = 'green'
        status = "SAFE: Normal Operations"
    
    # Add Marker to Map
    folium.Marker(
        location=[loc['lat'], loc['lon']],
        popup=f"<b>{loc['name']}</b><br>Status: {status}<br>AI Risk Index: {risk_score:.2f}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)

# Save the map
m.save('output/rail_safety_map.html')
print("Success! Interactive map generated at output/rail_safety_map.html")