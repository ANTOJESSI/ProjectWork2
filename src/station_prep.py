import json
import pandas as pd
import os

def process_station_geojson():
    # Build correct path to data folder
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "data", "stations.json")
    output_path = os.path.join(base_dir, "data", "tamilnadu_stations.csv")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for feature in data["features"]:
        geometry = feature.get("geometry")
        props = feature.get("properties")

        if geometry and geometry.get("coordinates"):
            lon, lat = geometry["coordinates"]
            rows.append({
                "station": props.get("name"),
                "code": props.get("code"),
                "state": props.get("state"),
                "latitude": lat,
                "longitude": lon
            })

    df = pd.DataFrame(rows)

    # Filter Tamil Nadu stations
    df_tn = df[df["state"] == "Tamil Nadu"]

    df_tn.to_csv(output_path, index=False)

    print("Tamil Nadu stations file created successfully ✅")
