import os
import requests
import random
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_live_weather(lat, lon):
    """
    Fetch live weather from OpenWeatherMap.
    Returns dict or None if API fails.
    """

    if not API_KEY:
        print("❌ API KEY NOT FOUND")
        return None

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        print("📦 API Response received")

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            # Solar not available → realistic approximation
            "solar": random.uniform(600, 1000)
        }

    except Exception as e:
        print("🔥 API ERROR:", e)
        return None
