import json
import requests

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# --------- Tool that fetches real-time rain probability ----------
@tool
def get_weather_for_location(city: str) -> str:
    """
    Get real-time weather data for a city using Open-Meteo API.
    Returns a JSON string with at least:
      - city
      - latitude
      - longitude
      - rain_probability_percent (max probability in next 24h)
      - raw_notes (any extra info)
    """
    # 1. Geocode city name to lat/long
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    try:
        geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to get location for city '{city}': {e}"
        })

    if not geo_data.get("results"):
        return json.dumps({
            "error": f"City not found: {city}"
        })

    first = geo_data["results"][0]
    lat = first["latitude"]
    lon = first["longitude"]
    resolved_name = first.get("name", city)
    country = first.get("country", "")

    # 2. Use Open-Meteo forecast API to get precipitation probability for next 24h
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation_probability",
        "forecast_days": 1,
        "timezone": "auto",
    }

    try:
        forecast_resp = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to get forecast for city '{city}': {e}"
        })

    hourly = forecast_data.get("hourly", {})
    probs = hourly.get("precipitation_probability", [])

    if not probs:
        return json.dumps({
            "error": "No precipitation probability data available.",
            "city": resolved_name,
            "country": country
        })

    # For simplicity: use the maximum probability in the next 24 hours
    max_prob = max(probs)

    result = {
        "city": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "rain_probability_percent": max_prob,
        "raw_notes": (
            "rain_probability_percent is the maximum precipitation probability "
            "over the next 24 hours from the Open-Meteo hourly forecast."
        ),
    }

    return json.dumps(result)


# --------- Prompt user for city name ----------
city = input("Enter a city name (e.g., Mumbai, Delhi, London): ").strip()

if not city:
    raise SystemExit("No city provided. Please run again and enter a valid city name.")


# --------- Initialize Chat Model (OpenAI) ----------
# Make sure you have OPENAI_API_KEY set in your environment:
#   export OPENAI_API_KEY="sk-proj-..."
model = init_chat_model(
    "gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0,
)


# --------- Create Agent using model + weather tool ----------
agent = create_agent(
    model,
    tools=[get_weather_for_location],
    system_prompt=(
        "You are a helpful weather assistant.\n"
        "You have a tool called 'get_weather_for_location' that returns JSON with "
        "the field 'rain_probability_percent' (0-100) for the next 24 hours.\n"
        "Your job is:\n"
        "1. Call this tool with the city name.\n"
        "2. Read the JSON result.\n"
        "3. If there is an 'error', explain it clearly to the user.\n"
        "4. Otherwise, explain the probability of rain in simple, user-friendly "
        "language, like: 'There is about a 60% chance of rain today in CITY.'\n"
        "5. Optionally give 1–2 short suggestions (e.g., whether to carry an umbrella)."
    ),
)


# --------- Ask Agent: probability of rain for the given city ----------
user_message = f"What is the probability it will rain today in {city}?"

result = agent.invoke(
    {"messages": [{"role": "user", "content": user_message}]}
)

# Grab the last AI message
final_message = result["messages"][-1]

print("\n🤖 AI Agent Response:")
print(final_message.content)
