import os
import json
import requests

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# ---------------------- Shared: Geocoding via Open-Meteo ----------------------


def geocode_city_open_meteo(city: str):
    """
    Use Open-Meteo geocoding API to convert city name -> (lat, lon, name, country).
    No API key required.
    """
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    resp = requests.get(geo_url, params=geo_params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        raise ValueError(f"City not found: {city}")

    first = data["results"][0]
    lat = first["latitude"]
    lon = first["longitude"]
    resolved_name = first.get("name", city)
    country = first.get("country", "")

    return lat, lon, resolved_name, country


# ---------------------- Tool 1: 15-day weather forecast ----------------------


@tool
def get_weather_for_location(city: str) -> str:
    """
    Get a 15-day daily weather forecast for a given city using Open-Meteo.

    Returns a JSON string with fields:
      - city, country, latitude, longitude
      - days: list of objects with:
          - date
          - temp_max_c
          - temp_min_c
          - precipitation_probability_max
          - precipitation_sum_mm
    """
    try:
        lat, lon, resolved_name, country = geocode_city_open_meteo(city)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to geocode city '{city}': {e}"
        })

    # Open-Meteo Weather Forecast API: up to 16 days; we’ll request 15.
    # We'll use daily:
    #   - temperature_2m_max / _min
    #   - precipitation_probability_max
    #   - precipitation_sum
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
        ]),
        "forecast_days": 15,
        "timezone": "auto",
    }

    try:
        f_resp = requests.get(forecast_url, params=forecast_params, timeout=10)
        f_resp.raise_for_status()
        f_data = f_resp.json()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch weather forecast for '{city}': {e}"
        })

    daily = f_data.get("daily", {})
    dates = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    pop_max = daily.get("precipitation_probability_max", [])
    precip_sum = daily.get("precipitation_sum", [])

    if not dates:
        return json.dumps({
            "error": "No daily forecast data returned.",
            "city": resolved_name,
            "country": country
        })

    days = []
    for i, date in enumerate(dates):
        days.append({
            "date": date,
            "temp_max_c": t_max[i] if i < len(t_max) else None,
            "temp_min_c": t_min[i] if i < len(t_min) else None,
            "precipitation_probability_max": pop_max[i] if i < len(pop_max) else None,
            "precipitation_sum_mm": precip_sum[i] if i < len(precip_sum) else None,
        })
    
    print("\n🤖 Weather forecast:\n")
    print(days)
    
    result = {
        "city": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "days": days,
        "notes": (
            "Forecast from Open-Meteo: 15 days of daily max/min temperature, "
            "max precipitation probability, and precipitation sum."
        ),
    }

    return json.dumps(result)


# ---------------------- Tool 2: Search tourist attractions ----------------------


@tool
def search_tourist_spots(city: str) -> str:
    """
    Search top ~20 tourist attractions in the given city using OpenTripMap.

    Returns a JSON string with:
      - city, country
      - attractions: list of up to 20 objects with:
          - name
          - kinds
          - dist_m (approx distance from city center)
          - osm, xid (IDs if needed)
    Requires OPEN_TRIPMAP_API_KEY to be set in environment.
    """
    api_key = os.getenv("OPEN_TRIPMAP_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "OPEN_TRIPMAP_API_KEY is not set in environment."
        })

    # 1) Geocode city with Open-Meteo (reusing function above)
    try:
        lat, lon, resolved_name, country = geocode_city_open_meteo(city)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to geocode city '{city}': {e}"
        })

    # 2) Use OpenTripMap "places/radius" endpoint to get POIs near that coordinate
    # Docs: https://dev.opentripmap.org/ (Places list, radius search)
    # We filter for main tourist-related categories & ask for rating
    radius = 15000  # 15km around city center
    places_url = "https://api.opentripmap.com/0.1/en/places/radius"
    places_params = {
        "radius": radius,
        "lon": lon,
        "lat": lat,
        "kinds": "tourist_facilities,interesting_places,cultural,historic,architecture,natural",
        "limit": 20,
        "rate": 2,          # 1-3; 3 is best, but we allow 2+ for enough results
        "format": "json",
        "apikey": api_key,
    }

    try:
        p_resp = requests.get(places_url, params=places_params, timeout=10)
        p_resp.raise_for_status()
        p_data = p_resp.json()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch tourist spots for '{city}': {e}"
        })

    attractions = []
    for item in p_data:
        attractions.append({
            "name": item.get("name"),
            "kinds": item.get("kinds"),
            "dist_m": item.get("dist"),
            "osm": item.get("osm"),
            "xid": item.get("xid"),
        })
    
    print("\n🤖 Top 20 spots:\n")
    print(attractions)

    result = {
        "city": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "attractions": attractions,
        "notes": (
            "Attractions from OpenTripMap within ~15km radius. "
            "Use name + kinds for itinerary planning."
        ),
    }

    return json.dumps(result)


# ---------------------- Main: Build itinerary with the agent ----------------------


def main():
    city = input("Enter a city name for a 15-day trip (e.g., Rome, Tokyo, Seattle): ").strip()
    if not city:
        raise SystemExit("No city provided. Please run again and enter a valid city name.")

    # Initialize OpenAI model
    model = init_chat_model(
        "gpt-4.1-mini",
        model_provider="openai",
        temperature=0.3,  # slight creativity for itinerary
    )

    # Create agent with both tools
    agent = create_agent(
        model,
        tools=[get_weather_for_location, search_tourist_spots],
        system_prompt=(
            "You are a meticulous travel planner.\n"
            "You have two tools:\n"
            "1) get_weather_for_location(city): returns a JSON string with 15 days of daily "
            "   weather forecast for that city (dates, temp_max_c, temp_min_c, "
            "   precipitation_probability_max, precipitation_sum_mm).\n"
            "2) search_tourist_spots(city): returns a JSON string with up to 20 top attractions "
            "   in that city (name, kinds, dist_m, etc.).\n\n"
            "Your task:\n"
            "- First, call BOTH tools for the given city.\n"
            "- Parse their JSON outputs.\n"
            "- Then construct a detailed 15-day sightseeing itinerary that:\n"
            "    * Maps specific attractions to specific dates.\n"
            "    * Uses weather data intelligently: on high-rain-probability days, prefer more "
            "      indoor/covered attractions; on good-weather days, prioritize outdoor or "
            "      view-heavy spots.\n"
            "    * Keeps each day realistic: usually 2–4 attractions per day depending on type.\n"
            "    * Mentions the date and a short description per day.\n"
            "    * If there are fewer attractions than 15 days, spread them thoughtfully and "
            "      add some 'flex/relax' days with suggestions.\n"
            "- If any tool returns an error, explain the problem clearly to the user.\n"
            "Output the final answer as a clearly structured day-by-day itinerary, "
            "WITHOUT showing raw JSON or tool logs."
        ),
    )

    user_message = (
        f"Plan a detailed 15-day sightseeing itinerary for {city}. "
        "Use both the weather forecast and top tourist attractions to decide "
        "which places to visit on which days."
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    # Extract only the final assistant message content
    final_message = result["messages"][-1]

    print("\n🤖 Suggested 15-Day Itinerary:\n")
    print(final_message.content)


if __name__ == "__main__":
    main()
