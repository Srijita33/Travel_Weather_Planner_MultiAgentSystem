"""
weather.py

Small, dependency-light helper around the Open-Meteo API.

Open-Meteo needs no API key. We use two of its free endpoints:
  1. Geocoding API  -> turn a place name into latitude/longitude
  2. Forecast API   -> get a daily weather forecast for those coordinates

This module exposes a single function, get_weather(), which is registered
as a tool for the Weather Agent in agents.py. AutoGen calls this function
directly, so its signature and docstring double as the tool description
the LLM sees.
"""

from __future__ import annotations

import datetime as dt

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo's free forecast endpoint only covers roughly the next 16 days.
MAX_FORECAST_DAYS = 16

# Rough mapping of Open-Meteo "weather codes" to human-readable conditions.
# https://open-meteo.com/en/docs (WMO Weather interpretation codes)
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Weather codes that generally make outdoor activities a bad idea.
BAD_OUTDOOR_CODES = {65, 66, 67, 75, 77, 81, 82, 86, 95, 96, 99}


def _geocode(destination: str) -> tuple[float, float, str] | None:
    """Look up latitude/longitude for a place name using Open-Meteo's
    geocoding API. Returns (lat, lon, resolved_name) or None if not found."""
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": destination, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None

    results = data.get("results") or []
    if not results:
        return None

    top = results[0]
    resolved_name = top.get("name", destination)
    country = top.get("country")
    if country:
        resolved_name = f"{resolved_name}, {country}"
    return top["latitude"], top["longitude"], resolved_name


def get_weather(destination: str, start_date: str, num_days: int) -> str:
    """Get a day-by-day weather forecast for a destination.

    Args:
        destination: City or place name, e.g. "Goa" or "Paris, France".
        start_date: Trip start date in YYYY-MM-DD format.
        num_days: Number of days of the trip (1-16).

    Returns:
        A plain-text summary of the forecast for each day of the trip,
        including max/min temperature (Celsius), precipitation chance,
        and a short condition description. If the destination cannot be
        geocoded, or the dates fall outside Open-Meteo's ~16 day forecast
        window, a clear explanatory message is returned instead of
        invented weather data.
    """
    num_days = max(1, min(int(num_days), MAX_FORECAST_DAYS))

    geo = _geocode(destination)
    if geo is None:
        return (
            f"Could not find coordinates for '{destination}' via the Open-Meteo "
            "geocoding API. Please ask the user to confirm the destination name/spelling."
        )
    lat, lon, resolved_name = geo

    try:
        start = dt.date.fromisoformat(start_date)
    except ValueError:
        return (
            f"'{start_date}' is not a valid date in YYYY-MM-DD format. "
            "Please ask the user for a valid travel start date."
        )

    today = dt.date.today()
    end = start + dt.timedelta(days=num_days - 1)
    forecast_horizon = today + dt.timedelta(days=MAX_FORECAST_DAYS)

    if end < today:
        return (
            f"The requested dates ({start} to {end}) are in the past, so no forecast "
            "is available. Please ask the user for upcoming travel dates."
        )

    if start > forecast_horizon:
        return (
            f"The trip start date ({start}) is more than {MAX_FORECAST_DAYS} days away. "
            "Open-Meteo's forecast API only covers about 16 days ahead, so no reliable "
            "day-by-day forecast can be retrieved yet. Recommend the user re-check closer "
            "to the trip, and plan with general seasonal expectations in mind for now."
        )

    # Clip the requested window to what the forecast API can actually return.
    query_start = max(start, today)
    query_end = min(end, forecast_horizon)

    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ",".join(
                    [
                        "weathercode",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                    ]
                ),
                "timezone": "auto",
                "start_date": query_start.isoformat(),
                "end_date": query_end.isoformat(),
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"Open-Meteo forecast request failed: {exc}"

    daily = data.get("daily")
    if not daily or "time" not in daily:
        return "Open-Meteo returned no forecast data for this location and date range."

    lines = [f"Weather forecast for {resolved_name} ({query_start} to {query_end}):"]
    any_bad_day = False
    for i, date_str in enumerate(daily["time"]):
        code = daily["weathercode"][i]
        tmax = daily["temperature_2m_max"][i]
        tmin = daily["temperature_2m_min"][i]
        rain_chance = daily.get("precipitation_probability_max", [None])[i]
        condition = WEATHER_CODES.get(code, "Unknown conditions")
        if code in BAD_OUTDOOR_CODES:
            any_bad_day = True
        rain_txt = f", {rain_chance}% chance of rain" if rain_chance is not None else ""
        lines.append(
            f"- {date_str}: {condition}, {tmin}-{tmax}C{rain_txt}"
        )

    if start < today or end > forecast_horizon:
        lines.append(
            "Note: part of the requested trip window is outside Open-Meteo's ~16-day "
            "forecast range, so only the dates above could be retrieved."
        )

    lines.append(
        "Outdoor-activity suitability: "
        + ("Some days have rain/storms/snow - plan indoor alternatives for those days."
           if any_bad_day else
           "Conditions look generally favorable for outdoor activities.")
    )

    return "\n".join(lines)
