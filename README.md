# AI Travel Weather Planner Multi Agent System

A multi-agent AI travel planner built with **Microsoft AutoGen**, **Gemini**,
**Chainlit**, and the **Open-Meteo** weather API.

Three agents collaborate in real time to produce a weather-aware travel
itinerary:

- **Travel Planner Agent** - drafts and revises the itinerary based on your
  destination, dates, budget, and interests.
- **Weather Agent** - calls the Open-Meteo API for real forecast data and
  tells the Planner which days are good/bad for outdoor activities.
- **Travel Critic Agent** - reviews the itinerary for budget fit, pacing,
  interest match, conflicts, and overall practicality, and sends feedback
  back to the Planner.

The agents actually talk to each other (via an AutoGen `SelectorGroupChat`),
not three independent LLM calls stitched together.

## 🎥 Demo

[▶️ Watch the demo](https://drive.google.com/file/d/1Zb2b69YdNmAi_1SQ5RIYdNgAEiqY1-n4/view?usp=sharing)

## 🏗️ Architecture

![Architecture Diagram](https://github.com/Srijita33/Travel_Weather_Planner_MultiAgentSystem/blob/main/Architecture_Diagram_Travel_Weather_planner.jpeg)

## Project structure

```text
travel-weather-planner/
│
├── app.py              # Chainlit frontend / chat entry point
├── agents.py            # Agent definitions + team (AutoGen + Gemini)
├── weather.py            # Open-Meteo geocoding + forecast helper (tool)
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 1. Prerequisites

- Python 3.10+ installed on Windows
- A free Gemini API key from https://aistudio.google.com/app/apikey

## 2. Setup (Windows / PowerShell or CMD)

```bat
cd travel-weather-planner

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env
:: now open .env and paste in your real GEMINI_API_KEY
```

## 3. Run

```bat
chainlit run app.py -w
```

This opens the chat UI in your browser (usually at http://localhost:8000).
Type a request like:

> Plan a 4-day trip to Goa. My budget is ₹20,000. I like beaches, food and
> sightseeing. I prefer a relaxed itinerary.

You'll see live status messages as each agent works:

- 🧭 Travel Planner is creating the itinerary...
- 🌦️ Weather Agent is checking the weather...
- 🔍 Travel Critic is reviewing the itinerary...
- ✈️ Final Travel Plan

## Notes

- Open-Meteo needs no API key and is called directly from `weather.py`.
- Open-Meteo's free forecast endpoint covers roughly the next 16 days; for
  dates further out, the Weather Agent will say so rather than inventing
  data.
- Only `GEMINI_API_KEY` is required in `.env`. Never commit your real `.env`
  file (it's already in `.gitignore`).
