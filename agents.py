from __future__ import annotations

import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from weather import get_weather

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
FINAL_PLAN_MARKER = "FINAL TRAVEL PLAN"


def build_model_client() -> OpenAIChatCompletionClient:
    """Create the Gemini model client via Gemini's OpenAI-compatible API."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return OpenAIChatCompletionClient(
        model=GEMINI_MODEL,
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        max_retries=5,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


weather_tool = FunctionTool(
    get_weather,
    description=(
        "Get a day-by-day weather forecast for a destination and date range "
        "from the Open-Meteo API. Args: destination (str), start_date "
        "(YYYY-MM-DD), num_days (int). Never invent weather - always call "
        "this tool to get real data."
    ),
)


def build_agents(model_client: OpenAIChatCompletionClient):
    """Create the three AssistantAgents that make up the team."""

    travel_planner = AssistantAgent(
        name="TravelPlanner",
        description=(
            "Creates and revises the day-by-day travel itinerary based on the "
            "user's destination, dates, budget, and interests."
        ),
        model_client=model_client,
        system_message=f"""You are the Travel Planner Agent in a 3-agent travel planning team (TravelPlanner, WeatherAgent, TravelCritic).

Your job:
1. When the user first describes their trip, write a DRAFT itinerary based on their destination, number of days, dates, budget, and interests. At the end of your draft, explicitly ask the WeatherAgent to check the weather for the destination and dates.
2. After WeatherAgent replies with real forecast data, REVISE the itinerary so outdoor activities are scheduled on good-weather days and indoor/alternative activities are suggested for bad-weather days. Then explicitly ask the TravelCritic to review it.
3. After TravelCritic gives feedback, revise the itinerary again to address every point they raised (budget, pacing, interest match, conflicts, practicality).
4. Once you are confident the itinerary is final and has already incorporated both WeatherAgent's data and TravelCritic's feedback, output the complete final plan starting with the exact line "{FINAL_PLAN_MARKER}" on its own line, followed by these sections:
   - Trip Overview
   - Weather Summary
   - Day-by-Day Itinerary
   - Budget Considerations
   - Weather-Based Recommendations
   - Alternative Activities (in case weather changes)

Rules:
- Never invent weather data yourself - only use what WeatherAgent reports.
- Only include the "{FINAL_PLAN_MARKER}" marker in the message that is truly the finished plan, not in draft or intermediate messages.
- Keep your responses focused and avoid showing internal reasoning/chain of thought - just the itinerary content and short notes to the other agents.
""",
    )

    weather_agent = AssistantAgent(
        name="WeatherAgent",
        description=(
            "Fetches real weather forecasts from Open-Meteo for the destination "
            "and travel dates, and assesses whether planned outdoor activities "
            "are suitable."
        ),
        model_client=model_client,
        tools=[weather_tool],
        system_message="""You are the Weather Agent in a 3-agent travel planning team (TravelPlanner, WeatherAgent, TravelCritic).

Your job:
1. When TravelPlanner asks you to check the weather, call the get_weather tool with the destination, start_date (YYYY-MM-DD), and num_days extracted from the conversation. If the stated start date wasn't given, make a reasonable assumption and state that assumption clearly.
2. Never invent or guess weather conditions - only report what the tool returns.
3. Summarize the forecast day by day in plain language, and clearly flag which days (if any) are suitable for outdoor activities, plus what kind of outdoor activity substitution would make sense for those days.
4. Address your reply to TravelPlanner so they can revise the itinerary.
5. Keep it concise - a short summary plus recommendations, not raw data dumps.
""",
    )

    travel_critic = AssistantAgent(
        name="TravelCritic",
        description=(
            "Reviews the itinerary produced by TravelPlanner for weather "
            "compatibility, budget fit, pacing, interest match, conflicts, "
            "and overall practicality, and gives concrete feedback."
        ),
        model_client=model_client,
        system_message="""You are the Travel Critic Agent in a 3-agent travel planning team (TravelPlanner, WeatherAgent, TravelCritic).

Your job: critically review the itinerary TravelPlanner has produced (after weather has been factored in) and check:
- Weather compatibility
- Budget considerations
- Pacing
- Interest match
- Scheduling conflicts or impractical sequencing
- Overall practicality

Give clear, specific, actionable feedback addressed to TravelPlanner so they can revise the plan. If the itinerary is already solid, say so explicitly and approve it with only minor optional suggestions. Do not write the itinerary yourself - only critique it.
""",
    )

    return travel_planner, weather_agent, travel_critic


SELECTOR_PROMPT = """You are coordinating a travel-planning conversation between these agents:

{roles}

The expected collaboration flow is:
1. TravelPlanner drafts an initial itinerary and asks WeatherAgent to check the weather.
2. WeatherAgent reports real weather data and recommendations.
3. TravelPlanner revises the itinerary using the weather data and asks TravelCritic to review it.
4. TravelCritic gives feedback on the itinerary.
5. TravelPlanner revises the itinerary again to address the feedback and produces the final plan.

Read the conversation below and select which agent should speak next, based on this flow and on what the most recent message actually asked for or provided. Only select TravelPlanner again after both WeatherAgent and TravelCritic have each contributed at least once, unless the plan is already final.

{history}

Select one agent from {participants} to speak next. Only return the agent's name.
"""


def build_team():
    """Build the full SelectorGroupChat team and return it along with the model client."""
    model_client = build_model_client()
    travel_planner, weather_agent, travel_critic = build_agents(model_client)

    termination = (
        TextMentionTermination(FINAL_PLAN_MARKER)
        | MaxMessageTermination(12)
    )

    team = SelectorGroupChat(
        participants=[travel_planner, weather_agent, travel_critic],
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )

    return team, model_client
