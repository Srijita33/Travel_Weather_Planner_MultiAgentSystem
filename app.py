from __future__ import annotations

import chainlit as cl
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv

from agents import FINAL_PLAN_MARKER, build_team

load_dotenv()


AGENT_STATUS = {
    "TravelPlanner": "🧳 Travel Planner is creating the itinerary...",
    "WeatherAgent": "🌤️ Weather Agent is checking the weather...",
    "TravelCritic": "🔎 Travel Critic is reviewing the itinerary...",
}

AGENT_ICON = {
    "TravelPlanner": "🧳",
    "WeatherAgent": "🌤️",
    "TravelCritic": "🔎",
}

PREVIEW_LINES = 3


def split_preview(content: str) -> tuple[str, str]:
    """Split content into a short preview and the remaining full text."""
    lines = content.strip().splitlines()
    preview = "\n".join(lines[:PREVIEW_LINES])
    rest = "\n".join(lines[PREVIEW_LINES:]).strip()
    return preview, rest


@cl.on_chat_start
async def on_chat_start():
    try:
        team, model_client = build_team()
    except Exception as exc:
        await cl.Message(content=f"⚠️ Setup error: {exc}").send()
        return

    cl.user_session.set("team", team)
    cl.user_session.set("model_client", model_client)

    await cl.Message(
        content=(
            "✈️ **AI Travel & Weather Planner**\n\n"
            "Tell me about your trip — destination, number of days, travel "
            "dates, budget, and interests. For example:\n\n"
            "> Plan a 4-day trip to Goa. My budget is ₹20,000. I like "
            "beaches, food and sightseeing. I prefer a relaxed itinerary."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    team = cl.user_session.get("team")

    if team is None:
        await cl.Message(
            content="⚠️ The planner isn't set up. Please refresh and try again."
        ).send()
        return

    seen_agents = set()
    final_plan_text = None

    try:
        async for event in team.run_stream(task=message.content):
            if isinstance(event, TaskResult):
                # Stream finished; nothing extra to send here.
                continue

            source = getattr(event, "source", None)
            content = getattr(event, "content", None)

            if source is None or source == "user" or not isinstance(content, str):
                # Skip tool-call/tool-result events and anything without plain
                # text - we only want to show real agent messages.
                continue

            if source not in AGENT_STATUS:
                continue

            if source not in seen_agents:
                seen_agents.add(source)
                await cl.Message(
                    content=AGENT_STATUS[source],
                    author="System",
                ).send()

            if FINAL_PLAN_MARKER in content:
                final_plan_text = content.replace(FINAL_PLAN_MARKER, "").strip()

                await cl.Message(
                    content=f"## ✈️ Final Travel Plan\n\n{final_plan_text}",
                    author="TravelPlanner",
                ).send()
            else:
                icon = AGENT_ICON.get(source, "")
                preview, rest = split_preview(content)

                await cl.Message(
                    content=f"{icon} {preview}",
                    author=source,
                ).send()

                if rest:
                    async with cl.Step(
                        name=f"{source} - full message",
                        type="run",
                    ) as step:
                        step.output = rest

    except Exception as exc:
        await cl.Message(
            content=(
                "⚠️ The planning team hit an error and had to stop - "
                f"**{exc}**\n\n"
                "This is often a Gemini API rate limit. Wait a moment and "
                "try again, or send a smaller/simpler request."
            )
        ).send()
        return

    if final_plan_text is None:
        await cl.Message(
            content=(
                "⚠️ I ran out of turns before finalizing the plan. Feel free "
                "to ask me to continue or reset, or start a new request."
            )
        ).send()


@cl.on_chat_end
async def on_chat_end():
    model_client = cl.user_session.get("model_client")

    if model_client is not None:
        await model_client.close()
