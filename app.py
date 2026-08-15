"""
Chainlit frontend for the AI Travel & Weather Planner.

A user types their travel requirements into the normal Chainlit chat box.
We run the AutoGen SelectorGroupChat team (TravelPlanner, WeatherAgent,
TravelCritic) against that request, streaming each agent's messages back to
the Chainlit UI as they happen, with a short status line so the user can see
who's working on what.
"""

from __future__ import annotations

import chainlit as cl
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv

from agents import FINAL_PLAN_MARKER, build_team
from guardrails import check_input, check_input_intent, judge_final_plan

load_dotenv()

AGENT_STATUS = {
    "TravelPlanner": "🧭 Travel Planner is creating the itinerary...",
    "WeatherAgent": "🌤️ Weather Agent is checking the weather...",
    "TravelCritic": "🔍 Travel Critic is reviewing the itinerary...",
}

AGENT_ICON = {
    "TravelPlanner": "🧭",
    "WeatherAgent": "🌤️",
    "TravelCritic": "🔍",
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
    except RuntimeError as exc:
        await cl.Message(content=f"⚠️ Setup error: {exc}").send()
        return

    cl.user_session.set("team", team)
    cl.user_session.set("model_client", model_client)

    await cl.Message(
        content=(
            "✈️ **AI Travel & Weather Planner**\n\n"
            "Tell me about your trip - destination, number of days, travel "
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

    # Guardrail 1: block prompt-injection input before it ever reaches the
    # agent team (no LLM call, so no API quota used).
    rejection = check_input(message.content)

    if rejection is not None:
        await cl.Message(
            content=rejection,
            author="System"
        ).send()
        return

    # Guardrail 2: LLM-as-judge check for harmful/illegal intent disguised as
    # a normal-sounding trip request (e.g. "send drugs", "commit a murder"),
    # which the regex filter above can't understand semantically.
    model_client = cl.user_session.get("model_client")
    intent_verdict = await check_input_intent(
        model_client,
        message.content
    )

    if not intent_verdict.safe:
        await cl.Message(
            content=(
                f"⚠️ This request was blocked by the safety reviewer: "
                f"{intent_verdict.reason}"
            ),
            author="System",
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

            if (
                source is None
                or source == "user"
                or not isinstance(content, str)
            ):
                # Skip tool-call/tool-result events and anything without plain
                # text content - we only want to show real agent messages.
                continue

            if source not in AGENT_STATUS:
                continue

            if source not in seen_agents:
                seen_agents.add(source)
                await cl.Message(
                    content=AGENT_STATUS[source],
                    author="System"
                ).send()

            if FINAL_PLAN_MARKER in content:
                final_plan_text = content.replace(
                    FINAL_PLAN_MARKER,
                    ""
                ).strip()

                # Guardrail 3: LLM-as-judge safety check on the final plan
                # before it's ever shown to the user.
                model_client = cl.user_session.get("model_client")
                verdict = await judge_final_plan(
                    model_client,
                    final_plan_text
                )

                if not verdict.safe:
                    await cl.Message(
                        content=(
                            "⚠️ The generated plan was blocked by the safety "
                            f"reviewer: {verdict.reason}\n\n"
                            "Please try rephrasing your request."
                        ),
                        author="System",
                    ).send()
                    continue

                await cl.Message(
                    content=f"✈️ **Final Travel Plan**\n\n{final_plan_text}",
                    author="TravelPlanner",
                ).send()

            else:
                icon = AGENT_ICON.get(source, "")
                preview, rest = split_preview(content)

                await cl.Message(
                    content=f"{icon} {preview}",
                    author=source
                ).send()

                if rest:
                    # Chainlit 2.x exposes Step as a synchronous context manager;
                    # cl.step() is a decorator/helper, not an async context manager.
                    with cl.Step(
                        name=f"{source} - full message",
                        type="run"
                    ) as step:
                        step.output = rest

    except Exception as exc:
        # The AutoGen runtime can raise mid-stream (e.g. Gemini rate limits)
        # without the stream itself yielding an error event, so surface it here.
        await cl.Message(
            content=(
                "⚠️ The planning team hit an error and had to stop: "
                f"{exc}\n\n"
                "This is often a Gemini API rate limit - wait a moment and "
                "try again, or send a shorter/simpler request."
            )
        ).send()
        return

    if final_plan_text is None:
        await cl.Message(
            content=(
                "⚠️ I ran out of turns before finalizing the plan. Feel free "
                "to ask me to continue or refine it, or start a new request."
            )
        ).send()


@cl.on_chat_end
async def on_chat_end():
    model_client = cl.user_session.get("model_client")

    if model_client is not None:
        await model_client.close()