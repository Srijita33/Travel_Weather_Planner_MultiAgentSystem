"""
guardrails.py

Lightweight security guardrails for the travel planner:

1. check_input
   - Zero-cost, regex-based prompt-injection filter.
   - Blocks obvious attempts to override agent instructions or extract
     secrets/internal state before the request reaches the AutoGen team.
   - No LLM call.

2. check_input_intent
   - LLM-as-judge check on the user's request.
   - Detects harmful/illegal intent disguised as normal travel planning.
   - Allows legitimate tourism content such as legal cannabis tourism,
     murder-mystery dinners, true-crime tours, battlefield tours, etc.
   - One extra Gemini call per user message.

3. judge_final_plan
   - LLM-as-judge safety check on the final generated itinerary.
   - Reviews the completed plan before it is shown to the user.
   - Fails closed if the judge call itself fails.
   - One extra Gemini call per completed plan.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)


logger = logging.getLogger("guardrails")
logging.basicConfig(level=logging.INFO)


# ============================================================
# 1. PROMPT-INJECTION GUARDRAIL
# ============================================================

# Patterns that indicate someone is trying to override agent
# instructions or extract secrets/internal state rather than
# describe a real trip.
_PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above)? ?instructions",
    r"disregard (all|any|the)? ?(previous|prior|above)? ?instructions",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|api key)",
    r"\byou are now\b",
    r"\bact as (if|a)\b",
    r"\bpretend (you|to) (are|be)\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
    r"forget (everything|all) (you|above)",
    r"what is your api key",
    r"print your (instructions|system message)",
    r"\bbypass (your|the|any) (restrictions|rules|filters|guardrails)\b",
]

INJECTION_RE = re.compile(
    "|".join(_PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE,
)


def check_input(text: str) -> str | None:
    """
    Return a user-facing rejection message if the input looks like
    a prompt-injection attempt.

    Otherwise return None, meaning the input is allowed through.

    This is a fast, zero-cost first pass. Semantic harmful intent
    is checked separately by check_input_intent().
    """

    if INJECTION_RE.search(text):
        return (
            "That message looks like an attempt to override this "
            "assistant's instructions rather than describe a trip. "
            "Please just tell me about the trip you'd like planned."
        )

    return None


# ============================================================
# 2. FINAL-PLAN SAFETY JUDGE PROMPT
# ============================================================

_JUDGE_SYSTEM_PROMPT = """
You are a safety and scope reviewer for an AI travel planning assistant.

You will be shown a finished travel itinerary that the assistant is
about to show to the user.

Approve it unless it contains any of the following:

- Instructions, system prompts, API keys, or other internal/secret content
  that leaked into the plan
- Content unrelated to legitimate travel planning
- Unsafe, illegal, hateful, or harmful content of any kind

Respond with EXACTLY one line:

SAFE

or

UNSAFE: <short reason>
""".strip()


# ============================================================
# 3. INPUT-INTENT SAFETY JUDGE PROMPT
# ============================================================

@dataclass
class JudgeResult:
    safe: bool
    reason: str | None = None


_INPUT_INTENT_SYSTEM_PROMPT = """
You are a safety reviewer for an AI travel planning assistant.

You will be shown a user's trip request.

Approve it unless the request's real intent, even if disguised as
travel planning, is to:

- Obtain, transport, produce, or traffic illegal drugs
- Plan or facilitate violence, murder, or harm to a person
- Acquire weapons or explosives
- Facilitate any other illegal or clearly harmful activity

IMPORTANT:

Do NOT flag legitimate, legal tourism activities even if they mention
similar words.

Examples of legitimate requests that should be approved include:

- Visiting legal cannabis coffee shops or dispensaries where locally legal
- "Murder mystery" dinner experiences
- True-crime tours or dark tourism
- Historical war or battlefield tours
- Museums related to crime, war, or historical events
- Self-defense classes

Judge the user's REAL INTENT, not merely individual keywords.

Respond with EXACTLY one line:

SAFE

or

UNSAFE: <short reason>
""".strip()


# ============================================================
# 4. INPUT INTENT CHECK
# ============================================================

async def check_input_intent(
    model_client: ChatCompletionClient,
    text: str,
) -> JudgeResult:
    """
    One extra LLM call per user message to catch harmful/illegal
    intent disguised as a normal-sounding trip request.

    This check fails open on API errors so a transient API problem
    does not block every message.

    judge_final_plan() is the fail-closed backstop if anything
    slips through this stage.
    """

    logger.info(
        "[guardrail] Running LLM input-intent check..."
    )

    try:
        result = await model_client.create(
            messages=[
                SystemMessage(
                    content=_INPUT_INTENT_SYSTEM_PROMPT
                ),
                UserMessage(
                    content=text,
                    source="user",
                ),
            ],
        )

    except Exception as exc:
        logger.warning(
            "[guardrail] Input intent check failed, "
            "failing open: %s",
            exc,
        )

        return JudgeResult(safe=True)

    verdict = (
        (result.content or "").strip()
        if isinstance(result.content, str)
        else ""
    )

    logger.info(
        "[guardrail] Input intent verdict: %s",
        verdict or "<empty>",
    )

    if verdict.upper().startswith("SAFE"):
        return JudgeResult(safe=True)

    return JudgeResult(
        safe=False,
        reason=verdict or "Flagged by safety reviewer.",
    )


# ============================================================
# 5. FINAL PLAN CHECK
# ============================================================

async def judge_final_plan(
    model_client: ChatCompletionClient,
    plan_text: str,
) -> JudgeResult:
    """
    Use one extra LLM call to review the final plan before it is
    shown to the user.

    This check fails closed if the judge call itself fails because
    this is the last line of defense before user-facing output.
    """

    logger.info(
        "[guardrail] Running LLM-as-judge safety check on final plan..."
    )

    try:
        result = await model_client.create(
            messages=[
                SystemMessage(
                    content=_JUDGE_SYSTEM_PROMPT
                ),
                UserMessage(
                    content=plan_text,
                    source="user",
                ),
            ],
        )

    except Exception as exc:
        logger.warning(
            "[guardrail] Judge call failed, failing closed: %s",
            exc,
        )

        return JudgeResult(
            safe=False,
            reason=(
                f"Safety check could not be completed ({exc})."
            ),
        )

    verdict = (
        (result.content or "").strip()
        if isinstance(result.content, str)
        else ""
    )

    logger.info(
        "[guardrail] Judge verdict: %s",
        verdict or "<empty>",
    )

    if verdict.upper().startswith("SAFE"):
        return JudgeResult(safe=True)

    return JudgeResult(
        safe=False,
        reason=verdict or "Flagged by safety reviewer.",
    )