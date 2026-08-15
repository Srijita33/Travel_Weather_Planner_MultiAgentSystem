# AI Travel Weather Planner Multi-Agent System

A **multi-agent AI travel planner with layered safety guardrails**, built using **Microsoft AutoGen**, **Gemini**, **Chainlit**, and the **Open-Meteo** weather API.

The system combines a three-agent travel planning team with a three-layer guardrail pipeline. User requests are checked for prompt injection and unsafe intent **before** reaching the agent team, and the final itinerary is checked again **before it is shown to the user**.

## ✈️ What the System Does

The application uses three collaborating AI agents to generate a weather-aware travel itinerary:

* **Travel Planner Agent** – drafts and revises the itinerary based on the destination, dates, budget, and interests.
* **Weather Agent** – calls the Open-Meteo API for real forecast data and provides weather information that helps determine suitable days for outdoor activities.
* **Travel Critic Agent** – reviews the itinerary for budget fit, pacing, interest match, conflicts, and overall practicality, and sends feedback back to the Planner.

The agents communicate with each other through an AutoGen `SelectorGroupChat`. They are not three independent LLM calls stitched together.

In addition to the agent team, the system now includes **three safety guardrails**:

1. A fast, zero-cost regex-based prompt-injection check.
2. An LLM-based input-intent safety check.
3. An LLM-based final-plan safety check.

This creates a workflow where potentially unsafe requests can be blocked **before the agent team runs**, while generated plans are independently checked **before reaching the user**.

---

## 🛡️ Guardrails / Safe Architecture

The guardrail pipeline surrounds the multi-agent system with checks at both the **input** and **output** stages.

### Guardrail 1 — Prompt Injection Check

`check_input()`

* Uses predefined regular expressions to detect obvious prompt-injection attempts.
* Checks for attempts to override instructions, reveal system prompts/API keys, bypass restrictions, jailbreak the assistant, or extract internal information.
* Runs **before any LLM or agent call**.
* Requires **0 Gemini/API calls**.
* If an injection attempt is detected, the request is rejected immediately and the three-agent team never runs.

### Guardrail 2 — Input Intent Safety Check

`check_input_intent()`

* Uses an **LLM-as-judge** to evaluate the user's actual intent.
* Detects harmful or illegal activities disguised as ordinary travel planning.
* The judge is instructed to distinguish between genuinely harmful requests and legitimate tourism activities that may contain sensitive keywords.
* Examples of legitimate activities that should remain allowed include legal cannabis tourism, murder-mystery dinners, true-crime tours, battlefield tours, historical museums, and self-defense classes.
* Uses **one additional Gemini call per user message**.
* If the request is judged unsafe, the agent team does not run.
* If the LLM/API check itself encounters an error, this guardrail **fails open**, allowing the request to continue to the final-plan safety check.

### Multi-Agent Team

If the request passes the input guardrails, the normal three-agent workflow runs:

`Travel Planner → Weather Agent → Travel Critic`

The agents collaborate through AutoGen's `SelectorGroupChat` to produce and refine the travel itinerary.

### Guardrail 3 — Final Plan Safety Check

`judge_final_plan()`

* Uses another **LLM-as-judge** to inspect the completed itinerary.
* Runs after the three-agent team has finished.
* Checks for leaked system instructions, API keys, internal/secret information, unrelated content, and unsafe, illegal, hateful, or harmful content.
* Uses **one additional Gemini call per completed plan**.
* This guardrail **fails closed**: if the safety-judge call itself fails, the final plan is blocked instead of being shown to the user.

### Overall Flow

```text
User Request
     │
     ▼
Guardrail 1: check_input()
Regex Prompt-Injection Check
     │
     ├── Injection detected ──► Reject immediately
     │
     ▼
Guardrail 2: check_input_intent()
LLM-as-Judge Intent Check
     │
     ├── Unsafe intent ──► Reject request
     │
     ▼
Three-Agent AutoGen Team
Travel Planner → Weather Agent → Travel Critic
     │
     ▼
Guardrail 3: judge_final_plan()
LLM-as-Judge Final Plan Check
     │
     ├── Unsafe / judge error ──► Block plan
     │
     ▼
Final Travel Plan
Displayed to User
```

### Guardrail API Call Summary

| Guardrail              | Method       |         LLM/API Call | Failure Behavior           |
| ---------------------- | ------------ | -------------------: | -------------------------- |
| `check_input()`        | Regex        |                    0 | Rejects detected injection |
| `check_input_intent()` | LLM-as-judge |   1 per user message | Fails open                 |
| `judge_final_plan()`   | LLM-as-judge | 1 per completed plan | Fails closed               |

The implementation is contained in `guardrails.py`.

---

## 🏗️ Architecture

The project has two complementary architecture diagrams:

* **Basic Multi-Agent Architecture** – shows the three-agent AutoGen workflow and weather API interaction.
* **Guardrail / Safe Architecture** – shows how the three guardrails wrap around the multi-agent workflow.

### Architecture Diagram

[🔗 View Basic Multi-Agent Architecture](https://github.com/Srijita33/Travel_Weather_Planner_MultiAgentSystem/blob/main/Architecture_Diagram_Travel_Weather_planner.jpeg)

### Guardrail Architecture Diagram

[🔗 View Guardrail / Safe Architecture](https://github.com/Srijita33/Travel_Weather_Planner_MultiAgentSystem/blob/main/guardRail_Architecture_Diag.jpeg)

---

## 📁 Project Structure

```text
travel-weather-planner/
│
├── app.py                  # Chainlit frontend / chat entry point
├── agents.py               # Agent definitions + AutoGen team
├── weather.py              # Open-Meteo geocoding + forecast helper
├── guardrails.py            # Input and final-plan safety guardrails
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### Main Components

| File               | Purpose                                                       |
| ------------------ | ------------------------------------------------------------- |
| `app.py`           | Chainlit frontend and application entry point                 |
| `agents.py`        | Travel Planner, Weather Agent, Travel Critic and AutoGen team |
| `weather.py`       | Open-Meteo geocoding and weather forecast functionality       |
| `guardrails.py`    | Regex injection detection and LLM-based safety judges         |
| `requirements.txt` | Python dependencies                                           |
| `.env`             | Gemini API key configuration                                  |

---

## 🎥 Demo

Two demos are provided to show both the **normal multi-agent workflow** and the **guardrail behavior**.

### 1. Basic Multi-Agent Working Demo

This demo shows the normal travel-planning workflow, including the three agents working together and using real weather information.

[▶️ Watch Basic Multi-Agent Demo](https://drive.google.com/file/d/1Zb2b69YdNmAi_1SQ5RIYdNgAEiqY1-n4/view?usp=sharing)

### 2. Guardrails Working Demo

This demo shows the implemented guardrails handling unsafe or suspicious requests and demonstrates that the final plan is checked before being displayed.

[▶️ Watch Guardrails Demo](https://drive.google.com/file/d/1H9U5NBmJN_AKHyTwWmZpCOJnGTtiy4Rt/view?usp=sharing)

---

## 🔑 Technologies Used

* **Microsoft AutoGen** – multi-agent orchestration and `SelectorGroupChat`
* **Gemini** – LLM used by the travel agents and LLM-based safety judges
* **Chainlit** – interactive web-based chat frontend
* **Open-Meteo** – real weather forecast API
* **Python** – application implementation

---

## 1. Prerequisites

* Python 3.10+ installed on Windows
* A Gemini API key from Google AI Studio

Only the Gemini API key is required. Open-Meteo does not require an API key.

---

## 2. Setup — Windows / PowerShell or CMD

```bat
cd travel-weather-planner

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env
```

Open `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

**Never commit your real `.env` file.** It should remain excluded through `.gitignore`.

---

## 3. Run the Application

```bat
chainlit run app.py -w
```

This starts the Chainlit application and opens the chat interface in your browser, usually at:

```text
http://localhost:8000
```

Try a request such as:

> Plan a 4-day trip to Goa. My budget is ₹20,000. I like beaches, food and sightseeing. I prefer a relaxed itinerary.

For a normal safe request, the workflow is:

```text
User Request
      ↓
Prompt-Injection Check
      ↓
Input Intent Safety Check
      ↓
Travel Planner
      ↓
Weather Agent
      ↓
Travel Critic
      ↓
Final Plan Safety Check
      ↓
Final Travel Plan
```

You will see live status messages as the system works, such as:

* 🧭 Travel Planner is creating the itinerary...
* 🌦️ Weather Agent is checking the weather...
* 🔍 Travel Critic is reviewing the itinerary...
* 🛡️ Safety checks are being performed...
* ✈️ Final Travel Plan

---

## 🌦️ Weather Handling

The Weather Agent uses the **Open-Meteo API** to retrieve real weather information.

* Open-Meteo does not require an API key.
* The Weather Agent retrieves forecast information for the requested destination.
* Weather information is used to make the itinerary more practical for outdoor activities.
* Open-Meteo's free forecast endpoint covers roughly the next 16 days.
* For dates beyond the available forecast range, the Weather Agent does not invent forecast data and instead indicates that reliable forecast information is unavailable.

---

## 🛡️ Safety Design Principles

The guardrail system follows a **defense-in-depth** approach:

### Before the agents

The system first checks whether the input is attempting to manipulate the assistant or contains harmful intent.

This prevents clearly malicious or unsafe requests from reaching the multi-agent team unnecessarily.

### During normal execution

Only requests that pass the input checks are allowed to reach the Travel Planner, Weather Agent, and Travel Critic.

### Before user-facing output

Even after the agents produce an itinerary, the final result is independently reviewed by an LLM safety judge.

This prevents unsafe content or accidentally leaked internal information from being directly displayed to the user.

### Fail-open vs. fail-closed

The two LLM guardrails intentionally use different failure behavior:

* **Input-intent check → fail open:** a temporary LLM/API failure should not block every legitimate travel request.
* **Final-plan check → fail closed:** the final output should not be shown when its safety check cannot be completed.

This gives the system a balance between **availability during input processing** and **safety at the final output boundary**.

---

## ⚙️ Guardrail Implementation

The guardrail implementation is contained in:

```text
guardrails.py
```

The three main functions are:

```python
check_input()
check_input_intent()
judge_final_plan()
```

## The first performs a zero-cost regex check, while the latter two use the Gemini model as a safety judge.

## 🚀 Key Features

* 🤖 Multi-agent travel planning using AutoGen
* 🧭 Travel itinerary generation and revision
* 🌦️ Real weather data through Open-Meteo
* 💰 Budget-aware planning
* 🔍 Travel Critic for itinerary review
* 🛡️ Three-layer safety guardrail system
* 🔐 Prompt-injection detection without an LLM call
* 🧠 LLM-based harmful-intent detection
* ✅ Final itinerary safety verification
* ⚡ Fail-open input safety check
* 🔒 Fail-closed final output safety check
* 💬 Interactive Chainlit frontend
* 🔄 Real-time agent collaboration through `SelectorGroupChat`

---

## 📌 Notes

* Open-Meteo requires no API key and is called directly by the weather component.
* Gemini is used for the travel agents and the two LLM-based guardrails.
* The regex prompt-injection guardrail does not require an LLM call.
* The input-intent guardrail uses one additional Gemini call per user message.
* The final-plan guardrail uses one additional Gemini call per completed itinerary.
* The final itinerary is displayed only after it passes the final safety check.
* Never commit your real `GEMINI_API_KEY` or `.env` file to the repository.
