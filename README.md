# Adversarial Population Sandbox (APS)

An advanced multi-agent simulation framework designed to stress-test strategic decisions, uncover black swan risks, and break groupthink by forcing adversarial dissent.

---

## 💡 Executive Summary

Strategic consensus is often a harbinger of blind spots. The **Adversarial Population Sandbox (APS)** simulates organizational and social environments where consensus is intentionally disrupted by a controlled dissident quota (e.g., 1% to 30%). 

Unlike conventional multi-agent systems that converge toward harmony, APS uses **Tiered LLM Inference** and **Conviction Cascading Mechanics** to model how contrarian arguments spread through a population. It measures the resilience of your decisions by simulating the viral spread of skepticism.

---

## 🛠️ Key Features

- **Forced Dissent Injection**: Dissidents challenge the group consensus each round by introducing persuasive, detail-oriented counter-arguments.
- **Big Five Personality Modeling**: Agent susceptibility to persuasion is governed by individualized Big Five personality traits (Openness, Agreeableness, Neuroticism, Conscientiousness, and Extraversion).
- **Conviction Cascading**: Conformist agents whose conviction score crosses the `0.5` threshold become **Sympathizers**, arguing the dissident position in subsequent rounds.
- **Tiered Inference Engine**:
  - **Tier 1 (Frontier)**: Powered by `gemini-2.5-pro` (or similar) for reasoning-heavy tasks: dissident persuasion, round synthesis, and final reports.
  - **Tier 2 (Swarm)**: Powered by `gemini-2.5-flash` for high-throughput, concise conformist discussions.
- **Interactive UI**: A sleek, dark-themed Streamlit dashboard providing real-time debate feeds, visual conviction graphs, and downloadable risk reports.
- **Mock Mode**: Complete system simulation run entirely on templates for local development and testing—no API keys required.

---

## 🗺️ Documentation Index

To explore the architecture and planning of the sandbox in detail, refer to the following documents:

* 📑 **[Technical Specifications](file:///Users/satwikdubey/dissent/projectspec.md)**: Details the domain schemas, state machine logic, and the conviction-spreading formulas.
* ⚙️ **[Tech Stack & Infrastructure](file:///Users/satwikdubey/dissent/techstackinfra.md)**: Defines the frameworks, LLM integrations, and planned Redis/Qdrant scale-out blueprints.
* 🚀 **[Development Roadmap](file:///Users/satwikdubey/dissent/roadmap.md)**: Tracks completed milestones and outlines persistent memory and swarm scaling features.

---

## 🚀 Quickstart Guide

### 1. Installation

Set up a virtual environment and install the package with dependencies:

```bash
# Clone the repository
git clone https://github.com/Satwik2112/Adversarial-Population-Sandbox.git
cd Adversarial-Population-Sandbox

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including developer and UI optional groups)
pip install -e ".[dev,ui]"
```

### 2. Configure Environment

Copy the environment example and add your API credentials:

```bash
cp .env.example .env
```

Open `.env` and configure your API keys (e.g., `GEMINI_API_KEY`) and execution modes:
- Set `APS_LLM_MODE=live` to make real LLM completions.
- Set `APS_LLM_MODE=mock` to run offline using pre-generated templates.

### 3. Run the Streamlit UI

Start the dashboard to configure and run simulations interactively:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the Test Suite

Run the full validation suite of 100+ tests:

```bash
pytest
```

---

## 📂 Project Directory Structure

```
dissent/
├── aps/                     # Core simulation logic
│   ├── dissident/           # Dissent injection and social persuasion (router.py)
│   ├── inference/           # Tiered LLM completions via LiteLLM (llm.py)
│   ├── memory/              # In-memory context storage (store.py)
│   ├── orchestrator/        # State-machine runner (graph.py)
│   ├── persona/             # Persona builder & Big Five generation (builder.py)
│   ├── reporting/           # Report agent for post-run synthesis (report_agent.py)
│   └── schemas/             # Pydantic schemas (agent.py, message.py, room.py)
├── tests/                   # Comprehensive pytest test suite
├── app.py                   # Streamlit Frontend application
├── pyproject.toml           # Package metadata and requirements
└── .env                     # Local settings and API keys
```

---

## 🧪 License

This project is open-source and licensed under the MIT License.
