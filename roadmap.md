# Development Roadmap: Adversarial Population Sandbox (APS)

This document tracks the feature development timeline, completed milestones, and upcoming architectural enhancements.

---

## 📅 Roadmap Overview

```mermaid
gantt
    title APS Project Roadmap
    dateFormat  YYYY-MM-DD
    section Core Infrastructure
    Phase 1: Schemas, Config, Logs    :active, 2026-06-01, 2026-06-03
    section Interactive Frontend
    Phase 2: Streamlit Dashboard & UI :active, 2026-06-04, 2026-06-06
    section Persistent Memory
    Phase 3: Qdrant Vector DB         :todo, 2026-06-07, 7d
    section Scale-out Swarm
    Phase 4: Celery & Redis Scaling   :todo, 2026-06-14, 10d
```

---

## 1. Completed Phases

### Phase 1: Core Engine & Domain Models (Completed)
- **Domain Modeling**: Established standard schemas for Rooms, Agent Profiles, Personality Weights, Messages, and Round Summaries.
- **Agent Generator**: Built the `PersonaBuilder` to dynamically create agents, assigning Big Five traits based on roles (outspoken dissidents vs. agreeable conformists).
- **Social Persuasion Engine**: Implemented `DissidentRouter` to calculate personality-based susceptibility, social pressure modifiers, and conviction accumulation.
- **Orchestrator Lifecycle**: Engineered `SimulationEngine` to coordinate debate turns, synthesize opinions, and output complete JSON logs.
- **Validation**: Wrote and verified a test suite containing 100 unit and integration tests covering configurations, schemas, and state updates.

### Phase 2: Interactive Frontend & UI (Completed)
- **Streamlit App**: Built a dashboard with dark-themed custom CSS configurations.
- **Provider Integrations**: Integrated LiteLLM supporting live runs with Google Gemini, Ollama, Hugging Face, or Mock mode.
- **Dynamics Analytics**: Added real-time visual progress bars tracking converted vs. resistant agents and individual conviction levels.
- **Automated Reporting**: Integrated `ReportAgent` using Tier 1 LLM reasoning to generate post-simulation analyst reports and executive summaries.

---

## 2. Upcoming Development

### Phase 3: Persistent Memory & Semantic Search (Next Up)
- Replace the in-memory cache with **Qdrant Vector DB** for semantic agent memory.
- Enable agents to search back through historical debates, referencing arguments made in previous runs.
- Integrate context metadata matching to enhance agent reasoning.

### Phase 4: Swarm Scaling with Celery & Redis
- Decouple the simulation runtime from the frontend thread using background workers.
- Leverage **Redis** as a task queue broker to handle 100+ concurrent LLM calls safely.
- Implement streaming updates from Celery workers back to the Streamlit UI using WebSocket status updates.

### Phase 5: Persuasion Rhetoric & Bias Modeling
- Add configuration settings to model specific cognitive biases (e.g., confirmation bias, status quo bias).
- Allow users to select different dissident personas (e.g., the logical pragmatist, the emotional whistleblower, the contrarian scientist).
