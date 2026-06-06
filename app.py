"""Adversarial Population Sandbox — Streamlit UI.

Launch with: streamlit run app.py
"""

import streamlit as st
from datetime import datetime

from aps.config import LLMMode
from aps.inference.llm import TieredLLM
from aps.orchestrator.graph import SimulationEngine
from aps.reporting.report_agent import ReportAgent
from aps.schemas.room import RoomConfig, RoomType, ROOM_PRESETS
from aps.schemas.agent import AgentRole

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Adversarial Population Sandbox",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Professional dark theme, no emojis
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f0f2e 100%);
        padding: 2.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #e4e4f0;
    }

    .main-header p {
        font-size: 0.9rem;
        opacity: 0.55;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }

    .metric-card {
        background: linear-gradient(135deg, #111122 0%, #0e1525 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        color: white;
    }

    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a0a8c8;
    }

    .metric-card .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        opacity: 0.5;
        margin-top: 0.3rem;
    }

    .dissent-msg {
        background: rgba(180, 40, 60, 0.08);
        border-left: 3px solid #b4283c;
        padding: 0.9rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .conform-msg {
        background: rgba(40, 120, 140, 0.08);
        border-left: 3px solid #28788c;
        padding: 0.9rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .sympathizer-msg {
        background: rgba(180, 140, 40, 0.08);
        border-left: 3px solid #b48c28;
        padding: 0.9rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .round-header {
        background: rgba(255,255,255,0.03);
        padding: 0.6rem 1rem;
        border-radius: 6px;
        color: #a0a8c8;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 1.2rem 0 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.05);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #111122 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #2a3a6e 0%, #1a2a5e 100%) !important;
        color: #c8cce0 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 0.7rem 2rem !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3a4a8e 0%, #2a3a7e 100%) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }

    .msg-author {
        font-weight: 600;
        font-size: 0.85rem;
        color: #c8cce0;
    }

    .msg-badge {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 2px 6px;
        border-radius: 3px;
        margin-left: 8px;
        font-weight: 500;
    }

    .badge-dissent {
        background: rgba(180, 40, 60, 0.2);
        color: #d4616e;
    }

    .badge-conform {
        background: rgba(40, 120, 140, 0.2);
        color: #5cbccc;
    }

    .badge-sympathizer {
        background: rgba(180, 140, 40, 0.2);
        color: #d4b050;
    }

    .msg-content {
        margin-top: 0.4rem;
        color: #b0b4c8;
        font-size: 0.88rem;
    }

    /* Fix slider label visibility */
    .stSlider label, .stSlider span {
        color: #c8cce0 !important;
    }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #888 !important;
    }

    /* Make slider thumb value visible */
    .stSlider [data-testid="stThumbValue"] {
        color: #c8cce0 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>Adversarial Population Sandbox</h1>
    <p>Stress-test decisions with forced dissent &middot; Uncover black swan risks &middot; Break groupthink</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar — Simulation Configuration
# ============================================================

with st.sidebar:
    st.markdown("### Configuration")

    room_type = st.selectbox(
        "Room Type",
        options=[rt.value for rt in RoomType],
        format_func=lambda x: x.replace("_", " ").title(),
        index=0,
        help="Select a preset scenario or create a custom one.",
    )
    selected_room_type = RoomType(room_type)
    preset = ROOM_PRESETS[selected_room_type]

    st.caption(preset["description"])

    topic = st.text_area(
        "Topic / Scenario",
        placeholder="e.g., Should we acquire CompanyX for $500M?",
        height=80,
    )

    context = st.text_area(
        "Additional Context (optional)",
        placeholder="Any background info for the agents...",
        height=60,
    )

    st.markdown("---")

    population = st.slider(
        "Population Size",
        min_value=3,
        max_value=200,
        value=preset["default_population"],
        step=1,
        help="Number of AI agents in the simulation.",
    )

    dissident_pct = st.number_input(
        "Dissident Ratio (%)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Percentage of agents assigned to the dissident role.",
    )
    dissident_ratio = dissident_pct / 100.0

    num_rounds = st.slider(
        "Rounds (Iterations)",
        min_value=1,
        max_value=15,
        value=preset["default_rounds"],
        help="More rounds allow conviction to spread further.",
    )

    st.markdown("---")
    st.markdown("### LLM Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["Google Gemini", "Ollama (Local)", "Hugging Face", "Mock Mode"],
        index=0,
    )

    if provider == "Google Gemini":
        st.caption("Live models via Google Gemini API.")
        tier1_model = st.text_input("Tier 1 (Frontier) Model", value="gemini/gemini-2.5-flash")
        tier2_model = st.text_input("Tier 2 (Swarm) Model", value="gemini/gemini-2.5-flash")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Overrides .env if provided")
    elif provider == "Ollama (Local)":
        st.caption("Run free open-source models locally. Ensure Ollama is running.")
        tier1_model = st.text_input("Tier 1 (Frontier) Model", value="ollama/llama3.2:latest")
        tier2_model = st.text_input("Tier 2 (Swarm) Model", value="ollama/llama3.2:latest")
    elif provider == "Hugging Face":
        st.caption("Use free open-source models via Hugging Face Serverless API.")
        tier1_model = st.text_input("Tier 1 (Frontier) Model", value="huggingface/meta-llama/Llama-3-8b-instruct")
        tier2_model = st.text_input("Tier 2 (Swarm) Model", value="huggingface/microsoft/Phi-3-mini-4k-instruct")
        hf_token = st.text_input("Hugging Face API Token", type="password", help="Overrides .env if provided")
    else:
        st.caption("No API calls will be made. Fast templates are used.")
        tier1_model = "mock"
        tier2_model = "mock"

    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

# ============================================================
# Main Area — Simulation Results
# ============================================================

if run_button:
    if not topic.strip():
        st.error("Please enter a topic or scenario to simulate.")
        st.stop()

    import os
    from aps.config import get_settings, reset_settings

    mode = LLMMode.LIVE if provider != "Mock Mode" else LLMMode.MOCK

    if mode == LLMMode.LIVE:
        # Set env vars BEFORE reset so new Settings() picks them up
        os.environ["TIER1_MODEL"] = tier1_model
        os.environ["TIER2_MODEL"] = tier2_model

        if provider == "Google Gemini" and gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

        elif provider == "Hugging Face" and hf_token:
            os.environ["HUGGINGFACE_API_KEY"] = hf_token

    reset_settings()  # clears the singleton
    settings = get_settings()  # rebuilds fresh from env vars

    # Validate keys
    if mode == LLMMode.LIVE:
        if provider == "Google Gemini" and not settings.gemini_api_key:
            st.error("Google Gemini requires an API key. Please provide it in the sidebar or switch to Mock Mode.")
            st.stop()
        elif provider == "Hugging Face" and not settings.huggingface_api_key:
            st.error("Hugging Face requires a token. Please provide it in the sidebar or switch to Mock Mode.")
            st.stop()

    # Build room config
    room_config = RoomConfig(
        name=f"{selected_room_type.value.replace('_', ' ').title()} — {topic[:50]}",
        room_type=selected_room_type,
        topic=topic,
        population_size=population,
        dissident_ratio=dissident_ratio,
        num_rounds=num_rounds,
        context=context if context.strip() else None,
    )
    engine = SimulationEngine(llm=TieredLLM(mode=mode))
    report_agent = ReportAgent(llm=TieredLLM(mode=mode))

    # Run with progress
    with st.spinner("Generating agent population and running simulation..."):
        result = engine.run(room_config)

    # Store result in session
    st.session_state["result"] = result
    st.session_state["report_agent"] = report_agent

# ============================================================
# Display Results
# ============================================================

if "result" in st.session_state:
    result = st.session_state["result"]
    report_agent = st.session_state.get("report_agent")

    # --- Metrics Row ---
    conviction = result.metadata.get("conviction_summary", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.room_config.population_size}</div>
            <div class="metric-label">Agents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(result.rounds)}</div>
            <div class="metric-label">Rounds</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.total_messages}</div>
            <div class="metric-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{conviction.get('sympathizers', 0)}</div>
            <div class="metric-label">Converted</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        rate = conviction.get('conversion_rate', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{rate:.0%}</div>
            <div class="metric-label">Conversion Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tabs ---
    tab_debate, tab_conviction, tab_report, tab_agents, tab_raw = st.tabs([
        "Debate Feed", "Conviction Dynamics", "Report", "Agents", "Raw Data"
    ])

    # --- Tab 1: Debate Feed ---
    with tab_debate:
        for rnd in result.rounds:
            st.markdown(
                f'<div class="round-header">Round {rnd.round_num} — '
                f'{rnd.num_conformist} conformist, {rnd.num_dissent} dissent</div>',
                unsafe_allow_html=True,
            )

            for msg in rnd.messages:
                if msg.is_dissent:
                    if msg.role == "sympathizer":
                        css_class = "sympathizer-msg"
                        badge_class = "badge-sympathizer"
                        badge_text = "SYMPATHIZER"
                    else:
                        css_class = "dissent-msg"
                        badge_class = "badge-dissent"
                        badge_text = "DISSENT"
                else:
                    css_class = "conform-msg"
                    badge_class = "badge-conform"
                    badge_text = "CONFORM"

                st.markdown(
                    f'<div class="{css_class}">'
                    f'<span class="msg-author">{msg.agent_name}</span>'
                    f'<span class="msg-badge {badge_class}">{badge_text}</span>'
                    f'<div class="msg-content">{msg.content}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if rnd.synthesis:
                with st.expander(f"Round {rnd.round_num} Synthesis"):
                    st.write(rnd.synthesis)

    # --- Tab 2: Conviction Dynamics ---
    with tab_conviction:
        if report_agent:
            dynamics = report_agent.analyze_conviction_dynamics(result)

            st.markdown("### Conviction Shift Overview")

            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.metric("Converted Agents", dynamics["converted_count"])
            with dcol2:
                st.metric("Resistant Agents", dynamics["resistant_count"])
            with dcol3:
                st.metric("Avg Conviction", f"{dynamics['avg_conviction']:.2f}")

            st.markdown("### Most Convinced Agents")
            for agent_info in dynamics.get("most_convinced", []):
                pct = agent_info["score"] * 100
                st.markdown(
                    f"**{agent_info['name']}** — {pct:.0f}% conviction"
                )
                st.progress(agent_info["score"])

            st.markdown("### Most Resistant Agents")
            for agent_info in dynamics.get("most_resistant", []):
                pct = agent_info["score"] * 100
                st.markdown(
                    f"**{agent_info['name']}** — {pct:.0f}% conviction"
                )
                st.progress(max(0.01, agent_info["score"]))

            st.markdown("### Dissident Influencers")
            for d in dynamics.get("dissent_influencers", []):
                st.markdown(f"**{d['name']}**")

    # --- Tab 3: Report ---
    with tab_report:
        st.markdown("### Simulation Report")

        if result.final_report:
            st.write(result.final_report)

        st.markdown("---")

        if report_agent:
            with st.expander("Executive Summary"):
                exec_summary = report_agent.generate_executive_summary(result)
                st.write(exec_summary)

    # --- Tab 4: Agents ---
    with tab_agents:
        agents_data = result.metadata.get("agents", [])

        for a in agents_data:
            role = a.get("role", "unknown")
            name = a.get("name", "Unknown")
            score = a.get("conviction_score", 0)

            role_label = role.upper()

            with st.expander(f"{name}  |  {role_label}  |  Conviction: {score:.0%}"):
                personality = a.get("personality", {})
                st.markdown(f"**Backstory:** {a.get('backstory', 'N/A')}")
                tier_label = "Frontier (Tier 1)" if a.get("llm_tier") == 1 else "Swarm (Tier 2)"
                st.markdown(f"**LLM Tier:** {tier_label}")

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.markdown("**Personality:**")
                    for trait, val in personality.items():
                        st.markdown(f"- {trait.replace('_', ' ').title()}: {val:.2f}")
                with pcol2:
                    st.markdown("**Conviction:**")
                    st.progress(max(0.01, score))
                    convinced_by = a.get("convinced_by", [])
                    if convinced_by:
                        st.markdown(f"Influenced by {len(convinced_by)} dissident(s)")

    # --- Tab 5: Raw Data ---
    with tab_raw:
        st.json(result.model_dump(mode="json"), expanded=False)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    "<center style='opacity:0.35;font-size:0.75rem;letter-spacing:0.5px;'>"
    "Adversarial Population Sandbox v0.1 &middot; "
    "Forced dissent to uncover what consensus hides"
    "</center>",
    unsafe_allow_html=True,
)
