"""ReportAgent — synthesizes simulation logs into structured analysis.

Uses Tier 1 (Frontier) LLM to produce a comprehensive report covering:
- Consensus points
- Dissenting views and their merit
- Black swan risks identified
- Conviction shift analysis
- Actionable recommendations
"""

from __future__ import annotations

from typing import Optional

from aps.config import LLMMode
from aps.inference.llm import TieredLLM
from aps.schemas.agent import AgentProfile, AgentRole, LLMTier
from aps.schemas.message import RoundSummary, SimulationResult
from aps.dissident.router import DissidentRouter


class ReportAgent:
    """Analyzes simulation results and produces structured reports.

    Uses Tier 1 (Frontier) LLM for high-quality synthesis.
    """

    def __init__(self, llm: Optional[TieredLLM] = None):
        self._llm = llm or TieredLLM(mode=LLMMode.MOCK)

    def analyze(self, result: SimulationResult) -> str:
        """Generate a comprehensive analysis report from simulation results.

        Args:
            result: Complete simulation result with all rounds and metadata.

        Returns:
            Structured report string.
        """
        # Build structured input for the LLM
        sections = self._build_report_context(result)
        prompt = self._build_report_prompt(result, sections)

        report = self._llm.invoke(
            prompt=prompt,
            system_prompt=(
                "You are a senior analyst producing a rigorous post-simulation report. "
                "Be specific, cite agent names and round numbers. Distinguish between "
                "genuine insights and noise. Grade the severity of identified risks."
            ),
            tier=LLMTier.TIER_1,
            role_hint="synthesizer",
        )

        return report

    def analyze_conviction_dynamics(self, result: SimulationResult) -> dict:
        """Analyze how convictions shifted over the simulation.

        Returns a structured dict with conviction dynamics data.
        """
        agents_data = result.metadata.get("agents", [])
        conviction_summary = result.metadata.get("conviction_summary", {})

        # Reconstruct agent profiles to analyze
        agents = []
        for a_data in agents_data:
            try:
                agents.append(AgentProfile.model_validate(a_data))
            except Exception:
                continue

        # Identify key dynamics
        conformists = [a for a in agents if a.role == AgentRole.CONFORMIST]
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]

        most_convinced = sorted(conformists, key=lambda a: a.conviction_score, reverse=True)
        most_resistant = sorted(conformists, key=lambda a: a.conviction_score)

        converted = [a for a in conformists if a.conviction_score >= 0.5]
        resistant = [a for a in conformists if a.conviction_score < 0.2]

        return {
            "total_conformists": len(conformists),
            "total_dissidents": len(dissidents),
            "converted_count": len(converted),
            "resistant_count": len(resistant),
            "conversion_rate": round(len(converted) / max(1, len(conformists)), 4),
            "avg_conviction": conviction_summary.get("avg_conviction", 0),
            "most_convinced": [
                {"name": a.name, "score": a.conviction_score}
                for a in most_convinced[:3]
            ],
            "most_resistant": [
                {"name": a.name, "score": a.conviction_score}
                for a in most_resistant[:3]
            ],
            "dissent_influencers": [
                {"name": a.name, "id": a.id} for a in dissidents
            ],
        }

    def generate_executive_summary(self, result: SimulationResult) -> str:
        """Generate a brief executive summary (1-2 paragraphs)."""
        conviction = result.metadata.get("conviction_summary", {})
        num_rounds = len(result.rounds)
        total_msgs = result.total_messages
        dissent_msgs = result.total_dissent_messages
        conversion_rate = conviction.get("conversion_rate", 0)

        prompt = (
            f"Simulation: '{result.room_config.name}'\n"
            f"Topic: {result.room_config.topic}\n"
            f"Population: {result.room_config.population_size} agents\n"
            f"Rounds: {num_rounds}\n"
            f"Total messages: {total_msgs} ({dissent_msgs} dissenting)\n"
            f"Conversion rate: {conversion_rate:.1%} of conformists shifted to dissent\n"
            f"Average conviction: {conviction.get('avg_conviction', 0):.2f}\n\n"
            f"Write a 2-paragraph executive summary. First paragraph: what happened. "
            f"Second paragraph: the most important insight or risk uncovered."
        )

        return self._llm.invoke(
            prompt=prompt,
            system_prompt="You are writing an executive briefing for senior leadership.",
            tier=LLMTier.TIER_1,
            role_hint="synthesizer",
        )

    def _build_report_context(self, result: SimulationResult) -> dict:
        """Build structured context sections from the simulation result."""
        sections = {
            "overview": {
                "room": result.room_config.name,
                "topic": result.room_config.topic,
                "population": result.room_config.population_size,
                "rounds": len(result.rounds),
                "total_messages": result.total_messages,
                "dissent_messages": result.total_dissent_messages,
            },
            "round_data": [],
        }

        for rnd in result.rounds:
            round_info = {
                "round_num": rnd.round_num,
                "conformist_msgs": rnd.num_conformist,
                "dissent_msgs": rnd.num_dissent,
                "synthesis": rnd.synthesis,
                "key_messages": [],
            }

            # Pick up to 3 notable messages per round
            for msg in rnd.messages[:6]:
                round_info["key_messages"].append({
                    "agent": msg.agent_name,
                    "is_dissent": msg.is_dissent,
                    "content_preview": msg.content[:200],
                })

            sections["round_data"].append(round_info)

        sections["conviction"] = result.metadata.get("conviction_summary", {})

        return sections

    def _build_report_prompt(self, result: SimulationResult, sections: dict) -> str:
        """Build the final LLM prompt for report generation."""
        overview = sections["overview"]

        rounds_text = ""
        for rd in sections["round_data"]:
            rounds_text += f"\n--- Round {rd['round_num']} ---\n"
            rounds_text += f"Conformist messages: {rd['conformist_msgs']}, Dissent: {rd['dissent_msgs']}\n"
            if rd["synthesis"]:
                rounds_text += f"Synthesis: {rd['synthesis'][:300]}\n"
            for km in rd["key_messages"]:
                tag = "[DISSENT]" if km["is_dissent"] else "[CONFORM]"
                rounds_text += f"  {km['agent']} {tag}: {km['content_preview']}\n"

        conviction = sections.get("conviction", {})

        return (
            f"# Adversarial Population Sandbox — Simulation Report\n\n"
            f"## Overview\n"
            f"- Room: {overview['room']}\n"
            f"- Topic: {overview['topic']}\n"
            f"- Population: {overview['population']} agents\n"
            f"- Rounds completed: {overview['rounds']}\n"
            f"- Total messages: {overview['total_messages']} "
            f"({overview['dissent_messages']} dissenting)\n\n"
            f"## Round-by-Round Discussion\n{rounds_text}\n\n"
            f"## Conviction Dynamics\n"
            f"- Sympathizers (converted): {conviction.get('sympathizers', 0)}\n"
            f"- Unconvinced: {conviction.get('unconvinced', 0)}\n"
            f"- Average conviction score: {conviction.get('avg_conviction', 0)}\n"
            f"- Conversion rate: {conviction.get('conversion_rate', 0):.1%}\n\n"
            f"## Your Task\n"
            f"Produce a structured analysis with these sections:\n"
            f"1. **CONSENSUS POINTS** — What the group agreed on\n"
            f"2. **DISSENTING VIEWS** — Key counter-arguments and their merit\n"
            f"3. **BLACK SWAN RISKS** — Hidden risks the dissidents uncovered\n"
            f"4. **CONVICTION SHIFT ANALYSIS** — How and why opinions changed\n"
            f"5. **RECOMMENDATIONS** — Actionable next steps for decision-makers\n"
            f"6. **RISK SEVERITY GRADES** — Rate each identified risk (LOW/MEDIUM/HIGH/CRITICAL)\n"
        )
