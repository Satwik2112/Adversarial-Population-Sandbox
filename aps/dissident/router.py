"""DissidentRouter — forced-dissent injection and conviction spreading.

Core mechanic:
1. Each round, dissident agents generate adversarial counter-arguments.
2. Each conformist agent is then "exposed" to the dissent and may shift
   their conviction_score based on personality susceptibility.
3. Over multiple rounds, dissent spreads: agents who cross the conviction
   threshold (0.5) effectively become sympathizers and argue the dissident
   position in subsequent rounds.
4. The user controls the number of rounds, allowing dissent to cascade.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from aps.schemas.agent import AgentProfile, AgentRole, LLMTier
from aps.schemas.message import Message
from aps.inference.llm import TieredLLM
from aps.config import LLMMode


# Conviction threshold: above this, a conformist starts sympathizing with dissent
CONVICTION_THRESHOLD = 0.5

# How much conviction can shift per round (base, before personality modifiers)
BASE_CONVICTION_SHIFT = 0.20


class ConvictionResult:
    """Result of a conviction-spreading round."""

    def __init__(self):
        self.shifts: dict[str, float] = {}       # agent_id -> new conviction_score
        self.newly_converted: list[str] = []      # agent_ids that crossed threshold
        self.dissent_messages: list[Message] = []  # adversarial messages generated


class DissidentRouter:
    """Handles forced-dissent injection and conviction spreading.

    Each round:
    1. Dissidents generate adversarial counter-arguments to the conformist consensus.
    2. Each conformist's conviction_score is updated based on:
       - Their personality (low agreeableness = harder to convince)
       - The "strength" of the dissident argument
       - Number of dissidents vs. conformists (social pressure)
    3. Agents crossing CONVICTION_THRESHOLD become sympathizers.
    """

    def __init__(
        self,
        llm: Optional[TieredLLM] = None,
        seed: Optional[int] = None,
    ):
        self._llm = llm or TieredLLM(mode=LLMMode.MOCK)
        self._rng = random.Random(seed)

    def inject_dissent(
        self,
        agents: list[AgentProfile],
        conformist_messages: list[Message],
        round_num: int,
        topic: str,
    ) -> tuple[list[Message], list[AgentProfile]]:
        """Run the dissent injection + conviction spreading for one round.

        Args:
            agents: All agents in the simulation.
            conformist_messages: Messages from conformist agents this round.
            round_num: Current round number.
            topic: The simulation topic.

        Returns:
            Tuple of (dissent_messages, updated_agents) where agents have
            updated conviction_scores.
        """
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]
        sympathizers = [
            a for a in agents
            if a.role == AgentRole.CONFORMIST and a.conviction_score >= CONVICTION_THRESHOLD
        ]
        conformists = [
            a for a in agents
            if a.role == AgentRole.CONFORMIST and a.conviction_score < CONVICTION_THRESHOLD
        ]

        # Ensure at least 1 dissident exists (safety check)
        if not dissidents:
            return [], agents

        # --- Step 1: Generate adversarial counter-arguments ---
        dissent_messages = self._generate_dissent(
            dissidents=dissidents,
            sympathizers=sympathizers,
            conformist_messages=conformist_messages,
            round_num=round_num,
            topic=topic,
        )

        # --- Step 2: Spread conviction ---
        updated_agents = self._spread_conviction(
            agents=agents,
            dissent_messages=dissent_messages,
            dissidents=dissidents,
            sympathizers=sympathizers,
            round_num=round_num,
        )

        return dissent_messages, updated_agents

    def _generate_dissent(
        self,
        dissidents: list[AgentProfile],
        sympathizers: list[AgentProfile],
        conformist_messages: list[Message],
        round_num: int,
        topic: str,
    ) -> list[Message]:
        """Generate adversarial counter-arguments from dissidents + sympathizers."""
        messages = []

        # Summarize the conformist consensus for context
        consensus_summary = "\n".join(
            f"- {m.agent_name}: {m.content[:200]}" for m in conformist_messages[:10]
        )

        # Dissidents argue their position
        for agent in dissidents:
            prompt = (
                f"Scenario/Topic: {topic}\n\n"
                f"The group consensus so far (Round {round_num}):\n{consensus_summary}\n\n"
                f"Analyze the scenario completely. You MUST challenge this consensus using unique, natural-language arguments based directly on the specific details of the scenario. "
                f"Find the blind spots, hidden risks, and uncomfortable truths. Steel-man the opposite position. "
                f"Be persuasive and authentic to your personality — your goal is to convince others to reconsider."
            )

            response = self._llm.invoke(
                prompt=prompt,
                system_prompt=agent.system_prompt,
                tier=agent.llm_tier,
                role_hint="dissident",
            )

            messages.append(Message(
                agent_id=agent.id,
                agent_name=agent.name,
                content=response,
                round_num=round_num,
                is_dissent=True,
                role=AgentRole.DISSIDENT.value,
            ))

        # Sympathizers (converted conformists) also voice dissent
        for agent in sympathizers:
            prompt = (
                f"Scenario/Topic: {topic}\n\n"
                f"You were initially part of the consensus, but you've been convinced "
                f"by the dissenting arguments. Your conviction score is {agent.conviction_score:.2f}. "
                f"The group consensus:\n{consensus_summary}\n\n"
                f"Analyze the scenario completely. Provide a unique, natural-language argument "
                f"sharing why you changed your mind based directly on the scenario's specific details. "
                f"Explain what persuaded you. Be authentic to your personality and try to convince others to also reconsider."
            )

            response = self._llm.invoke(
                prompt=prompt,
                system_prompt=agent.system_prompt,
                tier=agent.llm_tier,
                role_hint="conviction",
            )

            messages.append(Message(
                agent_id=agent.id,
                agent_name=agent.name,
                content=response,
                round_num=round_num,
                is_dissent=True,
                role="sympathizer",
            ))

        return messages

    def _spread_conviction(
        self,
        agents: list[AgentProfile],
        dissent_messages: list[Message],
        dissidents: list[AgentProfile],
        sympathizers: list[AgentProfile],
        round_num: int,
    ) -> list[AgentProfile]:
        """Update conviction scores for all conformist agents.

        Conviction shift is influenced by:
        1. Agent personality (low agreeableness = resistant, high openness = receptive)
        2. Social pressure (more dissidents + sympathizers = stronger influence)
        3. Round number (cumulative exposure increases susceptibility)
        4. Random variance (not everyone responds the same way)
        """
        num_dissenters = len(dissidents) + len(sympathizers)
        total_agents = len(agents)

        # Social pressure multiplier: more dissenters = stronger influence
        if total_agents > 0:
            social_pressure = num_dissenters / total_agents
        else:
            social_pressure = 0.0

        updated = []
        for agent in agents:
            if agent.role == AgentRole.DISSIDENT:
                # Dissidents don't shift — they maintain their position
                updated.append(agent)
                continue

            if agent.role == AgentRole.SYNTHESIZER:
                # Synthesizers stay neutral
                updated.append(agent)
                continue

            # --- Conformist conviction update ---
            susceptibility = self._calculate_susceptibility(agent)

            # Base shift modified by susceptibility and social pressure
            shift = (
                BASE_CONVICTION_SHIFT
                * susceptibility
                * (1.0 + social_pressure)
                * (1.0 + 0.1 * round_num)  # Slight increase over rounds
            )

            # Add random variance (±30%)
            shift *= self._rng.uniform(0.7, 1.3)

            new_score = min(1.0, agent.conviction_score + shift)

            # Track who influenced them
            influencer_ids = [d.id for d in dissidents]
            new_convinced_by = list(set(agent.convinced_by + influencer_ids))

            # Create updated agent (Pydantic models are immutable by default, so copy)
            updated_agent = agent.model_copy(update={
                "conviction_score": round(new_score, 4),
                "convinced_by": new_convinced_by,
            })

            updated.append(updated_agent)

        return updated

    def _calculate_susceptibility(self, agent: AgentProfile) -> float:
        """Calculate how susceptible an agent is to conviction shifting.

        Higher susceptibility = easier to convince.

        Factors:
        - High openness → more susceptible (receptive to new ideas)
        - High agreeableness → more susceptible (goes along with persuasion)
        - High neuroticism → more susceptible (anxiety about being wrong)
        - High conscientiousness → less susceptible (trusts their own analysis)
        - High extraversion → slightly more susceptible (engages more with dissent)
        """
        p = agent.personality

        susceptibility = (
            0.30 * p.openness           # Open minds shift easier
            + 0.25 * p.agreeableness    # Agreeable people are easier to persuade
            + 0.20 * p.neuroticism      # Anxious agents second-guess themselves
            - 0.10 * p.conscientiousness  # Conscientious agents are harder to sway
            + 0.10 * p.extraversion     # Extraverts engage more with dissent
        )

        # Clamp to [0.1, 1.0] — nobody is completely immune
        return max(0.1, min(1.0, susceptibility))

    def get_conviction_summary(self, agents: list[AgentProfile]) -> dict:
        """Get a summary of conviction states across the population."""
        conformists = [a for a in agents if a.role == AgentRole.CONFORMIST]
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]

        if not conformists:
            return {
                "total_agents": len(agents),
                "dissidents": len(dissidents),
                "conformists": 0,
                "sympathizers": 0,
                "unconvinced": 0,
                "avg_conviction": 0.0,
                "max_conviction": 0.0,
                "conversion_rate": 0.0,
            }

        scores = [a.conviction_score for a in conformists]
        sympathizers = [a for a in conformists if a.conviction_score >= CONVICTION_THRESHOLD]

        return {
            "total_agents": len(agents),
            "dissidents": len(dissidents),
            "conformists": len(conformists),
            "sympathizers": len(sympathizers),
            "unconvinced": len(conformists) - len(sympathizers),
            "avg_conviction": round(sum(scores) / len(scores), 4),
            "max_conviction": round(max(scores), 4),
            "conversion_rate": round(len(sympathizers) / len(conformists), 4) if conformists else 0.0,
        }
