"""PersonaBuilder — dynamically generate agent populations with personality weights.

Assigns roles (CONFORMIST, DISSIDENT, SYNTHESIZER) and generates Big Five
personality traits with weighted random distributions. Dissidents get
extreme openness + low agreeableness to be natural challengers.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from aps.schemas.agent import (
    AgentProfile,
    AgentRole,
    LLMTier,
    PersonalityWeights,
)
from aps.schemas.room import RoomConfig

# Name pools for random agent generation
_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Taylor",
    "Sage", "Reese", "Blake", "Cameron", "Dakota", "Emery", "Finley", "Harper",
    "Kai", "Lennox", "Marlow", "Nico", "Oakley", "Parker", "River", "Skyler",
    "Tatum", "Val", "Wren", "Yael", "Zara", "Ash", "Drew", "Ellis", "Flynn",
    "Gray", "Hollis", "Indigo", "Jules", "Kendall", "Lane", "Micah",
]

_BACKSTORIES_CONFORMIST = [
    "A seasoned industry veteran who values stability and proven strategies.",
    "An analytical thinker with 15 years of domain expertise.",
    "A collaborative team player who excels at building consensus.",
    "A pragmatic strategist focused on incremental improvements.",
    "A data-driven professional who trusts established methodologies.",
    "A risk-aware planner with deep institutional knowledge.",
    "A process-oriented leader who prioritizes operational efficiency.",
    "A stakeholder-focused advisor who weighs political dynamics.",
]

_BACKSTORIES_DISSIDENT = [
    "A contrarian thinker who made a career spotting blind spots others miss.",
    "A former whistleblower who believes uncomfortable truths must be spoken.",
    "An outsider brought in specifically to challenge the status quo.",
    "A crisis veteran who has seen 'consensus thinking' lead to catastrophe.",
    "A red-team specialist trained to find flaws in any argument.",
    "An independent analyst with no stake in the prevailing opinion.",
    "A maverick strategist who thrives on questioning assumptions.",
    "A cognitive bias researcher who sees groupthink patterns everywhere.",
]


class PersonaBuilder:
    """Build agent populations for simulations.

    Generates a mix of conformists, dissidents, and a synthesizer
    with randomized personality traits appropriate to their roles.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self._rng = random.Random(seed)

    def generate(self, room_config: RoomConfig) -> list[AgentProfile]:
        """Generate a full population of agents for a simulation room.

        Args:
            room_config: The room configuration specifying population size and dissident ratio.

        Returns:
            List of AgentProfile instances with assigned roles and personalities.
        """
        pop_size = room_config.population_size
        num_dissidents = room_config.min_dissidents

        # Reserve 1 slot for synthesizer (Tier 1 LLM)
        num_synthesizers = 1
        num_conformists = pop_size - num_dissidents - num_synthesizers

        if num_conformists < 0:
            # Very small populations: at least 1 conformist
            num_conformists = 1
            num_dissidents = pop_size - 2  # pop - conformist - synthesizer
            if num_dissidents < 1:
                num_dissidents = 1
                num_synthesizers = max(0, pop_size - 2)

        agents: list[AgentProfile] = []
        used_names: set[str] = set()

        # Generate dissidents
        for i in range(num_dissidents):
            name = self._pick_name(used_names)
            agents.append(self._build_dissident(name))

        # Generate conformists
        for i in range(num_conformists):
            name = self._pick_name(used_names)
            agents.append(self._build_conformist(name))

        # Generate synthesizer(s)
        for i in range(num_synthesizers):
            name = self._pick_name(used_names)
            agents.append(self._build_synthesizer(name))

        # Shuffle so dissidents aren't always first
        self._rng.shuffle(agents)

        return agents

    def _pick_name(self, used: set[str]) -> str:
        """Pick a unique name from the pool."""
        available = [n for n in _FIRST_NAMES if n not in used]
        if not available:
            # Fallback: generate numbered names
            name = f"Agent-{len(used) + 1}"
        else:
            name = self._rng.choice(available)
        used.add(name)
        return name

    def _build_dissident(self, name: str) -> AgentProfile:
        """Build a dissident agent with challenger personality."""
        personality = PersonalityWeights(
            openness=self._rng.uniform(0.75, 1.0),        # Very high: creative, unconventional
            conscientiousness=self._rng.uniform(0.4, 0.8),  # Moderate to high
            extraversion=self._rng.uniform(0.6, 1.0),      # High: outspoken
            agreeableness=self._rng.uniform(0.0, 0.25),    # Very low: confrontational
            neuroticism=self._rng.uniform(0.3, 0.7),       # Variable
        )
        backstory = self._rng.choice(_BACKSTORIES_DISSIDENT)
        return AgentProfile(
            name=name,
            role=AgentRole.DISSIDENT,
            personality=personality,
            backstory=backstory,
            llm_tier=LLMTier.TIER_1,
            conviction_score=1.0,  # Dissidents start fully convinced of their position
        )

    def _build_conformist(self, name: str) -> AgentProfile:
        """Build a conformist agent with mainstream personality."""
        personality = PersonalityWeights(
            openness=self._rng.uniform(0.2, 0.6),          # Low to moderate
            conscientiousness=self._rng.uniform(0.5, 0.9),  # Moderate to high
            extraversion=self._rng.uniform(0.3, 0.7),      # Moderate
            agreeableness=self._rng.uniform(0.6, 1.0),     # High: cooperative
            neuroticism=self._rng.uniform(0.2, 0.5),       # Low to moderate
        )
        backstory = self._rng.choice(_BACKSTORIES_CONFORMIST)
        return AgentProfile(
            name=name,
            role=AgentRole.CONFORMIST,
            personality=personality,
            backstory=backstory,
            llm_tier=LLMTier.TIER_2,
            conviction_score=0.0,  # Conformists start at 0
        )

    def _build_synthesizer(self, name: str) -> AgentProfile:
        """Build a synthesizer agent (uses Tier 1 LLM)."""
        personality = PersonalityWeights(
            openness=self._rng.uniform(0.6, 0.9),
            conscientiousness=self._rng.uniform(0.7, 1.0),
            extraversion=self._rng.uniform(0.4, 0.7),
            agreeableness=self._rng.uniform(0.5, 0.8),
            neuroticism=self._rng.uniform(0.1, 0.4),
        )
        return AgentProfile(
            name=name,
            role=AgentRole.SYNTHESIZER,
            personality=personality,
            backstory="An impartial analyst tasked with synthesizing all viewpoints into actionable insight.",
            llm_tier=LLMTier.TIER_1,  # Synthesizer gets Tier 1 (frontier) model
            conviction_score=0.5,  # Neutral starting position
        )
