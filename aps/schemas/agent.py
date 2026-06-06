"""Agent and Persona schemas — define agent profiles and personality."""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AgentRole(str, Enum):
    """Role assigned to an agent in the simulation."""
    CONFORMIST = "conformist"
    DISSIDENT = "dissident"
    SYNTHESIZER = "synthesizer"
    OBSERVER = "observer"


class LLMTier(int, Enum):
    """LLM inference tier for cost/quality tradeoff."""
    TIER_1 = 1  # Frontier: GPT-4o, Claude 3.5 — used for synthesis, reports
    TIER_2 = 2  # Swarm: Llama-3 via Ollama — used for worker agents


class PersonalityWeights(BaseModel):
    """Big Five personality traits — each on a 0.0 to 1.0 scale."""

    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0, description="Conscientiousness")
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="Neuroticism / emotional instability")

    def as_prompt_description(self) -> str:
        """Convert personality weights into a natural language description for LLM prompts."""
        descriptors = []

        if self.openness > 0.7:
            descriptors.append("highly creative and open to unconventional ideas")
        elif self.openness < 0.3:
            descriptors.append("practical and prefers proven approaches")

        if self.conscientiousness > 0.7:
            descriptors.append("meticulous and detail-oriented")
        elif self.conscientiousness < 0.3:
            descriptors.append("flexible and spontaneous")

        if self.extraversion > 0.7:
            descriptors.append("outspoken and assertive")
        elif self.extraversion < 0.3:
            descriptors.append("reserved and thoughtful")

        if self.agreeableness > 0.7:
            descriptors.append("cooperative and seeks consensus")
        elif self.agreeableness < 0.3:
            descriptors.append("skeptical and challenges assumptions")

        if self.neuroticism > 0.7:
            descriptors.append("sensitive to risks and worst-case scenarios")
        elif self.neuroticism < 0.3:
            descriptors.append("calm and emotionally stable under pressure")

        if not descriptors:
            return "balanced and moderate in temperament"

        return ", ".join(descriptors)


class AgentProfile(BaseModel):
    """Complete profile of a simulation agent."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique agent identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Agent display name",
    )
    role: AgentRole = Field(
        default=AgentRole.CONFORMIST,
        description="Role in the simulation",
    )
    personality: PersonalityWeights = Field(
        default_factory=PersonalityWeights,
        description="Big Five personality weights",
    )
    backstory: str = Field(
        default="",
        description="Brief backstory providing context for the agent's perspective",
    )
    llm_tier: LLMTier = Field(
        default=LLMTier.TIER_2,
        description="Which LLM tier to use for this agent's responses",
    )
    conviction_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How convinced this agent is by the dissident position. "
            "0.0 = fully conformist, 1.0 = fully converted to dissent. "
            "Shifts each round based on dissident persuasion."
        ),
    )
    convinced_by: list[str] = Field(
        default_factory=list,
        description="IDs of dissident agents who influenced this agent's conviction shift",
    )

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Agent name cannot be blank")
        return v.strip()

    @property
    def is_dissident(self) -> bool:
        """Check if this agent is a dissident."""
        return self.role == AgentRole.DISSIDENT

    @property
    def system_prompt(self) -> str:
        """Generate a system prompt for this agent based on their profile."""
        personality_desc = self.personality.as_prompt_description()
        role_instruction = {
            AgentRole.CONFORMIST: "Engage constructively with the group discussion. Share your genuine perspective.",
            AgentRole.DISSIDENT: (
                "You MUST challenge the emerging consensus. Play devil's advocate. "
                "Steel-man the opposite position. Identify blind spots, hidden risks, and "
                "uncomfortable truths that the group is ignoring. Be respectful but unflinching."
            ),
            AgentRole.SYNTHESIZER: (
                "Your role is to synthesize all perspectives — including dissenting ones — "
                "into a coherent analysis. Identify areas of genuine agreement vs. unresolved tensions."
            ),
            AgentRole.OBSERVER: "Observe and note patterns in the group dynamics without participating directly.",
        }

        parts = [
            f"You are {self.name}.",
            f"Personality: {personality_desc}.",
        ]
        if self.backstory:
            parts.append(f"Background: {self.backstory}")
        parts.append(f"Role instruction: {role_instruction[self.role]}")

        return "\n".join(parts)
