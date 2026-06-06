"""Room schemas — define the simulation environment."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RoomType(str, Enum):
    """Types of simulation rooms."""
    BOARDROOM = "boardroom"
    SOCIAL_SWARM = "social_swarm"
    POLITICAL_ARENA = "political_arena"
    CUSTOM = "custom"


# Default presets for each room type
ROOM_PRESETS: dict[RoomType, dict] = {
    RoomType.BOARDROOM: {
        "description": "Corporate board meeting — strategic decisions, risk analysis, M&A scenarios.",
        "default_population": 12,
        "default_rounds": 3,
    },
    RoomType.SOCIAL_SWARM: {
        "description": "Social media simulation — viral dynamics, opinion cascades, echo chambers.",
        "default_population": 100,
        "default_rounds": 5,
    },
    RoomType.POLITICAL_ARENA: {
        "description": "Political debate — policy analysis, coalition dynamics, public opinion.",
        "default_population": 50,
        "default_rounds": 4,
    },
    RoomType.CUSTOM: {
        "description": "User-defined scenario with custom parameters.",
        "default_population": 20,
        "default_rounds": 3,
    },
}


class RoomConfig(BaseModel):
    """Configuration for a simulation room."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name for this simulation room",
    )
    room_type: RoomType = Field(
        default=RoomType.CUSTOM,
        description="Type of simulation room",
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="The topic or scenario to stress-test",
    )
    population_size: int = Field(
        default=20,
        ge=3,
        le=10_000,
        description="Number of agents in the simulation (min 3 for meaningful dissent)",
    )
    dissident_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of population forced into dissident role (default 1%)",
    )
    num_rounds: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of debate rounds",
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context or background information for agents",
    )

    @field_validator("population_size")
    @classmethod
    def validate_population_for_dissent(cls, v: int) -> int:
        """Ensure population is large enough for at least 1 dissident."""
        # With minimum ratio 0.01 and ceil(), pop >= 3 guarantees at least 1 dissident
        return v

    @property
    def min_dissidents(self) -> int:
        """Minimum number of dissident agents (at least 1)."""
        import math
        return max(1, math.ceil(self.population_size * self.dissident_ratio))

    @classmethod
    def from_preset(cls, room_type: RoomType, topic: str, **overrides) -> RoomConfig:
        """Create a RoomConfig from a preset room type."""
        preset = ROOM_PRESETS[room_type]
        defaults = {
            "name": f"{room_type.value.replace('_', ' ').title()} — {topic[:50]}",
            "room_type": room_type,
            "topic": topic,
            "population_size": preset["default_population"],
            "num_rounds": preset["default_rounds"],
        }
        defaults.update(overrides)
        return cls(**defaults)
