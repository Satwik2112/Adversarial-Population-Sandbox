"""Message and logging schemas — define the communication protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from aps.schemas.room import RoomConfig


class Message(BaseModel):
    """A single message produced by an agent during a simulation round."""

    agent_id: str = Field(..., description="ID of the agent who sent this message")
    agent_name: str = Field(..., description="Display name of the agent")
    content: str = Field(..., description="The message content")
    round_num: int = Field(..., ge=1, description="Simulation round number")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    is_dissent: bool = Field(
        default=False,
        description="Whether this message is a forced-dissent injection",
    )
    role: str = Field(
        default="conformist",
        description="Role of the agent when this message was sent",
    )

    model_config = {"ser_json_timedelta": "iso8601"}


class RoundSummary(BaseModel):
    """Summary of a single round of debate."""

    round_num: int = Field(..., ge=1)
    messages: list[Message] = Field(default_factory=list)
    num_conformist: int = Field(default=0, ge=0)
    num_dissent: int = Field(default=0, ge=0)
    synthesis: Optional[str] = Field(
        default=None,
        description="Synthesized summary of this round's discussion",
    )

    @property
    def total_messages(self) -> int:
        return len(self.messages)


class SimulationResult(BaseModel):
    """Complete result of a simulation run."""

    simulation_id: str = Field(..., description="Unique simulation run ID")
    room_config: RoomConfig = Field(..., description="Room configuration used")
    rounds: list[RoundSummary] = Field(default_factory=list)
    final_report: Optional[str] = Field(
        default=None,
        description="Final synthesized report from the ReportAgent",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    completed_at: Optional[datetime] = Field(default=None)
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (agent profiles, config, etc.)",
    )

    @property
    def total_messages(self) -> int:
        return sum(r.total_messages for r in self.rounds)

    @property
    def total_dissent_messages(self) -> int:
        return sum(r.num_dissent for r in self.rounds)
