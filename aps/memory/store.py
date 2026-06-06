"""In-memory context store — Phase 1 substitute for Qdrant/Zep.

Simple storage for agent messages and round context.
Will be replaced with vector DB integration in production.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from aps.schemas.message import Message


class MemoryStore:
    """Simple in-memory message store for agent context retrieval.

    Stores messages by simulation, round, and agent. Provides retrieval
    methods for building conversation context in prompts.
    """

    def __init__(self):
        # simulation_id -> round_num -> list[Message]
        self._rounds: dict[str, dict[int, list[Message]]] = defaultdict(lambda: defaultdict(list))
        # simulation_id -> agent_id -> list[Message]
        self._by_agent: dict[str, dict[str, list[Message]]] = defaultdict(lambda: defaultdict(list))

    def store(self, simulation_id: str, message: Message) -> None:
        """Store a message."""
        self._rounds[simulation_id][message.round_num].append(message)
        self._by_agent[simulation_id][message.agent_id].append(message)

    def store_many(self, simulation_id: str, messages: list[Message]) -> None:
        """Store multiple messages."""
        for m in messages:
            self.store(simulation_id, m)

    def get_round(self, simulation_id: str, round_num: int) -> list[Message]:
        """Get all messages from a specific round."""
        return list(self._rounds.get(simulation_id, {}).get(round_num, []))

    def get_agent_history(self, simulation_id: str, agent_id: str) -> list[Message]:
        """Get all messages from a specific agent."""
        return list(self._by_agent.get(simulation_id, {}).get(agent_id, []))

    def get_recent_context(
        self,
        simulation_id: str,
        max_rounds: int = 2,
        max_messages_per_round: int = 10,
    ) -> str:
        """Build a text context string from recent rounds for prompts.

        Args:
            simulation_id: The simulation to pull context from.
            max_rounds: How many recent rounds to include.
            max_messages_per_round: Max messages per round to include.

        Returns:
            Formatted context string for LLM prompts.
        """
        rounds = self._rounds.get(simulation_id, {})
        if not rounds:
            return "No prior discussion."

        sorted_rounds = sorted(rounds.keys(), reverse=True)[:max_rounds]
        sorted_rounds.reverse()  # Back to chronological order

        parts = []
        for rnum in sorted_rounds:
            msgs = rounds[rnum][:max_messages_per_round]
            round_text = f"--- Round {rnum} ---\n"
            for m in msgs:
                tag = " [DISSENT]" if m.is_dissent else ""
                round_text += f"  {m.agent_name}{tag}: {m.content[:300]}\n"
            parts.append(round_text)

        return "\n".join(parts)

    def get_all_messages(self, simulation_id: str) -> list[Message]:
        """Get all messages for a simulation, in order."""
        all_msgs = []
        for round_msgs in self._rounds.get(simulation_id, {}).values():
            all_msgs.extend(round_msgs)
        all_msgs.sort(key=lambda m: (m.round_num, m.timestamp))
        return all_msgs

    def clear(self, simulation_id: Optional[str] = None) -> None:
        """Clear stored messages."""
        if simulation_id:
            self._rounds.pop(simulation_id, None)
            self._by_agent.pop(simulation_id, None)
        else:
            self._rounds.clear()
            self._by_agent.clear()
