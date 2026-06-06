"""LangGraph orchestrator — the core simulation state machine.

Full pipeline: generate_agents → discuss → inject_dissent → synthesize → (loop or report)

The conviction-spreading mechanic runs each round:
1. Conformists discuss the topic.
2. Dissidents + sympathizers inject adversarial counter-arguments.
3. Conviction scores shift — some conformists become sympathizers.
4. Synthesizer summarizes the round.
5. Repeat for N rounds (user-controlled iterations).
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from aps.config import LLMMode, get_settings
from aps.schemas.agent import AgentProfile, AgentRole, LLMTier
from aps.schemas.message import Message, RoundSummary, SimulationResult
from aps.schemas.room import RoomConfig
from aps.persona.builder import PersonaBuilder
from aps.inference.llm import TieredLLM
from aps.dissident.router import DissidentRouter
from aps.memory.store import MemoryStore
from aps.log_store import LogStore


class SimulationState(BaseModel):
    """State that flows through the LangGraph simulation pipeline."""

    # --- Configuration ---
    simulation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    room_config: Optional[RoomConfig] = Field(default=None)

    # --- Agents ---
    agents: list[AgentProfile] = Field(default_factory=list)

    # --- Round tracking ---
    current_round: int = Field(default=0)
    max_rounds: int = Field(default=3)

    # --- Messages ---
    current_round_messages: list[Message] = Field(default_factory=list)
    round_summaries: list[RoundSummary] = Field(default_factory=list)

    # --- Output ---
    final_report: Optional[str] = Field(default=None)
    status: str = Field(default="initialized")


def should_continue(state: SimulationState) -> str:
    """Conditional edge: 'discuss' for more rounds, 'report' when done."""
    if state.current_round < state.max_rounds:
        return "discuss"
    return "report"


class SimulationEngine:
    """Runs the full simulation pipeline.

    Manages the lifecycle: persona generation → multi-round debate
    with conviction spreading → final report synthesis.
    """

    def __init__(
        self,
        llm: Optional[TieredLLM] = None,
        persona_seed: Optional[int] = None,
        dissident_seed: Optional[int] = None,
    ):
        self._llm = llm or TieredLLM(mode=LLMMode.MOCK)
        self._builder = PersonaBuilder(seed=persona_seed)
        self._router = DissidentRouter(llm=self._llm, seed=dissident_seed)
        self._memory = MemoryStore()
        self._log_store = LogStore()

    def run(self, room_config: RoomConfig) -> SimulationResult:
        """Run a full simulation end-to-end.

        Args:
            room_config: Configuration for the simulation room.

        Returns:
            SimulationResult with all rounds, messages, and final report.
        """
        state = SimulationState(
            room_config=room_config,
            max_rounds=room_config.num_rounds,
        )

        # Step 1: Generate agents
        state = self._generate_agents(state)
        self._log_store.log_event(
            state.simulation_id, "agents_generated",
            count=len(state.agents),
            dissidents=sum(1 for a in state.agents if a.is_dissident),
        )

        # Step 2: Run rounds
        while should_continue(state) == "discuss":
            state = self._run_round(state)

        # Step 3: Generate final report
        state = self._generate_report(state)

        # Build result
        result = SimulationResult(
            simulation_id=state.simulation_id,
            room_config=room_config,
            rounds=state.round_summaries,
            final_report=state.final_report,
            metadata={
                "agents": [a.model_dump(mode="json") for a in state.agents],
                "conviction_summary": self._router.get_conviction_summary(state.agents),
            },
        )

        self._log_store.log_event(
            state.simulation_id, "simulation_complete",
            total_rounds=len(state.round_summaries),
            total_messages=result.total_messages,
        )

        return result

    def _generate_agents(self, state: SimulationState) -> SimulationState:
        """Generate the agent population."""
        agents = self._builder.generate(state.room_config)
        return state.model_copy(update={
            "agents": agents,
            "status": "agents_generated",
        })

    def _run_round(self, state: SimulationState) -> SimulationState:
        """Run a single round: discuss → inject dissent → synthesize."""
        round_num = state.current_round + 1

        self._log_store.log_event(
            state.simulation_id, "round_start", round_num=round_num,
        )

        # --- Conformists discuss ---
        conformist_messages = self._conformist_discussion(state, round_num)

        # --- Dissidents inject + conviction spreading ---
        dissent_messages, updated_agents = self._router.inject_dissent(
            agents=state.agents,
            conformist_messages=conformist_messages,
            round_num=round_num,
            topic=state.room_config.topic,
        )

        # --- Store all messages ---
        all_round_messages = conformist_messages + dissent_messages
        self._memory.store_many(state.simulation_id, all_round_messages)

        for msg in all_round_messages:
            self._log_store.log_message(state.simulation_id, msg)

        # --- Synthesize ---
        synthesis = self._synthesize_round(state, all_round_messages, round_num)

        # --- Build round summary ---
        num_dissent = sum(1 for m in all_round_messages if m.is_dissent)
        summary = RoundSummary(
            round_num=round_num,
            messages=all_round_messages,
            num_conformist=len(conformist_messages),
            num_dissent=num_dissent,
            synthesis=synthesis,
        )

        conviction_summary = self._router.get_conviction_summary(updated_agents)
        self._log_store.log_event(
            state.simulation_id, "round_complete",
            round_num=round_num,
            conviction_summary=conviction_summary,
        )

        return state.model_copy(update={
            "agents": updated_agents,
            "current_round": round_num,
            "current_round_messages": [],
            "round_summaries": state.round_summaries + [summary],
            "status": f"round_{round_num}_complete",
        })

    def _conformist_discussion(
        self, state: SimulationState, round_num: int
    ) -> list[Message]:
        """Generate messages from conformist agents (and unconvinced)."""
        messages = []
        context = self._memory.get_recent_context(state.simulation_id)

        for agent in state.agents:
            # Skip dissidents and synthesizer — they speak in other phases
            if agent.role == AgentRole.DISSIDENT:
                continue
            if agent.role == AgentRole.SYNTHESIZER:
                continue

            # Conformists with high conviction now argue from a shifted perspective
            if agent.conviction_score >= 0.5:
                role_hint = "conviction"
                extra_instruction = (
                    f"\nYour conviction has shifted (score: {agent.conviction_score:.2f}). "
                    f"You are now sympathetic to the dissenting view. "
                    f"Argue from your new perspective — explain what changed your mind."
                )
            else:
                role_hint = "conformist"
                extra_instruction = ""

            prompt = (
                f"Scenario/Topic: {state.room_config.topic}\n\n"
                f"Previous discussion:\n{context}\n\n"
                f"Round {round_num}: Analyze the scenario completely. Provide a unique, natural-language argument "
                f"based directly on the scenario provided. Be authentic to your personality and background. "
                f"Do not use generic statements; integrate the specific details of the topic."
                f"{extra_instruction}"
            )

            response = self._llm.invoke(
                prompt=prompt,
                system_prompt=agent.system_prompt,
                tier=agent.llm_tier,
                role_hint=role_hint,
            )

            messages.append(Message(
                agent_id=agent.id,
                agent_name=agent.name,
                content=response,
                round_num=round_num,
                is_dissent=agent.conviction_score >= 0.5,
                role=agent.role.value,
            ))

        return messages

    def _synthesize_round(
        self, state: SimulationState, messages: list[Message], round_num: int
    ) -> str:
        """Generate a round synthesis using the synthesizer agent."""
        synthesizer = next(
            (a for a in state.agents if a.role == AgentRole.SYNTHESIZER),
            None,
        )

        if not synthesizer:
            return "No synthesizer agent available."

        msg_summary = "\n".join(
            f"- {m.agent_name} ({'DISSENT' if m.is_dissent else 'CONFORM'}): {m.content[:200]}"
            for m in messages
        )

        conviction_summary = self._router.get_conviction_summary(state.agents)

        prompt = (
            f"Topic: {state.room_config.topic}\n\n"
            f"Round {round_num} discussion:\n{msg_summary}\n\n"
            f"Conviction state: {conviction_summary}\n\n"
            f"Synthesize all perspectives. Identify:\n"
            f"1. Points of genuine agreement\n"
            f"2. Unresolved tensions\n"
            f"3. Key insights from the dissenting voices\n"
            f"4. How convictions are shifting across the group"
        )

        return self._llm.invoke(
            prompt=prompt,
            system_prompt=synthesizer.system_prompt,
            tier=synthesizer.llm_tier,
            role_hint="synthesizer",
        )

    def _generate_report(self, state: SimulationState) -> SimulationState:
        """Generate the final report from all round summaries."""
        round_summaries_text = "\n\n".join(
            f"=== Round {rs.round_num} ===\n"
            f"Conformist messages: {rs.num_conformist}\n"
            f"Dissent messages: {rs.num_dissent}\n"
            f"Synthesis: {rs.synthesis or 'N/A'}"
            for rs in state.round_summaries
        )

        conviction_summary = self._router.get_conviction_summary(state.agents)

        prompt = (
            f"Topic: {state.room_config.topic}\n\n"
            f"Simulation completed after {len(state.round_summaries)} rounds.\n\n"
            f"{round_summaries_text}\n\n"
            f"Final conviction state: {conviction_summary}\n\n"
            f"Generate a comprehensive final report covering:\n"
            f"1. CONSENSUS POINTS — What did the group agree on?\n"
            f"2. DISSENTING VIEWS — What counter-arguments were raised?\n"
            f"3. BLACK SWAN RISKS — What hidden risks did the dissidents uncover?\n"
            f"4. CONVICTION SHIFTS — How did opinions change over rounds?\n"
            f"5. RECOMMENDATIONS — What should decision-makers take from this?"
        )

        report = self._llm.invoke(
            prompt=prompt,
            system_prompt="You are an impartial analyst producing the final simulation report.",
            tier=LLMTier.TIER_1,  # Final report uses frontier model
            role_hint="synthesizer",
        )

        return state.model_copy(update={
            "final_report": report,
            "status": "completed",
        })
