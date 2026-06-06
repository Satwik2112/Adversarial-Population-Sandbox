"""Tests for Phase 4: Full orchestrator integration."""

import pytest

from aps.config import LLMMode
from aps.inference.llm import TieredLLM
from aps.memory.store import MemoryStore
from aps.orchestrator.graph import SimulationEngine, SimulationState, should_continue
from aps.schemas.agent import AgentRole
from aps.schemas.message import Message
from aps.schemas.room import RoomConfig, RoomType


# ============================================================
# MemoryStore Tests
# ============================================================

class TestMemoryStore:
    def test_store_and_retrieve_by_round(self):
        store = MemoryStore()
        msg = Message(agent_id="a", agent_name="Alice", content="Hello", round_num=1)
        store.store("sim-1", msg)
        assert len(store.get_round("sim-1", 1)) == 1
        assert len(store.get_round("sim-1", 2)) == 0

    def test_store_and_retrieve_by_agent(self):
        store = MemoryStore()
        msg = Message(agent_id="a1", agent_name="Alice", content="Hello", round_num=1)
        store.store("sim-1", msg)
        assert len(store.get_agent_history("sim-1", "a1")) == 1

    def test_store_many(self):
        store = MemoryStore()
        msgs = [
            Message(agent_id="a", agent_name="A", content="msg1", round_num=1),
            Message(agent_id="b", agent_name="B", content="msg2", round_num=1),
        ]
        store.store_many("sim-1", msgs)
        assert len(store.get_round("sim-1", 1)) == 2

    def test_get_recent_context(self):
        store = MemoryStore()
        for i in range(1, 4):
            store.store("sim-1", Message(
                agent_id="a", agent_name="Alice", content=f"Round {i} message",
                round_num=i,
            ))
        context = store.get_recent_context("sim-1", max_rounds=2)
        assert "Round 2" in context or "Round 3" in context

    def test_empty_context(self):
        store = MemoryStore()
        context = store.get_recent_context("nonexistent")
        assert context == "No prior discussion."

    def test_clear_specific_simulation(self):
        store = MemoryStore()
        store.store("sim-1", Message(agent_id="a", agent_name="A", content="x", round_num=1))
        store.store("sim-2", Message(agent_id="b", agent_name="B", content="y", round_num=1))
        store.clear("sim-1")
        assert len(store.get_round("sim-1", 1)) == 0
        assert len(store.get_round("sim-2", 1)) == 1

    def test_get_all_messages(self):
        store = MemoryStore()
        store.store("sim-1", Message(agent_id="a", agent_name="A", content="r1", round_num=1))
        store.store("sim-1", Message(agent_id="a", agent_name="A", content="r2", round_num=2))
        all_msgs = store.get_all_messages("sim-1")
        assert len(all_msgs) == 2
        assert all_msgs[0].round_num <= all_msgs[1].round_num


# ============================================================
# SimulationState Tests
# ============================================================

class TestSimulationState:
    def test_default_state(self):
        state = SimulationState()
        assert state.current_round == 0
        assert state.status == "initialized"

    def test_should_continue_logic(self):
        state = SimulationState(current_round=1, max_rounds=3)
        assert should_continue(state) == "discuss"

        state2 = SimulationState(current_round=3, max_rounds=3)
        assert should_continue(state2) == "report"


# ============================================================
# Full Simulation Engine Tests (Mock LLM)
# ============================================================

class TestSimulationEngine:
    def test_mini_simulation(self):
        """Run a mini simulation with 5 agents, 2 rounds."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=42,
            dissident_seed=42,
        )
        room = RoomConfig(
            name="Test Boardroom",
            topic="Should we invest $100M in quantum computing?",
            population_size=5,
            dissident_ratio=0.2,
            num_rounds=2,
        )
        result = engine.run(room)

        assert result.simulation_id is not None
        assert len(result.rounds) == 2
        assert result.total_messages > 0
        assert result.final_report is not None
        assert len(result.final_report) > 0

    def test_full_boardroom_simulation(self):
        """Run a full boardroom preset simulation."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=123,
            dissident_seed=123,
        )
        room = RoomConfig.from_preset(RoomType.BOARDROOM, "Should we acquire StartupX?")
        result = engine.run(room)

        assert len(result.rounds) == room.num_rounds
        assert result.final_report is not None

    def test_conviction_spreads_during_simulation(self):
        """Verify conviction scores actually change during a multi-round simulation."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=42,
            dissident_seed=42,
        )
        room = RoomConfig(
            name="Conviction Test",
            topic="AI regulation policy",
            population_size=10,
            dissident_ratio=0.1,
            num_rounds=5,
        )
        result = engine.run(room)

        # Check conviction summary in metadata
        conviction = result.metadata.get("conviction_summary", {})
        assert conviction.get("avg_conviction", 0) > 0, "Conviction should have shifted"

    def test_dissent_messages_present_each_round(self):
        """Every round should contain dissent messages."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=42,
            dissident_seed=42,
        )
        room = RoomConfig(
            name="Dissent Check",
            topic="Climate policy",
            population_size=8,
            dissident_ratio=0.15,
            num_rounds=3,
        )
        result = engine.run(room)

        for rnd in result.rounds:
            assert rnd.num_dissent > 0, f"Round {rnd.round_num} has no dissent messages"

    def test_synthesis_present_each_round(self):
        """Every round should have a synthesis."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=42,
            dissident_seed=42,
        )
        room = RoomConfig(
            name="Synthesis Check",
            topic="Test topic",
            population_size=6,
            dissident_ratio=0.2,
            num_rounds=2,
        )
        result = engine.run(room)

        for rnd in result.rounds:
            assert rnd.synthesis is not None
            assert len(rnd.synthesis) > 0

    def test_agents_in_metadata(self):
        """Result metadata should contain agent profiles."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=42,
        )
        room = RoomConfig(name="Meta Test", topic="Test", population_size=5, num_rounds=1)
        result = engine.run(room)

        agents_data = result.metadata.get("agents", [])
        assert len(agents_data) == 5

    def test_single_round_simulation(self):
        """Even a 1-round simulation should complete successfully."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=1,
        )
        room = RoomConfig(name="Quick", topic="Quick test", population_size=4, num_rounds=1)
        result = engine.run(room)

        assert len(result.rounds) == 1
        assert result.final_report is not None

    def test_large_population_completes(self):
        """Simulation with many agents should complete without errors."""
        engine = SimulationEngine(
            llm=TieredLLM(mode=LLMMode.MOCK),
            persona_seed=99,
        )
        room = RoomConfig(
            name="Stress Test",
            topic="Market dynamics",
            population_size=50,
            num_rounds=2,
        )
        result = engine.run(room)

        assert len(result.rounds) == 2
        assert result.total_messages > 40  # Should have many messages
