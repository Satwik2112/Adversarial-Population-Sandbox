"""Tests for Phase 1: Schemas, Config, LogStore, and Orchestrator skeleton."""

import math
import tempfile
from pathlib import Path

import pytest

from aps.config import LLMMode, Settings, get_settings, reset_settings
from aps.schemas.room import RoomConfig, RoomType, ROOM_PRESETS
from aps.schemas.agent import (
    AgentProfile,
    AgentRole,
    LLMTier,
    PersonalityWeights,
)
from aps.schemas.message import Message, RoundSummary, SimulationResult
from aps.log_store import LogStore
from aps.orchestrator.graph import SimulationState, should_continue


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def setup_method(self):
        reset_settings()

    def test_default_settings(self):
        settings = Settings(
            _env_file=None,  # Don't load .env in tests
        )
        # aps_llm_mode may be overridden by env vars in test runs, skip that check
        assert settings.aps_default_dissident_ratio == 0.01
        assert settings.tier1_model == "gemini/gemini-2.5-pro"
        assert settings.tier2_model == "gemini/gemini-2.5-flash"

    def test_llm_mode_enum(self):
        assert LLMMode.MOCK.value == "mock"
        assert LLMMode.LIVE.value == "live"


# ============================================================
# Room Schema Tests
# ============================================================

class TestRoomConfig:
    def test_create_basic_room(self):
        room = RoomConfig(name="Test Room", topic="AI Safety")
        assert room.name == "Test Room"
        assert room.topic == "AI Safety"
        assert room.room_type == RoomType.CUSTOM
        assert room.population_size == 20
        assert room.dissident_ratio == 0.01
        assert room.num_rounds == 3

    def test_min_dissidents_calculation(self):
        room = RoomConfig(name="Test", topic="Test", population_size=100, dissident_ratio=0.01)
        assert room.min_dissidents == 1  # ceil(100 * 0.01) = 1

        room2 = RoomConfig(name="Test", topic="Test", population_size=100, dissident_ratio=0.05)
        assert room2.min_dissidents == 5  # ceil(100 * 0.05) = 5

        room3 = RoomConfig(name="Test", topic="Test", population_size=3, dissident_ratio=0.01)
        assert room3.min_dissidents == 1  # max(1, ceil(3 * 0.01)) = 1

    def test_from_preset_boardroom(self):
        room = RoomConfig.from_preset(RoomType.BOARDROOM, "Should we acquire CompanyX?")
        assert room.room_type == RoomType.BOARDROOM
        assert room.population_size == 12
        assert room.num_rounds == 3
        assert "CompanyX" in room.topic

    def test_from_preset_social_swarm(self):
        room = RoomConfig.from_preset(RoomType.SOCIAL_SWARM, "Cancel culture dynamics")
        assert room.population_size == 100
        assert room.num_rounds == 5

    def test_from_preset_with_overrides(self):
        room = RoomConfig.from_preset(
            RoomType.BOARDROOM,
            "Budget review",
            population_size=24,
            num_rounds=5,
        )
        assert room.population_size == 24
        assert room.num_rounds == 5

    def test_invalid_population_too_small(self):
        with pytest.raises(Exception):
            RoomConfig(name="Test", topic="Test", population_size=1)

    def test_invalid_dissident_ratio(self):
        with pytest.raises(Exception):
            RoomConfig(name="Test", topic="Test", dissident_ratio=1.5)

    def test_invalid_empty_name(self):
        with pytest.raises(Exception):
            RoomConfig(name="", topic="Test")

    def test_all_room_presets_exist(self):
        for room_type in RoomType:
            assert room_type in ROOM_PRESETS


# ============================================================
# Agent Schema Tests
# ============================================================

class TestPersonalityWeights:
    def test_default_personality(self):
        p = PersonalityWeights()
        assert p.openness == 0.5
        assert p.agreeableness == 0.5

    def test_custom_personality(self):
        p = PersonalityWeights(openness=0.9, agreeableness=0.1)
        assert p.openness == 0.9
        assert p.agreeableness == 0.1

    def test_out_of_range_rejected(self):
        with pytest.raises(Exception):
            PersonalityWeights(openness=1.5)
        with pytest.raises(Exception):
            PersonalityWeights(agreeableness=-0.1)

    def test_prompt_description_high_openness(self):
        p = PersonalityWeights(openness=0.9, agreeableness=0.2)
        desc = p.as_prompt_description()
        assert "creative" in desc.lower() or "unconventional" in desc.lower()
        assert "skeptical" in desc.lower() or "challenges" in desc.lower()

    def test_prompt_description_balanced(self):
        p = PersonalityWeights()  # all 0.5
        desc = p.as_prompt_description()
        assert "balanced" in desc.lower()


class TestAgentProfile:
    def test_create_agent(self):
        agent = AgentProfile(name="Alice")
        assert agent.name == "Alice"
        assert agent.role == AgentRole.CONFORMIST
        assert agent.llm_tier == LLMTier.TIER_2
        assert agent.id  # UUID generated

    def test_dissident_agent(self):
        agent = AgentProfile(name="Rebel", role=AgentRole.DISSIDENT)
        assert agent.is_dissident is True

    def test_conformist_not_dissident(self):
        agent = AgentProfile(name="Conformist")
        assert agent.is_dissident is False

    def test_system_prompt_contains_name(self):
        agent = AgentProfile(name="Dr. Smith", backstory="20 years in finance")
        prompt = agent.system_prompt
        assert "Dr. Smith" in prompt
        assert "20 years in finance" in prompt

    def test_dissident_system_prompt(self):
        agent = AgentProfile(name="Rebel", role=AgentRole.DISSIDENT)
        prompt = agent.system_prompt
        assert "challenge" in prompt.lower() or "devil" in prompt.lower()

    def test_blank_name_rejected(self):
        with pytest.raises(Exception):
            AgentProfile(name="   ")

    def test_unique_ids(self):
        a1 = AgentProfile(name="A")
        a2 = AgentProfile(name="B")
        assert a1.id != a2.id


# ============================================================
# Message Schema Tests
# ============================================================

class TestMessage:
    def test_create_message(self):
        msg = Message(
            agent_id="abc-123",
            agent_name="Alice",
            content="I think we should proceed.",
            round_num=1,
        )
        assert msg.agent_id == "abc-123"
        assert msg.content == "I think we should proceed."
        assert msg.is_dissent is False
        assert msg.timestamp is not None

    def test_dissent_message(self):
        msg = Message(
            agent_id="xyz",
            agent_name="Rebel",
            content="I disagree strongly.",
            round_num=2,
            is_dissent=True,
        )
        assert msg.is_dissent is True

    def test_invalid_round_num(self):
        with pytest.raises(Exception):
            Message(agent_id="x", agent_name="A", content="test", round_num=0)


class TestRoundSummary:
    def test_empty_round(self):
        rs = RoundSummary(round_num=1)
        assert rs.total_messages == 0
        assert rs.synthesis is None

    def test_round_with_messages(self):
        msgs = [
            Message(agent_id="a", agent_name="A", content="yes", round_num=1),
            Message(agent_id="b", agent_name="B", content="no", round_num=1, is_dissent=True),
        ]
        rs = RoundSummary(round_num=1, messages=msgs, num_conformist=1, num_dissent=1)
        assert rs.total_messages == 2


class TestSimulationResult:
    def test_create_result(self):
        room = RoomConfig(name="Test", topic="Test topic")
        result = SimulationResult(simulation_id="sim-001", room_config=room)
        assert result.total_messages == 0
        assert result.total_dissent_messages == 0
        assert result.final_report is None


# ============================================================
# LogStore Tests
# ============================================================

class TestLogStore:
    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            entry = store.append("sim-001", "test_event", {"key": "value"})
            assert entry.event_type == "test_event"
            assert entry.simulation_id == "sim-001"
            assert store.entry_count == 1

    def test_log_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            msg = Message(agent_id="a", agent_name="Alice", content="Hello", round_num=1)
            entry = store.log_message("sim-001", msg)
            assert entry.event_type == "message"
            assert entry.data["content"] == "Hello"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            store.append("sim-002", "event1", {"a": 1})
            store.append("sim-002", "event2", {"b": 2})

            # Load from file
            loaded = store.load_from_file("sim-002")
            assert len(loaded) == 2
            assert loaded[0].event_type == "event1"
            assert loaded[1].event_type == "event2"

    def test_filter_by_simulation_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            store.append("sim-A", "event", {"x": 1})
            store.append("sim-B", "event", {"x": 2})
            store.append("sim-A", "event", {"x": 3})

            entries = store.get_entries(simulation_id="sim-A")
            assert len(entries) == 2

    def test_filter_by_event_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            store.append("sim-001", "message", {"x": 1})
            store.append("sim-001", "event", {"x": 2})

            entries = store.get_entries(event_type="message")
            assert len(entries) == 1

    def test_immutability_file_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LogStore(log_dir=tmpdir)
            store.append("sim-001", "event", {"data": "important"})
            store.clear_buffer()
            assert store.entry_count == 0

            # But the file is still there
            loaded = store.load_from_file("sim-001")
            assert len(loaded) == 1


# ============================================================
# Orchestrator State Tests
# ============================================================

class TestSimulationState:
    def test_default_state(self):
        state = SimulationState()
        assert state.current_round == 0
        assert state.status == "initialized"
        assert len(state.agents) == 0

    def test_state_with_room(self):
        room = RoomConfig(name="Test", topic="Test topic")
        state = SimulationState(room_config=room, max_rounds=5)
        assert state.max_rounds == 5
        assert state.room_config.name == "Test"

    def test_should_continue_more_rounds(self):
        state = SimulationState(current_round=1, max_rounds=3)
        assert should_continue(state) == "discuss"

    def test_should_continue_done(self):
        state = SimulationState(current_round=3, max_rounds=3)
        assert should_continue(state) == "report"

    def test_should_continue_exactly_at_max(self):
        state = SimulationState(current_round=5, max_rounds=5)
        assert should_continue(state) == "report"
