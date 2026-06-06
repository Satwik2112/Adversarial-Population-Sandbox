"""Tests for Phase 2: PersonaBuilder."""

import math

import pytest

from aps.persona.builder import PersonaBuilder
from aps.schemas.agent import AgentProfile, AgentRole, LLMTier
from aps.schemas.room import RoomConfig, RoomType


class TestPersonaBuilder:
    def setup_method(self):
        self.builder = PersonaBuilder(seed=42)

    def test_generate_correct_population_size(self):
        room = RoomConfig(name="Test", topic="Test", population_size=20)
        agents = self.builder.generate(room)
        assert len(agents) == 20

    def test_generate_large_population(self):
        room = RoomConfig(name="Test", topic="Test", population_size=100)
        agents = self.builder.generate(room)
        assert len(agents) == 100

    def test_dissident_ratio_enforced(self):
        room = RoomConfig(name="Test", topic="Test", population_size=100, dissident_ratio=0.05)
        agents = self.builder.generate(room)
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]
        assert len(dissidents) >= math.ceil(100 * 0.05)

    def test_default_1_percent_dissident(self):
        room = RoomConfig(name="Test", topic="Test", population_size=100, dissident_ratio=0.01)
        agents = self.builder.generate(room)
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]
        assert len(dissidents) >= 1

    def test_at_least_one_synthesizer(self):
        room = RoomConfig(name="Test", topic="Test", population_size=20)
        agents = self.builder.generate(room)
        synthesizers = [a for a in agents if a.role == AgentRole.SYNTHESIZER]
        assert len(synthesizers) >= 1

    def test_synthesizer_uses_tier1(self):
        room = RoomConfig(name="Test", topic="Test", population_size=10)
        agents = self.builder.generate(room)
        synthesizers = [a for a in agents if a.role == AgentRole.SYNTHESIZER]
        for s in synthesizers:
            assert s.llm_tier == LLMTier.TIER_1

    def test_all_agents_have_unique_ids(self):
        room = RoomConfig(name="Test", topic="Test", population_size=30)
        agents = self.builder.generate(room)
        ids = [a.id for a in agents]
        assert len(ids) == len(set(ids))

    def test_all_agents_have_unique_names(self):
        room = RoomConfig(name="Test", topic="Test", population_size=30)
        agents = self.builder.generate(room)
        names = [a.name for a in agents]
        assert len(names) == len(set(names))

    def test_dissident_personality_traits(self):
        room = RoomConfig(name="Test", topic="Test", population_size=20)
        agents = self.builder.generate(room)
        dissidents = [a for a in agents if a.role == AgentRole.DISSIDENT]
        for d in dissidents:
            assert d.personality.openness >= 0.75  # High openness
            assert d.personality.agreeableness <= 0.25  # Low agreeableness

    def test_conformist_personality_traits(self):
        room = RoomConfig(name="Test", topic="Test", population_size=20)
        agents = self.builder.generate(room)
        conformists = [a for a in agents if a.role == AgentRole.CONFORMIST]
        for c in conformists:
            assert c.personality.agreeableness >= 0.6  # High agreeableness

    def test_all_agents_have_backstories(self):
        room = RoomConfig(name="Test", topic="Test", population_size=15)
        agents = self.builder.generate(room)
        for a in agents:
            assert len(a.backstory) > 0

    def test_reproducible_with_seed(self):
        room = RoomConfig(name="Test", topic="Test", population_size=10)
        builder1 = PersonaBuilder(seed=123)
        builder2 = PersonaBuilder(seed=123)
        agents1 = builder1.generate(room)
        agents2 = builder2.generate(room)
        # Same seed → same names and roles (IDs will differ due to uuid4)
        assert [a.name for a in agents1] == [a.name for a in agents2]
        assert [a.role for a in agents1] == [a.role for a in agents2]

    def test_minimum_population_3(self):
        room = RoomConfig(name="Test", topic="Test", population_size=3)
        agents = self.builder.generate(room)
        assert len(agents) == 3
        roles = [a.role for a in agents]
        assert AgentRole.DISSIDENT in roles

    def test_preset_boardroom_population(self):
        room = RoomConfig.from_preset(RoomType.BOARDROOM, "Should we merge?")
        agents = self.builder.generate(room)
        assert len(agents) == 12

    def test_conviction_scores_initial(self):
        """Dissidents start at 1.0, conformists at 0.0, synthesizers at 0.5."""
        room = RoomConfig(name="Test", topic="Test", population_size=20)
        agents = self.builder.generate(room)
        for a in agents:
            if a.role == AgentRole.DISSIDENT:
                assert a.conviction_score == 1.0
            elif a.role == AgentRole.CONFORMIST:
                assert a.conviction_score == 0.0
            elif a.role == AgentRole.SYNTHESIZER:
                assert a.conviction_score == 0.5
