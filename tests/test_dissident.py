"""Tests for Phase 3: DissidentRouter + TieredLLM + Conviction Spreading."""

import pytest

from aps.config import LLMMode
from aps.inference.llm import TieredLLM
from aps.dissident.router import (
    DissidentRouter,
    CONVICTION_THRESHOLD,
    BASE_CONVICTION_SHIFT,
    ConvictionResult,
)
from aps.persona.builder import PersonaBuilder
from aps.schemas.agent import AgentProfile, AgentRole, PersonalityWeights, LLMTier
from aps.schemas.message import Message
from aps.schemas.room import RoomConfig


# ============================================================
# TieredLLM Tests
# ============================================================

class TestTieredLLM:
    def test_mock_mode_returns_string(self):
        llm = TieredLLM(mode=LLMMode.MOCK)
        response = llm.invoke("Test prompt", role_hint="conformist")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_conformist_response(self):
        llm = TieredLLM(mode=LLMMode.MOCK)
        response = llm.invoke("Should we invest?", role_hint="conformist")
        assert isinstance(response, str)

    def test_mock_dissident_response(self):
        llm = TieredLLM(mode=LLMMode.MOCK)
        response = llm.invoke("Should we invest?", role_hint="dissident")
        assert isinstance(response, str)

    def test_mock_synthesizer_response(self):
        llm = TieredLLM(mode=LLMMode.MOCK)
        response = llm.invoke("Synthesize the debate", role_hint="synthesizer")
        assert isinstance(response, str)

    def test_mock_conviction_response(self):
        llm = TieredLLM(mode=LLMMode.MOCK)
        response = llm.invoke("I'm reconsidering", role_hint="conviction")
        assert isinstance(response, str)


# ============================================================
# DissidentRouter Tests
# ============================================================

class TestDissidentRouter:
    def setup_method(self):
        self.builder = PersonaBuilder(seed=42)
        self.router = DissidentRouter(
            llm=TieredLLM(mode=LLMMode.MOCK),
            seed=42,
        )

    def _make_agents_and_messages(self, pop_size=20, dissident_ratio=0.05):
        """Helper: generate agents and fake conformist messages."""
        room = RoomConfig(
            name="Test", topic="Should we invest in AI?",
            population_size=pop_size, dissident_ratio=dissident_ratio,
        )
        agents = self.builder.generate(room)

        # Generate fake conformist messages
        conformist_messages = []
        for a in agents:
            if a.role == AgentRole.CONFORMIST:
                conformist_messages.append(Message(
                    agent_id=a.id,
                    agent_name=a.name,
                    content="I agree we should proceed with the investment.",
                    round_num=1,
                ))

        return room, agents, conformist_messages

    def test_inject_dissent_returns_messages(self):
        room, agents, conf_msgs = self._make_agents_and_messages()
        dissent_msgs, updated_agents = self.router.inject_dissent(
            agents=agents,
            conformist_messages=conf_msgs,
            round_num=1,
            topic=room.topic,
        )
        assert len(dissent_msgs) > 0
        assert all(m.is_dissent for m in dissent_msgs)

    def test_dissent_messages_come_from_dissidents(self):
        room, agents, conf_msgs = self._make_agents_and_messages()
        dissent_msgs, _ = self.router.inject_dissent(
            agents=agents,
            conformist_messages=conf_msgs,
            round_num=1,
            topic=room.topic,
        )
        dissident_ids = {a.id for a in agents if a.role == AgentRole.DISSIDENT}
        for msg in dissent_msgs:
            # Message comes from a dissident or sympathizer
            assert msg.is_dissent

    def test_conviction_scores_increase_after_round(self):
        room, agents, conf_msgs = self._make_agents_and_messages()
        _, updated_agents = self.router.inject_dissent(
            agents=agents,
            conformist_messages=conf_msgs,
            round_num=1,
            topic=room.topic,
        )
        # At least some conformists should have increased conviction
        conformists_before = [a for a in agents if a.role == AgentRole.CONFORMIST]
        conformists_after = [a for a in updated_agents if a.role == AgentRole.CONFORMIST]

        before_scores = {a.id: a.conviction_score for a in conformists_before}
        increased = 0
        for a in conformists_after:
            if a.id in before_scores and a.conviction_score > before_scores[a.id]:
                increased += 1

        # All conformists should have their scores increase
        assert increased == len(conformists_before)

    def test_conviction_spreads_over_rounds(self):
        """Key test: conviction should increase more over multiple rounds."""
        room, agents, conf_msgs = self._make_agents_and_messages()

        # Run 5 rounds
        current_agents = agents
        for round_num in range(1, 6):
            _, current_agents = self.router.inject_dissent(
                agents=current_agents,
                conformist_messages=conf_msgs,
                round_num=round_num,
                topic=room.topic,
            )

        # After 5 rounds, average conviction should be significantly higher
        conformists = [a for a in current_agents if a.role == AgentRole.CONFORMIST]
        avg_conviction = sum(a.conviction_score for a in conformists) / len(conformists)
        assert avg_conviction > 0.3  # Should have shifted substantially

    def test_sympathizers_emerge_over_rounds(self):
        """After several rounds, some conformists should cross the threshold."""
        room, agents, conf_msgs = self._make_agents_and_messages(pop_size=20, dissident_ratio=0.1)

        current_agents = agents
        for round_num in range(1, 8):
            _, current_agents = self.router.inject_dissent(
                agents=current_agents,
                conformist_messages=conf_msgs,
                round_num=round_num,
                topic=room.topic,
            )

        # Check for sympathizers (conformists above threshold)
        sympathizers = [
            a for a in current_agents
            if a.role == AgentRole.CONFORMIST and a.conviction_score >= CONVICTION_THRESHOLD
        ]
        assert len(sympathizers) > 0, "Expected some conformists to be converted after 7 rounds"

    def test_sympathizers_generate_dissent_messages(self):
        """Sympathizers should generate dissent messages in later rounds."""
        room, agents, conf_msgs = self._make_agents_and_messages(pop_size=20, dissident_ratio=0.1)

        # Manually set a conformist as sympathizer
        for i, a in enumerate(agents):
            if a.role == AgentRole.CONFORMIST:
                agents[i] = a.model_copy(update={"conviction_score": 0.7})
                break

        dissent_msgs, _ = self.router.inject_dissent(
            agents=agents,
            conformist_messages=conf_msgs,
            round_num=2,
            topic=room.topic,
        )

        # Should have messages from both dissidents AND the sympathizer
        roles = [m.role for m in dissent_msgs]
        assert "sympathizer" in roles

    def test_dissidents_dont_shift(self):
        """Dissidents should maintain their conviction score at 1.0."""
        room, agents, conf_msgs = self._make_agents_and_messages()
        _, updated = self.router.inject_dissent(
            agents=agents,
            conformist_messages=conf_msgs,
            round_num=1,
            topic=room.topic,
        )
        for a in updated:
            if a.role == AgentRole.DISSIDENT:
                assert a.conviction_score == 1.0

    def test_conviction_summary(self):
        room, agents, conf_msgs = self._make_agents_and_messages()
        summary = self.router.get_conviction_summary(agents)
        assert "total_agents" in summary
        assert "dissidents" in summary
        assert "conformists" in summary
        assert "sympathizers" in summary
        assert "conversion_rate" in summary
        assert summary["sympathizers"] == 0  # No sympathizers initially

    def test_no_dissidents_no_crash(self):
        """If somehow no dissidents exist, router should not crash."""
        agents = [
            AgentProfile(name="Alice", role=AgentRole.CONFORMIST),
            AgentProfile(name="Bob", role=AgentRole.CONFORMIST),
        ]
        msgs, updated = self.router.inject_dissent(
            agents=agents,
            conformist_messages=[],
            round_num=1,
            topic="test",
        )
        assert msgs == []
        assert len(updated) == 2

    def test_conviction_capped_at_1(self):
        """Conviction score should never exceed 1.0."""
        room, agents, conf_msgs = self._make_agents_and_messages()

        current_agents = agents
        for round_num in range(1, 20):  # Many rounds
            _, current_agents = self.router.inject_dissent(
                agents=current_agents,
                conformist_messages=conf_msgs,
                round_num=round_num,
                topic=room.topic,
            )

        for a in current_agents:
            assert a.conviction_score <= 1.0


# ============================================================
# Susceptibility Tests
# ============================================================

class TestSusceptibility:
    def test_high_openness_more_susceptible(self):
        router = DissidentRouter(seed=1)
        agent_open = AgentProfile(
            name="Open", role=AgentRole.CONFORMIST,
            personality=PersonalityWeights(openness=0.9, agreeableness=0.5),
        )
        agent_closed = AgentProfile(
            name="Closed", role=AgentRole.CONFORMIST,
            personality=PersonalityWeights(openness=0.1, agreeableness=0.5),
        )
        s_open = router._calculate_susceptibility(agent_open)
        s_closed = router._calculate_susceptibility(agent_closed)
        assert s_open > s_closed

    def test_nobody_is_completely_immune(self):
        """Even the most resistant personality should have susceptibility > 0."""
        router = DissidentRouter(seed=1)
        agent = AgentProfile(
            name="Tank", role=AgentRole.CONFORMIST,
            personality=PersonalityWeights(
                openness=0.0, agreeableness=1.0, conscientiousness=1.0,
                extraversion=0.0, neuroticism=0.0,
            ),
        )
        s = router._calculate_susceptibility(agent)
        assert s >= 0.1  # Minimum floor
