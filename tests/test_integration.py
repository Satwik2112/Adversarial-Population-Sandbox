"""Tests for Phase 5: ReportAgent + Full Integration."""

import pytest

from aps.config import LLMMode
from aps.inference.llm import TieredLLM
from aps.orchestrator.graph import SimulationEngine
from aps.reporting.report_agent import ReportAgent
from aps.schemas.agent import AgentRole
from aps.schemas.room import RoomConfig, RoomType


# ============================================================
# ReportAgent Tests
# ============================================================

class TestReportAgent:
    def setup_method(self):
        self.llm = TieredLLM(mode=LLMMode.MOCK)
        self.engine = SimulationEngine(llm=self.llm, persona_seed=42, dissident_seed=42)
        self.reporter = ReportAgent(llm=self.llm)

    def _run_simulation(self, pop=10, rounds=3, ratio=0.1):
        room = RoomConfig(
            name="Test Room",
            topic="Should we pivot our business model?",
            population_size=pop,
            dissident_ratio=ratio,
            num_rounds=rounds,
        )
        return self.engine.run(room)

    def test_analyze_returns_string(self):
        result = self._run_simulation()
        report = self.reporter.analyze(result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_conviction_dynamics(self):
        result = self._run_simulation(rounds=5)
        dynamics = self.reporter.analyze_conviction_dynamics(result)

        assert "total_conformists" in dynamics
        assert "total_dissidents" in dynamics
        assert "converted_count" in dynamics
        assert "conversion_rate" in dynamics
        assert "most_convinced" in dynamics
        assert "most_resistant" in dynamics
        assert "dissent_influencers" in dynamics

    def test_executive_summary(self):
        result = self._run_simulation()
        summary = self.reporter.generate_executive_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_report_with_no_rounds(self):
        """Edge case: handle an empty simulation result."""
        room = RoomConfig(name="Empty", topic="Nothing", population_size=3, num_rounds=1)
        result = self.engine.run(room)
        report = self.reporter.analyze(result)
        assert isinstance(report, str)


# ============================================================
# Full End-to-End Integration Tests
# ============================================================

class TestFullIntegration:
    """Complete end-to-end tests exercising the full pipeline."""

    def test_boardroom_scenario(self):
        """Full boardroom scenario from creation through report."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=1)
        reporter = ReportAgent(llm=llm)

        room = RoomConfig.from_preset(RoomType.BOARDROOM, "Should we acquire CompanyX for $2B?")
        result = engine.run(room)

        # Verify simulation completed
        assert result.final_report is not None
        assert len(result.rounds) == room.num_rounds
        assert result.total_messages > 0

        # Verify report
        report = reporter.analyze(result)
        assert len(report) > 0

        # Verify conviction dynamics
        dynamics = reporter.analyze_conviction_dynamics(result)
        assert dynamics["total_conformists"] > 0
        assert dynamics["total_dissidents"] > 0

    def test_social_swarm_scenario(self):
        """Social swarm with large population."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=42)

        room = RoomConfig.from_preset(
            RoomType.SOCIAL_SWARM,
            "Is cancel culture beneficial or harmful?",
            population_size=30,  # Scale down for test speed
            num_rounds=3,
        )
        result = engine.run(room)

        assert len(result.rounds) == 3
        assert result.total_messages > 20

    def test_political_arena_scenario(self):
        """Political debate scenario."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=7)

        room = RoomConfig.from_preset(
            RoomType.POLITICAL_ARENA,
            "Should AI development be paused for 6 months?",
            population_size=15,
            num_rounds=4,
        )
        result = engine.run(room)

        assert len(result.rounds) == 4

    def test_conviction_cascade_over_many_rounds(self):
        """Test that conviction genuinely cascades with enough rounds."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=42, dissident_seed=42)

        room = RoomConfig(
            name="Cascade Test",
            topic="Should we abandon our core product?",
            population_size=15,
            dissident_ratio=0.15,  # 15% dissidents
            num_rounds=8,  # Many rounds for cascade
        )
        result = engine.run(room)

        conviction = result.metadata.get("conviction_summary", {})
        # After 8 rounds with 15% dissidents, significant conviction shift expected
        assert conviction.get("avg_conviction", 0) > 0.2

    def test_minimum_viable_simulation(self):
        """Smallest possible valid simulation."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=1)

        room = RoomConfig(
            name="Minimal",
            topic="Test",
            population_size=3,
            num_rounds=1,
        )
        result = engine.run(room)

        assert len(result.rounds) == 1
        assert result.final_report is not None

    def test_high_dissident_ratio(self):
        """Simulation with 30% dissidents."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=42)

        room = RoomConfig(
            name="High Dissent",
            topic="Is the earth flat?",
            population_size=20,
            dissident_ratio=0.3,
            num_rounds=3,
        )
        result = engine.run(room)

        # 30% of 20 = 6 dissidents
        conviction = result.metadata.get("conviction_summary", {})
        assert conviction.get("dissidents", 0) >= 6
        assert result.total_dissent_messages > 0

    def test_result_serialization(self):
        """Verify simulation result can be serialized to JSON."""
        llm = TieredLLM(mode=LLMMode.MOCK)
        engine = SimulationEngine(llm=llm, persona_seed=42)

        room = RoomConfig(name="Serialize Test", topic="Test", population_size=5, num_rounds=1)
        result = engine.run(room)

        # Should not raise
        json_data = result.model_dump_json()
        assert len(json_data) > 100

        # Should round-trip
        from aps.schemas.message import SimulationResult
        restored = SimulationResult.model_validate_json(json_data)
        assert restored.simulation_id == result.simulation_id
        assert len(restored.rounds) == len(result.rounds)
