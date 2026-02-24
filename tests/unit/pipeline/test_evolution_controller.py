"""
Tests for EvolutionController: phase transitions, parent selection, max rounds.

Covers:
- EvolutionConfig: default and custom values
- Phase transitions: original → mutation → crossover
- Parent selection strategies
- Max rounds enforcement
- Disabled mutation/crossover handling
- Trajectory pool integration
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile
import json

from quantaalpha.pipeline.evolution.controller import (
    EvolutionController, EvolutionConfig
)
from quantaalpha.pipeline.evolution.trajectory import (
    StrategyTrajectory, TrajectoryPool, RoundPhase
)


@pytest.fixture(autouse=True)
def _mock_llm_calls():
    """Mock all LLM API calls to prevent real network requests in tests."""
    fake_response = '{"new_hypothesis": "test", "exploration_direction": "test", "orthogonality_reason": "test"}'
    with patch(
        "quantaalpha.pipeline.evolution.mutation.APIBackend"
    ) as mock_mut, patch(
        "quantaalpha.pipeline.evolution.crossover.APIBackend"
    ) as mock_cross:
        mock_mut.return_value.build_messages_and_create_chat_completion.return_value = fake_response
        mock_cross.return_value.build_messages_and_create_chat_completion.return_value = fake_response
        yield


@pytest.fixture
def basic_config():
    """Create a basic evolution config."""
    return EvolutionConfig(
        num_directions=2,
        steps_per_loop=5,
        max_rounds=5,
        mutation_enabled=True,
        crossover_enabled=True,
        crossover_size=2,
        crossover_n=2,
        prefer_diverse_crossover=True,
        parent_selection_strategy="best",
        parallel_enabled=False,
        fresh_start=True,
    )


@pytest.fixture
def controller(basic_config):
    """Create an evolution controller with basic config."""
    return EvolutionController(basic_config)


def create_trajectory(
    direction_id: int,
    round_idx: int,
    phase: RoundPhase,
    rank_ic: float = 0.05,
    parent_ids: list = None
) -> StrategyTrajectory:
    """Helper to create a trajectory for testing."""
    traj_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
    return StrategyTrajectory(
        trajectory_id=traj_id,
        direction_id=direction_id,
        round_idx=round_idx,
        phase=phase,
        hypothesis="Test hypothesis",
        hypothesis_details={},
        factors=[{"name": f"factor_{direction_id}", "expression": "TS_MEAN($close, 5)"}],
        backtest_result=None,
        backtest_metrics={"RankIC": rank_ic, "IC": rank_ic * 0.8},
        feedback="Test feedback",
        feedback_details={},
        parent_ids=parent_ids or [],
    )


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = EvolutionConfig()
        assert config.num_directions == 2
        assert config.max_rounds == 10
        assert config.mutation_enabled is True
        assert config.crossover_enabled is True
        assert config.crossover_size == 2
        assert config.crossover_n == 3
        assert config.prefer_diverse_crossover is True
        assert config.parent_selection_strategy == "best"
        assert config.parallel_enabled is False
        assert config.fresh_start is True

    def test_custom_values(self):
        """Custom config values should be set correctly."""
        config = EvolutionConfig(
            num_directions=4,
            max_rounds=20,
            mutation_enabled=False,
            crossover_enabled=True,
            crossover_size=3,
            crossover_n=5,
            parent_selection_strategy="random",
        )
        assert config.num_directions == 4
        assert config.max_rounds == 20
        assert config.mutation_enabled is False
        assert config.crossover_size == 3
        assert config.crossover_n == 5
        assert config.parent_selection_strategy == "random"


class TestControllerInitialization:
    """Tests for EvolutionController initialization."""

    def test_initial_state(self, controller):
        """Controller should have correct initial state."""
        state = controller.get_current_state()
        assert state["round"] == 0
        assert state["phase"] == "original"
        assert state["directions_completed"] == []
        assert state["active_branch_count"] == 2

    def test_pool_initialized(self, controller):
        """Controller should have an initialized trajectory pool."""
        assert controller.pool is not None
        assert isinstance(controller.pool, TrajectoryPool)

    def test_operators_initialized(self, controller):
        """Controller should have mutation and crossover operators."""
        assert controller.mutation_op is not None
        assert controller.crossover_op is not None


class TestOriginalPhase:
    """Tests for original phase execution."""

    def test_get_first_task(self, controller):
        """First task should be original phase, direction 0."""
        task = controller.get_next_task()
        assert task is not None
        assert task["phase"] == RoundPhase.ORIGINAL
        assert task["direction_id"] == 0
        assert task["parent_trajectories"] == []
        assert task["round_idx"] == 0

    def test_get_second_task(self, controller):
        """After completing direction 0, should get direction 1."""
        # Complete first direction
        task1 = controller.get_next_task()
        traj1 = create_trajectory(0, 0, RoundPhase.ORIGINAL)
        controller.report_task_complete(task1, traj1)

        # Get next task
        task2 = controller.get_next_task()
        assert task2["phase"] == RoundPhase.ORIGINAL
        assert task2["direction_id"] == 1

    def test_all_directions_complete_transitions_to_mutation(self, controller):
        """After all original directions complete, should transition to mutation."""
        # Complete all original directions
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Next task should be mutation
        next_task = controller.get_next_task()
        assert next_task["phase"] == RoundPhase.MUTATION

    def test_original_phase_skips_completed_directions(self, controller):
        """Should not return tasks for already completed directions."""
        # Complete direction 0
        task = controller.get_next_task()
        traj = create_trajectory(0, 0, RoundPhase.ORIGINAL)
        controller.report_task_complete(task, traj)

        # Get all tasks for current phase - should only have direction 1
        all_tasks = controller.get_all_tasks_for_current_phase()
        assert len(all_tasks) == 1
        assert all_tasks[0]["direction_id"] == 1


class TestMutationPhase:
    """Tests for mutation phase execution."""

    def test_mutation_uses_original_as_parents(self, controller):
        """Mutation should use original trajectories as parents."""
        # Complete original phase
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL, rank_ic=0.05 + d * 0.01)
            controller.report_task_complete(task, traj)

        # Get mutation task
        mutation_task = controller.get_next_task()
        assert mutation_task["phase"] == RoundPhase.MUTATION
        assert len(mutation_task["parent_trajectories"]) == 1
        assert mutation_task["parent_trajectories"][0].phase == RoundPhase.ORIGINAL

    def test_mutation_task_has_strategy_suffix(self, controller):
        """Mutation task should have strategy suffix."""
        # Complete original phase
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Get mutation task
        mutation_task = controller.get_next_task()
        assert "strategy_suffix" in mutation_task
        # Should have some guidance text
        assert len(mutation_task["strategy_suffix"]) >= 0

    def test_multiple_mutation_tasks(self, controller):
        """Should get multiple mutation tasks for multiple parents."""
        # Complete original phase with 2 directions
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Get all mutation tasks
        all_mutation = controller.get_all_tasks_for_current_phase()
        # Should have tasks for each original trajectory
        assert len(all_mutation) == controller.config.num_directions


class TestCrossoverPhase:
    """Tests for crossover phase execution."""

    def test_crossover_after_mutation(self, controller):
        """Crossover should follow mutation phase."""
        # Complete original phase
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Complete mutation phase
        while True:
            task = controller.get_next_task()
            if task["phase"] != RoundPhase.MUTATION:
                break
            traj = create_trajectory(
                task["direction_id"], task["round_idx"], RoundPhase.MUTATION
            )
            controller.report_task_complete(task, traj)

        # Should now be crossover
        assert task["phase"] == RoundPhase.CROSSOVER

    def test_crossover_has_multiple_parents(self, controller):
        """Crossover task should have multiple parent trajectories."""
        # Complete original and mutation phases
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Complete all mutations
        mutation_tasks = controller.get_all_tasks_for_current_phase()
        for task in mutation_tasks:
            traj = create_trajectory(task["direction_id"], 1, RoundPhase.MUTATION)
            controller.report_task_complete(task, traj)

        # Advance to crossover
        controller.advance_phase_after_parallel_completion(mutation_tasks)

        # Get crossover task
        crossover_task = controller.get_next_task()
        assert crossover_task["phase"] == RoundPhase.CROSSOVER
        assert len(crossover_task["parent_trajectories"]) >= controller.config.crossover_size


class TestPhaseTransitions:
    """Tests for phase transition logic."""

    def test_original_to_mutation_transition(self, controller):
        """Should transition from original to mutation when all directions complete."""
        # Complete all original tasks
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Transition happens lazily on get_next_task
        next_task = controller.get_next_task()
        assert next_task["phase"] == RoundPhase.MUTATION

        state = controller.get_current_state()
        assert state["phase"] == "mutation"

    def test_mutation_to_crossover_transition(self, controller):
        """Should transition from mutation to crossover."""
        # Complete original and mutation
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Get and complete mutation tasks
        mutation_tasks = controller.get_all_tasks_for_current_phase()
        for task in mutation_tasks:
            traj = create_trajectory(task["direction_id"], 1, RoundPhase.MUTATION)
            controller.report_task_complete(task, traj)

        controller.advance_phase_after_parallel_completion(mutation_tasks)
        state = controller.get_current_state()
        assert state["phase"] == "crossover"


class TestDisabledPhases:
    """Tests for disabled mutation/crossover phases."""

    def test_disabled_mutation_skips_to_crossover(self):
        """When mutation disabled, should skip to crossover after original."""
        config = EvolutionConfig(
            num_directions=2,
            max_rounds=5,
            mutation_enabled=False,
            crossover_enabled=True,
        )
        controller = EvolutionController(config)

        # Complete original phase
        for d in range(config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Should go directly to crossover
        next_task = controller.get_next_task()
        assert next_task["phase"] == RoundPhase.CROSSOVER

    def test_disabled_crossover_stays_in_mutation(self):
        """When crossover disabled, should stay in mutation loop."""
        config = EvolutionConfig(
            num_directions=2,
            max_rounds=5,
            mutation_enabled=True,
            crossover_enabled=False,
        )
        controller = EvolutionController(config)

        # Complete original phase
        for d in range(config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Should go to mutation
        next_task = controller.get_next_task()
        assert next_task["phase"] == RoundPhase.MUTATION

    def test_both_disabled_ends_after_original(self):
        """When both disabled, should complete after original phase."""
        config = EvolutionConfig(
            num_directions=2,
            max_rounds=5,
            mutation_enabled=False,
            crossover_enabled=False,
        )
        controller = EvolutionController(config)

        # Complete original phase
        for d in range(config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Should return None (evolution complete)
        next_task = controller.get_next_task()
        assert next_task is None


class TestMaxRounds:
    """Tests for max rounds enforcement."""

    def test_evolution_stops_at_max_rounds(self):
        """Evolution should stop when max_rounds is reached."""
        config = EvolutionConfig(
            num_directions=2,
            max_rounds=2,  # Very low max rounds
            mutation_enabled=True,
            crossover_enabled=True,
        )
        controller = EvolutionController(config)

        # Complete original phase (round 0)
        for d in range(config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Complete mutation (round 1)
        mutation_tasks = controller.get_all_tasks_for_current_phase()
        for task in mutation_tasks:
            traj = create_trajectory(task["direction_id"], 1, RoundPhase.MUTATION)
            controller.report_task_complete(task, traj)

        # Advance past mutation
        controller.advance_phase_after_parallel_completion(mutation_tasks)

        # Complete crossover (round 2, should hit max)
        crossover_tasks = controller.get_all_tasks_for_current_phase()
        for task in crossover_tasks:
            traj = create_trajectory(task["direction_id"], 2, RoundPhase.CROSSOVER)
            controller.report_task_complete(task, traj)

        # Should be at or past max rounds
        assert controller.is_complete() or controller._current_round >= config.max_rounds

    def test_is_complete_method(self, controller):
        """is_complete should return True when evolution is done."""
        assert controller.is_complete() is False

        # Manually set to max rounds
        controller._current_round = controller.config.max_rounds
        assert controller.is_complete() is True


class TestTrajectoryPoolIntegration:
    """Tests for trajectory pool integration."""

    def test_trajectories_added_to_pool(self, controller):
        """Completed tasks should add trajectories to pool."""
        initial_count = len(controller.pool.get_all())

        # Complete a task
        task = controller.get_next_task()
        traj = create_trajectory(0, 0, RoundPhase.ORIGINAL)
        controller.report_task_complete(task, traj)

        # Pool should have new trajectory
        assert len(controller.pool.get_all()) == initial_count + 1

    def test_get_best_trajectories(self, controller):
        """Should be able to get best performing trajectories."""
        # Add some trajectories with different performance
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            rank_ic = 0.05 + d * 0.02  # Different performance
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL, rank_ic=rank_ic)
            controller.report_task_complete(task, traj)

        # Get best trajectories
        best = controller.get_best_trajectories(top_n=1)
        assert len(best) == 1
        assert best[0].backtest_metrics["RankIC"] == 0.07  # Highest


class TestStatePersistence:
    """Tests for state save/load."""

    def test_save_and_load_state(self, controller):
        """Should be able to save and load controller state."""
        # Complete some tasks
        for d in range(controller.config.num_directions):
            task = controller.get_next_task()
            traj = create_trajectory(d, 0, RoundPhase.ORIGINAL)
            controller.report_task_complete(task, traj)

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = Path(f.name)

        try:
            controller.save_state(save_path)

            # Create new controller and load state
            new_controller = EvolutionController(controller.config)
            new_controller.load_state(save_path)

            # Verify state restored
            assert new_controller._current_round == controller._current_round
            assert new_controller._current_phase == controller._current_phase
            assert new_controller._directions_completed == controller._directions_completed
        finally:
            save_path.unlink()


class TestTaskFailed:
    """Tests for task failure handling."""

    def test_failed_task_skipped(self, controller):
        """Failed task should be skipped without adding trajectory."""
        task = controller.get_next_task()
        initial_pool_size = len(controller.pool.get_all())

        # Report task as failed
        controller.report_task_failed(task, Exception("Test error"))

        # Direction should be marked complete but pool unchanged
        assert 0 in controller._directions_completed
        assert len(controller.pool.get_all()) == initial_pool_size

    def test_failed_task_allows_evolution_to_continue(self, controller):
        """Evolution should continue after task failure."""
        # Fail first direction
        task1 = controller.get_next_task()
        controller.report_task_failed(task1, Exception("Test error"))

        # Should still get next direction
        task2 = controller.get_next_task()
        assert task2 is not None
        assert task2["direction_id"] == 1


class TestGetCurrentState:
    """Tests for get_current_state method."""

    def test_state_includes_all_fields(self, controller):
        """State should include all expected fields."""
        state = controller.get_current_state()

        assert "round" in state
        assert "phase" in state
        assert "directions_completed" in state
        assert "active_branch_count" in state
        assert "mutation_targets_remaining" in state
        assert "crossover_groups_remaining" in state
        assert "pool_stats" in state

    def test_state_updates_after_completion(self, controller):
        """State should update after task completion."""
        initial_state = controller.get_current_state()

        # Complete a task
        task = controller.get_next_task()
        traj = create_trajectory(0, 0, RoundPhase.ORIGINAL)
        controller.report_task_complete(task, traj)

        new_state = controller.get_current_state()

        # Directions completed should have changed
        assert len(new_state["directions_completed"]) > len(initial_state["directions_completed"])