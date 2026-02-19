"""
Factor workflow with session control and evolution support.

Supports three round phases:
- Original: Initial exploration in each direction
- Mutation: Orthogonal exploration from parent trajectories
- Crossover: Hybrid strategies from multiple parents

Supports parallel execution within each phase when enabled.
"""

from typing import Any
from pathlib import Path
import fire
import signal
import sys
import threading
from multiprocessing import Process, Queue, Semaphore
from functools import wraps
import time
import ctypes
import os
import pickle
from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
from quantaalpha.pipeline.planning import generate_parallel_directions
from quantaalpha.pipeline.planning import load_run_config
from quantaalpha.pipeline.loop import AlphaAgentLoop
from quantaalpha.pipeline.evolution import (
    EvolutionController, 
    EvolutionConfig,
    StrategyTrajectory,
    RoundPhase,
)
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.log import logger
from quantaalpha.log.time import measure_time
from quantaalpha.llm.config import LLM_SETTINGS




def force_timeout():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            seconds = LLM_SETTINGS.factor_mining_timeout
            def handle_timeout(signum, frame):
                logger.error(f"Process terminated: timeout exceeded ({seconds}s)")
                sys.exit(1)

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def _run_branch(
    direction: str | None,
    step_n: int,
    use_local: bool,
    idx: int,
    log_root: str,
    log_prefix: str,
    quality_gate_cfg: dict = None,
):
    if log_root:
        branch_name = f"{log_prefix}_{idx:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=None,
        use_local=use_local,
        quality_gate_config=quality_gate_cfg or {},
    )
    model_loop.user_initial_direction = direction
    model_loop.run(step_n=step_n, stop_event=None)


def _run_branch_with_semaphore(
    semaphore: Semaphore,
    direction: str | None,
    step_n: int,
    use_local: bool,
    idx: int,
    log_root: str,
    log_prefix: str,
    quality_gate_cfg: dict = None,
):
    """Wrapper around _run_branch that releases the semaphore when done."""
    try:
        _run_branch(direction, step_n, use_local, idx, log_root, log_prefix, quality_gate_cfg)
    finally:
        semaphore.release()


def _run_evolution_task(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    stop_event: threading.Event | None,
    quality_gate_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single evolution task (one small loop).

    Args:
        task: Evolution task descriptor
        directions: List of original directions
        step_n: Steps per round
        use_local: Use local backtest
        user_direction: User initial direction
        log_root: Log root directory
        stop_event: Stop event
        quality_gate_cfg: Quality gate config

    Returns:
        Dict containing trajectory data
    """
    phase = task["phase"]
    direction_id = task["direction_id"]
    strategy_suffix = task.get("strategy_suffix", "")
    round_idx = task["round_idx"]
    parent_trajectories = task.get("parent_trajectories", [])
    
    # Resolve direction by phase
    if phase == RoundPhase.ORIGINAL:
        direction = directions[direction_id] if direction_id < len(directions) else None
    elif phase == RoundPhase.MUTATION:
        direction = directions[direction_id] if direction_id < len(directions) else None
    else:  # CROSSOVER
        direction = None

    trajectory_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
    parent_ids = [p.trajectory_id for p in parent_trajectories]

    if log_root:
        branch_name = f"{phase.value}_{round_idx:02d}_{direction_id:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)

    logger.info(f"Starting evolution task: phase={phase.value}, round={round_idx}, direction={direction_id}")

    # Create and run loop
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=stop_event,
        use_local=use_local,
        strategy_suffix=strategy_suffix,
        evolution_phase=phase.value,
        trajectory_id=trajectory_id,
        parent_trajectory_ids=parent_ids,
        direction_id=direction_id,
        round_idx=round_idx,
        quality_gate_config=quality_gate_cfg or {},
    )
    model_loop.user_initial_direction = user_direction
    
    # Run one small loop (5 steps)
    model_loop.run(step_n=step_n, stop_event=stop_event)

    traj_data = model_loop._get_trajectory_data()
    traj_data["task"] = task
    
    return traj_data


def _parallel_task_worker(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    result_queue: Queue,
    task_idx: int,
    semaphore: Any = None,
):
    """
    Worker for parallel evolution tasks. Runs one evolution task in a separate process and puts result in queue.
    Args: task, directions, step_n, use_local, user_direction, log_root, result_queue, task_idx.
    The optional semaphore is released when this worker finishes (used for bounded concurrency).
    """
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS
        RD_AGENT_SETTINGS.use_file_lock = False
        # Place worker cache alongside the main experiment cache, not inside log/
        base_cache = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str)
        RD_AGENT_SETTINGS.pickle_cache_folder_path_str = str(
            base_cache.parent / f"{base_cache.name}_w{task_idx}"
        )

        traj_data = _run_evolution_task(
            task=task,
            directions=directions,
            step_n=step_n,
            use_local=use_local,
            user_direction=user_direction,
            log_root=log_root,
            stop_event=None,
        )
        result_queue.put({
            "success": True,
            "task_idx": task_idx,
            "task": task,
            "traj_data": traj_data,
        })
    except Exception as e:
        import traceback
        result_queue.put({
            "success": False,
            "task_idx": task_idx,
            "task": task,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        if semaphore is not None:
            semaphore.release()


def _serialize_task_for_parallel(task: dict[str, Any]) -> dict[str, Any]:
    """Serialize task for use in child process (parent_trajectories are complex objects)."""
    serialized = task.copy()
    
    # RoundPhase -> string
    if "phase" in serialized and isinstance(serialized["phase"], RoundPhase):
        serialized["phase"] = serialized["phase"]
    
    # Convert parent_trajectories to serializable info
    if "parent_trajectories" in serialized:
        serialized["parent_trajectory_ids"] = [
            p.trajectory_id for p in serialized.get("parent_trajectories", [])
        ]
        # Child process does not need full trajectory objects; strategy_suffix has required info
        serialized["parent_trajectories"] = []
    
    return serialized


def _run_tasks_parallel(
    tasks: list[dict[str, Any]],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    max_workers: int = 0,
) -> list[dict[str, Any]]:
    """
    Run multiple evolution tasks with bounded concurrency.
    max_workers: 0 or 1 = fully sequential (in-process); N>1 = up to N subprocesses at once.
    Returns list of results, each with task and traj_data.
    """
    if not tasks:
        return []

    # Sequential mode: run tasks one at a time in the current process
    effective_workers = max(0, max_workers)
    if effective_workers <= 1:
        results = []
        result_queue: Queue = Queue()
        for idx, task in enumerate(tasks):
            serialized_task = _serialize_task_for_parallel(task)
            _parallel_task_worker(
                serialized_task, directions, step_n, use_local,
                user_direction, log_root, result_queue, idx, semaphore=None,
            )
            result = result_queue.get()
            if result["success"]:
                result["task"] = tasks[idx]
                result["traj_data"]["task"] = tasks[idx]
                results.append(result)
                logger.info(f"Task {idx} completed (sequential)")
            else:
                logger.error(f"Task {idx} failed: {result['error']}")
                logger.error(result.get("traceback", ""))
        return results

    # Bounded-parallel mode: semaphore limits concurrency to max_workers
    sem = Semaphore(effective_workers)
    result_queue = Queue()
    processes = []

    logger.info(f"Starting {len(tasks)} tasks (max_parallel_workers={effective_workers})")

    for idx, task in enumerate(tasks):
        sem.acquire()  # blocks when effective_workers slots are all occupied
        serialized_task = _serialize_task_for_parallel(task)
        p = Process(
            target=_parallel_task_worker,
            args=(
                serialized_task, directions, step_n, use_local,
                user_direction, log_root, result_queue, idx, sem,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started task {idx}: phase={task['phase'].value}, direction={task['direction_id']}")

    results = []
    for _ in range(len(processes)):
        result = result_queue.get()
        if result["success"]:
            original_task = tasks[result["task_idx"]]
            result["task"] = original_task
            result["traj_data"]["task"] = original_task
            results.append(result)
            logger.info(f"Task {result['task_idx']} completed")
        else:
            logger.error(f"Task {result['task_idx']} failed: {result['error']}")
            logger.error(result.get("traceback", ""))

    for p in processes:
        p.join()

    logger.info(f"Parallel tasks done: {len(results)}/{len(tasks)} succeeded")
    return results


def run_evolution_loop(
    initial_direction: str | None,
    evolution_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
    stop_event: threading.Event | None = None,
    quality_gate_cfg: dict[str, Any] | None = None,
):
    """
    Run evolution loop: Original -> Mutation -> Crossover -> Mutation -> ...
    Supports parallel execution per phase.
    """
    quality_gate_cfg = quality_gate_cfg or {}
    from quantaalpha.core.conf import RD_AGENT_SETTINGS
    RD_AGENT_SETTINGS.use_file_lock = False
    logger.info("Evolution mode: file lock disabled to avoid deadlock")

    # Fail fast: CONDA_DEFAULT_ENV must be set when use_local=True so that
    # rdagent's LocalEnv can construct CondaConf.  Without this guard the loop
    # catches the ValidationError and retries the same task infinitely.
    _use_local_check = bool(exec_cfg.get("use_local", True))
    if _use_local_check and not os.environ.get("CONDA_DEFAULT_ENV"):
        raise EnvironmentError(
            "CONDA_DEFAULT_ENV is not set. "
            "Export it before launching: export CONDA_DEFAULT_ENV=quantaalpha-ollama"
        )

    # Parse config
    num_directions = int(planning_cfg.get("num_directions", 2))
    max_rounds = int(evolution_cfg.get("max_rounds", 10))
    crossover_size = int(evolution_cfg.get("crossover_size", 2))
    crossover_n = int(evolution_cfg.get("crossover_n", 3))
    steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
    use_local = bool(exec_cfg.get("use_local", True))
    
    mutation_enabled = bool(evolution_cfg.get("mutation_enabled", True))
    crossover_enabled = bool(evolution_cfg.get("crossover_enabled", True))
    parent_selection_strategy = str(evolution_cfg.get("parent_selection_strategy", "best"))
    top_percent_threshold = float(evolution_cfg.get("top_percent_threshold", 0.3))
    log_root = str(logger.log_trace_path)
    parallel_enabled = bool(evolution_cfg.get("parallel_enabled", False))
    max_parallel_workers = int(evolution_cfg.get("max_parallel_workers", 0) or 0)
    fresh_start = bool(evolution_cfg.get("fresh_start", True))
    cleanup_on_finish = bool(evolution_cfg.get("cleanup_on_finish", False))

    # Generate initial directions
    planning_enabled = bool(planning_cfg.get("enabled", False))
    prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
    prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
    
    if planning_enabled and initial_direction:
        directions = generate_parallel_directions(
            initial_direction=initial_direction,
            n=num_directions,
            prompt_file=prompt_path,
            max_attempts=int(planning_cfg.get("max_attempts", 5)),
            use_llm=bool(planning_cfg.get("use_llm", True)),
            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
        )
    elif planning_enabled:
        directions = [None] * num_directions
    else:
        directions = [initial_direction] if initial_direction else [None]

    logger.info(f"Generated {len(directions)} exploration directions")
    for i, d in enumerate(directions):
        logger.info(f"  Direction {i}: {d}")

    pool_save_path = Path(log_root) / "trajectory_pool.json"
    mutation_prompt_path = Path(__file__).parent / "prompts" / "evolution_prompts.yaml"
    
    logger.info(f"Trajectory pool path: {pool_save_path} (fresh_start={fresh_start})")

    config = EvolutionConfig(
        num_directions=len(directions),
        steps_per_loop=steps_per_loop,
        max_rounds=max_rounds,
        mutation_enabled=mutation_enabled,
        crossover_enabled=crossover_enabled,
        crossover_size=crossover_size,
        crossover_n=crossover_n,
        prefer_diverse_crossover=True,
        parent_selection_strategy=parent_selection_strategy,
        top_percent_threshold=top_percent_threshold,
        parallel_enabled=parallel_enabled,
        pool_save_path=str(pool_save_path),
        mutation_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        crossover_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        fresh_start=fresh_start,
    )

    controller = EvolutionController(config)

    logger.info("="*60)
    logger.info("Starting evolution loop")
    logger.info(f"Config: directions={len(directions)}, max_rounds={max_rounds}, "
               f"crossover_size={crossover_size}, crossover_n={crossover_n}")
    logger.info(f"Phases: mutation={'on' if mutation_enabled else 'off'}, "
               f"crossover={'on' if crossover_enabled else 'off'}")
    if mutation_enabled and not crossover_enabled:
        logger.info("Mode: mutation only (Original -> Mutation -> ...)")
    elif crossover_enabled and not mutation_enabled:
        logger.info("Mode: crossover only (Original -> Crossover -> ...)")
    elif mutation_enabled and crossover_enabled:
        logger.info("Mode: full evolution (Original -> Mutation -> Crossover -> ...)")
    else:
        logger.info("Mode: original only (no evolution)")
    logger.info(f"Parent selection: {parent_selection_strategy}" +
               (f" (top_percent={top_percent_threshold})" if parent_selection_strategy == "top_percent_plus_random" else ""))
    logger.info(f"Parallel execution: {'on' if parallel_enabled else 'off'}" +
               (f" (max_workers={max_parallel_workers})" if parallel_enabled and max_parallel_workers > 1 else ""))
    logger.info("="*60)

    if parallel_enabled:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            tasks = controller.get_all_tasks_for_current_phase()
            if not tasks:
                logger.info("Evolution complete: no more tasks")
                break

            current_phase = tasks[0]["phase"]
            current_round = tasks[0]["round_idx"]
            logger.info(f"Parallel phase: phase={current_phase.value}, round={current_round}, tasks={len(tasks)}")

            results = _run_tasks_parallel(
                tasks=tasks,
                directions=directions,
                step_n=steps_per_loop,
                use_local=use_local,
                user_direction=initial_direction,
                log_root=log_root,
                max_workers=max_parallel_workers,
            )
            
            completed_tasks = []
            for result in results:
                if result["success"]:
                    task = result["task"]
                    traj_data = result["traj_data"]
                    trajectory = controller.create_trajectory_from_loop_result(
                        task=task,
                        hypothesis=traj_data.get("hypothesis"),
                        experiment=traj_data.get("experiment"),
                        feedback=traj_data.get("feedback"),
                    )
                    controller.report_task_complete(task, trajectory)
                    completed_tasks.append(task)
                    logger.info(f"Trajectory done: {trajectory.trajectory_id}, RankIC={trajectory.get_primary_metric()}")

            controller.advance_phase_after_parallel_completion(completed_tasks)

    else:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            task = controller.get_next_task()
            if task is None:
                logger.info("Evolution complete: no more tasks")
                break

            logger.info(f"Running task: phase={task['phase'].value}, round={task['round_idx']}, direction={task['direction_id']}")

            try:
                traj_data = _run_evolution_task(
                    task=task,
                    directions=directions,
                    step_n=steps_per_loop,
                    use_local=use_local,
                    user_direction=initial_direction,
                    log_root=log_root,
                    stop_event=stop_event,
                    quality_gate_cfg=quality_gate_cfg,
                )
                trajectory = controller.create_trajectory_from_loop_result(
                    task=task,
                    hypothesis=traj_data.get("hypothesis"),
                    experiment=traj_data.get("experiment"),
                    feedback=traj_data.get("feedback"),
                )
                controller.report_task_complete(task, trajectory)
                logger.info(f"Task done: trajectory_id={trajectory.trajectory_id}, RankIC={trajectory.get_primary_metric()}")
            except Exception as e:
                logger.error(f"Task failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                controller.report_task_failed(task, e)
                continue

    state_path = Path(log_root) / "evolution_state.json"
    controller.save_state(state_path)
    best_trajs = controller.get_best_trajectories(top_n=5)
    logger.info("="*60)
    logger.info(f"Evolution complete. Top {len(best_trajs)} trajectories:")
    for i, t in enumerate(best_trajs):
        metric = t.get_primary_metric()
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        logger.info(f"  {i+1}. {t.trajectory_id}: phase={t.phase.value}, RankIC={metric_str}")
    logger.info(f"Pool stats: {controller.pool.get_statistics()}")
    logger.info("="*60)
    if cleanup_on_finish:
        logger.info("Cleaning up trajectory pool file...")
        controller.pool.cleanup_file()


@force_timeout()
def main(path=None, step_n=100, direction=None, stop_event=None, config_path=None, evolution_mode=None, factor_lib_suffix=None):
    """
    Autonomous alpha factor mining with optional evolution support.

    Args:
        path: Session path (for resume)
        step_n: Number of steps (default 100 = 20 loops * 5 steps/loop)
        direction: Initial direction
        stop_event: Stop event
        config_path: Run config file path
        evolution_mode: Enable evolution (None=from config, True/False=override)
        factor_lib_suffix: Suffix for the factor library filename
            (e.g. "exp_v1" → all_factors_library_exp_v1.json).
            Overrides the FACTOR_LIBRARY_SUFFIX environment variable.

    Evolution flow: Original -> Mutation -> Crossover -> Mutation -> ...

    You can continue running session by

    .. code-block:: python

        quantaalpha mine --direction "[Initial Direction]" --config_path configs/experiment.yaml

    """
    if factor_lib_suffix:
        os.environ["FACTOR_LIBRARY_SUFFIX"] = factor_lib_suffix
    try:
        from quantaalpha.core.conf import RD_AGENT_SETTINGS
        logger.info("="*60)
        logger.info("Experiment config")
        logger.info(f"  Workspace: {RD_AGENT_SETTINGS.workspace_path}")
        logger.info(f"  Cache dir: {RD_AGENT_SETTINGS.pickle_cache_folder_path_str}")
        logger.info(f"  Cache enabled: {RD_AGENT_SETTINGS.cache_with_pickle}")
        logger.info("="*60)

        # Config file default: project_root/configs/
        _project_root = Path(__file__).resolve().parents[2]
        config_default = _project_root / "configs" / "experiment.yaml"
        config_file = Path(config_path) if config_path else config_default
        run_cfg = load_run_config(config_file)
        planning_cfg = (run_cfg.get("planning") or {}) if isinstance(run_cfg, dict) else {}
        exec_cfg = (run_cfg.get("execution") or {}) if isinstance(run_cfg, dict) else {}
        evolution_cfg = (run_cfg.get("evolution") or {}) if isinstance(run_cfg, dict) else {}
        quality_gate_cfg = (run_cfg.get("quality_gate") or {}) if isinstance(run_cfg, dict) else {}

        if evolution_mode is not None:
            use_evolution = evolution_mode
        else:
            use_evolution = bool(evolution_cfg.get("enabled", False))

        if step_n is None or step_n == 100:
            if exec_cfg.get("step_n") is not None:
                step_n = exec_cfg.get("step_n")
            else:
                max_loops = int(exec_cfg.get("max_loops", 10))
                steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
                step_n = max_loops * steps_per_loop

        use_local = os.getenv("USE_LOCAL", "True").lower()
        use_local = True if use_local in ["true", "1"] else False
        if exec_cfg.get("use_local") is not None:
            use_local = bool(exec_cfg.get("use_local"))
        exec_cfg["use_local"] = use_local
        
        logger.info(f"Use {'Local' if use_local else 'Docker container'} to execute factor backtest")
        
        if use_evolution and path is None:
            logger.info("="*60)
            logger.info("Evolution mode: Original -> Mutation -> Crossover loop")
            logger.info("="*60)
            
            run_evolution_loop(
                initial_direction=direction,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
                quality_gate_cfg=quality_gate_cfg,
            )
        
        elif path is None:
            planning_enabled = bool(planning_cfg.get("enabled", False))
            n_dirs = int(planning_cfg.get("num_directions", 1))
            max_attempts = int(planning_cfg.get("max_attempts", 5))
            use_llm = bool(planning_cfg.get("use_llm", True))
            allow_fallback = bool(planning_cfg.get("allow_fallback", True))
            prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
            prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
            if planning_enabled and direction:
                directions = generate_parallel_directions(
                    initial_direction=direction,
                    n=n_dirs,
                    prompt_file=prompt_path,
                    max_attempts=max_attempts,
                    use_llm=use_llm,
                    allow_fallback=allow_fallback,
                )
            else:
                directions = [direction] if direction else [None]

            log_root = exec_cfg.get("branch_log_root") or "log"
            log_prefix = exec_cfg.get("branch_log_prefix") or "branch"
            use_branch_logs = planning_enabled and len(directions) > 1
            parallel_execution = bool(exec_cfg.get("parallel_execution", False))
            branch_max_workers = int(exec_cfg.get("max_parallel_workers", 0) or 0)

            if parallel_execution and len(directions) > 1:
                # Bounded-parallel branch launch: semaphore limits concurrency
                effective_workers = branch_max_workers if branch_max_workers > 1 else len(directions)
                sem = Semaphore(effective_workers)
                procs: list[Process] = []
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    sem.acquire()
                    p = Process(
                        target=_run_branch_with_semaphore,
                        args=(sem, dir_text, step_n, use_local, idx, log_root if use_branch_logs else "", log_prefix),
                    )
                    p.start()
                    procs.append(p)
                    logger.info(f"Started branch {idx} (max_workers={effective_workers})")
                for p in procs:
                    p.join()
            else:
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    if use_branch_logs:
                        branch_name = f"{log_prefix}_{idx:02d}"
                        branch_log = Path(log_root) / branch_name
                        branch_log.mkdir(parents=True, exist_ok=True)
                        logger.set_trace_path(branch_log)
                    model_loop = AlphaAgentLoop(
                        ALPHA_AGENT_FACTOR_PROP_SETTING,
                        potential_direction=dir_text,
                        stop_event=stop_event,
                        use_local=use_local,
                        quality_gate_config=quality_gate_cfg,
                    )
                    model_loop.user_initial_direction = direction
                    model_loop.run(step_n=step_n, stop_event=stop_event)
        else:
            model_loop = AlphaAgentLoop.load(path, use_local=use_local)
            model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
    finally:
        logger.info("Run finished or terminated")

if __name__ == "__main__":
    fire.Fire(main)
