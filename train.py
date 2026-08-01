"""Training and paired evaluation entry point for the Chapter-3 project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from base_env import UAVEnv as BaseUAVEnv
from ch3_constants import CH3_MISSION_V1
from env import UAVEnv
from registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
    assert_ch3_method,
)
from registry.ch3_efficiency_v3_registry import CH3_EFFICIENCY_V3_SCREEN
from utils.ch3_buffer import CH3ReplayBuffer
from utils.provenance import (
    assert_algorithm_source_unchanged,
    capture_provenance_snapshot,
    file_sha256,
    json_file_sha256,
    repository_source_fingerprint,
    runtime_versions,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FORMAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "chapter3_final" / "runs"
DEFAULT_PILOT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chapter3_final" / "pilot" / "runs"
DEFAULT_EFFICIENCY_V2_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v2"
DEFAULT_EFFICIENCY_V2_OUTPUT_DIR = DEFAULT_EFFICIENCY_V2_ROOT / "runs"
DEFAULT_EFFICIENCY_V3_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v3_screen"
DEFAULT_EFFICIENCY_V3_OUTPUT_DIR = DEFAULT_EFFICIENCY_V3_ROOT / "runs"
CHECKPOINT_SCHEMA_VERSION = 2
CH3_PILOT_V1 = "ch3_pilot_v1"
CH3_EFFICIENCY_V2 = "ch3_efficiency_v2"

CH3_TRAINING_BUDGETS = {
    "smoke": 2,
    "short": 20,
    "pilot": 200,
    "medium": 1000,
    "formal": 3000,
}
CH3_CHECKPOINT_INTERVALS = {
    "smoke": 0,
    "short": 10,
    "pilot": 50,
    "medium": 100,
    "formal": 250,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


_RUNTIME_ONLY_CONFIG_KEYS = {
    "checkpoint_interval",
    "training_episodes",
    "pilot_episodes",
}


def _config_hash(config):
    """Hash the complete resolved run configuration.

    This hash is useful for provenance, but it is intentionally *not* used to
    decide whether a checkpoint can be evaluated or resumed because operational
    settings such as checkpoint cadence do not change the learned algorithm.
    """
    payload = json.dumps(
        _json_safe(dict(config)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _algorithm_config(config):
    """Return only configuration fields that affect environment or learning."""
    return {
        key: value
        for key, value in dict(config).items()
        if key not in _RUNTIME_ONLY_CONFIG_KEYS
        and not (key == "protocol" and value == CH3_PILOT_V1)
    }


def _algorithm_config_hash(config):
    return _config_hash(_algorithm_config(config))


_TRAINING_ONLY_CONFIG_KEYS = _RUNTIME_ONLY_CONFIG_KEYS | {
    "optimizer",
    "lr_actor",
    "lr_critic",
    "gamma",
    "tau",
    "batch_size",
    "replay_size",
    "update_frequency",
    "policy_delay",
    "updates_per_train",
    "warmup_steps",
    "initial_sigma",
    "min_sigma",
    "sigma_decay",
    "sigma_hold_episodes",
    "seed_protocol",
}


def _evaluation_config(config):
    """Configuration that must match to execute a checkpoint in an environment."""
    return {
        key: value
        for key, value in dict(config).items()
        if key not in _TRAINING_ONLY_CONFIG_KEYS
        and not (key == "protocol" and value == CH3_PILOT_V1)
    }


def _evaluation_config_hash(config):
    return _config_hash(_evaluation_config(config))


def _run_config_hash(config, runtime_args=None):
    """Hash the resolved algorithm config plus operational run arguments.

    ``algorithm_config_hash`` deliberately excludes cadence and budget fields so
    compatible training can resume. This run-level hash records those fields,
    output placement and evaluation selection for provenance only.
    """
    payload = {
        "resolved_config": dict(config),
        "runtime_args": dict(runtime_args or {}),
    }
    return _config_hash(payload)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


CH3_BASE_CONFIG = {
    "ch3_config": True,
    "use_pse_planner": False,
    "pse_use_belief": False,
    "pse_use_exec_cost": False,
    "pse_use_standby": False,
    "pse_fixed_standby_mode": "space_center",
    "pse_use_exec_cost_schedule": False,
    "pse_lazy_standby": False,
    "use_residual_prior": True,
    "prior_strength_search": 1.0,
    "prior_strength_executor": 1.0,
    "residual_scale_search": 0.45,
    "residual_scale_executor": 0.35,
}

CH3_SHARED_TRAINING_CONFIG = {
    "actor_type": "plain_mlp",
    "optimizer": "adam",
    "hidden_dim": 128,
    "lr_actor": 1e-3,
    "lr_critic": 5e-4,
    "gamma": 0.95,
    "tau": 5e-3,
    "batch_size": 128,
    "replay_size": 500_000,
    "update_frequency": 4,
    "policy_delay": 2,
    "updates_per_train": 2,
    "warmup_steps": 256,
    "training_episodes": 6000,
    "pilot_episodes": 200,
    "max_steps": 400,
    "initial_sigma": 0.2,
    "min_sigma": 0.05,
    "sigma_decay": 0.9997,
    "sigma_hold_episodes": 1200,
    "checkpoint_interval": 1000,
    "seed_protocol": "python+numpy+torch+torch_cuda",
}

CH3_PROTOCOL_CONFIGS = {
    CH3_PILOT_V1: {
        "protocol": CH3_PILOT_V1,
        "reward_profile": CH3_PILOT_V1,
        "default_output_dir": str(DEFAULT_PILOT_OUTPUT_DIR),
    },
    CH3_EFFICIENCY_V2: {
        "protocol": CH3_EFFICIENCY_V2,
        "reward_profile": "task_efficiency_v2",
        "reward_scale": 400.0,
        "team_find_bonus": 50.0,
        "finder_extra_bonus": 100.0,
        "mission_complete_bonus": 500.0,
        "time_penalty": 0.20,
        "lambda_a": 0.005,
        "lambda_da": 0.015,
        "search_detect_bonuses": (200.0, 250.0, 300.0),
        "early_find_bonus_gain": 100.0,
        "early_success_bonus_gain": 200.0,
        "residual_scale_search": 0.20,
        "residual_scale_executor": 0.15,
        "residual_action_reg": 0.05,
        "pse_belief_weight": 0.60,
        "pse_standby_start_step": 80,
        "pse_standby_update_interval": 10,
        "pse_standby_move_weight": 0.80,
        "pse_standby_hysteresis_weight": 0.30,
        "pse_standby_min_relative_gain": 0.05,
        "pse_standby_max_target_shift": 3.0,
        "use_obstacles": False,
        "default_output_dir": str(DEFAULT_EFFICIENCY_V2_OUTPUT_DIR),
    },
}


def _ch3_method(overrides=None, *, run_type="learning"):
    config = dict(CH3_BASE_CONFIG)
    config.update(CH3_SHARED_TRAINING_CONFIG)
    config["run_type"] = str(run_type)
    config.update(dict(overrides or {}))
    return config


CH3_METHOD_CONFIGS = {
    "ch3_pheromone_prior": _ch3_method(
        {"residual_scale_search": 0.0, "residual_scale_executor": 0.0},
        run_type="controller_only",
    ),
    "ch3_pheromone_rmaddpg": _ch3_method(),
    "ch3_pse_rmaddpg": _ch3_method({
        "use_pse_planner": True,
        "pse_use_belief": True,
        "pse_use_exec_cost": True,
        "pse_use_standby": True,
    }),
}
CH3_METHOD_CONFIGS["ch3_pse_no_belief"] = dict(
    CH3_METHOD_CONFIGS["ch3_pse_rmaddpg"], pse_use_belief=False
)
CH3_METHOD_CONFIGS["ch3_pse_no_exec_cost"] = dict(
    CH3_METHOD_CONFIGS["ch3_pse_rmaddpg"], pse_use_exec_cost=False
)
CH3_METHOD_CONFIGS["ch3_pse_no_standby"] = dict(
    CH3_METHOD_CONFIGS["ch3_pse_rmaddpg"], pse_use_standby=False
)
CH3_METHOD_CONFIGS["ch3_pse_no_residual"] = dict(
    CH3_METHOD_CONFIGS["ch3_pse_rmaddpg"],
    residual_scale_search=0.0,
    residual_scale_executor=0.0,
    run_type="controller_only",
)


def _build_efficiency_v2_method_configs():
    common = {
        key: value
        for key, value in CH3_PROTOCOL_CONFIGS[CH3_EFFICIENCY_V2].items()
        if key != "default_output_dir"
    }
    base = dict(CH3_METHOD_CONFIGS["ch3_pheromone_rmaddpg"])
    base.update(common)
    configs = {
        "ch3_pheromone_prior": dict(
            base,
            residual_scale_search=0.0,
            residual_scale_executor=0.0,
            run_type="controller_only",
        ),
        "ch3_pheromone_rmaddpg": dict(base),
        "ch3_pse_rmaddpg": dict(
            base,
            use_pse_planner=True,
            pse_use_belief=True,
            pse_use_exec_cost=True,
            pse_use_standby=True,
        ),
    }
    configs["ch3_pse_no_belief"] = dict(configs["ch3_pse_rmaddpg"], pse_use_belief=False)
    configs["ch3_pse_no_exec_cost"] = dict(configs["ch3_pse_rmaddpg"], pse_use_exec_cost=False)
    configs["ch3_pse_no_standby"] = dict(configs["ch3_pse_rmaddpg"], pse_use_standby=False)
    configs["ch3_pse_no_residual"] = dict(
        configs["ch3_pse_rmaddpg"],
        residual_scale_search=0.0,
        residual_scale_executor=0.0,
        run_type="controller_only",
    )
    return configs


CH3_EFFICIENCY_V2_METHOD_CONFIGS = _build_efficiency_v2_method_configs()

if tuple(CH3_METHOD_CONFIGS) != ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
    raise RuntimeError("Chapter-3 registry/config mismatch")


def get_ch3_method_config(method, protocol=CH3_PILOT_V1):
    method = assert_ch3_method(method)
    protocol = str(protocol)
    if protocol == CH3_PILOT_V1:
        return dict(CH3_METHOD_CONFIGS[method], protocol=CH3_PILOT_V1)
    if protocol == CH3_EFFICIENCY_V2:
        return dict(CH3_EFFICIENCY_V2_METHOD_CONFIGS[method])
    raise ValueError(f"unknown Chapter-3 protocol={protocol!r}")


_BASE_PROTOCOLS = {
    CH3_PILOT_V1,
    CH3_EFFICIENCY_V2,
    CH3_EFFICIENCY_V3_SCREEN,
}


def _environment_class_for_protocol(protocol):
    protocol = str(protocol)
    if protocol in _BASE_PROTOCOLS:
        return BaseUAVEnv
    if protocol == CH3_MISSION_V1:
        return UAVEnv
    raise ValueError(f"unknown Chapter-3 environment protocol={protocol!r}")


def _build_ch3_env_kwargs(env_device, max_steps, config):
    resolved = dict(config or {})
    if not resolved.get("ch3_config"):
        raise ValueError("a resolved Chapter-3 config is required")
    if "protocol" not in resolved:
        raise ValueError("resolved Chapter-3 config must explicitly declare protocol")
    env_class = _environment_class_for_protocol(resolved["protocol"])
    constructors = (
        (BaseUAVEnv.__init__, UAVEnv.__init__)
        if env_class is UAVEnv
        else (BaseUAVEnv.__init__,)
    )
    valid = {
        name
        for constructor in constructors
        for name, parameter in inspect.signature(constructor).parameters.items()
        if name not in {"self", "kwargs"}
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    kwargs = {key: value for key, value in resolved.items() if key in valid}
    kwargs.update(device=env_device, return_numpy=False, max_steps=int(max_steps))
    return kwargs


def _build_train_env(env_device, max_steps, config):
    kwargs = _build_ch3_env_kwargs(env_device, max_steps, config)
    env_class = _environment_class_for_protocol(config["protocol"])
    return env_class(**kwargs), kwargs


@dataclass
class CH3Runtime:
    method: str
    config: Dict[str, Any]
    env: Any
    maddpg: Optional[MADDPG]
    replay_buffer: Optional[CH3ReplayBuffer]
    run_type: str
    train_device: torch.device


def set_ch3_determinism(seed: int) -> Dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def build_ch3_runtime(
    method: str,
    *,
    seed: int,
    max_steps: int,
    device: str | torch.device = "cpu",
    replay_size: Optional[int] = None,
    protocol: str = CH3_PILOT_V1,
) -> CH3Runtime:
    method = assert_ch3_method(method)
    config = get_ch3_method_config(method, protocol=protocol)
    config.setdefault("protocol", str(protocol))
    return build_ch3_runtime_from_resolved_config(
        method, config, seed=seed, max_steps=max_steps, device=device,
        replay_size=replay_size,
    )


def build_ch3_runtime_from_resolved_config(
    artifact_label: str,
    resolved_config: Dict[str, Any],
    *,
    seed: int,
    max_steps: int,
    device: str | torch.device = "cpu",
    replay_size: Optional[int] = None,
) -> CH3Runtime:
    """Build a runtime from an explicit, provenance-hashable configuration."""
    if not artifact_label:
        raise ValueError("artifact_label is required")
    set_ch3_determinism(seed)
    config = dict(resolved_config)
    if not config.get("ch3_config"):
        raise ValueError("resolved_config must be a Chapter-3 configuration")
    config["max_steps"] = int(max_steps)
    if replay_size is not None:
        config["replay_size"] = int(replay_size)
    train_device = _resolve_device(device)
    env, _ = _build_train_env(torch.device("cpu"), max_steps, config)
    run_type = str(config["run_type"])
    if run_type == "controller_only":
        return CH3Runtime(artifact_label, config, env, None, None, run_type, train_device)
    if run_type != "learning":
        raise RuntimeError(f"Unexpected run_type={run_type!r} for {artifact_label}")
    maddpg = MADDPG.init_from_env(
        env,
        gamma=float(config["gamma"]),
        tau=float(config["tau"]),
        lr_actor=float(config["lr_actor"]),
        lr_critic=float(config["lr_critic"]),
        hidden_dim=int(config["hidden_dim"]),
        residual_action_reg=float(config.get("residual_action_reg", 1e-2)),
    )
    maddpg.prep_rollouts(device=train_device)
    replay_buffer = CH3ReplayBuffer(
        max_steps=int(config["replay_size"]),
        num_agents=env.num_agents,
        obs_dims=tuple(env.observation_space[f"agent_{i}"].shape[0] for i in range(4)),
        ac_dims=tuple(env.action_space[f"agent_{i}"].shape[0] for i in range(4)),
        storage_device="cpu",
    )
    return CH3Runtime(
        artifact_label, config, env, maddpg, replay_buffer, run_type, train_device
    )


def _as_env_actions(runtime, observations, *, explore):
    if runtime.run_type == "controller_only":
        return torch.zeros((4, 3), dtype=torch.float32, device=runtime.env.device)
    actions = runtime.maddpg.step(observations, explore=explore)
    return torch.stack([action.squeeze(0) for action in actions]).to(runtime.env.device)


def add_planner_mechanism_metrics(
    row,
    env,
    *,
    gated_names,
    gated_values,
    gated_at_found,
    exec_reference_distances,
):
    """Add shared v3/mission planner diagnostics to one episode row."""

    row.update({
        "exec_cost_reference_mode": env.pse_exec_cost_reference_mode,
        "exec_cost_reference_position": json.dumps(
            env.exec_cost_reference_position.detach().cpu().tolist()
        ),
        "physical_executor_position": json.dumps(
            env.physical_executor_position.detach().cpu().tolist()
        ),
        "exec_cost_reference_to_executor_distance": float(
            np.mean(exec_reference_distances) if exec_reference_distances else 0.0
        ),
    })
    for name in gated_names:
        row[name] = (
            float(np.mean(gated_values[name])) if gated_values[name] else 0.0
        )
        row[f"{name}_at_found"] = gated_at_found[name]
    return row


def _run_episode(
    runtime,
    *,
    explore,
    scenario=None,
    train_updates=False,
    global_step=0,
    update_step=0,
):
    env = runtime.env
    observations = env.reset(scenario=scenario)
    if runtime.maddpg is not None:
        runtime.maddpg.prep_rollouts(device=runtime.train_device)
        runtime.maddpg.reset_noise()
    total_reward = 0.0
    energy_cost = 0.0
    residual_norms = []
    prior_norms = []
    residual_ratios = []
    residual_ratios_search = []
    residual_ratios_executor = []
    reward_components = {name: 0.0 for name in env.reward_component_names()}
    collision = False
    minimum_separation_violation = False
    found_step = None
    success_step = None
    coverage_at_found = claim_overlap = standby_distance = float("nan")
    belief_entropy_at_found = belief_peak_at_found = float("nan")
    gated_names = (
        "gated_belief_time_factor", "gated_belief_entropy_confidence",
        "gated_belief_total_confidence", "gated_belief_uniform_mix",
        "gated_belief_effective_weight", "gated_belief_mix_entropy",
        "gated_belief_mix_peak_probability",
    )
    gated_values = {name: [] for name in gated_names}
    gated_at_found = {name: float("nan") for name in gated_names}
    exec_reference_distances = []
    executor_path = 0.0
    search_distance_before_found = 0.0
    total_agent_distance = 0.0
    previous_positions = env._agent_pos.clone()
    actor_seconds = 0.0
    actor_calls = 0
    critic_losses = []
    actor_losses = []

    for step in range(1, env.max_steps + 1):
        was_found = bool(env.task_found)
        started = time.perf_counter()
        actions = _as_env_actions(runtime, observations, explore=explore)
        if runtime.run_type == "learning":
            actor_seconds += time.perf_counter() - started
            actor_calls += 1
        next_observations, rewards, dones = env.step(actions)
        for name in gated_names:
            gated_values[name].append(float(getattr(env, f"last_{name}", 0.0)))
        exec_reference_distances.append(
            float(getattr(env, "exec_cost_reference_to_executor_distance", 0.0))
        )
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
        total_reward += float(reward_tensor.sum())
        residual_norms.append(float(env.last_residual_norm))
        prior_norms.append(float(env.last_prior_term_norm))
        residual_ratios.append(float(env.last_residual_contribution_ratio))
        residual_ratios_search.append(float(env.last_residual_contribution_ratio_search))
        residual_ratios_executor.append(float(env.last_residual_contribution_ratio_executor))
        for name in reward_components:
            reward_components[name] += float(env.last_reward_components[name].sum().item())
        energy_cost += float(
            torch.sum(env._energy_coeff * torch.sum(env._agent_acc ** 2, dim=1)) * env.dt
        )
        collision = collision or bool(torch.any(env._collision_flags))
        movement = torch.norm(env._agent_pos - previous_positions, dim=1)
        total_agent_distance += float(movement.sum().item())
        if not was_found:
            search_distance_before_found += float(movement[:3].sum().item())
        else:
            executor_path += float(movement[env.executor_idx].item())
        previous_positions = env._agent_pos.clone()
        pairwise = torch.cdist(env._agent_pos, env._agent_pos)
        pairwise.fill_diagonal_(float("inf"))
        minimum_separation_violation = minimum_separation_violation or bool(
            torch.any(pairwise < env.safe_dist)
        )
        if found_step is None and env.task_found:
            found_step = env.found_step
            coverage_at_found = env._current_coverage_ratio_internal()
            claim_overlap = float(env.last_pse_claim_overlap)
            standby_distance = float(torch.norm(env._executor_wait_point - env._task_target))
            belief_entropy_at_found = float(env.last_belief_entropy)
            belief_peak_at_found = float(env.last_belief_peak_probability)
            for name in gated_names:
                gated_at_found[name] = float(getattr(env, f"last_{name}", 0.0))
        if success_step is None and env.mission_complete:
            success_step = int(env.success_step if env.success_step is not None else step)

        if train_updates:
            if runtime.maddpg is None or runtime.replay_buffer is None:
                raise RuntimeError("train_updates requires a learning runtime")
            success_flags = [bool(env.mission_complete)] * 4
            runtime.replay_buffer.push(
                observations, actions, reward_tensor, next_observations, dones, success_flags
            )
            global_step += 1
            config = runtime.config
            if (
                global_step >= int(config["warmup_steps"])
                and global_step % int(config["update_frequency"]) == 0
                and len(runtime.replay_buffer) >= int(config["batch_size"])
            ):
                runtime.maddpg.prep_training(device=runtime.train_device)
                for _ in range(int(config["updates_per_train"])):
                    sample = runtime.replay_buffer.sample(
                        int(config["batch_size"]), norm_rews=False, device=runtime.train_device
                    )
                    indices, success_batch = sample[6], sample[7]
                    errors = []
                    update_actor = update_step % int(config["policy_delay"]) == 0
                    for agent_i in range(4):
                        if update_actor:
                            critic_loss, actor_loss, error = runtime.maddpg.update(sample, agent_i)
                            actor_losses.append(float(actor_loss))
                        else:
                            critic_loss, error = runtime.maddpg.update_critic_only(sample, agent_i)
                        critic_losses.append(float(critic_loss))
                        errors.append(error.detach())
                    runtime.replay_buffer.update_priorities(
                        indices, torch.stack(errors).mean(dim=0), success_batch
                    )
                    runtime.maddpg.update_all_targets(compute_diff=False)
                    update_step += 1
                runtime.maddpg.prep_rollouts(device=runtime.train_device)
        observations = next_observations
        if all(dones):
            break

    handoff = env.get_ch3_communication_metrics()
    execution_delay = float("nan")
    if success_step is not None and env.executor_received_target_step is not None:
        execution_delay = float(success_step - env.executor_received_target_step)
    row = {
        "scenario_id": None if scenario is None else scenario["scenario_id"],
        "scenario_seed": None if scenario is None else int(scenario["scenario_seed"]),
        "steps": int(env.step_count),
        "found": int(env.task_found),
        "success": int(env.mission_complete),
        "found_step": float("nan") if found_step is None else int(found_step),
        "success_step": float("nan") if success_step is None else int(success_step),
        "succ_if_found": int(env.mission_complete) if env.task_found else float("nan"),
        "success_given_found": int(env.mission_complete) if env.task_found else float("nan"),
        "collision": int(collision),
        "minimum_separation_violation": int(minimum_separation_violation),
        "energy_cost": energy_cost,
        "reward": total_reward,
        "raw_reward": float(sum(reward_components.values())),
        "residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "mean_prior_norm": float(np.mean(prior_norms)) if prior_norms else 0.0,
        "mean_residual_contribution_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
        "mean_residual_contribution_ratio_search": float(np.mean(residual_ratios_search)) if residual_ratios_search else 0.0,
        "mean_residual_contribution_ratio_executor": float(np.mean(residual_ratios_executor)) if residual_ratios_executor else 0.0,
        "residual_contribution_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
        "coverage_at_found": coverage_at_found,
        "belief_entropy_at_start": float(env.belief_entropy_at_start),
        "belief_entropy_at_found": belief_entropy_at_found,
        "belief_peak_at_found": belief_peak_at_found,
        "claim_overlap": claim_overlap,
        "exec_delay": execution_delay,
        "execution_delay": execution_delay,
        "executor_path_after_found": executor_path,
        "executor_distance_after_found": executor_path,
        "executor_distance_after_found_unconditional": executor_path,
        "search_distance_before_found": search_distance_before_found,
        "search_distance_until_found_or_horizon": search_distance_before_found,
        "total_agent_distance": total_agent_distance,
        "standby_to_target_dist_at_found": standby_distance,
        "standby_update_attempt_count": int(env.standby_update_attempt_count),
        "standby_update_accept_count": int(env.standby_update_accept_count),
        "standby_update_reject_count": int(env.standby_update_reject_count),
        "standby_total_target_shift": float(env.standby_total_target_shift),
        "standby_executor_travel_distance": float(env.standby_executor_travel_distance),
        "standby_current_response_cost": float(env.standby_current_response_cost),
        "standby_candidate_response_cost": float(env.standby_candidate_response_cost),
        "standby_relative_gain": float(env.standby_relative_gain),
        "standby_mean_accepted_gain": float(env.standby_mean_accepted_gain),
        "handoff_count": handoff["handoff_count"],
        "handoff_delay": handoff["handoff_delay"],
        "actor_runtime_ms": (
            0.0 if runtime.run_type == "controller_only"
            else 1000.0 * actor_seconds / max(1, actor_calls)
        ),
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
    }
    protocol = runtime.config.get("protocol")
    if protocol == CH3_EFFICIENCY_V3_SCREEN:
        penalty = int(runtime.config.get("failure_penalty_steps", 100))
        failure_value = int(env.max_steps + penalty)
        penalized_completion = failure_value if success_step is None else int(success_step)
        penalized_found = failure_value if found_step is None else int(found_step)
        row.update({
            "penalized_completion_step": penalized_completion,
            "normalized_penalized_completion": penalized_completion / float(env.max_steps),
            "penalized_found_step": penalized_found,
            "completion_failure": int(success_step is None),
            "search_failure": int(found_step is None),
        })
    if protocol in (CH3_EFFICIENCY_V3_SCREEN, CH3_MISSION_V1):
        add_planner_mechanism_metrics(
            row,
            env,
            gated_names=gated_names,
            gated_values=gated_values,
            gated_at_found=gated_at_found,
            exec_reference_distances=exec_reference_distances,
        )
        # v3 CSVs use blank cells, never textual nan/inf, for unavailable
        # conditional metrics. Penalized metrics above remain unconditional.
        row = {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in row.items()
        }
    row.update(reward_components)
    return row, global_step, update_step


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]):
    rows = [
        {
            key: (
                None
                if isinstance(value, (float, np.floating))
                and not np.isfinite(float(value))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


CH3_PRIMARY_METRICS = (
    "success_rate",
    "mean_success_step",
    "found_rate",
    "mean_found_step",
    "success_given_found",
    "mean_execution_delay",
)
CH3_EFFICIENCY_METRICS = (
    "search_distance_before_found",
    "search_distance_until_found_or_horizon",
    "executor_distance_after_found",
    "executor_distance_after_found_unconditional",
    "total_agent_distance",
    "energy_cost",
    "standby_executor_travel_distance",
)
CH3_MECHANISM_METRICS = (
    "coverage_at_found",
    "belief_entropy_at_found",
    "claim_overlap",
    "residual_contribution_ratio",
    "standby_update_accept_count",
    "standby_mean_accepted_gain",
)


def _finite_values(rows, key, predicate=None):
    values = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def summarize_evaluation_rows(rows):
    rows = list(rows)
    count = len(rows)
    found = sum(int(float(row.get("found", 0))) for row in rows)
    success = sum(int(float(row.get("success", 0))) for row in rows)

    def mean_or_none(key, predicate=None):
        values = _finite_values(rows, key, predicate)
        return None if not values else float(np.mean(values))

    found_only = lambda row: bool(int(float(row.get("found", 0))))
    success_only = lambda row: bool(int(float(row.get("success", 0))))
    metrics = {
        "evaluation_count": count,
        "success_rate": None if count == 0 else success / count,
        "mean_success_step": mean_or_none("success_step", success_only),
        "found_rate": None if count == 0 else found / count,
        "mean_found_step": mean_or_none("found_step", found_only),
        "success_given_found": None if found == 0 else success / found,
        "mean_execution_delay": mean_or_none("execution_delay", success_only),
    }
    # Path metrics are reported with explicit conditioning.  A method that
    # rarely finds the target must not appear to have a shorter executor path
    # merely because non-found episodes contain zero post-discovery distance.
    metrics["search_distance_before_found"] = mean_or_none(
        "search_distance_before_found", found_only
    )
    metrics["search_distance_until_found_or_horizon"] = mean_or_none(
        "search_distance_until_found_or_horizon"
    )
    metrics["executor_distance_after_found"] = mean_or_none(
        "executor_distance_after_found", found_only
    )
    metrics["executor_distance_after_found_unconditional"] = mean_or_none(
        "executor_distance_after_found_unconditional"
    )
    for key in (
        "total_agent_distance", "energy_cost", "standby_executor_travel_distance"
    ):
        metrics[key] = mean_or_none(key)
    for key in CH3_MECHANISM_METRICS:
        predicate = found_only if key in {
            "coverage_at_found", "belief_entropy_at_found", "claim_overlap"
        } else None
        metrics[key] = mean_or_none(key, predicate)
    metrics["collision_rate"] = mean_or_none("collision")
    metrics["minimum_separation_violation_rate"] = mean_or_none(
        "minimum_separation_violation"
    )
    if rows and "penalized_completion_step" in rows[0]:
        metrics.update({
            "mean_penalized_completion_step": mean_or_none("penalized_completion_step"),
            "mean_normalized_penalized_completion": mean_or_none(
                "normalized_penalized_completion"
            ),
            "mean_penalized_found_step": mean_or_none("penalized_found_step"),
            "completion_failure_rate": mean_or_none("completion_failure"),
            "search_failure_rate": mean_or_none("search_failure"),
        })
        for key in (
            "gated_belief_effective_weight", "gated_belief_uniform_mix",
            "gated_belief_time_factor", "gated_belief_entropy_confidence",
            "gated_belief_total_confidence", "gated_belief_mix_entropy",
            "gated_belief_mix_peak_probability",
            "exec_cost_reference_to_executor_distance",
        ):
            metrics[key] = mean_or_none(key)
    metrics["metric_semantics"] = {
        "rates": "unconditional over all paired scenarios unless explicitly conditional",
        "means": "conditional on the named event for found/success timing metrics",
    }
    return metrics


_REQUIRED_SCENARIO_KEYS = {
    "scenario_id", "scenario_seed", "initial_agent_positions", "target_position",
    "initial_executor_wait_point", "planner_seed",
}


def load_scenario_manifest(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = list(data.get("scenarios", []))
    if not scenarios:
        raise ValueError("scenario manifest contains no scenarios")
    ids = []
    for index, scenario in enumerate(scenarios):
        missing = sorted(_REQUIRED_SCENARIO_KEYS - set(scenario))
        if missing:
            raise ValueError(f"scenario {index} missing required keys: {missing}")
        ids.append(str(scenario["scenario_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("scenario manifest contains duplicate scenario_id values")
    declared = data.get("scenario_count")
    if declared is not None and int(declared) != len(scenarios):
        raise ValueError(
            f"scenario_count={declared} does not match scenarios={len(scenarios)}"
        )
    data["manifest_sha256"] = json_file_sha256(path)
    return data, scenarios


def _checkpoint_metadata(
    runtime: CH3Runtime,
    *,
    method: str,
    seed: int,
    requested_episodes: int,
    max_steps: int,
    pilot: bool,
    checkpoint_episode: int,
    checkpoint_kind: str,
    global_step: int,
    update_step: int,
    manifest_id: str | None,
    manifest_sha256: str | None,
    provenance_snapshot: dict | None = None,
    run_config_hash: str | None = None,
    artifact_metadata: dict | None = None,
) -> dict:
    if provenance_snapshot is None:
        provenance_snapshot = capture_provenance_snapshot(PROJECT_ROOT)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "residual_maddpg_twin_critic_v1",
        "method": method,
        "protocol": runtime.config.get("protocol", CH3_PILOT_V1),
        "run_type": runtime.run_type,
        "seed": int(seed),
        "config": _json_safe(runtime.config),
        # config_hash is retained for compatibility with existing v1 files.
        "config_hash": _config_hash(runtime.config),
        "run_config_hash": str(
            _run_config_hash(runtime.config)
            if run_config_hash is None
            else run_config_hash
        ),
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "evaluation_config_hash": _evaluation_config_hash(runtime.config),
        "requested_episodes": int(requested_episodes),
        "episodes": int(requested_episodes),
        "max_steps": int(max_steps),
        "pilot": bool(pilot),
        "checkpoint_episode": int(checkpoint_episode),
        "checkpoint_kind": str(checkpoint_kind),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "scenario_manifest_id": manifest_id,
        "scenario_manifest_sha256": manifest_sha256,
        "reward_profile": runtime.config.get("reward_profile", CH3_PILOT_V1),
        "residual_action_reg": float(runtime.config.get("residual_action_reg", 1e-2)),
        "provenance_schema_version": provenance_snapshot["provenance_schema_version"],
        "algorithm_source_fingerprint": provenance_snapshot["algorithm_source_fingerprint"],
        "repository_source_fingerprint": provenance_snapshot["repository_source_fingerprint"],
        "algorithm_source_files": list(provenance_snapshot["algorithm_source_files"]),
        "provenance_captured_at_utc": provenance_snapshot["captured_at_utc"],
        "source_fingerprint": provenance_snapshot["repository_source_fingerprint"],
        "source_fingerprint_semantics": "legacy_repository_alias",
        "runtime_versions": runtime_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observation_dims": [
            int(runtime.env.observation_space[f"agent_{i}"].shape[0])
            for i in range(runtime.env.num_agents)
        ],
        "action_dims": [
            int(runtime.env.action_space[f"agent_{i}"].shape[0])
            for i in range(runtime.env.num_agents)
        ],
    }
    metadata.update(_json_safe(dict(artifact_metadata or {})))
    return metadata


def _read_csv_rows(path: Path):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rng_state_dict():
    return {
        "python_random_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda_rng_state"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])


def _resume_identity(
    runtime, *, method, protocol, seed, max_steps, manifest_id,
    manifest_sha256, provenance_snapshot,
):
    return {
        "method": method,
        "protocol": protocol,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "algorithm_source_fingerprint": provenance_snapshot[
            "algorithm_source_fingerprint"
        ],
        "scenario_manifest_id": manifest_id,
        "scenario_manifest_sha256": manifest_sha256,
    }


def save_resume_state(
    path,
    runtime,
    *,
    episode,
    global_step,
    update_step,
    sigma,
    method,
    protocol,
    seed,
    max_steps,
    manifest_id,
    manifest_sha256,
    provenance_snapshot,
):
    if runtime.maddpg is None or runtime.replay_buffer is None:
        raise ValueError("controller-only methods must not create resume_state.pt")
    state = {
        "schema_version": 1,
        "episode": int(episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "sigma": float(sigma),
        **_resume_identity(
            runtime,
            method=method,
            protocol=protocol,
            seed=seed,
            max_steps=max_steps,
            manifest_id=manifest_id,
            manifest_sha256=manifest_sha256,
            provenance_snapshot=provenance_snapshot,
        ),
        "provenance_schema_version": provenance_snapshot["provenance_schema_version"],
        "repository_source_fingerprint": provenance_snapshot[
            "repository_source_fingerprint"
        ],
        "algorithm_source_files": list(provenance_snapshot["algorithm_source_files"]),
        "provenance_captured_at_utc": provenance_snapshot["captured_at_utc"],
        "maddpg": runtime.maddpg.training_state_dict(),
        "replay_buffer": runtime.replay_buffer.state_dict(),
        **_rng_state_dict(),
    }
    torch.save(state, Path(path))
    return state


def load_resume_state(
    path,
    runtime,
    *,
    method,
    protocol,
    seed,
    max_steps,
    manifest_id,
    manifest_sha256,
    provenance_snapshot,
):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    expected = _resume_identity(
        runtime,
        method=method,
        protocol=protocol,
        seed=seed,
        max_steps=max_steps,
        manifest_id=manifest_id,
        manifest_sha256=manifest_sha256,
        provenance_snapshot=provenance_snapshot,
    )
    mismatches = {
        key: {"expected": value, "resume_state": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume identity mismatch: {mismatches}")
    resume_repository = state.get("repository_source_fingerprint")
    current_repository = provenance_snapshot["repository_source_fingerprint"]
    runtime.maddpg.load_training_state_dict(state["maddpg"])
    runtime.replay_buffer.load_state_dict(state["replay_buffer"])
    runtime.maddpg.prep_rollouts(device=runtime.train_device)
    _restore_rng_state(state)
    result = dict(state)
    result.update({
        "repository_source_matches_resume": resume_repository == current_repository,
        "resumed_from_repository_source_fingerprint": resume_repository,
        "current_repository_source_fingerprint": current_repository,
        "repository_source_changed_since_resume": resume_repository != current_repository,
    })
    return result


def train_and_evaluate_resolved_config(
    artifact_label,
    resolved_config,
    *,
    seed,
    episodes,
    max_steps,
    device,
    output_dir,
    pilot,
    scenario_manifest,
    protocol=CH3_PILOT_V1,
    resume=False,
    checkpoint_interval=None,
    evaluation_limit=None,
    replay_size=None,
    artifact_metadata=None,
):
    method = str(artifact_label)
    protocol = str(protocol)
    if str(resolved_config.get("protocol", protocol)) != protocol:
        raise ValueError("resolved config protocol does not match requested protocol")
    process_provenance_snapshot = capture_provenance_snapshot(PROJECT_ROOT)
    provenance_snapshot = process_provenance_snapshot
    episodes = int(episodes)
    max_steps = int(max_steps)
    runtime = build_ch3_runtime_from_resolved_config(
        method, dict(resolved_config, protocol=protocol),
        seed=seed,
        max_steps=max_steps,
        device=device,
        replay_size=replay_size,
    )
    if checkpoint_interval is not None:
        runtime.config["checkpoint_interval"] = max(0, int(checkpoint_interval))
    elif protocol == CH3_EFFICIENCY_V2:
        runtime.config["checkpoint_interval"] = CH3_CHECKPOINT_INTERVALS["pilot"]
    method_dir = Path(output_dir) / method / f"seed_{int(seed)}"
    method_dir.mkdir(parents=True, exist_ok=True)
    determinism = set_ch3_determinism(seed)

    manifest_id = manifest_sha256 = None
    scenarios = []
    if scenario_manifest is not None:
        manifest, scenarios = load_scenario_manifest(scenario_manifest)
        manifest_id = manifest.get("manifest_id")
        manifest_sha256 = manifest["manifest_sha256"]

    effective_episodes = episodes if runtime.run_type == "learning" else 0
    run_config_hash = _run_config_hash(
        runtime.config,
        {
            "method": method,
            "protocol": protocol,
            "seed": int(seed),
            "requested_episodes": int(effective_episodes),
            "pilot": bool(pilot),
            "output_directory": str(method_dir.resolve()),
            "evaluation_limit": (
                None if evaluation_limit is None else int(evaluation_limit)
            ),
            "scenario_manifest_id": manifest_id,
            "scenario_manifest_sha256": manifest_sha256,
        },
    )

    metrics_path = method_dir / "episode_metrics.csv"
    resume_path = method_dir / "resume_state.pt"
    training_rows = []
    global_step = update_step = 0
    started = time.perf_counter()
    checkpoint_paths = []
    checkpoint_metadata = None
    resume_audit = {
        "repository_source_matches_resume": True,
        "resumed_from_repository_source_fingerprint": None,
        "current_repository_source_fingerprint": process_provenance_snapshot[
            "repository_source_fingerprint"
        ],
        "repository_source_changed_since_resume": False,
    }
    if runtime.run_type == "learning":
        if episodes <= 0:
            raise ValueError("learning methods require episodes > 0")
        sigma = float(runtime.config["initial_sigma"])
        checkpoint_interval = max(0, int(runtime.config.get("checkpoint_interval", 0)))
        start_episode = 1
        if resume:
            if protocol not in (CH3_EFFICIENCY_V2, CH3_EFFICIENCY_V3_SCREEN):
                raise ValueError("--resume is supported only for efficiency v2/v3")
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume state does not exist: {resume_path}")
            state = load_resume_state(
                resume_path,
                runtime,
                method=method,
                protocol=protocol,
                seed=seed,
                max_steps=max_steps,
                manifest_id=manifest_id,
                manifest_sha256=manifest_sha256,
                provenance_snapshot=process_provenance_snapshot,
            )
            resume_audit = {
                key: state[key]
                for key in resume_audit
            }
            provenance_snapshot = {
                "provenance_schema_version": state["provenance_schema_version"],
                "captured_at_utc": state["provenance_captured_at_utc"],
                "algorithm_source_fingerprint": state[
                    "algorithm_source_fingerprint"
                ],
                "repository_source_fingerprint": state[
                    "repository_source_fingerprint"
                ],
                "algorithm_source_files": list(state["algorithm_source_files"]),
            }
            completed_episode = int(state["episode"])
            global_step = int(state["global_step"])
            update_step = int(state["update_step"])
            sigma = float(state["sigma"])
            training_rows = _read_csv_rows(metrics_path)
            recorded = [int(row["episode"]) for row in training_rows]
            if len(recorded) != len(set(recorded)):
                raise ValueError("episode_metrics.csv contains duplicate episode values")
            expected_recorded = list(range(1, completed_episode + 1))
            if recorded != expected_recorded:
                raise ValueError(
                    "episode_metrics.csv is not contiguous with resume state: "
                    f"expected {expected_recorded}, got {recorded}"
                )
            start_episode = completed_episode + 1
        elif protocol in (CH3_EFFICIENCY_V2, CH3_EFFICIENCY_V3_SCREEN) and resume_path.exists():
            # A fresh run intentionally replaces only this v2 method/seed run.
            training_rows = []

        for episode in range(start_episode, episodes + 1):
            runtime.maddpg.scale_noise(sigma, multiply=False)
            row, global_step, update_step = _run_episode(
                runtime,
                explore=True,
                train_updates=True,
                global_step=global_step,
                update_step=update_step,
            )
            row.update(method=method, seed=int(seed), episode=episode)
            training_rows.append(row)
            if episode > int(runtime.config["sigma_hold_episodes"]):
                sigma = max(
                    float(runtime.config["min_sigma"]),
                    sigma * float(runtime.config["sigma_decay"]),
                )
            if checkpoint_interval and episode % checkpoint_interval == 0:
                assert_algorithm_source_unchanged(PROJECT_ROOT, provenance_snapshot)
                periodic_path = method_dir / f"checkpoint_ep{episode:06d}.pt"
                metadata = _checkpoint_metadata(
                    runtime,
                    method=method,
                    seed=seed,
                    requested_episodes=episodes,
                    max_steps=max_steps,
                    pilot=pilot,
                    checkpoint_episode=episode,
                    checkpoint_kind="periodic",
                    global_step=global_step,
                    update_step=update_step,
                    manifest_id=manifest_id,
                    manifest_sha256=manifest_sha256,
                    provenance_snapshot=provenance_snapshot,
                    run_config_hash=run_config_hash,
                    artifact_metadata=artifact_metadata,
                )
                runtime.maddpg.save(str(periodic_path), metadata=metadata)
                checkpoint_paths.append(str(periodic_path))
            if protocol in (CH3_EFFICIENCY_V2, CH3_EFFICIENCY_V3_SCREEN):
                save_resume_state(
                    resume_path,
                    runtime,
                    episode=episode,
                    global_step=global_step,
                    update_step=update_step,
                    sigma=sigma,
                    method=method,
                    protocol=protocol,
                    seed=seed,
                    max_steps=max_steps,
                    manifest_id=manifest_id,
                    manifest_sha256=manifest_sha256,
                    provenance_snapshot=provenance_snapshot,
                )
                _write_csv(metrics_path, training_rows)
        training_time = time.perf_counter() - started
        checkpoint_path = method_dir / ("pilot_model.pt" if pilot else "model_final.pt")
        assert_algorithm_source_unchanged(PROJECT_ROOT, provenance_snapshot)
        metadata = _checkpoint_metadata(
            runtime,
            method=method,
            seed=seed,
            requested_episodes=episodes,
            max_steps=max_steps,
            pilot=pilot,
            checkpoint_episode=episodes,
            checkpoint_kind="final",
            global_step=global_step,
            update_step=update_step,
            manifest_id=manifest_id,
            manifest_sha256=manifest_sha256,
            provenance_snapshot=provenance_snapshot,
            run_config_hash=run_config_hash,
            artifact_metadata=artifact_metadata,
        )
        checkpoint_metadata = metadata
        runtime.maddpg.save(str(checkpoint_path), metadata=metadata)
        if str(checkpoint_path) not in checkpoint_paths:
            checkpoint_paths.append(str(checkpoint_path))
    else:
        # Controller-only baselines never train.  The caller may pass a nominal
        # episode budget, but it is deliberately ignored and reported as zero.
        if resume:
            raise ValueError("controller-only methods do not have resumable training state")
        training_time = 0.0
        checkpoint_path = None

    evaluation_rows = []
    selected_scenarios = scenarios[
        : None if evaluation_limit is None else max(0, int(evaluation_limit))
    ]
    for scenario in selected_scenarios:
        row, _, _ = _run_episode(runtime, explore=False, scenario=scenario)
        row.update(method=method, seed=int(seed))
        evaluation_rows.append(row)

    _write_csv(metrics_path, training_rows)
    _write_csv(method_dir / "evaluation_metrics.csv", evaluation_rows)
    checkpoint_sha256 = (
        None if checkpoint_path is None else file_sha256(checkpoint_path)
    )
    assert_algorithm_source_unchanged(PROJECT_ROOT, provenance_snapshot)
    final_repository_source = repository_source_fingerprint(PROJECT_ROOT)
    summary = {
        "method": method,
        "protocol": protocol,
        "seed": int(seed),
        "pilot": bool(pilot),
        "episodes": episodes if runtime.run_type == "learning" else 0,
        "max_steps": max_steps,
        "run_type": runtime.run_type,
        "run_directory": str(method_dir),
        "resolved_device": str(runtime.train_device),
        "checkpoint_path": "N/A" if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_metadata": _json_safe(checkpoint_metadata),
        "checkpoint_interval": int(runtime.config.get("checkpoint_interval", 0)),
        "resolved_config": _json_safe(runtime.config),
        "config_hash": _config_hash(runtime.config),
        "run_config_hash": run_config_hash,
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "evaluation_config_hash": _evaluation_config_hash(runtime.config),
        "reward_profile": runtime.config.get("reward_profile", CH3_PILOT_V1),
        "residual_action_reg": float(runtime.config.get("residual_action_reg", 1e-2)),
        "resume_state_path": str(resume_path) if runtime.run_type == "learning" and protocol in (CH3_EFFICIENCY_V2, CH3_EFFICIENCY_V3_SCREEN) else "N/A",
        "global_step": int(global_step),
        "update_step": int(update_step),
        "training_time": float(training_time),
        "actor_runtime_ms": (
            0.0 if runtime.run_type == "controller_only"
            else float(np.mean([row["actor_runtime_ms"] for row in evaluation_rows]))
            if evaluation_rows else float("nan")
        ),
        "evaluation_scenarios": len(evaluation_rows),
        "scenario_ids": [str(s["scenario_id"]) for s in selected_scenarios],
        "scenario_manifest": None if scenario_manifest is None else str(scenario_manifest),
        "scenario_manifest_id": manifest_id,
        "scenario_manifest_sha256": manifest_sha256,
        "provenance_schema_version": provenance_snapshot["provenance_schema_version"],
        "algorithm_source_fingerprint": provenance_snapshot[
            "algorithm_source_fingerprint"
        ],
        "repository_source_fingerprint": provenance_snapshot[
            "repository_source_fingerprint"
        ],
        "algorithm_source_files": list(provenance_snapshot["algorithm_source_files"]),
        "provenance_captured_at_utc": provenance_snapshot["captured_at_utc"],
        "source_fingerprint": provenance_snapshot["repository_source_fingerprint"],
        "source_fingerprint_semantics": "legacy_repository_alias",
        "repository_source_fingerprint_at_start": process_provenance_snapshot[
            "repository_source_fingerprint"
        ],
        "repository_source_fingerprint_at_end": final_repository_source,
        "final_repository_source_fingerprint": final_repository_source,
        "repository_changed_during_run": (
            process_provenance_snapshot["repository_source_fingerprint"]
            != final_repository_source
        ),
        **resume_audit,
        "runtime_versions": runtime_versions(),
        "determinism": determinism,
        "communication_model": runtime.env.communication_model_id,
    }
    summary.update(_json_safe(dict(artifact_metadata or {})))
    summary.update(summarize_evaluation_rows(evaluation_rows))
    (method_dir / "training_summary.json").write_text(
        json.dumps(
            _json_safe(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return summary, training_rows, evaluation_rows


def train_and_evaluate_method(
    method,
    *,
    seed,
    episodes,
    max_steps,
    device,
    output_dir,
    pilot,
    scenario_manifest,
    protocol=CH3_PILOT_V1,
    resume=False,
    checkpoint_interval=None,
    evaluation_limit=None,
    replay_size=None,
):
    """Backward-compatible v1/v2 entry point delegating to resolved config."""
    method = assert_ch3_method(method)
    config = get_ch3_method_config(method, protocol=protocol)
    return train_and_evaluate_resolved_config(
        method,
        config,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        device=device,
        output_dir=output_dir,
        pilot=pilot,
        scenario_manifest=scenario_manifest,
        protocol=protocol,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        evaluation_limit=evaluation_limit,
        replay_size=replay_size,
    )


def train(method, **kwargs):
    return train_and_evaluate_method(assert_ch3_method(method), **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pure Chapter-3 training entry point")
    parser.add_argument("--method", required=True, choices=ACTIVE_CH3_FINAL_EXPERIMENT_MODES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--protocol", choices=tuple(CH3_PROTOCOL_CONFIGS), default=CH3_PILOT_V1)
    parser.add_argument("--budget", choices=tuple(CH3_TRAINING_BUDGETS), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args(argv)
    method_config = get_ch3_method_config(args.method, protocol=args.protocol)
    if args.protocol == CH3_EFFICIENCY_V2:
        resolved_episodes = (
            int(args.episodes)
            if args.episodes is not None
            else CH3_TRAINING_BUDGETS[args.budget]
        )
        if args.method in CONTROLLER_ONLY_METHODS:
            resolved_episodes = 0
        output_dir = args.output_dir or DEFAULT_EFFICIENCY_V2_OUTPUT_DIR
        resolved_checkpoint_interval = (
            int(args.checkpoint_interval)
            if args.checkpoint_interval is not None
            else CH3_CHECKPOINT_INTERVALS[args.budget]
        )
    else:
        resolved_episodes = (
            int(method_config["pilot_episodes"] if args.pilot else method_config["training_episodes"])
            if args.episodes is None and args.method not in CONTROLLER_ONLY_METHODS
            else int(args.episodes or 0)
        )
        output_dir = args.output_dir or (
            DEFAULT_PILOT_OUTPUT_DIR if args.pilot else DEFAULT_FORMAL_OUTPUT_DIR
        )
        resolved_checkpoint_interval = args.checkpoint_interval
    summary, _, _ = train_and_evaluate_method(
        args.method,
        seed=args.seed,
        episodes=resolved_episodes,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=output_dir,
        pilot=args.pilot,
        scenario_manifest=args.scenario_manifest,
        protocol=args.protocol,
        resume=args.resume,
        checkpoint_interval=resolved_checkpoint_interval,
        evaluation_limit=args.evaluation_limit,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
