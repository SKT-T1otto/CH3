"""Unified runtime builder for known- and unknown-map Chapter-3 profiles."""

from __future__ import annotations

import torch

from algorithms.maddpg import MADDPG
from ch3_config import build_ch3_config
from ch3_constants import CH3_MISSION_V1, CH3_UNKNOWN_MAP_V1, UNKNOWN_MAP_PROFILES
from env import UAVEnv, _BaseUAVEnv
from train import CH3Runtime, _resolve_device, set_ch3_determinism
from utils.ch3_buffer import CH3ReplayBuffer


def _declared_parameter_names(function) -> set[str]:
    """Read the real Python signature, ignoring compatibility ``__signature__``."""

    code = function.__code__
    count = int(code.co_argcount + code.co_kwonlyargcount)
    return {
        name
        for name in code.co_varnames[:count]
        if name not in {"self", "kwargs"}
    }


ENV_KEYS = (
    _declared_parameter_names(_BaseUAVEnv.__init__)
    | _declared_parameter_names(UAVEnv.__init__)
)


def build_runtime(
    base_candidate,
    scenario_profile,
    *,
    seed,
    max_steps=400,
    device="cpu",
    replay_size=None,
    resolved_config=None,
):
    """Build one runtime for any registered S- or M-profile.

    The artifact protocol is ``ch3_mission_v1`` for S-profiles and
    ``ch3_unknown_map_v1`` for M-profiles, while both execute through the same
    merged :class:`env.UAVEnv` implementation.
    """

    set_ch3_determinism(seed)
    config = (
        build_ch3_config(base_candidate, scenario_profile)
        if resolved_config is None
        else dict(resolved_config)
    )
    if config.get("protocol") != CH3_MISSION_V1:
        raise ValueError("runtime requires base protocol=ch3_mission_v1")

    artifact_protocol = config.get("artifact_protocol", CH3_MISSION_V1)
    expected_artifact = (
        CH3_UNKNOWN_MAP_V1
        if scenario_profile in UNKNOWN_MAP_PROFILES
        else CH3_MISSION_V1
    )
    if artifact_protocol != expected_artifact:
        raise ValueError(
            "scenario profile and artifact protocol disagree: "
            f"profile={scenario_profile!r}, protocol={artifact_protocol!r}"
        )

    config["max_steps"] = int(max_steps)
    if replay_size is not None:
        config["replay_size"] = int(replay_size)

    env_kwargs = {
        key: value for key, value in config.items() if key in ENV_KEYS
    }
    env_kwargs.update(
        max_steps=int(max_steps),
        device=torch.device("cpu"),
        return_numpy=False,
    )
    env = UAVEnv(**env_kwargs)
    train_device = _resolve_device(device)

    if config.get("run_type") != "learning":
        raise ValueError(
            "unified mission runtime currently supports learning v3 candidates only"
        )
    maddpg = MADDPG.init_from_env(
        env,
        gamma=float(config["gamma"]),
        tau=float(config["tau"]),
        lr_actor=float(config["lr_actor"]),
        lr_critic=float(config["lr_critic"]),
        hidden_dim=int(config["hidden_dim"]),
        residual_action_reg=float(config["residual_action_reg"]),
    )
    maddpg.prep_rollouts(device=train_device)
    replay = CH3ReplayBuffer(
        max_steps=int(config["replay_size"]),
        num_agents=env.num_agents,
        obs_dims=(28,) * 4,
        ac_dims=(3,) * 4,
        storage_device="cpu",
    )
    prefix = "unknown__" if artifact_protocol == CH3_UNKNOWN_MAP_V1 else ""
    label = f"{prefix}{base_candidate}__{scenario_profile}"
    return CH3Runtime(label, config, env, maddpg, replay, "learning", train_device)
