from __future__ import annotations

import pytest
import torch

from algorithms.maddpg import MADDPG
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES
from train import (
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    _build_train_env,
    get_ch3_method_config,
)
from utils.networks import MLPNetwork


def _build(method, protocol, max_steps=8):
    return _build_train_env(
        torch.device("cpu"),
        max_steps,
        get_ch3_method_config(method, protocol=protocol),
    )[0]


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_all_ch3_methods_share_28d_local_observation_without_neighbor_blocks(protocol):
    for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
        env = _build(method, protocol)
        obs = env.reset()
        layout = env.get_observation_layout()
        assert env.obs_dim == 28
        assert [item.numel() for item in obs] == [28] * 4
        assert layout[-1]["end"] == 28
        assert all("neighbor" not in field["name"] for field in layout)


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_true_target_is_not_exposed_before_detection(protocol):
    env = _build("ch3_pse_rmaddpg", protocol)
    env.reset()
    before = [item.clone() for item in env._get_obs()]
    env._task_target.copy_(torch.tensor([0.75, 18.5, 7.0], dtype=env.dtype))
    after = env._get_obs()
    for left, right in zip(before, after):
        assert torch.allclose(left, right, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_learning_methods_instantiate_plain_mlp_actor(protocol):
    for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
        cfg = get_ch3_method_config(method, protocol=protocol)
        if cfg["run_type"] == "controller_only":
            continue
        maddpg = MADDPG.init_from_env(
            _build(method, protocol),
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            lr_actor=cfg["lr_actor"],
            lr_critic=cfg["lr_critic"],
            hidden_dim=cfg["hidden_dim"],
            residual_action_reg=float(cfg.get("residual_action_reg", 1e-2)),
        )
        assert all(type(agent.policy) is MLPNetwork for agent in maddpg.agents)
