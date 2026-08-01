from __future__ import annotations

import torch

from env import UAVEnv
from map.map_module import ProbabilisticTaskMapPlanner
from train import build_ch3_runtime


def _forced_detection_scenario(planner_seed=1234):
    return {
        "scenario_id": "forced_detection",
        "scenario_seed": 1,
        "initial_agent_positions": [
            [2.0, 2.0, 1.0],
            [8.0, 2.0, 2.0],
            [2.0, 8.0, 3.0],
            [16.0, 16.0, 4.0],
        ],
        "target_position": [2.0, 2.0, 1.0],
        "flow_phase_x": 0.0,
        "flow_phase_y": 0.0,
        "initial_executor_wait_point": [10.0, 10.0, 4.0],
        "planner_seed": planner_seed,
    }


def test_discovery_reward_is_preserved():
    env = UAVEnv(max_steps=4, return_numpy=False)
    env.reset(scenario=_forced_detection_scenario())
    _, rewards, _ = env.step(torch.zeros((4, 3)))

    assert env.task_found
    assert env.finder_idx == 0
    assert float(rewards[env.finder_idx]) > 0.5
    assert bool(torch.all(rewards[:3] > 0.0))


def test_uniform_belief_support_spans_the_grid():
    planner = ProbabilisticTaskMapPlanner(
        space_size=(20, 20, 8),
        grid_size=(10, 10, 8),
        z_range=(0.5, 7.5),
        device="cpu",
    )
    points, probabilities = planner.topk_belief_points(48)

    assert points.shape[0] >= 40
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.unique(points[:, 0]).numel() >= 8
    assert torch.unique(points[:, 1]).numel() >= 5
    assert torch.unique(points[:, 2]).numel() >= 4


def test_standby_ablation_does_not_shift_search_rng():
    scenario = _forced_detection_scenario(planner_seed=5678)
    full = build_ch3_runtime(
        "ch3_pse_rmaddpg", seed=11, max_steps=5, device="cpu", replay_size=8
    ).env
    no_standby = build_ch3_runtime(
        "ch3_pse_no_standby", seed=11, max_steps=5, device="cpu", replay_size=8
    ).env

    full.reset(scenario=scenario)
    no_standby.reset(scenario=scenario)

    assert torch.allclose(full._search_waypoints, no_standby._search_waypoints)
