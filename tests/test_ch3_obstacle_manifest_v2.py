from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from tools.build_ch3_efficiency_scenarios import (
    MANIFEST_SPECS,
    MIN_AGENT_DISTANCE,
    build_efficiency_manifest,
)
from train import CH3_EFFICIENCY_V2, build_ch3_runtime


_EXPECTED_OBSTACLES = (
    (np.asarray([5.0, 5.0, 2.0]), np.asarray([2.5, 2.5, 2.0])),
    (np.asarray([11.0, 10.0, 4.0]), np.asarray([3.0, 3.0, 2.5])),
    (np.asarray([15.5, 6.0, 5.5]), np.asarray([2.0, 3.0, 2.0])),
)
_SPACE_SIZE = np.asarray([20.0, 20.0, 8.0])


def _independent_inside_obstacle(point):
    point = np.asarray(point, dtype=np.float64)
    for center, size in _EXPECTED_OBSTACLES:
        lower = center - size / 2.0
        upper = center + size / 2.0
        if np.all(point >= lower) and np.all(point <= upper):
            return True
    return False


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_official_efficiency_manifests_are_reproducible_and_geometrically_valid():
    assert {key: spec["count"] for key, spec in MANIFEST_SPECS.items()} == {
        "validation": 50,
        "test": 100,
        "obstacle": 20,
    }

    for kind in ("validation", "test", "obstacle"):
        first = build_efficiency_manifest(kind, count=4)
        second = build_efficiency_manifest(kind, count=4)
        assert first == second
        assert _canonical_sha256(first) == _canonical_sha256(second)
        assert first["scenario_count"] == len(first["scenarios"]) == 4
        ids = [row["scenario_id"] for row in first["scenarios"]]
        assert len(ids) == len(set(ids))

        for scenario in first["scenarios"]:
            assert scenario["flow_phase_x"] == 0.0
            assert scenario["flow_phase_y"] == 0.0
            positions = np.asarray(scenario["initial_agent_positions"], dtype=np.float64)
            distances = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            )
            distances += np.eye(4) * 1e9
            assert float(distances.min()) >= MIN_AGENT_DISTANCE - 1e-6

            points = [
                *scenario["initial_agent_positions"],
                scenario["target_position"],
                scenario["initial_executor_wait_point"],
            ]
            for point in points:
                point_array = np.asarray(point, dtype=np.float64)
                assert np.all(point_array >= 0.0)
                assert np.all(point_array <= _SPACE_SIZE)
                if scenario["use_obstacles"]:
                    assert not _independent_inside_obstacle(point_array)

        assert first["use_obstacles"] is (kind == "obstacle")
        assert first["obstacle_layout_id"] == (
            "default_fixed_v1" if kind == "obstacle" else "none"
        )


def test_obstacle_scenario_resets_steps_and_rebuilds_valid_mask():
    scenario = build_efficiency_manifest("obstacle", count=1)["scenarios"][0]
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg",
        seed=1,
        max_steps=4,
        device="cpu",
        replay_size=32,
        protocol=CH3_EFFICIENCY_V2,
    )
    obs = runtime.env.reset(scenario)
    assert runtime.env.use_obstacles
    assert runtime.env.obstacle_layout_id == "default_fixed_v1"
    assert not bool(runtime.env.map_module.valid_mask.all())
    assert all(item.shape == (28,) for item in obs)

    next_obs, rewards, dones = runtime.env.step(torch.zeros((4, 3)))
    assert all(torch.isfinite(item).all() for item in next_obs)
    assert torch.isfinite(torch.as_tensor(rewards)).all()
    assert len(dones) == 4

    non_obstacle = build_efficiency_manifest("validation", count=1)["scenarios"][0]
    runtime.env.reset(non_obstacle)
    assert not runtime.env.use_obstacles
    assert runtime.env.obstacle_layout_id == "none"
    assert runtime.env._obstacle_lower is None
    assert bool(runtime.env.map_module.valid_mask.all())


def test_environment_rejects_manifest_point_inside_obstacle():
    scenario = build_efficiency_manifest("obstacle", count=1)["scenarios"][0]
    scenario = dict(scenario)
    scenario["target_position"] = [5.0, 5.0, 2.0]
    assert _independent_inside_obstacle(scenario["target_position"])

    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg",
        seed=1,
        max_steps=4,
        device="cpu",
        replay_size=32,
        protocol=CH3_EFFICIENCY_V2,
    )
    with pytest.raises(ValueError, match="target_position.*inside obstacle"):
        runtime.env.reset(scenario)
