import json

import numpy as np
import pytest

from tools.build_ch3_scenarios import (
    build_scenario_manifests,
    validate_scenario_reachability,
)


def test_four_manifests_are_reproducible_and_strictly_paired():
    first = build_scenario_manifests(1, 71001)
    second = build_scenario_manifests(1, 71001)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    profiles = tuple(first)
    assert set(profiles) == {
        "S00_STATIC_CLEAR", "S10_MOVING_CLEAR",
        "S01_STATIC_OBSTACLE", "S11_MOVING_OBSTACLE",
    }
    for index in range(1):
        rows = {profile: first[profile]["scenarios"][index] for profile in profiles}
        shared = (
            "pair_group_id", "scenario_seed", "planner_seed",
            "initial_agent_positions", "target_initial_position",
            "initial_executor_wait_point",
        )
        assert all(rows[profile][key] == rows["S00_STATIC_CLEAR"][key] for profile in profiles for key in shared)
        assert not rows["S00_STATIC_CLEAR"]["use_obstacles"]
        assert not rows["S10_MOVING_CLEAR"]["use_obstacles"]
        assert rows["S01_STATIC_OBSTACLE"]["obstacles"] == rows["S11_MOVING_OBSTACLE"]["obstacles"]
        assert rows["S00_STATIC_CLEAR"]["target_initial_velocity"] == [0.0, 0.0, 0.0]
        assert rows["S01_STATIC_OBSTACLE"]["target_initial_velocity"] == [0.0, 0.0, 0.0]
        assert rows["S10_MOVING_CLEAR"]["target_initial_velocity"] == rows["S11_MOVING_OBSTACLE"]["target_initial_velocity"]
        assert rows["S10_MOVING_CLEAR"]["target_trajectory_distance"] > 10


def test_all_initial_points_are_legal_for_obstacle_profiles():
    manifests = build_scenario_manifests(1)
    for profile in ("S01_STATIC_OBSTACLE", "S11_MOVING_OBSTACLE"):
        for row in manifests[profile]["scenarios"]:
            obstacle = row["obstacles"][0]
            center = np.asarray(obstacle["center"])
            half = np.asarray(obstacle["size"]) / 2
            points = (
                list(row["initial_agent_positions"])
                + [row["target_initial_position"], row["initial_executor_wait_point"]]
            )
            assert all(not np.all(np.asarray(point) >= center - half) or not np.all(np.asarray(point) <= center + half) for point in points)


def test_manifest_declares_connectivity_and_trajectory_identity_inputs():
    manifests = build_scenario_manifests(1, 71001)
    for manifest in manifests.values():
        row = manifest["scenarios"][0]
        assert row["planner_grid_size"] == [10, 10, 8]
        assert row["planner_obstacle_clearance"] == 0.4
        assert row["target_obstacle_clearance"] == 0.2
        assert row["target_capture_radius"] == 0.8
        assert row["connectivity_component_id"]
        assert row["target_trajectory_reachable"] is True
        assert row["scenario_schema_version"] == 3
        assert row["target_trajectory_exact_endpoint_reachable"] is True
        assert row["target_trajectory_connector_failure_count"] == 0
        assert len(row["target_trajectory_sha256"]) == 64


def test_partitioned_world_and_uncapturable_target_are_rejected():
    wall = {"center": [10, 10, 4], "size": [2, 20, 8]}
    with pytest.raises(RuntimeError):
        validate_scenario_reachability(
            [[2, 2, 2], [2, 4, 2], [18, 2, 2], [18, 4, 2]],
            [2, 6, 2],
            [2, 8, 2],
            [[2, 8, 2]],
            [wall],
        )
    obstacle = {"center": [10, 10, 4], "size": [4, 4, 4]}
    with pytest.raises(RuntimeError):
        validate_scenario_reachability(
            [[2, 2, 2], [3, 2, 2], [2, 3, 2], [3, 3, 2]],
            [4, 4, 2],
            [5, 5, 2],
            [[10, 10, 4]],
            [obstacle],
        )
