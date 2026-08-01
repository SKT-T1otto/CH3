import math

import torch

from map.path_planner import ObstacleAwareTaskMapPlanner


def _planner(obstacles=()):
    planner = ObstacleAwareTaskMapPlanner(
        space_size=(20, 20, 8),
        grid_size=(10, 10, 8),
        z_range=(0.5, 7.5),
        device="cpu",
        planner_obstacle_clearance=0.4,
    )
    planner.reset(None, list(obstacles))
    return planner


def test_real_endpoints_are_included_and_same_cell_cost_is_not_zero():
    planner = _planner()
    start = [2.10, 2.10, 2.10]
    goal = [2.35, 2.30, 2.25]
    result = planner.grid_astar_path(start, goal)
    assert result["reachable"]
    assert result["start_connector_valid"]
    assert result["exact_goal_reachable"]
    assert result["cost"] > 0
    assert result["cost"] == (
        result["start_connector_cost"]
        + result["grid_cost"]
        + result["goal_connector_cost"]
    )
    assert torch.allclose(torch.tensor(result["points"][0]), torch.tensor(start))
    assert torch.allclose(torch.tensor(result["points"][-1]), torch.tensor(goal))


def test_every_path_edge_including_connectors_is_collision_free():
    obstacle = {"center": [10, 10, 4], "size": [2, 8, 6]}
    planner = _planner([obstacle])
    result = planner.grid_astar_path([4.2, 10.1, 4.1], [15.8, 9.9, 3.9])
    assert result["reachable"]
    assert result["goal_connector_valid"]
    assert all(
        planner.segment_is_free(left, right)
        for left, right in zip(result["points"], result["points"][1:])
    )
    subgoals = planner.path_to_subgoals(result, [15.8, 9.9, 3.9])
    assert torch.allclose(subgoals[-1], torch.tensor([15.8, 9.9, 3.9]))


def test_unreachable_cost_and_estimate_are_infinite_and_masked():
    wall = {"center": [10, 10, 4], "size": [2, 20, 8]}
    planner = _planner([wall])
    result = planner.grid_astar_path([2, 10, 4], [18, 10, 4])
    assert not result["reachable"] and math.isinf(result["cost"])
    estimate = planner.estimate_travel_time(
        [2, 10, 4], torch.tensor([[18, 10, 4], [4, 10, 4]])
    )
    assert torch.isinf(estimate[0])
    assert torch.isfinite(estimate[1])
    score = planner.score_search_candidates(
        0,
        torch.tensor([[18, 10, 4], [4, 10, 4]], dtype=torch.float32),
        torch.ones(2),
        torch.tensor([2, 10, 4], dtype=torch.float32),
    )
    assert torch.isneginf(score[0])
    assert torch.isfinite(score[1])


def test_astar_is_deterministic_and_continuous_endpoints_are_not_cached():
    planner = _planner()
    start = [2.1, 2.1, 2.1]
    goal_a = [6.1, 2.1, 2.1]
    goal_b = [6.4, 2.4, 2.4]
    first = planner.grid_astar_path(start, goal_a)
    again = planner.grid_astar_path(start, goal_a)
    other = planner.grid_astar_path(start, goal_b)
    assert first == again
    assert torch.allclose(torch.tensor(first["points"][-1]), torch.tensor(goal_a))
    assert torch.allclose(torch.tensor(other["points"][-1]), torch.tensor(goal_b))
    assert first["cost"] != other["cost"]


def test_astar_uses_shared_component_not_each_endpoints_nearest_connector():
    planner = _planner()
    start = torch.tensor([0.0, 0.0, 0.0])
    goal = torch.tensor([20.0, 20.0, 8.0])
    start_wrong = (0, 0, 0)
    start_shared = (1, 0, 0)
    goal_wrong = (9, 9, 7)
    goal_shared = (8, 9, 7)
    labels = {
        start_wrong: 0,
        start_shared: 1,
        goal_wrong: 2,
        goal_shared: 1,
    }

    def connectors(point, role="searcher", *, per_component=True):
        point = torch.as_tensor(point)
        if torch.allclose(point, start):
            return [(0.1, start_wrong), (0.2, start_shared)]
        return [(0.1, goal_wrong), (0.2, goal_shared)]

    planner._connector_candidates = connectors
    planner._component_labels = lambda: labels
    planner._core_cell_path = lambda left, right, role: {
        "reachable": True,
        "cells": [left, right],
        "cost": 1.0,
    }
    result = planner.grid_astar_path(start, goal)
    assert result["reachable"]
    assert result["resolved_component_id"] == 1
    assert result["resolved_start_cell"] == start_shared
    assert result["resolved_goal_cell"] == goal_shared
    assert math.isclose(result["cost"], 1.4)
