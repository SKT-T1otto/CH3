import math

import torch

from map.path_planner import ObstacleAwareTaskMapPlanner


def _planner(obstacles):
    planner = ObstacleAwareTaskMapPlanner(
        space_size=(20, 20, 8), grid_size=(10, 10, 8),
        z_range=(0.5, 7.5), device="cpu",
        planner_obstacle_clearance=0.4,
    )
    planner.reset(None, obstacles)
    return planner


def test_astar_geodesic_routes_around_obstacle_deterministically():
    obstacle = {"center": [10, 10, 4], "size": [2, 8, 6]}
    planner = _planner([obstacle])
    start, goal = [4, 10, 4], [16, 10, 4]
    state = torch.random.get_rng_state().clone()
    first = planner.grid_astar_path(start, goal)
    second = planner.grid_astar_path(start, goal)
    assert torch.equal(state, torch.random.get_rng_state())
    assert first == second and first["reachable"]
    direct_time = 12.0
    assert first["cost"] > direct_time
    for point in first["points"]:
        assert not (
            8.6 <= point[0] <= 11.4
            and 5.6 <= point[1] <= 14.4
            and 0.6 <= point[2] <= 7.4
        )
    subgoals = planner.path_to_subgoals(first, goal)
    assert subgoals
    assert all(torch.isfinite(point).all() for point in subgoals)


def test_same_cell_cost_zero_and_partitioned_world_is_unreachable():
    planner = _planner([])
    same = planner.grid_astar_path([2, 2, 2], [2, 2, 2])
    assert same["reachable"] and same["cost"] == 0.0
    wall = {"center": [10, 10, 4], "size": [2, 20, 8]}
    blocked = _planner([wall]).grid_astar_path([2, 10, 4], [18, 10, 4])
    assert not blocked["reachable"]
    assert math.isinf(blocked["cost"])


def test_belief_transition_static_and_diffusion_constraints():
    static = _planner([])
    before = static.belief_map.clone()
    state = torch.random.get_rng_state().clone()
    returned = static.predict_belief_motion()
    assert torch.equal(before, returned)
    assert torch.equal(state, torch.random.get_rng_state())
    moving = _planner([{"center": [10, 10, 4], "size": [2, 4, 4]}])
    moving.target_belief_transition_mode = "isotropic_diffusion_v1"
    moving.target_belief_diffusion_rate = 0.12
    moving.belief_map.zero_()
    index = int(torch.nonzero(moving.flat_valid_mask, as_tuple=False)[0])
    moving.belief_map.reshape(-1)[index] = 1.0
    entropy_before = float(moving.belief_entropy().item())
    moving.predict_belief_motion()
    assert torch.all(moving.belief_map >= 0)
    assert torch.isclose(moving.belief_map.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(moving.belief_map[~moving.valid_mask] == 0)
    assert float(moving.belief_entropy().item()) >= entropy_before
