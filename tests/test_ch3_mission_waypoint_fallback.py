import numpy as np
import torch

from map.path_planner import ObstacleAwareTaskMapPlanner


def _planner():
    planner = ObstacleAwareTaskMapPlanner(
        space_size=(20, 20, 8),
        grid_size=(5, 5, 4),
        z_range=(0.5, 7.5),
        device="cpu",
    )
    planner.reset(None, [])
    return planner


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_all_unreachable_keeps_position_claims_and_rng_unchanged():
    planner = _planner()
    current = torch.tensor([2.0, 2.0, 2.0])
    points = torch.tensor([[6.0, 2.0, 2.0], [10.0, 2.0, 2.0]])
    planner._candidate_points = lambda *args, **kwargs: (
        points,
        torch.full((2,), -torch.inf),
    )
    claims_before = planner.claim_count.clone()
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    chosen = planner.sample_next_waypoint(0, current)
    assert torch.equal(chosen, current)
    assert torch.equal(planner.claim_count, claims_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert planner.last_all_search_candidates_unreachable is True
    assert (
        planner.last_waypoint_failure_reason
        == "all_candidates_unreachable"
    )


def test_partial_reachability_never_selects_negative_infinity():
    planner = _planner()
    current = torch.tensor([2.0, 2.0, 2.0])
    points = torch.tensor([[6.0, 2.0, 2.0], [10.0, 2.0, 2.0]])
    planner._candidate_points = lambda *args, **kwargs: (
        points,
        torch.tensor([-torch.inf, 0.25]),
    )
    planner.stochastic_eps = 0.0
    chosen = planner.sample_next_waypoint(0, current)
    assert torch.equal(chosen, points[1])
    assert planner.last_waypoint_failure_reason is None
