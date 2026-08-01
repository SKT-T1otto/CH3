import math

import numpy as np
import pytest
import torch

from target_motion import (
    TargetState,
    advance_target_state,
    predict_target_state,
    simulate_target_trajectory,
    solve_intercept_point,
    swept_relative_min_distance,
)


BOUNDS = np.asarray([20.0, 20.0, 8.0])


def test_static_and_free_constant_velocity_are_exact():
    static = TargetState([1, 2, 3], [1, 0, 0], 0, "static")
    advanced = advance_target_state(static, 0.2, BOUNDS)
    assert np.array_equal(advanced.position, static.position)
    moving = TargetState([1, 2, 3], [0.5, -0.25, 0.1], 0, "constant_velocity_reflect_v1")
    advanced = advance_target_state(moving, 0.2, BOUNDS)
    assert np.allclose(advanced.position, [1.1, 1.95, 3.02])
    assert np.array_equal(advanced.velocity, moving.velocity)


def test_boundary_and_obstacle_reflection_prevent_tunnelling():
    boundary = TargetState([19.9, 3, 3], [2, 0, 0], 0, "constant_velocity_reflect_v1")
    reflected = advance_target_state(boundary, 0.2, BOUNDS)
    assert reflected.position[0] < 20.0
    assert reflected.velocity[0] == -2.0
    obstacle = {"center": [10, 10, 4], "size": [2, 6, 6]}
    fast = TargetState([5, 10, 4], [30, 0, 0], 0, "constant_velocity_reflect_v1")
    reflected = advance_target_state(fast, 0.3, BOUNDS, [obstacle], clearance=0.2)
    assert reflected.velocity[0] < 0
    assert not (8.8 < reflected.position[0] < 11.2)
    assert reflected.reflection_count >= 1


def test_trajectory_is_deterministic_finite_and_rng_free():
    state = TargetState([3, 4, 2], [0.4, 0.2, 0.1], 0, "constant_velocity_reflect_v1")
    rng_before = np.random.get_state()
    first = simulate_target_trajectory(state, 100, 0.2, BOUNDS)
    second = simulate_target_trajectory(state, 100, 0.2, BOUNDS)
    rng_after = np.random.get_state()
    assert all(np.array_equal(a.position, b.position) for a, b in zip(first, second))
    assert all(np.all(np.isfinite(item.position)) for item in first)
    assert rng_before[0] == rng_after[0]
    assert np.array_equal(rng_before[1], rng_after[1])


def test_swept_detection_catches_between_endpoint_crossing():
    distance, tau = swept_relative_min_distance(
        [0, 0, 0], [0, 0, 0], [-2, 0, 0], [2, 0, 0]
    )
    assert distance == 0.0
    assert tau == 0.5
    miss, _ = swept_relative_min_distance(
        [0, 0, 0], [0, 0, 0], [-2, 3, 0], [2, 3, 0]
    )
    assert miss == 3.0


def test_target_state_payload_aliases_metadata_and_deep_copy():
    nested = {"route": {"points": [[1, 2, 3]]}}
    state = TargetState(
        [1, 2, 3], [0.5, 0.0, 0.0], 2,
        "constant_velocity_reflect_v1", metadata=nested,
    )
    copied = state.copy()
    payload = state.to_payload()
    restored = TargetState.from_payload(payload)
    copied.metadata["route"]["points"][0][0] = 99
    restored.metadata["route"]["points"][0][1] = 88
    payload["metadata"]["route"]["points"][0][2] = 77
    assert state.metadata == nested
    assert TargetState.from_payload({
        "position": [1, 2, 3],
        "velocity": [0, 0, 0],
        "sample_step": 0,
        "motion_mode": "static",
        "state_schema": "moving_target_state_v1",
    }).position.tolist() == [1.0, 2.0, 3.0]
    assert TargetState.from_payload({
        "position_at_detection": [4, 5, 6],
        "velocity_at_detection": [0, 0, 0],
        "sample_step": 0,
        "motion_mode": "static",
        "state_schema": "moving_target_state_v1",
    }).position.tolist() == [4.0, 5.0, 6.0]


def test_target_state_rejects_reserved_metadata_and_invalid_numbers():
    with pytest.raises(ValueError, match="reserved"):
        TargetState([1, 2, 3], [0, 0, 0], 0, "static", metadata={"position": [9, 9, 9]})
    with pytest.raises(ValueError, match="sample_step"):
        TargetState([1, 2, 3], [0, 0, 0], -1, "static")
    with pytest.raises(ValueError, match="finite"):
        TargetState([1, 2, np.inf], [0, 0, 0], 0, "static")


def test_corner_and_obstacle_edge_reflect_all_simultaneous_axes():
    corner = TargetState(
        [19.9, 19.9, 4], [2, 2, 0], 0,
        "constant_velocity_reflect_v1",
    )
    reflected = advance_target_state(corner, 0.1, BOUNDS)
    assert reflected.velocity[:2].tolist() == [-2.0, -2.0]
    obstacle = {"center": [5, 5, 4], "size": [2, 2, 2]}
    edge = TargetState(
        [3.9, 3.9, 4], [2, 2, 0], 0,
        "constant_velocity_reflect_v1",
    )
    reflected = advance_target_state(edge, 0.2, BOUNDS, [obstacle])
    assert reflected.velocity[:2].tolist() == [-2.0, -2.0]


def test_obstacle_surface_outward_continues_and_inward_reflects():
    obstacle = {"center": [5, 5, 4], "size": [2, 2, 2]}
    outward = TargetState(
        [4, 5, 4], [-1, 0, 0], 0, "constant_velocity_reflect_v1"
    )
    moved = advance_target_state(outward, 0.1, BOUNDS, [obstacle])
    assert moved.position[0] < 4 and moved.velocity[0] == -1
    inward = TargetState(
        [4, 5, 4], [1, 0, 0], 0, "constant_velocity_reflect_v1"
    )
    reflected = advance_target_state(inward, 0.1, BOUNDS, [obstacle])
    assert reflected.position[0] < 4 and reflected.velocity[0] == -1


def test_prediction_limits_nonfinite_travel_and_rng_free():
    state = TargetState(
        [1, 1, 1], [0.5, 0, 0], 0,
        "constant_velocity_reflect_v1",
    )
    with pytest.raises(ValueError, match="max_prediction_steps"):
        predict_target_state(state, 6, 0.2, BOUNDS, max_prediction_steps=5)
    calls = {"count": 0}

    def eventually_infinite(_start, _goal):
        calls["count"] += 1
        return 1.0 if calls["count"] < 2 else math.inf

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    result = solve_intercept_point(
        [0, 1, 1], state, 0, eventually_infinite,
        dt=0.2, bounds=BOUNDS, max_iterations=1,
    )
    assert not result["reachable"]
    assert result["travel_time"] is None
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())
