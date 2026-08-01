import numpy as np

from comm.basic_communication import FixedReliableHandoff
from target_motion import TargetState, solve_intercept_point


def test_target_state_handoff_is_one_step_one_shot_and_deep_copied():
    service = FixedReliableHandoff(delay_steps=1)
    state = TargetState([1, 2, 3], [0.5, 0, 0], 7, "constant_velocity_reflect_v1")
    assert service.publish_target(found_step=7, finder_idx=1, target=state)
    assert not service.publish_target(found_step=7, finder_idx=2, target=state)
    state.position[0] = 99
    state.velocity[0] = 99
    assert service.advance(entering_step=7) is None
    event = service.advance(entering_step=8)
    delivered = event["target"]
    assert event["delivery_step"] - event["found_step"] == 1
    assert delivered.position[0] == 1
    assert delivered.velocity[0] == 0.5
    delivered.position[0] = -5
    assert service.state_dict()["target"].position[0] == 1
    assert service.advance(entering_step=9) is None


def test_intercept_prediction_depends_only_on_delivered_payload():
    delivered = TargetState([1, 1, 1], [0.5, 0, 0], 0, "constant_velocity_reflect_v1")
    travel = lambda start, goal: float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
    one = solve_intercept_point(
        [0, 1, 1], delivered, 2, travel, dt=0.2,
        bounds=[20, 20, 8], max_iterations=4,
    )
    unrelated_true_state = TargetState([19, 19, 7], [-1, 0, 0], 2, "constant_velocity_reflect_v1")
    two = solve_intercept_point(
        [0, 1, 1], delivered, 2, travel, dt=0.2,
        bounds=[20, 20, 8], max_iterations=4,
    )
    unrelated_true_state.position[:] = 0
    assert one["reachable"] and two["reachable"]
    assert np.array_equal(one["position"], two["position"])
