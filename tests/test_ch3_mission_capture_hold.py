import numpy as np

from runtime import build_runtime


def _capture_env():
    env = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=101,
        max_steps=20,
        device="cpu",
        replay_size=8,
    ).env
    env.reset()
    env.task_found = True
    env.executor_target_assigned = True
    return env


def _set_endpoints(env, executor_start, executor_end, target_start, target_end):
    starts = env._agent_pos.detach().cpu().numpy().copy()
    starts[env.executor_idx] = np.asarray(executor_start)
    env._agent_pos[env.executor_idx].copy_(env._vec(executor_end))
    env.target_state.position = np.asarray(target_end, dtype=np.float64)
    env._update_capture(starts, np.asarray(target_start, dtype=np.float64))


def test_swept_crossing_records_contact_but_not_hold():
    env = _capture_env()
    _set_endpoints(
        env,
        [0, 5, 4],
        [10, 5, 4],
        [5, 5, 4],
        [5, 5, 4],
    )
    assert env.capture_contact_step_count == 1
    assert env.capture_full_hold_step_count == 0
    assert env._capture_hold_counter == 0
    assert not env.mission_complete


def test_five_full_hold_steps_succeed_and_interruption_resets():
    env = _capture_env()
    for _ in range(2):
        _set_endpoints(env, [5, 5, 4], [5, 5, 4], [5, 5, 4], [5, 5, 4])
    assert env._capture_hold_counter == 2
    _set_endpoints(env, [5, 5, 4], [7, 5, 4], [5, 5, 4], [5, 5, 4])
    assert env._capture_hold_counter == 0
    for _ in range(5):
        _set_endpoints(env, [5, 5, 4], [5, 5, 4], [5, 5, 4], [5, 5, 4])
    assert env.mission_complete
    assert env.capture_hold_counter_max == 5
    assert np.array_equal(env.target_position_at_capture, [5, 5, 4])
