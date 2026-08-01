import torch

from train import CH3_EFFICIENCY_V2, build_ch3_runtime


def _scenario():
    return {
        "scenario_id": "standby_test",
        "scenario_seed": 1,
        "planner_seed": 2,
        "use_obstacles": False,
        "obstacle_layout_id": "none",
        "initial_agent_positions": [
            [2.0, 2.0, 1.0], [8.0, 2.0, 2.0],
            [2.0, 8.0, 3.0], [15.0, 15.0, 4.0],
        ],
        "target_position": [18.0, 18.0, 6.0],
        "initial_executor_wait_point": [10.0, 10.0, 4.0],
        "flow_phase_x": 0.0,
        "flow_phase_y": 0.0,
    }


def _env(method="ch3_pse_rmaddpg"):
    return build_ch3_runtime(
        method, seed=1, max_steps=400, device="cpu", replay_size=32,
        protocol=CH3_EFFICIENCY_V2,
    ).env


def test_v2_reset_and_start_gate_do_not_move_wait_point():
    env = _env()
    env.reset(_scenario())
    expected = torch.tensor([10.0, 10.0, 4.0])
    assert torch.allclose(env._executor_wait_point, expected)
    env.step_count = 79
    env._update_pse_executor_standby_v2()
    assert torch.allclose(env._executor_wait_point, expected)
    assert env.standby_update_attempt_count == 0


def test_v2_gain_threshold_interval_and_shift_limit(monkeypatch):
    env = _env()
    env.reset(_scenario())
    old = env._executor_wait_point.clone()
    candidate = old + torch.tensor([8.0, 0.0, 0.0])
    monkeypatch.setattr(
        env.map_module, "plan_executor_standby",
        lambda *args, **kwargs: candidate.clone(),
    )
    monkeypatch.setattr(
        env.map_module, "expected_response_cost",
        lambda point: (
            10.0 if torch.allclose(point, old)
            else 8.0 if torch.allclose(point, candidate)
            else 9.6
        ),
    )
    env.step_count = 80
    env._update_pse_executor_standby_v2()
    assert torch.allclose(env._executor_wait_point, old)
    assert env.standby_update_reject_count == 1
    assert env.standby_candidate_response_cost == 9.6

    monkeypatch.setattr(
        env.map_module, "expected_response_cost",
        lambda point: 10.0 if torch.allclose(point, old) else 8.0,
    )
    env.step_count = 85
    env._update_pse_executor_standby_v2()
    assert torch.allclose(env._executor_wait_point, old)  # interval gate
    env.step_count = 90
    env._update_pse_executor_standby_v2()
    assert env.standby_update_accept_count == 1
    assert float(torch.norm(env._executor_wait_point - old)) <= 3.0 + 1e-6
    assert env.standby_update_attempt_count == 2
    assert env.standby_update_attempt_count == (
        env.standby_update_accept_count + env.standby_update_reject_count
    )
    assert 0.0 < env.standby_total_target_shift <= 3.0 + 1e-6
    assert env.standby_current_response_cost == 10.0
    assert env.standby_candidate_response_cost == 8.0
    assert abs(env.standby_relative_gain - 0.2) < 1e-6


def test_standby_update_preserves_torch_rng(monkeypatch):
    env = _env()
    env.reset(_scenario())
    old = env._executor_wait_point.clone()
    monkeypatch.setattr(
        env.map_module, "plan_executor_standby",
        lambda *args, **kwargs: old + torch.tensor([1.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        env.map_module, "expected_response_cost",
        lambda point: 10.0 if torch.allclose(point, old) else 8.0,
    )
    env.step_count = 80
    before = torch.get_rng_state().clone()
    env._update_pse_executor_standby_v2()
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_no_standby_never_updates():
    env = _env("ch3_pse_no_standby")
    env.reset(_scenario())
    old = env._executor_wait_point.clone()
    env.step_count = 100
    env._update_pse_executor_standby_v2()
    assert torch.allclose(env._executor_wait_point, old)
    assert env.standby_update_attempt_count == 0
