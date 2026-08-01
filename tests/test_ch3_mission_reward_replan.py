import pytest
import torch

from runtime import build_runtime
from tools.build_ch3_scenarios import build_scenario_manifests


def _stationary_dynamics(env):
    def apply(_actions):
        env._collision_flags.zero_()
        env._agent_vel.zero_()
        env._agent_acc.zero_()
        env.step_count += 1

    env._apply_agent_dynamics = apply


def _environment(seed=91):
    scenario = build_scenario_manifests(1)["S00_STATIC_CLEAR"]["scenarios"][0]
    env = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=seed,
        max_steps=12,
        device="cpu",
        replay_size=8,
    ).env
    env.reset(scenario)
    return env, scenario


def test_replan_without_motion_has_zero_progress_reward():
    env, _ = _environment()
    _stationary_dynamics(env)
    env.path_replan_interval = 1
    env._path_last_replan_steps[:] = [-100] * 4
    before = env.path_replan_count
    env.step(torch.zeros((4, 3)))
    assert env.path_replan_count > before
    assert torch.allclose(
        env.last_reward_components["reward_progress"],
        torch.zeros(4),
        atol=1e-8,
    )


def test_same_motion_has_same_progress_for_different_replan_intervals():
    fast, scenario = _environment(seed=92)
    slow = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=92,
        max_steps=12,
        device="cpu",
        replay_size=8,
    ).env
    slow.reset(scenario)
    _stationary_dynamics(fast)
    _stationary_dynamics(slow)
    fast.path_replan_interval = 1
    slow.path_replan_interval = 1000
    fast._path_last_replan_steps[:] = [-100] * 4
    fast.step(torch.zeros((4, 3)))
    slow.step(torch.zeros((4, 3)))
    assert torch.equal(
        fast.last_reward_components["reward_progress"],
        slow.last_reward_components["reward_progress"],
    )


def test_finished_searchers_do_not_replan_and_wait_hold_progress_grows():
    env, _ = _environment(seed=93)
    _stationary_dynamics(env)
    env._publish_detection(0)
    searcher_steps = list(env._path_last_replan_steps[:3])
    env.step(torch.zeros((4, 3)))
    assert env._path_last_replan_steps[:3] == searcher_steps

    env, _ = _environment(seed=94)
    _stationary_dynamics(env)
    env._agent_pos[env.executor_idx].copy_(env._executor_wait_point)
    env._agent_vel[env.executor_idx].zero_()
    env._update_nav_targets(force=True)
    observation, _, _ = env.step(torch.zeros((4, 3)))
    assert bool(env.current_target_arrived[env.executor_idx].item())
    assert int(env.hold_counters[env.executor_idx].item()) == 1
    hold_field = next(
        field for field in env.get_observation_layout()
        if field["name"] == "hold_progress"
    )
    observed = float(observation[env.executor_idx][hold_field["start"]].item())
    assert observed == pytest.approx(1.0 / env.executor_hold_steps)
