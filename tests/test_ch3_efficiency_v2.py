import torch

from tools.validate_ch3_config import validate_ch3_configs

from train import (
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    _algorithm_config_hash,
    _evaluation_config_hash,
    _run_config_hash,
    build_ch3_runtime,
    get_ch3_method_config,
)


def _reward_at_find_step(step):
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg", seed=1, max_steps=400, device="cpu",
        replay_size=32, protocol=CH3_EFFICIENCY_V2,
    )
    env = runtime.env
    env.reset()
    env.task_found = True
    env.finder_idx = 2
    env.found_step = int(step)
    env.step_count = int(step)
    env._found_event = True
    env.executor_target_assigned = False
    previous = env._compute_nav_distances().clone()
    reward = env._calculate_mission_rewards(previous)
    return env, reward


def test_v1_v2_algorithm_hashes_and_parameters_are_distinct():
    v1 = get_ch3_method_config("ch3_pse_rmaddpg", protocol=CH3_PILOT_V1)
    v2 = get_ch3_method_config("ch3_pse_rmaddpg", protocol=CH3_EFFICIENCY_V2)
    assert _algorithm_config_hash(v1) != _algorithm_config_hash(v2)
    assert build_ch3_runtime(
        "ch3_pse_rmaddpg", seed=1, max_steps=400, device="cpu",
        replay_size=32, protocol=CH3_PILOT_V1,
    ).env.reward_scale == 100.0
    assert v2["reward_scale"] == 400.0
    assert v2["residual_scale_search"] == 0.20
    assert v2["residual_scale_executor"] == 0.15
    assert v2["residual_action_reg"] == 0.05
    assert v2["pse_belief_weight"] == 0.60


def test_hash_scopes_separate_algorithm_evaluation_and_runtime_fields():
    base = get_ch3_method_config("ch3_pse_rmaddpg", protocol=CH3_EFFICIENCY_V2)
    base["max_steps"] = 40
    cadence = dict(base, checkpoint_interval=7, training_episodes=4)
    assert _algorithm_config_hash(base) == _algorithm_config_hash(cadence)
    assert _evaluation_config_hash(base) == _evaluation_config_hash(cadence)
    assert _run_config_hash(base, {"output_directory": "left", "evaluation_limit": 2}) != (
        _run_config_hash(base, {"output_directory": "right", "evaluation_limit": 1})
    )

    replay = dict(base, replay_size=32)
    horizon = dict(base, max_steps=41)
    reward = dict(base, reward_scale=401.0)
    assert _algorithm_config_hash(replay) != _algorithm_config_hash(base)
    assert _evaluation_config_hash(replay) == _evaluation_config_hash(base)
    assert _algorithm_config_hash(horizon) != _algorithm_config_hash(base)
    assert _evaluation_config_hash(horizon) != _evaluation_config_hash(base)
    assert _algorithm_config_hash(reward) != _algorithm_config_hash(base)
    assert _evaluation_config_hash(reward) != _evaluation_config_hash(base)


def test_early_discovery_has_meaningfully_higher_reward_and_components_sum():
    early_env, early_reward = _reward_at_find_step(20)
    late_env, late_reward = _reward_at_find_step(380)
    assert float(early_reward[2]) > float(late_reward[2]) + 0.03
    component_sum = torch.stack(
        tuple(early_env.last_reward_components.values()), dim=0
    ).sum(dim=0)
    assert torch.allclose(component_sum, early_env.last_raw_reward, atol=1e-6)
    assert float(early_env.last_reward_components["reward_find_event"][2]) > 0
    assert float(early_env.last_reward_components["reward_early_find"][2]) > 0


def test_inactive_searchers_receive_no_post_discovery_penalties():
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg", seed=1, max_steps=400, device="cpu",
        replay_size=32, protocol=CH3_EFFICIENCY_V2,
    )
    env = runtime.env
    env.reset()
    env.task_found = True
    env._found_event = False
    env.executor_target_assigned = False
    previous = env._compute_nav_distances().clone()
    reward = env._calculate_mission_rewards(previous)
    assert torch.count_nonzero(reward[:3]) == 0
    for name in (
        "reward_time_penalty", "reward_energy_penalty",
        "reward_smoothness_penalty", "reward_collision_penalty",
        "reward_separation_penalty",
    ):
        assert torch.count_nonzero(env.last_reward_components[name][:3]) == 0


def test_v2_discovery_and_completion_event_rewards_are_one_shot():
    env, _ = _reward_at_find_step(20)
    assert float(env.last_reward_components["reward_find_event"].sum()) > 0.0
    env._found_event = False
    previous = env._compute_nav_distances().clone()
    env._calculate_mission_rewards(previous)
    assert float(env.last_reward_components["reward_find_event"].sum()) == 0.0
    assert float(env.last_reward_components["reward_early_find"].sum()) == 0.0

    env.executor_target_assigned = True
    env.mission_complete = True
    env.success_step = 20
    env._mission_complete_event = True
    env._calculate_mission_rewards(previous)
    assert float(env.last_reward_components["reward_completion_event"][3]) == 500.0
    assert float(env.last_reward_components["reward_early_completion"][3]) > 0.0
    env._mission_complete_event = False
    env._calculate_mission_rewards(previous)
    assert float(env.last_reward_components["reward_completion_event"][3]) == 0.0
    assert float(env.last_reward_components["reward_early_completion"][3]) == 0.0


def test_v2_learning_and_controller_residual_scales_are_fair():
    for method in (
        "ch3_pheromone_rmaddpg", "ch3_pse_rmaddpg", "ch3_pse_no_belief",
        "ch3_pse_no_exec_cost", "ch3_pse_no_standby",
    ):
        cfg = get_ch3_method_config(method, protocol=CH3_EFFICIENCY_V2)
        assert cfg["residual_scale_search"] == 0.20
        assert cfg["residual_scale_executor"] == 0.15
        assert cfg["residual_action_reg"] == 0.05
    for method in ("ch3_pheromone_prior", "ch3_pse_no_residual"):
        cfg = get_ch3_method_config(method, protocol=CH3_EFFICIENCY_V2)
        assert cfg["residual_scale_search"] == 0.0
        assert cfg["residual_scale_executor"] == 0.0


def _post_discovery_reward(protocol):
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg",
        seed=7,
        max_steps=400,
        device="cpu",
        replay_size=32,
        protocol=protocol,
    )
    env = runtime.env
    env.reset()
    env.task_found = True
    env._found_event = False
    env.executor_target_assigned = False
    env._agent_acc.zero_()
    env._prev_acc.zero_()
    previous = env._compute_nav_distances().clone()
    reward = env._calculate_mission_rewards(previous)
    return env, reward


def test_v1_reward_behavior_is_preserved_while_v2_freezes_inactive_searchers():
    v1_env, v1_reward = _post_discovery_reward(CH3_PILOT_V1)
    v2_env, v2_reward = _post_discovery_reward(CH3_EFFICIENCY_V2)

    assert torch.all(v1_reward[:3] < 0.0)
    assert torch.count_nonzero(v1_env.last_reward_components["reward_time_penalty"][:3]) == 3
    assert torch.count_nonzero(v2_reward[:3]) == 0
    assert torch.count_nonzero(v2_env.last_reward_components["reward_time_penalty"][:3]) == 0


def test_validator_covers_both_protocols_and_v2_parameters():
    errors, summary = validate_ch3_configs()
    assert errors == []
    assert summary["protocols"] == [CH3_PILOT_V1, CH3_EFFICIENCY_V2]
    assert set(summary["actor_types"]) == {CH3_PILOT_V1, CH3_EFFICIENCY_V2}
    assert set(summary["residual_scales"]) == {CH3_PILOT_V1, CH3_EFFICIENCY_V2}
    assert set(summary["algorithm_config_hashes"]) == {
        CH3_PILOT_V1, CH3_EFFICIENCY_V2
    }
    assert (
        summary["algorithm_config_hashes"][CH3_PILOT_V1]["ch3_pse_rmaddpg"]
        != summary["algorithm_config_hashes"][CH3_EFFICIENCY_V2]["ch3_pse_rmaddpg"]
    )
