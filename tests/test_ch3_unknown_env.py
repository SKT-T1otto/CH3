import torch

from runtime import build_runtime


def _scenario(profile, knowledge, obstacles):
    return {
        "scenario_id": f"test_{profile}",
        "scenario_seed": 101,
        "planner_seed": 202,
        "scenario_profile": profile,
        "scenario_role": "smoke_train",
        "scenario_split": "smoke_train",
        "protocol": "ch3_unknown_map_v1",
        "initial_agent_positions": [
            [2.0, 2.0, 1.0],
            [2.0, 18.0, 2.0],
            [18.0, 2.0, 3.0],
            [18.0, 18.0, 4.0],
        ],
        "initial_executor_wait_point": [15.0, 15.0, 4.0],
        "target_position": [4.0, 10.0, 4.0],
        "target_initial_position": [4.0, 10.0, 4.0],
        "target_initial_velocity": [0.35, 0.20, 0.10],
        "target_motion_mode": "constant_velocity_reflect_v1",
        "target_motion_known": True,
        "target_state_schema": "moving_target_state_v1",
        "use_obstacles": bool(obstacles),
        "obstacle_layout_id": "custom_aabb_v1" if obstacles else "none",
        "obstacle_knowledge_mode": knowledge,
        "obstacles": obstacles,
        "flow_phase_x": 0.0,
        "flow_phase_y": 0.0,
    }


def test_unknown_environment_keeps_truth_private_from_online_planner():
    profile = "M10_MOVING_UNKNOWN_SINGLE"
    obstacle = [
        {"center": [10.0, 10.0, 4.0], "size": [2.0, 4.0, 4.0]}
    ]
    runtime = build_runtime(
        "ch3_v3_full_reference",
        profile,
        seed=1,
        max_steps=10,
        device="cpu",
        replay_size=32,
    )
    env = runtime.env
    observations = env.reset(
        _scenario(profile, "online_unknown", obstacle)
    )
    assert len(env.ground_truth_obstacles) == 1
    assert env.map_module.obstacles == []
    assert env.target_state.motion_mode == "constant_velocity_reflect_v1"
    assert all(torch.as_tensor(item).numel() == 28 for item in observations)

    for _ in range(3):
        observations, rewards, _ = env.step(torch.zeros((4, 3)))
        assert all(
            torch.isfinite(torch.as_tensor(item)).all()
            for item in observations
        )
        assert torch.isfinite(torch.as_tensor(rewards)).all()


def test_oracle_profile_preserves_known_map_baseline():
    profile = "M90_MOVING_KNOWN_ORACLE"
    obstacles = [
        {"center": [8.0, 10.0, 3.0], "size": [2.0, 3.0, 3.0]},
        {"center": [13.0, 10.0, 5.0], "size": [2.5, 3.0, 3.0]},
    ]
    runtime = build_runtime(
        "ch3_v3_full_reference",
        profile,
        seed=1,
        max_steps=10,
        device="cpu",
        replay_size=32,
    )
    env = runtime.env
    env.reset(_scenario(profile, "oracle", obstacles))
    assert len(env.map_module.obstacles) == len(obstacles)
    metrics = env.get_unknown_map_metrics()
    assert metrics["map_known_fraction"] == 1.0
    assert metrics["obstacle_map_iou"] == 1.0
