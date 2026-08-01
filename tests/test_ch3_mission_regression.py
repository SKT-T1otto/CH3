from ch3_config import build_mission_config, scenario_profile_diff
from ch3_constants import CH3_MISSION_V1
from env import UAVEnv


def test_primary_environment_is_mission_environment():
    env = UAVEnv(max_steps=2, device="cpu", return_numpy=False)
    assert env.protocol == CH3_MISSION_V1
    assert hasattr(env, "target_state")
    assert hasattr(env.map_module, "grid_astar_path")


def test_profile_pairs_change_only_motion_or_obstacle_fields():
    static_clear = build_mission_config(
        "ch3_v3_full_reference", "S00_STATIC_CLEAR"
    )
    moving_clear = build_mission_config(
        "ch3_v3_full_reference", "S10_MOVING_CLEAR"
    )
    static_obstacle = build_mission_config(
        "ch3_v3_full_reference", "S01_STATIC_OBSTACLE"
    )
    assert static_clear["protocol"] == CH3_MISSION_V1
    assert scenario_profile_diff(static_clear, moving_clear) == {
        "scenario_profile",
        "target_belief_diffusion_rate",
        "target_belief_transition_mode",
        "target_motion_mode",
    }
    assert scenario_profile_diff(static_clear, static_obstacle) == {
        "scenario_profile",
        "use_obstacles",
    }
