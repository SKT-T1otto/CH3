from ch3_config import build_unknown_map_config
from ch3_constants import UNKNOWN_MAP_PROFILES


def test_all_unknown_map_profiles_use_moving_targets():
    for profile in UNKNOWN_MAP_PROFILES:
        config = build_unknown_map_config(
            "ch3_v3_full_reference", profile
        )
        assert config["artifact_protocol"] == "ch3_unknown_map_v1"
        assert config["target_motion_known"] is True
        assert (
            config["target_motion_mode"]
            == "constant_velocity_reflect_v1"
        )


def test_unknown_and_oracle_profiles_have_explicit_knowledge_identity():
    unknown = build_unknown_map_config(
        "ch3_v3_full_reference",
        "M20_MOVING_UNKNOWN_MULTI",
    )
    oracle = build_unknown_map_config(
        "ch3_v3_full_reference",
        "M90_MOVING_KNOWN_ORACLE",
    )
    assert unknown["obstacle_knowledge_mode"] == "online_unknown"
    assert unknown["planner_mode"] == "online_astar_v1"
    assert oracle["obstacle_knowledge_mode"] == "oracle"
    assert oracle["planner_mode"] == "oracle_astar_v1"
