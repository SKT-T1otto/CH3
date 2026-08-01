from pathlib import Path

import pytest

from ch3_config import build_ch3_config
from map.path_planner import (
    ObstacleAwareTaskMapPlanner,
    OnlineUnknownMapTaskPlanner,
)
from runtime import build_runtime
from tests.ch3_profile_cases import PROFILE_CASES
from tools.build_ch3_scenarios import build_scenario_manifests
from training import _checkpoint_metadata, _protocol_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "case", PROFILE_CASES, ids=lambda case: case["profile"]
)
def test_unified_profile_config_and_runtime_contract(case):
    profile = case["profile"]
    config = build_ch3_config("ch3_v3_full_reference", profile)
    runtime = build_runtime(
        "ch3_v3_full_reference",
        profile,
        seed=41,
        max_steps=2,
        device="cpu",
        replay_size=8,
        resolved_config=config,
    )
    env = runtime.env
    assert config.get("artifact_protocol", "ch3_mission_v1") == (
        case["artifact_protocol"]
    )
    assert env.artifact_protocol == case["artifact_protocol"]
    assert (env.target_motion_mode != "static") is case["moving"]
    assert bool(config["use_obstacles"]) is case["use_obstacles"]
    assert env.obstacle_knowledge_mode == case["knowledge_mode"]
    if profile.startswith("M"):
        assert env.planner_mode == case["planner_mode"]
        if case["knowledge_mode"] == "online_unknown":
            assert isinstance(env.map_module, OnlineUnknownMapTaskPlanner)
        else:
            assert isinstance(env.map_module, ObstacleAwareTaskMapPlanner)
    else:
        assert isinstance(env.map_module, ObstacleAwareTaskMapPlanner)
    assert [
        env.observation_space[f"agent_{index}"].shape
        for index in range(4)
    ] == [(28,)] * 4
    assert [
        env.action_space[f"agent_{index}"].shape for index in range(4)
    ] == [(3,)] * 4
    family = "unknown" if profile.startswith("M") else "mission"
    manifest = build_scenario_manifests(
        1, 88101 if family == "mission" else 98101, "smoke_train",
        profiles=family,
    )[profile]
    manifest = dict(manifest, manifest_sha256="fixture-manifest-sha256")
    spec = _protocol_spec(profile)
    snapshot = spec.provenance_snapshot(PROJECT_ROOT)
    metadata = _checkpoint_metadata(
        runtime,
        base_candidate="ch3_v3_full_reference",
        scenario_profile=profile,
        seed=41,
        episodes=1,
        max_steps=2,
        manifest=manifest,
        snapshot=snapshot,
        episode=1,
        obstacle_layout_identity=str(
            manifest["scenarios"][0].get("obstacle_layout_id", "none")
        ),
        spec=spec,
    )
    assert manifest["protocol"] == case["artifact_protocol"]
    assert manifest["scenario_role"] == "smoke_train"
    assert manifest["scenario_split"] == "smoke_train"
    assert metadata["protocol"] == case["artifact_protocol"]
    assert metadata["scenario_profile"] == profile
    assert metadata["observation_dims"] == [28] * 4
    assert metadata["action_dims"] == [3] * 4
    assert metadata["checkpoint_episode"] == 1
    assert metadata["algorithm_config_hash"]
    assert metadata["evaluation_config_hash"]
