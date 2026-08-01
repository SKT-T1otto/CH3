import shutil
import inspect
from pathlib import Path

import torch

from base_env import UAVEnv as BaseUAVEnv
from ch3_config import build_mission_config, build_unknown_map_config
from ch3_constants import (
    CH3_MISSION_V1,
    CH3_UNKNOWN_MAP_V1,
    MISSION_SCENARIO_PROFILES,
    UNKNOWN_MAP_PROFILES,
)
from env import UAVEnv as MissionUAVEnv
from map.path_planner import (
    ObstacleAwareTaskMapPlanner,
    OnlineUnknownMapTaskPlanner,
)
from metrics import augment_episode_metrics
from registry.ch3_efficiency_v3_registry import resolve_ch3_efficiency_v3_config
from runtime import build_runtime
from train import (
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    build_ch3_runtime,
    build_ch3_runtime_from_resolved_config,
)
from training import (
    run_episode,
    train_and_evaluate,
)
from utils.provenance import (
    base_algorithm_source_files,
    base_algorithm_source_fingerprint,
    mission_algorithm_source_files,
    mission_algorithm_source_fingerprint,
    capture_mission_provenance_snapshot,
    capture_unknown_map_provenance_snapshot,
    unknown_map_algorithm_source_files,
    unknown_map_algorithm_source_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_protocols_route_to_exact_environment_classes():
    legacy_files = (
        "unknown_env.py",
        "unknown_runtime.py",
        "unknown_training.py",
        "unknown_metrics.py",
        "ch3_unknown_config.py",
        "ch3_unknown_constants.py",
        "map/unknown_map_planner.py",
        "utils/unknown_provenance.py",
    )
    for relative in legacy_files:
        assert not (PROJECT_ROOT / relative).exists()
    for symbol in (
        MissionUAVEnv,
        build_runtime,
        run_episode,
        train_and_evaluate,
        augment_episode_metrics,
        build_mission_config,
        build_unknown_map_config,
        ObstacleAwareTaskMapPlanner,
        OnlineUnknownMapTaskPlanner,
        capture_mission_provenance_snapshot,
        capture_unknown_map_provenance_snapshot,
    ):
        assert callable(symbol)
    assert CH3_MISSION_V1 == "ch3_mission_v1"
    assert CH3_UNKNOWN_MAP_V1 == "ch3_unknown_map_v1"
    assert len(MISSION_SCENARIO_PROFILES) == 4
    assert len(UNKNOWN_MAP_PROFILES) == 4

    for protocol in (CH3_PILOT_V1, CH3_EFFICIENCY_V2):
        runtime = build_ch3_runtime(
            "ch3_pse_rmaddpg",
            seed=4,
            max_steps=2,
            device="cpu",
            replay_size=8,
            protocol=protocol,
        )
        assert type(runtime.env) is BaseUAVEnv
    v3 = build_ch3_runtime_from_resolved_config(
        "ch3_v3_full_reference",
        resolve_ch3_efficiency_v3_config("ch3_v3_full_reference"),
        seed=4,
        max_steps=2,
        device="cpu",
        replay_size=8,
    )
    assert type(v3.env) is BaseUAVEnv
    mission = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=4,
        max_steps=2,
        device="cpu",
        replay_size=8,
    )
    assert type(mission.env) is MissionUAVEnv


def test_v1_routing_matches_direct_base_environment_zero_actions():
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg",
        seed=55,
        max_steps=3,
        device="cpu",
        replay_size=8,
        protocol=CH3_PILOT_V1,
    )
    kwargs = {
        key: value
        for key, value in runtime.config.items()
        if key in inspect.signature(BaseUAVEnv.__init__).parameters
    }
    kwargs.update(max_steps=3, device=torch.device("cpu"), return_numpy=False)
    direct = BaseUAVEnv(**kwargs)
    scenario = {
        "scenario_id": "routing_static_clear",
        "scenario_seed": 55,
        "planner_seed": 56,
        "initial_agent_positions": [
            [1, 1, 1], [4, 1, 1], [1, 4, 1], [4, 4, 1],
        ],
        "target_position": [18, 18, 6],
        "initial_executor_wait_point": [10, 10, 4],
    }
    runtime.env.reset(scenario)
    direct.reset(scenario)
    for _ in range(3):
        left = runtime.env.step(torch.zeros((4, 3)))
        right = direct.step(torch.zeros((4, 3)))
        assert torch.equal(runtime.env._agent_pos, direct._agent_pos)
        assert torch.equal(torch.as_tensor(left[1]), torch.as_tensor(right[1]))
        assert left[2] == right[2]


def test_base_and_mission_source_changes_have_expected_dependency(tmp_path):
    files = set(base_algorithm_source_files(PROJECT_ROOT))
    files.update(mission_algorithm_source_files(PROJECT_ROOT))
    files.update(unknown_map_algorithm_source_files(PROJECT_ROOT))
    for relative in files:
        source = PROJECT_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def fingerprints():
        return (
            base_algorithm_source_fingerprint(tmp_path),
            mission_algorithm_source_fingerprint(tmp_path),
            unknown_map_algorithm_source_fingerprint(tmp_path),
        )

    before = fingerprints()
    with (tmp_path / "env.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# shared environment identity test\n")
    after_env = fingerprints()
    assert all(left != right for left, right in zip(before, after_env))

    before_map = after_env
    with (tmp_path / "map/map_module.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# shared map identity test\n")
    after_map = fingerprints()
    assert all(left != right for left, right in zip(before_map, after_map))

    before_path = after_map
    with (tmp_path / "map/path_planner.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# mission path-planner identity test\n")
    after_path = fingerprints()
    assert after_path[0] == before_path[0]
    assert after_path[1] != before_path[1]
    assert after_path[2] != before_path[2]

    before_generator = after_path
    with (
        tmp_path / "tools/build_ch3_scenarios.py"
    ).open("a", encoding="utf-8") as handle:
        handle.write("\n# unified scenario generator identity test\n")
    after_generator = fingerprints()
    assert after_generator[0] == before_generator[0]
    assert after_generator[1] != before_generator[1]
    assert after_generator[2] != before_generator[2]
