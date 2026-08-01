from pathlib import Path

from utils.provenance import (
    algorithm_source_files,
    algorithm_source_fingerprint,
    mission_algorithm_source_files,
    mission_algorithm_source_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_and_mission_identity_scopes_are_isolated():
    base_files = set(algorithm_source_files(PROJECT_ROOT))
    mission_files = set(mission_algorithm_source_files(PROJECT_ROOT))
    assert {
        "base_env.py",
        "env.py",
        "train.py",
        "map/map_module.py",
    } <= base_files
    mission_only = {
        "ch3_config.py",
        "ch3_constants.py",
        "target_motion.py",
        "runtime.py",
        "training.py",
        "metrics.py",
        "map/path_planner.py",
        "tools/build_ch3_scenarios.py",
    }
    assert mission_only <= mission_files
    assert mission_only.isdisjoint(base_files)
    assert algorithm_source_fingerprint(PROJECT_ROOT)
    assert mission_algorithm_source_fingerprint(PROJECT_ROOT)
    assert (
        algorithm_source_fingerprint(PROJECT_ROOT)
        != mission_algorithm_source_fingerprint(PROJECT_ROOT)
    )
