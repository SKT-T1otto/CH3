import json
from types import SimpleNamespace

import pytest

import training
from ch3_config import build_mission_config
from training import train_and_evaluate
from utils.provenance import json_file_sha256


def _manifest(path, role, identifier, offset):
    scenario = {
        "scenario_id": f"{identifier}-scenario",
        "scenario_seed": 100 + offset,
        "planner_seed": 200 + offset,
        "scenario_profile": "S00_STATIC_CLEAR",
        "scenario_role": role,
        "scenario_split": role,
        "initial_agent_positions": [
            [1 + offset, 1, 1],
            [4 + offset, 1, 1],
            [1 + offset, 4, 1],
            [4 + offset, 4, 1],
        ],
        "initial_executor_wait_point": [10, 10, 4],
        "target_position": [18 - offset, 18, 6],
        "target_initial_position": [18 - offset, 18, 6],
        "target_initial_velocity": [0, 0, 0],
        "target_motion_mode": "static",
        "obstacle_layout_id": "none",
        "obstacles": [],
    }
    manifest = {
        "protocol": "ch3_mission_v1",
        "manifest_id": identifier,
        "scenario_profile": "S00_STATIC_CLEAR",
        "scenario_role": role,
        "scenario_split": role,
        "generator_seed": 70000 + offset,
        "scenario_count": 1,
        "scenarios": [scenario],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_deprecated_alias_is_training_only(tmp_path):
    training_path = _manifest(
        tmp_path / "training.json", "train", "training", 1
    )
    with pytest.raises(ValueError, match="requires evaluation_limit=0"):
        train_and_evaluate(
            "ch3_v3_full_reference",
            "S00_STATIC_CLEAR",
            seed=1,
            episodes=1,
            max_steps=1,
            device="cpu",
            output_dir=tmp_path / "runs",
            scenario_manifest=training_path,
            evaluation_limit=1,
            replay_size=8,
        )
    with pytest.raises(ValueError, match="cannot both be set"):
        train_and_evaluate(
            "ch3_v3_full_reference",
            "S00_STATIC_CLEAR",
            seed=1,
            episodes=1,
            max_steps=1,
            device="cpu",
            output_dir=tmp_path / "runs",
            training_manifest=training_path,
            scenario_manifest=training_path,
            evaluation_limit=0,
            replay_size=8,
        )


def test_completed_noop_cannot_silently_replace_evaluation_identity(
    tmp_path, monkeypatch
):
    training_path = _manifest(
        tmp_path / "training.json", "smoke_train", "training", 1
    )
    evaluation_a = _manifest(
        tmp_path / "evaluation-a.json",
        "smoke_validation",
        "evaluation-a",
        2,
    )
    evaluation_b = _manifest(
        tmp_path / "evaluation-b.json",
        "smoke_validation",
        "evaluation-b",
        3,
    )
    run_dir = (
        tmp_path
        / "runs"
        / "ch3_v3_full_reference"
        / "S00_STATIC_CLEAR"
        / "seed_7"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "resume_state.pt").write_bytes(b"bounded-test-state")
    (run_dir / "episode_metrics.csv").write_text(
        "episode\n1\n", encoding="utf-8"
    )
    summary = {
        "training_manifest_id": "training",
        "training_manifest_sha256": json_file_sha256(training_path),
        "training_scenario_ids": ["training-scenario"],
        "evaluation_manifest_id": "evaluation-a",
        "evaluation_manifest_sha256": json_file_sha256(evaluation_a),
        "evaluation_scenario_ids": ["evaluation-a-scenario"],
        "evaluation_limit": 1,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    config = build_mission_config(
        "ch3_v3_full_reference", "S00_STATIC_CLEAR"
    )
    config.update(max_steps=1, replay_size=8)
    monkeypatch.setattr(
        training,
        "build_runtime",
        lambda *args, **kwargs: SimpleNamespace(config=dict(config)),
    )
    monkeypatch.setattr(
        training,
        "_load_resume",
        lambda *args, **kwargs: {
            "episode": 1,
            "global_step": 1,
            "update_step": 0,
            "sigma": config["initial_sigma"],
            "repository_source_fingerprint": None,
        },
    )
    assert summary["training_manifest_id"] == "training"
    assert summary["evaluation_manifest_id"] == "evaluation-a"
    assert not set(summary["training_scenario_ids"]) & set(
        summary["evaluation_scenario_ids"]
    )
    kwargs = dict(
        seed=7,
        episodes=1,
        max_steps=1,
        device="cpu",
        output_dir=tmp_path / "runs",
        training_manifest=training_path,
        evaluation_limit=1,
        checkpoint_interval=0,
        replay_size=8,
    )
    with pytest.raises(ValueError, match="independent evaluate phase"):
        train_and_evaluate(
            "ch3_v3_full_reference",
            "S00_STATIC_CLEAR",
            evaluation_manifest=evaluation_b,
            resume=True,
            **kwargs,
        )
