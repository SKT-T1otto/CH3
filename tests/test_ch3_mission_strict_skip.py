import csv
import json
from pathlib import Path

import pytest
import torch

import tools.run_ch3 as runner
from tools.run_ch3 import _strict_skip, resolve_runtime_config
from train import _algorithm_config_hash, _evaluation_config_hash
from utils.provenance import (
    base_algorithm_source_fingerprint,
    file_sha256,
    json_file_sha256,
    mission_algorithm_source_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "ch3_v3_full_reference"
PROFILE = "S00_STATIC_CLEAR"


def _scenario(identifier, seed, pair, offset):
    return {
        "scenario_id": identifier,
        "scenario_seed": seed,
        "planner_seed": seed + 100,
        "target_motion_seed": seed + 200,
        "pair_group_id": pair,
        "scenario_profile": PROFILE,
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


def _write_manifest(path, identifier, role, seed, scenario):
    scenario = dict(scenario, scenario_role=role, scenario_split=role)
    manifest = {
        "protocol": "ch3_mission_v1",
        "manifest_id": identifier,
        "scenario_profile": PROFILE,
        "scenario_role": role,
        "scenario_split": role,
        "generator_seed": seed,
        "scenario_count": 1,
        "scenarios": [scenario],
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def _artifact(root, replay_size):
    root.mkdir(parents=True)
    training_path = root / "training.json"
    evaluation_path = root / "evaluation.json"
    training = _write_manifest(
        training_path,
        "training",
        "train",
        71001,
        _scenario("training-1", 1, "training-pair", 0),
    )
    evaluation = _write_manifest(
        evaluation_path,
        "evaluation",
        "validation",
        72001,
        _scenario("evaluation-1", 2, "evaluation-pair", 0.5),
    )
    training_sha = json_file_sha256(training_path)
    evaluation_sha = json_file_sha256(evaluation_path)
    config = resolve_runtime_config(
        CANDIDATE, PROFILE, 4, replay_size=replay_size
    )
    config["checkpoint_interval"] = 0
    config = json.loads(json.dumps(config))
    base = base_algorithm_source_fingerprint(PROJECT_ROOT)
    mission = mission_algorithm_source_fingerprint(PROJECT_ROOT)
    common = {
        "protocol": "ch3_mission_v1",
        "base_candidate": CANDIDATE,
        "scenario_profile": PROFILE,
        "seed": 1,
        "episodes": 2,
        "max_steps": 4,
        "replay_size": replay_size,
        "base_algorithm_source_fingerprint": base,
        "mission_algorithm_source_fingerprint": mission,
        "algorithm_config_hash": _algorithm_config_hash(config),
        "evaluation_config_hash": _evaluation_config_hash(config),
        "target_motion_mode": "static",
        "obstacle_layout_identity": "none",
        "training_manifest_id": training["manifest_id"],
        "training_manifest_sha256": training_sha,
        "training_scenario_ids": ["training-1"],
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
    }
    run = root / CANDIDATE / PROFILE / "seed_1"
    run.mkdir(parents=True)
    checkpoint = run / "model_final.pt"
    torch.save({"metadata": {**common, "config": config}}, checkpoint)
    torch.save(
        {
            **{
                key: common[key]
                for key in (
                    "protocol", "base_candidate", "scenario_profile", "seed",
                    "max_steps", "replay_size",
                    "base_algorithm_source_fingerprint",
                    "mission_algorithm_source_fingerprint",
                    "algorithm_config_hash", "evaluation_config_hash",
                    "target_motion_mode", "obstacle_layout_identity",
                    "training_manifest_id", "training_manifest_sha256",
                    "training_scenario_ids",
                )
            },
            "episode": 2,
        },
        run / "resume_state.pt",
    )
    with (run / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode"])
        writer.writeheader()
        writer.writerows(({"episode": 1}, {"episode": 2}))
    with (run / "evaluation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id"])
        writer.writeheader()
        writer.writerow({"scenario_id": "evaluation-1"})
    summary = {
        **common,
        "resolved_config": config,
        "evaluation_manifest_id": evaluation["manifest_id"],
        "evaluation_manifest_sha256": evaluation_sha,
        "evaluation_scenario_ids": ["evaluation-1"],
        "evaluated_scenario_ids": ["evaluation-1"],
        "evaluation_count": 1,
        "evaluation_limit": None,
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    (run / "training_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return training_path, evaluation_path


def test_default_500000_and_explicit_32_strict_skip_without_buffer_allocation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner,
        "build_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("strict skip must not build runtime/replay")
        ),
    )
    training, evaluation = _artifact(tmp_path / "formal", 500000)
    assert _strict_skip(
        tmp_path / "formal", CANDIDATE, PROFILE, 1, 2, 4,
        training_manifest_path=training,
        evaluation_manifest_path=evaluation,
        evaluation_limit=None,
    )
    with pytest.raises(ValueError, match="incompatible occupied"):
        _strict_skip(
            tmp_path / "formal", CANDIDATE, PROFILE, 1, 2, 4,
            requested_replay_size=32,
            training_manifest_path=training,
            evaluation_manifest_path=evaluation,
            evaluation_limit=None,
        )

    smoke_training, smoke_evaluation = _artifact(tmp_path / "smoke", 32)
    assert _strict_skip(
        tmp_path / "smoke", CANDIDATE, PROFILE, 1, 2, 4,
        requested_replay_size=32,
        training_manifest_path=smoke_training,
        evaluation_manifest_path=smoke_evaluation,
        evaluation_limit=None,
    )


def test_strict_skip_rejects_pair_seed_and_physical_leakage(tmp_path):
    training, evaluation = _artifact(tmp_path / "leak", 32)
    train_data = json.loads(training.read_text(encoding="utf-8"))
    eval_data = json.loads(evaluation.read_text(encoding="utf-8"))
    eval_data["scenarios"][0]["scenario_seed"] = train_data["scenarios"][0][
        "scenario_seed"
    ]
    evaluation.write_text(json.dumps(eval_data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="leakage"):
        _strict_skip(
            tmp_path / "leak", CANDIDATE, PROFILE, 1, 2, 4,
            requested_replay_size=32,
            training_manifest_path=training,
            evaluation_manifest_path=evaluation,
            evaluation_limit=None,
        )
