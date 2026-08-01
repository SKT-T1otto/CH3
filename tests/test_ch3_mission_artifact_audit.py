import csv
import json
from pathlib import Path

import torch

from ch3_config import build_mission_config
from tools.audit_ch3_provenance import audit_runs, main as audit_main
from train import _algorithm_config_hash, _evaluation_config_hash
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    base_algorithm_source_fingerprint,
    file_sha256,
    json_file_sha256,
    mission_algorithm_source_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "ch3_v3_full_reference"
PROFILE = "S00_STATIC_CLEAR"


def _scenario(identifier, seed, pair, offset, role):
    return {
        "scenario_id": identifier,
        "scenario_seed": seed,
        "planner_seed": seed + 100,
        "target_motion_seed": seed + 200,
        "pair_group_id": pair,
        "scenario_profile": PROFILE,
        "scenario_role": role,
        "scenario_split": role,
        "protocol": "ch3_mission_v1",
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


def _write_manifest(path, identifier, role, generator_seed, scenarios):
    manifest = {
        "protocol": "ch3_mission_v1",
        "manifest_id": identifier,
        "scenario_profile": PROFILE,
        "scenario_role": role,
        "scenario_split": role,
        "generator_seed": generator_seed,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def _write_evaluation_csv(path, ids):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id"])
        writer.writeheader()
        for identifier in ids:
            writer.writerow({"scenario_id": identifier})


def _write_summary(run, summary):
    (run / "training_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )


def _resave_checkpoint(run, metadata, summary):
    checkpoint = run / "model_final.pt"
    torch.save({"metadata": metadata}, checkpoint)
    summary["checkpoint_sha256"] = file_sha256(checkpoint)
    _write_summary(run, summary)


def _artifact(root):
    run = root / CANDIDATE / PROFILE / "seed_1"
    run.mkdir(parents=True)
    base = base_algorithm_source_fingerprint(PROJECT_ROOT)
    mission = mission_algorithm_source_fingerprint(PROJECT_ROOT)

    training_manifest_path = run / "training.json"
    evaluation_manifest_path = run / "evaluation.json"
    training_manifest = _write_manifest(
        training_manifest_path,
        "training-manifest",
        "smoke_train",
        73001,
        [_scenario("training-1", 1, "training-pair", 0.0, "smoke_train")],
    )
    evaluation_manifest = _write_manifest(
        evaluation_manifest_path,
        "evaluation-manifest",
        "smoke_validation",
        74001,
        [
            _scenario(
                "evaluation-1", 2, "evaluation-pair-1", 0.5,
                "smoke_validation",
            ),
            _scenario(
                "evaluation-2", 3, "evaluation-pair-2", 1.0,
                "smoke_validation",
            ),
        ],
    )

    config = build_mission_config(CANDIDATE, PROFILE)
    config.update(max_steps=4, replay_size=8, checkpoint_interval=0)
    config = json.loads(json.dumps(config))
    common = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol": "ch3_mission_v1",
        "base_candidate": CANDIDATE,
        "scenario_profile": PROFILE,
        "seed": 1,
        "episodes": 2,
        "max_steps": 4,
        "replay_size": 8,
        "algorithm_config_hash": _algorithm_config_hash(config),
        "evaluation_config_hash": _evaluation_config_hash(config),
        "base_algorithm_source_fingerprint": base,
        "mission_algorithm_source_fingerprint": mission,
        "training_manifest_id": training_manifest["manifest_id"],
        "training_manifest_sha256": json_file_sha256(training_manifest_path),
        "training_scenario_ids": ["training-1"],
        "target_motion_mode": "static",
        "obstacle_layout_identity": "none",
    }
    metadata = {
        **common,
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
        "config": config,
    }
    checkpoint = run / "model_final.pt"
    torch.save({"metadata": metadata}, checkpoint)
    resume = {**common, "episode": 2}
    torch.save(resume, run / "resume_state.pt")

    with (run / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode"])
        writer.writeheader()
        writer.writerows([{"episode": 1}, {"episode": 2}])
    _write_evaluation_csv(run / "evaluation_metrics.csv", ["evaluation-1"])

    summary = {
        **common,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "resume_state_path": str(run / "resume_state.pt"),
        "training_manifest": str(training_manifest_path),
        "evaluation_manifest": str(evaluation_manifest_path),
        "evaluation_manifest_id": evaluation_manifest["manifest_id"],
        "evaluation_manifest_sha256": json_file_sha256(
            evaluation_manifest_path
        ),
        "evaluation_scenario_ids": ["evaluation-1", "evaluation-2"],
        "evaluated_scenario_ids": ["evaluation-1"],
        "evaluation_limit": 1,
        "evaluation_count": 1,
        "resolved_config": config,
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
    }
    _write_summary(run, summary)
    return run, summary, metadata


def _status(root):
    report = audit_runs(root)
    assert report["run_count"] == 1
    return report["runs"][0]["status"]


def test_audit_verifies_complete_artifact_and_detects_checkpoint_tamper(tmp_path):
    run, _, _ = _artifact(tmp_path)
    assert _status(tmp_path) == "verified"
    with (run / "model_final.pt").open("ab") as handle:
        handle.write(b"tamper")
    assert _status(tmp_path) == "checkpoint_hash_mismatch"


def test_audit_detects_base_fingerprint_mismatch(tmp_path):
    run, summary, metadata = _artifact(tmp_path)
    summary["base_algorithm_source_fingerprint"] = "wrong-base"
    metadata["base_algorithm_source_fingerprint"] = "wrong-base"
    _resave_checkpoint(run, metadata, summary)
    assert _status(tmp_path) == "base_algorithm_mismatch"


def test_audit_detects_resume_replay_mismatch(tmp_path):
    run, _, _ = _artifact(tmp_path)
    resume_path = run / "resume_state.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume["replay_size"] = 32
    torch.save(resume, resume_path)
    assert _status(tmp_path) == "replay_config_mismatch"


def test_audit_detects_illegal_evaluated_id_selection(tmp_path):
    run, summary, _ = _artifact(tmp_path)
    summary["evaluated_scenario_ids"] = ["evaluation-2"]
    summary["evaluation_count"] = 1
    _write_summary(run, summary)
    _write_evaluation_csv(run / "evaluation_metrics.csv", ["evaluation-2"])
    assert _status(tmp_path) == "evaluation_manifest_mismatch"


def test_audit_detects_evaluation_csv_missing_row_and_order(tmp_path):
    run, summary, _ = _artifact(tmp_path / "missing")
    summary["evaluation_limit"] = 2
    summary["evaluated_scenario_ids"] = ["evaluation-1", "evaluation-2"]
    summary["evaluation_count"] = 2
    _write_summary(run, summary)
    _write_evaluation_csv(run / "evaluation_metrics.csv", ["evaluation-1"])
    assert _status(tmp_path / "missing") == "evaluation_csv_mismatch"

    run, summary, _ = _artifact(tmp_path / "order")
    summary["evaluation_limit"] = 2
    summary["evaluated_scenario_ids"] = ["evaluation-1", "evaluation-2"]
    summary["evaluation_count"] = 2
    _write_summary(run, summary)
    _write_evaluation_csv(
        run / "evaluation_metrics.csv", ["evaluation-2", "evaluation-1"]
    )
    assert _status(tmp_path / "order") == "evaluation_csv_mismatch"


def test_audit_detects_summary_dimension_tamper(tmp_path):
    run, summary, _ = _artifact(tmp_path)
    summary["observation_dims"] = [27] * 4
    _write_summary(run, summary)
    assert _status(tmp_path) == "config_mismatch"


def test_audit_main_returns_nonzero_when_any_artifact_fails(tmp_path):
    run, summary, _ = _artifact(tmp_path)
    summary["action_dims"] = [2] * 4
    _write_summary(run, summary)
    output = tmp_path / "audit.json"
    assert audit_main([
        "--runs-root", str(tmp_path), "--output", str(output)
    ]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["all_verified"] is False


def test_audit_detects_manifest_mismatch_and_legacy_artifact(tmp_path):
    run, summary, metadata = _artifact(tmp_path / "mission")
    metadata["training_manifest_sha256"] = "different"
    _resave_checkpoint(run, metadata, summary)
    assert _status(tmp_path / "mission") == "manifest_mismatch"

    legacy = tmp_path / "legacy" / "candidate" / "profile" / "seed_2"
    legacy.mkdir(parents=True)
    (legacy / "training_summary.json").write_text(
        json.dumps({"algorithm_source_fingerprint": "legacy"}),
        encoding="utf-8",
    )
    assert _status(tmp_path / "legacy") == "legacy_schema"
