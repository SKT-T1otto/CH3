import csv
from pathlib import Path

import pytest
import torch

import evaluate_pse
import train
from evaluate_pse import _build_evaluation_runtime
from tests.test_ch3_provenance_v3 import fake_repository
from train import (
    CH3_EFFICIENCY_V2,
    CH3_EFFICIENCY_V2_METHOD_CONFIGS,
    CH3_PILOT_V1,
    train_and_evaluate_method,
)


def _episodes(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [int(row["episode"]) for row in csv.DictReader(handle)]


def test_resume_across_checkpoint_cadence_and_checkpoint_evaluation(tmp_path):
    common = dict(
        method="ch3_pse_rmaddpg",
        seed=1,
        max_steps=4,
        device="cpu",
        output_dir=tmp_path,
        pilot=False,
        scenario_manifest=None,
        protocol=CH3_EFFICIENCY_V2,
        evaluation_limit=None,
        replay_size=32,
    )
    train_and_evaluate_method(
        episodes=2, resume=False, checkpoint_interval=1, **common
    )
    summary, _, _ = train_and_evaluate_method(
        episodes=4, resume=True, checkpoint_interval=2, **common
    )
    metrics = tmp_path / "ch3_pse_rmaddpg" / "seed_1" / "episode_metrics.csv"
    assert _episodes(metrics) == [1, 2, 3, 4]
    checkpoint = Path(summary["checkpoint_path"])
    runtime = _build_evaluation_runtime(
        "ch3_pse_rmaddpg",
        model_path=checkpoint,
        seed=1,
        max_steps=4,
        device="cpu",
        protocol=CH3_EFFICIENCY_V2,
    )
    assert runtime.maddpg.checkpoint_metadata["algorithm_config_hash"] == summary["algorithm_config_hash"]


def test_resume_rejects_changed_horizon(tmp_path):
    common = dict(
        method="ch3_pse_rmaddpg", seed=1, device="cpu", output_dir=tmp_path,
        pilot=False, scenario_manifest=None, protocol=CH3_EFFICIENCY_V2,
        evaluation_limit=None, replay_size=32,
    )
    train_and_evaluate_method(
        episodes=1, max_steps=4, resume=False, checkpoint_interval=1, **common
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train_and_evaluate_method(
            episodes=2, max_steps=5, resume=True, checkpoint_interval=1, **common
        )


def test_resume_rejects_changed_reward_profile(tmp_path, monkeypatch):
    common = dict(
        method="ch3_pse_rmaddpg", seed=1, max_steps=4, device="cpu",
        output_dir=tmp_path, pilot=False, scenario_manifest=None,
        protocol=CH3_EFFICIENCY_V2, evaluation_limit=None, replay_size=32,
    )
    train_and_evaluate_method(
        episodes=1, resume=False, checkpoint_interval=1, **common
    )
    monkeypatch.setitem(
        CH3_EFFICIENCY_V2_METHOD_CONFIGS["ch3_pse_rmaddpg"],
        "reward_scale",
        401.0,
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train_and_evaluate_method(
            episodes=2, resume=True, checkpoint_interval=2, **common
        )


def test_efficiency_checkpoint_cannot_load_as_v1(tmp_path):
    summary, _, _ = train_and_evaluate_method(
        "ch3_pse_rmaddpg", seed=1, episodes=1, max_steps=2, device="cpu",
        output_dir=tmp_path, pilot=False, scenario_manifest=None,
        protocol=CH3_EFFICIENCY_V2, evaluation_limit=None, replay_size=32,
        resume=False, checkpoint_interval=1,
    )
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        _build_evaluation_runtime(
            "ch3_pse_rmaddpg", model_path=Path(summary["checkpoint_path"]),
            seed=1, max_steps=2, device="cpu", protocol=CH3_PILOT_V1,
        )


def test_v3_resume_and_evaluation_ignore_repository_only_changes(
    tmp_path, monkeypatch
):
    repository = fake_repository(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(train, "PROJECT_ROOT", repository)
    monkeypatch.setattr(evaluate_pse, "PROJECT_ROOT", repository)
    common = dict(
        method="ch3_pse_rmaddpg",
        seed=9,
        max_steps=2,
        device="cpu",
        output_dir=runs,
        pilot=False,
        scenario_manifest=None,
        protocol=CH3_EFFICIENCY_V2,
        evaluation_limit=None,
        replay_size=32,
    )
    first, _, _ = train.train_and_evaluate_method(
        episodes=2, resume=False, checkpoint_interval=1, **common
    )
    original_algorithm = first["algorithm_source_fingerprint"]
    original_repository = first["repository_source_fingerprint"]

    test_file = repository / "tests/test_x.py"
    test_file.write_text("repository audit changed\n", encoding="utf-8")
    resumed, _, _ = train.train_and_evaluate_method(
        episodes=4, resume=True, checkpoint_interval=2, **common
    )
    metrics = runs / "ch3_pse_rmaddpg" / "seed_9" / "episode_metrics.csv"
    assert _episodes(metrics) == [1, 2, 3, 4]
    assert resumed["algorithm_source_fingerprint"] == original_algorithm
    assert resumed["repository_source_fingerprint"] == original_repository
    assert resumed["repository_source_changed_since_resume"] is True
    assert resumed["repository_source_matches_resume"] is False
    assert resumed["repository_changed_during_run"] is False

    metadata_fingerprints = set()
    for checkpoint_path in resumed["checkpoint_paths"]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata_fingerprints.add(
            payload["metadata"]["algorithm_source_fingerprint"]
        )
    assert metadata_fingerprints == {original_algorithm}

    runtime = evaluate_pse._build_evaluation_runtime(
        "ch3_pse_rmaddpg",
        model_path=Path(resumed["checkpoint_path"]),
        seed=1,
        max_steps=2,
        device="cpu",
        protocol=CH3_EFFICIENCY_V2,
    )
    audit = runtime.maddpg.checkpoint_metadata["evaluation_source_audit"]
    assert audit["algorithm_source_matches_checkpoint"] is True
    assert audit["repository_source_matches_checkpoint"] is False
    evaluation_summary, _ = evaluate_pse.evaluate(
        "ch3_pse_rmaddpg",
        model_path=Path(resumed["checkpoint_path"]),
        episodes=1,
        seed=19,
        max_steps=2,
        device="cpu",
        result_dir=tmp_path / "evaluation",
        protocol=CH3_EFFICIENCY_V2,
    )
    assert evaluation_summary["algorithm_source_matches_checkpoint"] is True
    assert evaluation_summary["repository_source_matches_checkpoint"] is False
    assert evaluation_summary["legacy_provenance_unverified"] is False

    legacy_path = tmp_path / "legacy_checkpoint.pt"
    legacy_payload = torch.load(
        resumed["checkpoint_path"], map_location="cpu", weights_only=True
    )
    for key in (
        "provenance_schema_version",
        "algorithm_source_fingerprint",
        "repository_source_fingerprint",
        "algorithm_source_files",
        "provenance_captured_at_utc",
    ):
        legacy_payload["metadata"].pop(key, None)
    torch.save(legacy_payload, legacy_path)
    with pytest.raises(ValueError, match="legacy provenance"):
        evaluate_pse._build_evaluation_runtime(
            "ch3_pse_rmaddpg", model_path=legacy_path, seed=1, max_steps=2,
            device="cpu", protocol=CH3_EFFICIENCY_V2,
        )
    legacy_runtime = evaluate_pse._build_evaluation_runtime(
        "ch3_pse_rmaddpg", model_path=legacy_path, seed=1, max_steps=2,
        device="cpu", allow_legacy_provenance=True,
        protocol=CH3_EFFICIENCY_V2,
    )
    assert legacy_runtime.maddpg.checkpoint_metadata[
        "evaluation_source_audit"
    ]["legacy_provenance_unverified"] is True

    base_env_file = repository / "base_env.py"
    base_env_file.write_text("algorithm changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="algorithm_source_fingerprint"):
        evaluate_pse._build_evaluation_runtime(
            "ch3_pse_rmaddpg",
            model_path=Path(resumed["checkpoint_path"]),
            seed=1,
            max_steps=2,
            device="cpu",
            allow_source_mismatch=True,
            protocol=CH3_EFFICIENCY_V2,
        )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.train_and_evaluate_method(
            episodes=5, resume=True, checkpoint_interval=3, **common
        )
