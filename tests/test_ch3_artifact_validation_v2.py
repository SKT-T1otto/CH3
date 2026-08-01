from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

import tools.aggregate_ch3_efficiency_v2 as aggregator
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES
from tools.aggregate_ch3_efficiency_v2 import aggregate, write_aggregate_outputs
from tools.build_ch3_efficiency_scenarios import build_efficiency_manifest
from tools.run_ch3_efficiency_v2 import (
    _completed_summary_matches,
    _expected_run_identity,
)
from train import CH3_EFFICIENCY_V2, train_and_evaluate_method


def test_v2_skip_and_aggregate_validate_complete_artifact_identity(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(
        json.dumps(
            build_efficiency_manifest("validation", count=1),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    summary, _, _ = train_and_evaluate_method(
        "ch3_pse_rmaddpg",
        seed=1,
        episodes=1,
        max_steps=2,
        device="cpu",
        output_dir=runs_root,
        pilot=False,
        scenario_manifest=manifest_path,
        protocol=CH3_EFFICIENCY_V2,
        resume=False,
        checkpoint_interval=1,
        evaluation_limit=1,
        replay_size=32,
    )
    args = argparse.Namespace(
        method="ch3_pse_rmaddpg",
        seed=1,
        max_steps=2,
        evaluation_limit=1,
        replay_size=32,
    )
    expected = _expected_run_identity(args, 1, manifest_path)
    method_dir = runs_root / "ch3_pse_rmaddpg" / "seed_1"
    assert _completed_summary_matches(summary, expected, method_dir) == {}
    repository_changed = dict(expected)
    repository_changed["current_repository_source_fingerprint"] = "repository-changed"
    assert _completed_summary_matches(summary, repository_changed, method_dir) == {}

    report = aggregate(
        runs_root,
        seed=1,
        bootstrap_samples=8,
        expected_episodes=1,
        expected_max_steps=2,
        expected_scenarios=1,
        scenario_manifest=manifest_path,
        expected_replay_size=32,
        allow_partial=True,
    )
    assert report["partial_debug_report"] is True
    assert report["complete_method_set"] is False
    assert [row["method"] for row in report["methods"]] == ["ch3_pse_rmaddpg"]
    assert report["algorithm_source_fingerprint"] == summary[
        "algorithm_source_fingerprint"
    ]
    assert report["legacy_provenance_unverified"] is False
    output_report = write_aggregate_outputs(report, tmp_path / "summaries")
    assert set(output_report["output_files"]) == {
        "aggregate_json",
        "method_summary_csv",
        "paired_comparisons_csv",
        "findings_md",
    }
    assert all(
        Path(path).is_file() for path in output_report["output_files"].values()
    )
    stored_report = json.loads(
        Path(output_report["output_files"]["aggregate_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert stored_report["output_files"] == output_report["output_files"]
    findings = Path(output_report["output_files"]["findings_md"]).read_text(
        encoding="utf-8"
    )
    assert findings.startswith("# DEBUG PARTIAL REPORT")
    assert "不能用于论文结论" in findings

    monkeypatch.setattr(
        aggregator, "repository_source_fingerprint", lambda root: "changed-repository"
    )
    repository_mismatch_report = aggregate(
        runs_root,
        seed=1,
        bootstrap_samples=8,
        expected_episodes=1,
        expected_max_steps=2,
        expected_scenarios=1,
        scenario_manifest=manifest_path,
        expected_replay_size=32,
        allow_partial=True,
    )
    assert repository_mismatch_report["methods"][0][
        "repository_source_matches_current"
    ] is False
    assert "Algorithm source identity remained identical" in (
        repository_mismatch_report["repository_source_mismatch_warning"]
    )

    summary_path = method_dir / "training_summary.json"
    incompatible = copy.deepcopy(summary)
    incompatible["algorithm_source_fingerprint"] = "different-algorithm"
    summary_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(RuntimeError, match="algorithm_source_fingerprint"):
        aggregate(
            runs_root,
            seed=1,
            bootstrap_samples=8,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=manifest_path,
            expected_replay_size=32,
            allow_partial=True,
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    legacy = copy.deepcopy(summary)
    for key in (
        "provenance_schema_version",
        "algorithm_source_fingerprint",
        "repository_source_fingerprint",
    ):
        legacy.pop(key, None)
    summary_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy provenance cannot establish"):
        aggregate(
            runs_root,
            seed=1,
            bootstrap_samples=8,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=manifest_path,
            expected_replay_size=32,
            allow_partial=True,
        )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="requires --allow-partial"):
        aggregate(
            runs_root,
            seed=1,
            scenario_manifest=manifest_path,
            allow_legacy_provenance=True,
        )
    with pytest.raises(FileNotFoundError, match="requires all seven methods"):
        aggregate(
            runs_root,
            seed=1,
            bootstrap_samples=8,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=manifest_path,
            expected_replay_size=32,
            allow_partial=False,
        )

    tampered = copy.deepcopy(summary)
    tampered["checkpoint_metadata"]["max_steps"] = 3
    mismatches = _completed_summary_matches(tampered, expected, method_dir)
    assert "checkpoint_metadata.summary_copy" in mismatches


def test_formal_aggregation_allows_repository_variation_but_not_algorithm_variation(
    tmp_path, monkeypatch
):
    runs = tmp_path / "runs"
    for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
        method_dir = runs / method / "seed_1"
        method_dir.mkdir(parents=True)
        (method_dir / "training_summary.json").write_text("{}", encoding="utf-8")

    scenario = {"scenario_id": "validation_1", "flow_phase_x": 0.0, "flow_phase_y": 0.0}
    manifest = {
        "protocol": CH3_EFFICIENCY_V2,
        "scenario_role": "validation",
        "use_obstacles": False,
        "obstacle_layout_id": "none",
        "manifest_id": "manifest-v3",
        "manifest_sha256": "manifest-sha",
    }
    monkeypatch.setattr(aggregator, "load_scenario_manifest", lambda path: (manifest, [scenario]))
    monkeypatch.setattr(aggregator, "algorithm_source_fingerprint", lambda root: "algorithm-v3")
    monkeypatch.setattr(aggregator, "repository_source_fingerprint", lambda root: "repository-current")
    algorithm_by_method = {
        method: "algorithm-v3" for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES
    }

    def fake_validate(method, method_dir, **kwargs):
        row = {
            "scenario_id": "validation_1",
            "success": "0",
            "found": "0",
            "energy_cost": "1",
            "collision": "0",
            "minimum_separation_violation": "0",
        }
        return (
            {
                "run_type": (
                    "controller_only"
                    if method in {"ch3_pheromone_prior", "ch3_pse_no_residual"}
                    else "learning"
                ),
                "episodes": 0,
                "scenario_manifest_id": "manifest-v3",
                "scenario_manifest_sha256": "manifest-sha",
            },
            [],
            [row],
            {
                "legacy_provenance_unverified": False,
                "algorithm_source_fingerprint": algorithm_by_method[method],
                "repository_source_fingerprint": f"repository-{method}",
                "repository_source_matches_current": False,
            },
        )

    monkeypatch.setattr(aggregator, "_validate_method_run", fake_validate)
    report = aggregate(
        runs,
        seed=1,
        expected_episodes=1,
        expected_max_steps=2,
        expected_scenarios=1,
        scenario_manifest=tmp_path / "manifest.json",
    )
    assert report["complete_method_set"] is True
    assert report["repository_sources_all_equal"] is False
    assert "Algorithm source identity remained identical" in (
        report["repository_source_mismatch_warning"]
    )

    algorithm_by_method["ch3_pse_no_belief"] = "different-algorithm"
    with pytest.raises(RuntimeError, match="identical current algorithm"):
        aggregate(
            runs,
            seed=1,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=tmp_path / "manifest.json",
        )
