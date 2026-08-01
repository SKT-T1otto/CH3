from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import tools.aggregate_ch3_efficiency_v2 as aggregator
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES


def _metric_result(delta, *, treatment_mean=1.0, control_mean=1.0, count=2):
    return {
        "paired_count": count,
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "delta": delta,
        "bootstrap_95_ci": [delta, delta],
    }


def _partial_report():
    comparison = {
        "treatment": "ch3_pse_rmaddpg",
        "control": "ch3_pheromone_prior",
        "paired_metrics": {
            "success": _metric_result(0.5, treatment_mean=1.0, control_mean=0.5),
            "success_step": _metric_result(-2.0, treatment_mean=8.0, control_mean=10.0),
            "found": _metric_result(-0.5, treatment_mean=0.5, control_mean=1.0),
            "found_step": _metric_result(1.0, treatment_mean=7.0, control_mean=6.0),
            "execution_delay": _metric_result(
                None, treatment_mean=None, control_mean=None, count=0
            ),
            "energy_cost": _metric_result(-3.0, treatment_mean=9.0, control_mean=12.0),
        },
    }
    for metric, result in comparison["paired_metrics"].items():
        result["better_direction"] = (
            "higher" if metric in aggregator.HIGHER_IS_BETTER else "lower"
        )
        result["directional_result"] = aggregator._directional_result(
            metric, result["delta"]
        )
    comparison["single_seed_interpretation"] = aggregator._interpret(comparison)
    method_rows = [
        {
            "method": "ch3_pse_rmaddpg",
            "run_type": "learning",
            "trained_episodes": 1,
            "evaluation_scenarios": 2,
            "success_rate": 1.0,
            "mean_success_step": 8.0,
            "found_rate": 0.5,
            "mean_found_step": 7.0,
            "success_given_found": 1.0,
            "mean_execution_delay": float("nan"),
            "energy_cost": 9.0,
            "checkpoint_path": "model_final.pt",
            "checkpoint_sha256": "abc",
            "algorithm_config_hash": "algorithm-hash",
            "scenario_manifest_id": "validation-v1",
            "scenario_manifest_sha256": "manifest-hash",
        },
        {
            "method": "ch3_pheromone_prior",
            "run_type": "controller_only",
            "trained_episodes": 0,
            "evaluation_scenarios": 2,
            "success_rate": 0.5,
            "mean_success_step": 10.0,
            "found_rate": 1.0,
            "mean_found_step": 6.0,
            "success_given_found": 0.5,
            "mean_execution_delay": float("inf"),
            "energy_cost": 12.0,
            "checkpoint_path": "N/A",
            "checkpoint_sha256": None,
            "algorithm_config_hash": "controller-hash",
            "scenario_manifest_id": "validation-v1",
            "scenario_manifest_sha256": "manifest-hash",
        },
    ]
    return {
        "protocol": "ch3_efficiency_v2",
        "seed": 1,
        "expected_episodes": 1,
        "expected_max_steps": 2,
        "expected_scenarios": 2,
        "source_fingerprint": "source-hash",
        "partial_debug_report": True,
        "complete_method_set": False,
        "missing_methods": [
            method
            for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES
            if method not in {"ch3_pheromone_prior", "ch3_pse_rmaddpg"}
        ],
        "manifest_id": "validation-v1",
        "manifest_sha256": "manifest-hash",
        # Intentionally reversed: the writer must restore the formal order.
        "methods": method_rows,
        "paired_comparisons": [comparison],
    }


def test_partial_outputs_are_complete_ordered_and_finite(tmp_path):
    output_report = aggregator.write_aggregate_outputs(_partial_report(), tmp_path)
    paths = {key: Path(value) for key, value in output_report["output_files"].items()}

    assert set(paths) == {
        "aggregate_json",
        "method_summary_csv",
        "paired_comparisons_csv",
        "findings_md",
    }
    assert all(path.is_file() for path in paths.values())
    stored = json.loads(paths["aggregate_json"].read_text(encoding="utf-8"))
    assert stored["output_files"] == output_report["output_files"]

    with paths["method_summary_csv"].open(newline="", encoding="utf-8") as handle:
        method_rows = list(csv.DictReader(handle))
    assert [row["method"] for row in method_rows] == [
        "ch3_pheromone_prior",
        "ch3_pse_rmaddpg",
    ]
    controller = method_rows[0]
    assert controller["trained_episodes"] == "0"
    assert controller["checkpoint_path"] == "N/A"
    assert {
        "algorithm_source_fingerprint",
        "repository_source_fingerprint",
        "repository_source_matches_current",
    }.issubset(method_rows[0])

    for key in ("method_summary_csv", "paired_comparisons_csv"):
        csv_text = paths[key].read_text(encoding="utf-8").lower()
        assert "nan" not in csv_text
        assert "inf" not in csv_text
        assert "infinity" not in csv_text

    findings = paths["findings_md"].read_text(encoding="utf-8")
    assert findings.startswith("# DEBUG PARTIAL REPORT")
    assert "不能用于论文结论" in findings
    assert "mixed" in findings
    assert (
        "`ch3_pse_rmaddpg` vs `ch3_pheromone_prior`: **mixed**" in findings
    )
    assert "significant" not in findings.lower()
    assert "显著优于" not in findings
    assert "显著提升" not in findings


@pytest.mark.parametrize(
    ("metric", "delta", "expected"),
    [
        ("success", 0.1, "positive trend"),
        ("found", -0.1, "control better"),
        ("success_step", -1.0, "positive trend"),
        ("found_step", 1.0, "control better"),
        ("execution_delay", -1.0, "positive trend"),
        ("energy_cost", 1.0, "control better"),
        ("energy_cost", 0.0, "no clear difference"),
        ("energy_cost", None, "no clear difference"),
    ],
)
def test_paired_metric_direction(metric, delta, expected):
    assert aggregator._directional_result(metric, delta) == expected


def test_paired_statistics_use_matching_finite_scenarios_only():
    treatment = [
        {"scenario_id": "a", "energy_cost": "8"},
        {"scenario_id": "b", "energy_cost": "nan"},
        {"scenario_id": "c", "energy_cost": "4"},
    ]
    control = [
        {"scenario_id": "c", "energy_cost": "10"},
        {"scenario_id": "a", "energy_cost": "12"},
        {"scenario_id": "other", "energy_cost": "1"},
    ]
    result = aggregator._paired_delta(
        treatment, control, "energy_cost", bootstrap_samples=8
    )
    assert result["paired_count"] == 2
    assert result["treatment_mean"] == pytest.approx(6.0)
    assert result["control_mean"] == pytest.approx(11.0)
    assert result["delta"] == pytest.approx(-5.0)


def test_main_prints_all_four_output_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(aggregator, "aggregate", lambda *args, **kwargs: _partial_report())
    assert aggregator.main([
        "--allow-partial",
        "--summary-root",
        str(tmp_path),
    ]) == 0
    output = capsys.readouterr().out
    assert "[CH3 efficiency v2] aggregate JSON:" in output
    assert "[CH3 efficiency v2] method summary CSV:" in output
    assert "[CH3 efficiency v2] paired comparisons CSV:" in output
    assert "[CH3 efficiency v2] findings Markdown:" in output


def test_legacy_partial_report_is_explicitly_unverified(tmp_path):
    report = _partial_report()
    report["legacy_provenance_unverified"] = True
    report["algorithm_source_fingerprint"] = None
    output = aggregator.write_aggregate_outputs(report, tmp_path)
    findings = Path(output["output_files"]["findings_md"]).read_text(
        encoding="utf-8"
    )
    assert findings.startswith("# LEGACY PROVENANCE UNVERIFIED")
    assert "不能用于论文结论" in findings
    assert output["algorithm_source_fingerprint"] is None
