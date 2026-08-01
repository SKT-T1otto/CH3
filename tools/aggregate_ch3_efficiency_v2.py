"""Aggregate strictly paired Chapter-3 efficiency-v2 runs.

The default mode requires all seven methods to have completed the same protocol.
Use --allow-partial only for debugging incomplete runs; partial reports are never
accepted as formal ablation evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
)
from train import (
    CH3_EFFICIENCY_METRICS,
    CH3_EFFICIENCY_V2,
    CH3_MECHANISM_METRICS,
    CH3_PRIMARY_METRICS,
    _algorithm_config_hash,
    _config_hash,
    _evaluation_config_hash,
    _json_safe,
    get_ch3_method_config,
    load_scenario_manifest,
    summarize_evaluation_rows,
)
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_fingerprint,
    file_sha256,
    repository_source_fingerprint,
)


V2_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v2"
RUNS_ROOT = V2_ROOT / "runs"
SUMMARY_ROOT = V2_ROOT / "summaries"
DEFAULT_VALIDATION_MANIFEST = (
    V2_ROOT / "manifests" / "efficiency_v2_validation_scenarios.json"
)

PAIRED_METRICS = (
    "success",
    "success_step",
    "found",
    "found_step",
    "execution_delay",
    "energy_cost",
)
HIGHER_IS_BETTER = frozenset({"success", "found"})
METHOD_SUMMARY_FIELDS = (
    "method",
    "run_type",
    "trained_episodes",
    "evaluation_scenarios",
    "success_rate",
    "mean_success_step",
    "found_rate",
    "mean_found_step",
    "success_given_found",
    "mean_execution_delay",
    "search_distance_before_found",
    "search_distance_until_found_or_horizon",
    "executor_distance_after_found",
    "executor_distance_after_found_unconditional",
    "total_agent_distance",
    "energy_cost",
    "standby_executor_travel_distance",
    "coverage_at_found",
    "belief_entropy_at_found",
    "claim_overlap",
    "residual_contribution_ratio",
    "standby_update_accept_count",
    "standby_mean_accepted_gain",
    "collision_rate",
    "minimum_separation_violation_rate",
    "training_time",
    "actor_runtime_ms",
    "checkpoint_path",
    "checkpoint_sha256",
    "algorithm_config_hash",
    "algorithm_source_fingerprint",
    "repository_source_fingerprint",
    "repository_source_matches_current",
    "scenario_manifest_id",
    "scenario_manifest_sha256",
)
PAIRED_COMPARISON_FIELDS = (
    "treatment",
    "control",
    "metric",
    "paired_count",
    "treatment_mean",
    "control_mean",
    "delta",
    "bootstrap_95_ci_low",
    "bootstrap_95_ci_high",
    "better_direction",
    "directional_result",
    "single_seed_interpretation",
)


def _read_csv(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _rows_match_int(rows, key, expected):
    try:
        return all(int(row.get(key, -1)) == int(expected) for row in rows)
    except (TypeError, ValueError):
        return False


def _paired_delta(rows_a, rows_b, key, *, bootstrap_samples=2000, seed=73001):
    by_id_b = {row["scenario_id"]: row for row in rows_b}
    paired_a = []
    paired_b = []
    for row_a in rows_a:
        row_b = by_id_b.get(row_a["scenario_id"])
        left = _number(row_a, key)
        right = None if row_b is None else _number(row_b, key)
        if left is not None and right is not None:
            paired_a.append(left)
            paired_b.append(right)
    if not paired_a:
        return {
            "paired_count": 0,
            "treatment_mean": None,
            "control_mean": None,
            "delta": None,
            "bootstrap_95_ci": [None, None],
        }
    treatment_values = np.asarray(paired_a, dtype=np.float64)
    control_values = np.asarray(paired_b, dtype=np.float64)
    values = treatment_values - control_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(bootstrap_samples), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "paired_count": int(len(values)),
        "treatment_mean": float(treatment_values.mean()),
        "control_mean": float(control_values.mean()),
        "delta": float(values.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def _directional_result(metric, delta):
    if delta is None or abs(float(delta)) <= 1e-12:
        return "no clear difference"
    treatment_better = (
        float(delta) > 0 if metric in HIGHER_IS_BETTER else float(delta) < 0
    )
    return "positive trend" if treatment_better else "control better"


def _interpret(comparison):
    signals = []
    for metric in PAIRED_METRICS:
        delta = comparison["paired_metrics"][metric]["delta"]
        result = _directional_result(metric, delta)
        if result == "positive trend":
            signals.append(1)
        elif result == "control better":
            signals.append(-1)
    if not signals:
        return "no clear difference"
    if all(signal > 0 for signal in signals):
        return "positive trend"
    if all(signal < 0 for signal in signals):
        return "control better"
    return "mixed"


def _checkpoint_metadata(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint has no identity metadata: {path}")
    return metadata


def _validate_method_run(
    method,
    method_dir,
    *,
    seed,
    expected_episodes,
    expected_max_steps,
    expected_scenarios,
    expected_manifest_id,
    expected_manifest_sha256,
    expected_scenario_ids,
    expected_replay_size,
    current_algorithm_source,
    current_repository_source,
    allow_legacy_provenance,
):
    summary_path = method_dir / "training_summary.json"
    evaluation_path = method_dir / "evaluation_metrics.csv"
    training_path = method_dir / "episode_metrics.csv"
    if not summary_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError(f"incomplete efficiency-v2 run: {method_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_algorithm_source = summary.get("algorithm_source_fingerprint")
    summary_repository_source = summary.get("repository_source_fingerprint")
    legacy_provenance = (
        summary.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION
        or summary_algorithm_source is None
        or summary_repository_source is None
    )
    if legacy_provenance and not allow_legacy_provenance:
        raise RuntimeError(
            f"{method}: legacy provenance cannot establish algorithm identity"
        )
    training = _read_csv(training_path)
    evaluation = _read_csv(evaluation_path)
    expected_run_type = "controller_only" if method in CONTROLLER_ONLY_METHODS else "learning"
    trained_episodes = 0 if expected_run_type == "controller_only" else int(expected_episodes)
    expected_config = get_ch3_method_config(method, protocol=CH3_EFFICIENCY_V2)
    expected_config["max_steps"] = int(expected_max_steps)
    if expected_replay_size is not None:
        expected_config["replay_size"] = int(expected_replay_size)
    expected_algorithm_hash = _algorithm_config_hash(expected_config)
    expected_evaluation_hash = _evaluation_config_hash(expected_config)
    expected_reward_profile = expected_config["reward_profile"]
    actual_ids = [row.get("scenario_id") for row in evaluation]
    try:
        training_episodes = [int(row["episode"]) for row in training]
    except (KeyError, TypeError, ValueError):
        training_episodes = []
    resolved_config = summary.get("resolved_config")
    checks = {
        "method": summary.get("method") == method,
        "protocol": summary.get("protocol") == CH3_EFFICIENCY_V2,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "run_type": summary.get("run_type") == expected_run_type,
        "episodes": int(summary.get("episodes", -1)) == trained_episodes,
        "max_steps": int(summary.get("max_steps", -1)) == int(expected_max_steps),
        "algorithm_config_hash": summary.get("algorithm_config_hash") == expected_algorithm_hash,
        "evaluation_config_hash": summary.get("evaluation_config_hash") == expected_evaluation_hash,
        "reward_profile": summary.get("reward_profile") == expected_reward_profile,
        "training_count": len(training) == trained_episodes,
        "training_episode_sequence": training_episodes == list(range(1, trained_episodes + 1)),
        "training_method": all(row.get("method") == method for row in training),
        "training_seed": _rows_match_int(training, "seed", seed),
        "evaluation_count": len(evaluation) == int(expected_scenarios),
        "summary_evaluation_count": int(summary.get("evaluation_scenarios", -1)) == int(expected_scenarios),
        "scenario_manifest_id": summary.get("scenario_manifest_id") == expected_manifest_id,
        "scenario_manifest_sha256": summary.get("scenario_manifest_sha256") == expected_manifest_sha256,
        "scenario_ids": summary.get("scenario_ids") == list(expected_scenario_ids),
        "evaluation_scenario_ids": actual_ids == list(expected_scenario_ids),
        "scenario_ids_unique": len(actual_ids) == len(set(actual_ids)),
        "evaluation_method": all(row.get("method") == method for row in evaluation),
        "evaluation_seed": _rows_match_int(evaluation, "seed", seed),
        "resolved_config": isinstance(resolved_config, dict),
        "communication_model": summary.get("communication_model") == "fixed_reliable_one_step_v1",
    }
    if not legacy_provenance:
        checks.update({
            "provenance_schema_version": summary.get("provenance_schema_version")
            == PROVENANCE_SCHEMA_VERSION,
            "algorithm_source_fingerprint": summary_algorithm_source
            == current_algorithm_source,
            "source_fingerprint_legacy_alias": summary.get("source_fingerprint")
            == summary_repository_source,
        })
    if isinstance(resolved_config, dict):
        checks.update({
            "config_hash_self_consistent": summary.get("config_hash")
            == _config_hash(resolved_config),
            "algorithm_hash_self_consistent": summary.get("algorithm_config_hash")
            == _algorithm_config_hash(resolved_config),
            "evaluation_hash_self_consistent": summary.get("evaluation_config_hash")
            == _evaluation_config_hash(resolved_config),
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{method} protocol mismatch: {failed}")

    checkpoint_path = summary.get("checkpoint_path")
    if expected_run_type == "controller_only":
        controller_checks = {
            "checkpoint_path": checkpoint_path == "N/A",
            "checkpoint_sha256": summary.get("checkpoint_sha256") is None,
            "checkpoint_paths": summary.get("checkpoint_paths") == [],
            "checkpoint_metadata": summary.get("checkpoint_metadata") is None,
            "resume_state_path": summary.get("resume_state_path") in {None, "N/A"},
            "no_checkpoint_files": not any(method_dir.glob("*.pt")),
        }
        bad_controller = [name for name, passed in controller_checks.items() if not passed]
        if bad_controller:
            raise RuntimeError(
                f"{method} controller-only artifact mismatch: {bad_controller}"
            )
    else:
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{method} checkpoint missing: {checkpoint}")
        if checkpoint.resolve().parent != method_dir.resolve():
            raise RuntimeError(f"{method} checkpoint is outside its method directory")
        listed_checkpoints = {
            str(Path(str(path)).resolve()) for path in summary.get("checkpoint_paths", [])
        }
        if str(checkpoint.resolve()) not in listed_checkpoints:
            raise RuntimeError(f"{method} final checkpoint is absent from checkpoint_paths")
        if summary.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise RuntimeError(f"{method} checkpoint hash differs from summary")
        metadata = _checkpoint_metadata(checkpoint)
        metadata_legacy = (
            metadata.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION
            or metadata.get("algorithm_source_fingerprint") is None
            or metadata.get("repository_source_fingerprint") is None
        )
        if metadata_legacy and not allow_legacy_provenance:
            raise RuntimeError(
                f"{method}: legacy provenance cannot establish algorithm identity"
            )
        metadata_config = metadata.get("config")
        metadata_checks = {
            "schema_version": metadata.get("schema_version") == 2,
            "algorithm": metadata.get("algorithm") == "residual_maddpg_twin_critic_v1",
            "method": metadata.get("method") == method,
            "protocol": metadata.get("protocol") == CH3_EFFICIENCY_V2,
            "run_type": metadata.get("run_type") == expected_run_type,
            "seed": int(metadata.get("seed", -1)) == int(seed),
            "max_steps": int(metadata.get("max_steps", -1)) == int(expected_max_steps),
            "requested_episodes": int(metadata.get("requested_episodes", -1)) == int(expected_episodes),
            "episodes": int(metadata.get("episodes", -1)) == int(expected_episodes),
            "checkpoint_episode": int(metadata.get("checkpoint_episode", -1)) == int(expected_episodes),
            "checkpoint_kind": metadata.get("checkpoint_kind") == "final",
            "algorithm_config_hash": metadata.get("algorithm_config_hash") == expected_algorithm_hash,
            "evaluation_config_hash": metadata.get("evaluation_config_hash") == expected_evaluation_hash,
            "run_config_hash": metadata.get("run_config_hash") == summary.get("run_config_hash"),
            "reward_profile": metadata.get("reward_profile") == expected_reward_profile,
            "scenario_manifest_id": metadata.get("scenario_manifest_id") == expected_manifest_id,
            "scenario_manifest_sha256": metadata.get("scenario_manifest_sha256") == expected_manifest_sha256,
            "observation_dims": metadata.get("observation_dims") == [28, 28, 28, 28],
            "action_dims": metadata.get("action_dims") == [3, 3, 3, 3],
            "config_dictionary": isinstance(metadata_config, dict),
            "summary_copy": summary.get("checkpoint_metadata") == _json_safe(metadata),
        }
        if not metadata_legacy:
            metadata_checks.update({
                "provenance_schema_version": metadata.get("provenance_schema_version")
                == PROVENANCE_SCHEMA_VERSION,
                "algorithm_source_matches_current": metadata.get(
                    "algorithm_source_fingerprint"
                ) == current_algorithm_source,
                "algorithm_source_matches_summary": metadata.get(
                    "algorithm_source_fingerprint"
                ) == summary_algorithm_source,
                "repository_source_matches_summary": metadata.get(
                    "repository_source_fingerprint"
                ) == summary_repository_source,
                "source_fingerprint_legacy_alias": metadata.get("source_fingerprint")
                == metadata.get("repository_source_fingerprint"),
            })
        if isinstance(metadata_config, dict):
            metadata_checks.update({
                "config_hash_self_consistent": metadata.get("config_hash")
                == _config_hash(metadata_config),
                "algorithm_hash_self_consistent": metadata.get("algorithm_config_hash")
                == _algorithm_config_hash(metadata_config),
                "evaluation_hash_self_consistent": metadata.get("evaluation_config_hash")
                == _evaluation_config_hash(metadata_config),
            })
        bad_metadata = [name for name, passed in metadata_checks.items() if not passed]
        if bad_metadata:
            raise RuntimeError(f"{method} checkpoint metadata mismatch: {bad_metadata}")
        resume_path = Path(str(summary.get("resume_state_path", "")))
        if not resume_path.is_file() or resume_path.resolve().parent != method_dir.resolve():
            raise RuntimeError(f"{method} resume state is missing or misplaced")
    return summary, training, evaluation, {
        "legacy_provenance_unverified": legacy_provenance,
        "algorithm_source_fingerprint": summary_algorithm_source,
        "repository_source_fingerprint": (
            summary_repository_source
            if summary_repository_source is not None
            else summary.get("source_fingerprint")
        ),
        "repository_source_matches_current": (
            summary_repository_source == current_repository_source
            if summary_repository_source is not None
            else None
        ),
    }


def aggregate(
    runs_root=RUNS_ROOT,
    seed=1,
    bootstrap_samples=2000,
    *,
    expected_episodes=200,
    expected_max_steps=400,
    expected_scenarios=50,
    scenario_manifest=DEFAULT_VALIDATION_MANIFEST,
    expected_replay_size=None,
    allow_partial=False,
    allow_legacy_provenance=False,
):
    runs_root = Path(runs_root)
    if allow_legacy_provenance and not allow_partial:
        raise ValueError(
            "--allow-legacy-provenance requires --allow-partial; legacy artifacts "
            "cannot produce a complete report"
        )
    current_algorithm_source = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository_source = repository_source_fingerprint(PROJECT_ROOT)
    manifest, manifest_scenarios = load_scenario_manifest(scenario_manifest)
    manifest_checks = {
        "protocol": manifest.get("protocol") == CH3_EFFICIENCY_V2,
        "scenario_role": manifest.get("scenario_role") == "validation",
        "use_obstacles": manifest.get("use_obstacles") is False,
        "obstacle_layout_id": manifest.get("obstacle_layout_id") == "none",
        "scenario_count": len(manifest_scenarios) >= int(expected_scenarios),
        "flow_phases": all(
            float(row.get("flow_phase_x", 0.0)) == 0.0
            and float(row.get("flow_phase_y", 0.0)) == 0.0
            for row in manifest_scenarios
        ),
    }
    bad_manifest = [name for name, passed in manifest_checks.items() if not passed]
    if bad_manifest:
        raise ValueError(f"invalid efficiency-v2 validation manifest: {bad_manifest}")
    expected_scenario_ids = [
        str(row["scenario_id"])
        for row in manifest_scenarios[: int(expected_scenarios)]
    ]
    available = [
        method
        for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES
        if (runs_root / method / f"seed_{int(seed)}" / "training_summary.json").is_file()
    ]
    missing = [m for m in ACTIVE_CH3_FINAL_EXPERIMENT_MODES if m not in available]
    if missing and not allow_partial:
        raise FileNotFoundError(
            "formal efficiency-v2 aggregation requires all seven methods; "
            f"missing={missing}"
        )
    methods = available if allow_partial else list(ACTIVE_CH3_FINAL_EXPERIMENT_MODES)
    if not methods:
        raise FileNotFoundError("no efficiency-v2 runs are available to aggregate")
    summaries = []
    evaluations = {}
    reference_identity = None
    reference_ids = None
    method_repository_sources = {}
    method_algorithm_sources = {}
    legacy_provenance_unverified = False
    for method in methods:
        method_dir = runs_root / method / f"seed_{int(seed)}"
        summary, training, rows, provenance_audit = _validate_method_run(
            method,
            method_dir,
            seed=seed,
            expected_episodes=expected_episodes,
            expected_max_steps=expected_max_steps,
            expected_scenarios=expected_scenarios,
            expected_manifest_id=manifest.get("manifest_id"),
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_scenario_ids=expected_scenario_ids,
            expected_replay_size=expected_replay_size,
            current_algorithm_source=current_algorithm_source,
            current_repository_source=current_repository_source,
            allow_legacy_provenance=allow_legacy_provenance,
        )
        legacy_provenance_unverified = (
            legacy_provenance_unverified
            or provenance_audit["legacy_provenance_unverified"]
        )
        method_repository_sources[method] = provenance_audit[
            "repository_source_fingerprint"
        ]
        method_algorithm_sources[method] = provenance_audit[
            "algorithm_source_fingerprint"
        ]
        ids = [row["scenario_id"] for row in rows]
        identity = (
            summary.get("scenario_manifest_id"),
            summary.get("scenario_manifest_sha256"),
        )
        if reference_identity is None:
            reference_identity, reference_ids = identity, ids
        elif identity != reference_identity or ids != reference_ids:
            raise ValueError(f"{method} does not use the paired reference manifest/order")
        summaries.append({
            "method": method,
            "run_type": summary.get("run_type"),
            "trained_episodes": int(summary.get("episodes", 0)),
            "evaluation_scenarios": len(rows),
            **summarize_evaluation_rows(rows),
            "training_time": summary.get("training_time"),
            "actor_runtime_ms": summary.get("actor_runtime_ms"),
            "checkpoint_path": summary.get("checkpoint_path"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "algorithm_config_hash": summary.get("algorithm_config_hash"),
            "algorithm_source_fingerprint": provenance_audit[
                "algorithm_source_fingerprint"
            ],
            "repository_source_fingerprint": provenance_audit[
                "repository_source_fingerprint"
            ],
            "repository_source_matches_current": provenance_audit[
                "repository_source_matches_current"
            ],
            "scenario_manifest_id": summary.get("scenario_manifest_id"),
            "scenario_manifest_sha256": summary.get("scenario_manifest_sha256"),
        })
        evaluations[method] = rows

    verified_algorithm_sources = {
        value for value in method_algorithm_sources.values() if value is not None
    }
    if not legacy_provenance_unverified and verified_algorithm_sources != {
        current_algorithm_source
    }:
        raise RuntimeError(
            "formal aggregation requires one identical current algorithm source "
            f"fingerprint; found={sorted(verified_algorithm_sources)}"
        )
    repository_values = [
        value for value in method_repository_sources.values() if value is not None
    ]
    repository_sources_all_equal = len(set(repository_values)) <= 1
    repository_mismatch = (
        not repository_sources_all_equal
        or any(value != current_repository_source for value in repository_values)
    )
    repository_warning = (
        "" if not repository_mismatch else
        "Non-algorithm repository files differed across runs. "
        "Algorithm source identity remained identical."
    )

    comparisons = []
    full = "ch3_pse_rmaddpg"
    if full in evaluations:
        for method in methods:
            if method == full:
                continue
            paired_metrics = {
                key: _paired_delta(
                    evaluations[full],
                    evaluations[method],
                    key,
                    bootstrap_samples=bootstrap_samples,
                    seed=73001 + index,
                )
                for index, key in enumerate(PAIRED_METRICS)
            }
            for metric, metric_result in paired_metrics.items():
                metric_result["better_direction"] = (
                    "higher" if metric in HIGHER_IS_BETTER else "lower"
                )
                metric_result["directional_result"] = _directional_result(
                    metric, metric_result["delta"]
                )
            comparison = {
                "treatment": full,
                "control": method,
                "paired_metrics": paired_metrics,
            }
            comparison["single_seed_interpretation"] = _interpret(comparison)
            comparisons.append(comparison)
    return {
        "protocol": CH3_EFFICIENCY_V2,
        "seed": int(seed),
        "expected_episodes": int(expected_episodes),
        "expected_max_steps": int(expected_max_steps),
        "expected_scenarios": int(expected_scenarios),
        "expected_replay_size": (
            None if expected_replay_size is None else int(expected_replay_size)
        ),
        "scenario_manifest": str(Path(scenario_manifest)),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": (
            None if legacy_provenance_unverified else current_algorithm_source
        ),
        "current_repository_source_fingerprint": current_repository_source,
        "method_repository_source_fingerprints": method_repository_sources,
        "repository_sources_all_equal": repository_sources_all_equal,
        "repository_source_mismatch_warning": repository_warning,
        "source_fingerprint": current_repository_source,
        "source_fingerprint_semantics": "legacy_repository_alias",
        "partial_debug_report": bool(allow_partial),
        "legacy_provenance_unverified": legacy_provenance_unverified,
        "complete_method_set": not missing,
        "missing_methods": missing,
        "manifest_id": reference_identity[0],
        "manifest_sha256": reference_identity[1],
        "primary_metric_order": list(CH3_PRIMARY_METRICS),
        "secondary_efficiency_metric_order": list(CH3_EFFICIENCY_METRICS),
        "mechanism_metric_order": list(CH3_MECHANISM_METRICS),
        "metric_semantics": {
            "rates": "unconditional rates use all paired scenarios",
            "conditional_means": "found/success timing and post-found path means use only scenarios with the event",
            "inference": "single-seed labels are directional and are not significance claims",
        },
        "methods": summaries,
        "paired_comparisons": comparisons,
    }


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _ordered_method_rows(report):
    by_method = {row["method"]: row for row in report.get("methods", [])}
    return [
        by_method[method]
        for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES
        if method in by_method
    ]


def _paired_comparison_rows(report):
    rows = []
    for comparison in report.get("paired_comparisons", []):
        for metric in PAIRED_METRICS:
            result = comparison.get("paired_metrics", {}).get(metric, {})
            interval = result.get("bootstrap_95_ci", [None, None])
            rows.append({
                "treatment": comparison.get("treatment"),
                "control": comparison.get("control"),
                "metric": metric,
                "paired_count": result.get("paired_count", 0),
                "treatment_mean": result.get("treatment_mean"),
                "control_mean": result.get("control_mean"),
                "delta": result.get("delta"),
                "bootstrap_95_ci_low": interval[0],
                "bootstrap_95_ci_high": interval[1],
                "better_direction": result.get(
                    "better_direction",
                    "higher" if metric in HIGHER_IS_BETTER else "lower",
                ),
                "directional_result": result.get(
                    "directional_result",
                    _directional_result(metric, result.get("delta")),
                ),
                "single_seed_interpretation": comparison.get(
                    "single_seed_interpretation", "no clear difference"
                ),
            })
    return rows


def _markdown_value(value):
    value = _csv_value(value)
    if value == "":
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _render_findings(report):
    method_rows = _ordered_method_rows(report)
    comparison_rows = _paired_comparison_rows(report)
    lines = []
    if report.get("legacy_provenance_unverified"):
        lines.extend([
            "# LEGACY PROVENANCE UNVERIFIED",
            "",
            "**不能用于论文结论。**",
            "",
            "## DEBUG PARTIAL REPORT",
            "",
        ])
    elif report.get("partial_debug_report"):
        lines.extend(["# DEBUG PARTIAL REPORT", "", "**不能用于论文结论。**", ""])
    else:
        lines.extend(["# Chapter-3 Efficiency v2 Findings", ""])

    lines.extend([
        "## 实验身份",
        "",
        f"- protocol: `{report.get('protocol', '')}`",
        f"- seed: `{report.get('seed', '')}`",
        f"- episodes: `{report.get('expected_episodes', '')}`",
        f"- max_steps: `{report.get('expected_max_steps', '')}`",
        f"- scenario count: `{report.get('expected_scenarios', '')}`",
        f"- manifest ID: `{report.get('manifest_id', '')}`",
        f"- manifest SHA256: `{report.get('manifest_sha256', '')}`",
        f"- source fingerprint: `{report.get('source_fingerprint', '')}`",
        "- algorithm source fingerprint: `"
        + str(report.get("algorithm_source_fingerprint", ""))
        + "`",
        "- current repository source fingerprint: `"
        + str(report.get("current_repository_source_fingerprint", ""))
        + "`",
        "- repository source consistency: `"
        + str(bool(report.get("repository_sources_all_equal"))).lower()
        + "`",
        f"- complete: `{str(bool(report.get('complete_method_set'))).lower()}`",
        "- missing methods: `"
        + (", ".join(report.get("missing_methods", [])) or "none")
        + "`",
        "",
        "## 方法总体结果",
        "",
    ])
    if report.get("repository_source_mismatch_warning"):
        lines.extend([
            report["repository_source_mismatch_warning"],
            "",
        ])
    lines.extend(_markdown_table(
        [
            "method", "success rate", "mean success step", "found rate",
            "mean found step", "success given found", "mean execution delay",
            "energy cost",
        ],
        [
            (
                row.get("method"), row.get("success_rate"),
                row.get("mean_success_step"), row.get("found_rate"),
                row.get("mean_found_step"), row.get("success_given_found"),
                row.get("mean_execution_delay"), row.get("energy_cost"),
            )
            for row in method_rows
        ],
    ))
    lines.extend([
        "",
        "## 主指标排序说明",
        "",
        "1. success_rate：越高越好",
        "2. mean_success_step：越低越好",
        "3. found_rate：越高越好",
        "4. mean_found_step：越低越好",
        "5. mean_execution_delay：越低越好",
        "6. energy_cost：越低越好",
        "",
        "## PSE-RMADDPG 配对结果",
        "",
    ])
    lines.extend(_markdown_table(
        [
            "treatment", "control", "metric", "paired count",
            "treatment mean", "control mean", "delta", "95% CI low",
            "95% CI high", "better direction", "directional result",
            "single-seed interpretation",
        ],
        [tuple(row.get(field) for field in PAIRED_COMPARISON_FIELDS) for row in comparison_rows],
    ))
    lines.extend(["", "## 自动方向性结论", ""])
    if report.get("paired_comparisons"):
        lines.extend(
            f"- `{comparison['treatment']}` vs `{comparison['control']}`: "
            f"**{comparison['single_seed_interpretation']}**"
            for comparison in report["paired_comparisons"]
        )
    else:
        lines.append("- no clear difference")
    lines.extend([
        "",
        "## 边界声明",
        "",
        "- 本报告是单 seed Pilot。",
        "- 本报告不构成正式统计证据。",
        "- 未进行多 seed 显著性检验。",
        "- validation 结果不能替代最终 test。",
        "- obstacle 结果不包含在本主表内。",
        "",
    ])
    return "\n".join(lines)


def write_aggregate_outputs(report, summary_root=SUMMARY_ROOT):
    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    seed = int(report["seed"])
    suffix = "partial" if report.get("partial_debug_report") else "complete"
    prefix = f"efficiency_v2_seed_{seed}_{suffix}"
    paths = {
        "aggregate_json": summary_root / f"{prefix}_aggregate.json",
        "method_summary_csv": summary_root / f"{prefix}_method_summary.csv",
        "paired_comparisons_csv": summary_root / f"{prefix}_paired_comparisons.csv",
        "findings_md": summary_root / f"{prefix}_findings.md",
    }
    output_files = {key: str(path) for key, path in paths.items()}
    output_report = {**report, "output_files": output_files}
    _write_csv(
        paths["method_summary_csv"], METHOD_SUMMARY_FIELDS,
        _ordered_method_rows(output_report),
    )
    _write_csv(
        paths["paired_comparisons_csv"], PAIRED_COMPARISON_FIELDS,
        _paired_comparison_rows(output_report),
    )
    paths["findings_md"].write_text(
        _render_findings(output_report), encoding="utf-8"
    )
    paths["aggregate_json"].write_text(
        json.dumps(
            _json_safe(output_report), ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--expected-episodes", type=int, default=200)
    parser.add_argument("--expected-max-steps", type=int, default=400)
    parser.add_argument("--expected-scenarios", type=int, default=50)
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--expected-replay-size", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-legacy-provenance", action="store_true")
    args = parser.parse_args(argv)
    report = aggregate(
        args.runs_root,
        args.seed,
        args.bootstrap_samples,
        expected_episodes=args.expected_episodes,
        expected_max_steps=args.expected_max_steps,
        expected_scenarios=args.expected_scenarios,
        scenario_manifest=args.scenario_manifest,
        expected_replay_size=args.expected_replay_size,
        allow_partial=args.allow_partial,
        allow_legacy_provenance=args.allow_legacy_provenance,
    )
    report = write_aggregate_outputs(report, args.summary_root)
    output_files = report["output_files"]
    print(f"[CH3 efficiency v2] aggregate JSON: {output_files['aggregate_json']}")
    print(
        "[CH3 efficiency v2] method summary CSV: "
        f"{output_files['method_summary_csv']}"
    )
    print(
        "[CH3 efficiency v2] paired comparisons CSV: "
        f"{output_files['paired_comparisons_csv']}"
    )
    print(f"[CH3 efficiency v2] findings Markdown: {output_files['findings_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
