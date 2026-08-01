"""Aggregate and strictly validate Chapter-3 efficiency-v3 screening runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch3_efficiency_v3_registry import (  # noqa: E402
    CH3_EFFICIENCY_V3_SCREEN,
    CH3_EFFICIENCY_V3_SCREEN_METHODS,
    get_ch3_efficiency_v3_candidate,
    resolve_ch3_efficiency_v3_config,
)
from train import (  # noqa: E402
    _algorithm_config_hash,
    _config_hash,
    _evaluation_config_hash,
    _json_safe,
    load_scenario_manifest,
    summarize_evaluation_rows,
)
from utils.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_fingerprint,
    file_sha256,
    repository_source_fingerprint,
)


V3_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v3_screen"
DEFAULT_VALIDATION_MANIFEST = (
    V3_ROOT / "manifests" / "efficiency_v3_screen_validation_scenarios.json"
)
PAIR_METRICS = (
    "success",
    "found",
    "penalized_completion_step",
    "penalized_found_step",
    "energy_cost",
    "total_agent_distance",
    "minimum_separation_violation",
)
HIGHER_IS_BETTER = frozenset({"success", "found"})


def _read_csv(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ""
                if isinstance(row.get(key), (float, np.floating))
                and not math.isfinite(float(row[key]))
                else row.get(key, "")
                for key in fieldnames
            })


def _rows_match_int(rows, key, expected):
    try:
        return all(int(row.get(key, -1)) == int(expected) for row in rows)
    except (TypeError, ValueError):
        return False


def _load_checkpoint_metadata(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint has no identity metadata: {path}")
    return metadata


def _expected_manifest(path, *, expected_role, expected_scenarios):
    manifest, scenarios = load_scenario_manifest(path)
    checks = {
        "protocol": manifest.get("protocol") == CH3_EFFICIENCY_V3_SCREEN,
        "scenario_role": manifest.get("scenario_role") == expected_role,
        "use_obstacles": manifest.get("use_obstacles") is False,
        "obstacle_layout_id": manifest.get("obstacle_layout_id") == "none",
        "scenario_count": len(scenarios) >= int(expected_scenarios),
        "flow_phases": all(
            float(row.get("flow_phase_x", 0.0)) == 0.0
            and float(row.get("flow_phase_y", 0.0)) == 0.0
            for row in scenarios
        ),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid efficiency-v3 {expected_role} manifest: {failed}")
    selected = scenarios[: int(expected_scenarios)]
    return manifest, selected


def validate_v3_run(
    method,
    method_dir,
    *,
    seed,
    expected_episodes,
    expected_max_steps,
    expected_scenarios,
    scenario_manifest,
    expected_role="validation",
    expected_replay_size=None,
    current_algorithm_source=None,
    current_repository_source=None,
):
    """Validate one completed v3 candidate and return its exact artifacts."""

    method = str(method)
    if method not in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        raise ValueError(f"unknown v3 candidate={method!r}")
    method_dir = Path(method_dir)
    summary_path = method_dir / "training_summary.json"
    training_path = method_dir / "episode_metrics.csv"
    evaluation_path = method_dir / "evaluation_metrics.csv"
    if not summary_path.is_file() or not training_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError(f"incomplete efficiency-v3 run: {method_dir}")

    current_algorithm_source = (
        algorithm_source_fingerprint(PROJECT_ROOT)
        if current_algorithm_source is None
        else str(current_algorithm_source)
    )
    current_repository_source = (
        repository_source_fingerprint(PROJECT_ROOT)
        if current_repository_source is None
        else str(current_repository_source)
    )
    manifest, selected_scenarios = _expected_manifest(
        scenario_manifest,
        expected_role=expected_role,
        expected_scenarios=expected_scenarios,
    )
    expected_ids = [str(row["scenario_id"]) for row in selected_scenarios]

    entry = get_ch3_efficiency_v3_candidate(method)
    expected_config = resolve_ch3_efficiency_v3_config(method)
    expected_config["max_steps"] = int(expected_max_steps)
    if expected_replay_size is not None:
        expected_config["replay_size"] = int(expected_replay_size)
    expected_algorithm_hash = _algorithm_config_hash(expected_config)
    expected_evaluation_hash = _evaluation_config_hash(expected_config)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training = _read_csv(training_path)
    evaluation = _read_csv(evaluation_path)
    try:
        training_episodes = [int(row["episode"]) for row in training]
    except (KeyError, TypeError, ValueError):
        training_episodes = []
    evaluation_ids = [row.get("scenario_id") for row in evaluation]
    resolved_config = summary.get("resolved_config")
    expected_overrides = _json_safe(entry["config_overrides"])
    expected_changed = _json_safe(entry["changed_mechanisms"])

    checks = {
        "method": summary.get("method") == method,
        "candidate_label": summary.get("candidate_label") == method,
        "base_method": summary.get("base_method") == entry["base_method"],
        "config_overrides": summary.get("config_overrides") == expected_overrides,
        "changed_mechanisms": summary.get("changed_mechanisms") == expected_changed,
        "screening_role": summary.get("screening_role") == entry["screening_role"],
        "protocol": summary.get("protocol") == CH3_EFFICIENCY_V3_SCREEN,
        "seed": int(summary.get("seed", -1)) == int(seed),
        "run_type": summary.get("run_type") == "learning",
        "episodes": int(summary.get("episodes", -1)) == int(expected_episodes),
        "max_steps": int(summary.get("max_steps", -1)) == int(expected_max_steps),
        "provenance_schema_version": summary.get("provenance_schema_version")
        == PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": summary.get("algorithm_source_fingerprint")
        == current_algorithm_source,
        "repository_source_fingerprint": isinstance(
            summary.get("repository_source_fingerprint"), str
        ),
        "source_fingerprint_alias": summary.get("source_fingerprint")
        == summary.get("repository_source_fingerprint"),
        "algorithm_config_hash": summary.get("algorithm_config_hash")
        == expected_algorithm_hash,
        "evaluation_config_hash": summary.get("evaluation_config_hash")
        == expected_evaluation_hash,
        "reward_profile": summary.get("reward_profile") == "task_efficiency_v2",
        "resolved_config": isinstance(resolved_config, dict),
        "training_count": len(training) == int(expected_episodes),
        "training_sequence": training_episodes
        == list(range(1, int(expected_episodes) + 1)),
        "training_method": all(row.get("method") == method for row in training),
        "training_seed": _rows_match_int(training, "seed", seed),
        "evaluation_count": len(evaluation) == int(expected_scenarios),
        "summary_evaluation_count": int(summary.get("evaluation_scenarios", -1))
        == int(expected_scenarios),
        "evaluation_method": all(row.get("method") == method for row in evaluation),
        "evaluation_seed": _rows_match_int(evaluation, "seed", seed),
        "scenario_manifest_id": summary.get("scenario_manifest_id")
        == manifest.get("manifest_id"),
        "scenario_manifest_sha256": summary.get("scenario_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "scenario_ids": summary.get("scenario_ids") == expected_ids,
        "evaluation_scenario_ids": evaluation_ids == expected_ids,
        "evaluation_ids_unique": len(evaluation_ids) == len(set(evaluation_ids)),
        "communication_model": summary.get("communication_model")
        == "fixed_reliable_one_step_v1",
    }
    if isinstance(resolved_config, dict):
        checks.update({
            "config_hash_self_consistent": summary.get("config_hash")
            == _config_hash(resolved_config),
            "algorithm_hash_self_consistent": summary.get("algorithm_config_hash")
            == _algorithm_config_hash(resolved_config),
            "evaluation_hash_self_consistent": summary.get("evaluation_config_hash")
            == _evaluation_config_hash(resolved_config),
        })
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{method} v3 protocol mismatch: {failed}")

    checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{method} final checkpoint missing: {checkpoint_path}")
    if checkpoint_path.resolve().parent != method_dir.resolve():
        raise RuntimeError(f"{method} final checkpoint is outside its method directory")
    listed = {
        str(Path(str(path)).resolve()) for path in summary.get("checkpoint_paths", [])
    }
    if str(checkpoint_path.resolve()) not in listed:
        raise RuntimeError(f"{method} final checkpoint is absent from checkpoint_paths")
    checkpoint_sha = file_sha256(checkpoint_path)
    if summary.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"{method} checkpoint SHA256 differs from summary")

    metadata = _load_checkpoint_metadata(checkpoint_path)
    metadata_config = metadata.get("config")
    metadata_checks = {
        "schema_version": metadata.get("schema_version") == 2,
        "algorithm": metadata.get("algorithm") == "residual_maddpg_twin_critic_v1",
        "method": metadata.get("method") == method,
        "candidate_label": metadata.get("candidate_label") == method,
        "base_method": metadata.get("base_method") == entry["base_method"],
        "config_overrides": metadata.get("config_overrides") == expected_overrides,
        "changed_mechanisms": metadata.get("changed_mechanisms") == expected_changed,
        "screening_role": metadata.get("screening_role") == entry["screening_role"],
        "protocol": metadata.get("protocol") == CH3_EFFICIENCY_V3_SCREEN,
        "run_type": metadata.get("run_type") == "learning",
        "seed": int(metadata.get("seed", -1)) == int(seed),
        "requested_episodes": int(metadata.get("requested_episodes", -1))
        == int(expected_episodes),
        "episodes": int(metadata.get("episodes", -1)) == int(expected_episodes),
        "checkpoint_episode": int(metadata.get("checkpoint_episode", -1))
        == int(expected_episodes),
        "checkpoint_kind": metadata.get("checkpoint_kind") == "final",
        "max_steps": int(metadata.get("max_steps", -1)) == int(expected_max_steps),
        "provenance_schema_version": metadata.get("provenance_schema_version")
        == PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": metadata.get("algorithm_source_fingerprint")
        == current_algorithm_source,
        "algorithm_matches_summary": metadata.get("algorithm_source_fingerprint")
        == summary.get("algorithm_source_fingerprint"),
        "repository_matches_summary": metadata.get("repository_source_fingerprint")
        == summary.get("repository_source_fingerprint"),
        "source_fingerprint_alias": metadata.get("source_fingerprint")
        == metadata.get("repository_source_fingerprint"),
        "algorithm_config_hash": metadata.get("algorithm_config_hash")
        == expected_algorithm_hash,
        "evaluation_config_hash": metadata.get("evaluation_config_hash")
        == expected_evaluation_hash,
        "run_config_hash": metadata.get("run_config_hash")
        == summary.get("run_config_hash"),
        "scenario_manifest_id": metadata.get("scenario_manifest_id")
        == manifest.get("manifest_id"),
        "scenario_manifest_sha256": metadata.get("scenario_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "observation_dims": metadata.get("observation_dims") == [28, 28, 28, 28],
        "action_dims": metadata.get("action_dims") == [3, 3, 3, 3],
        "config_dictionary": isinstance(metadata_config, dict),
        "summary_copy": summary.get("checkpoint_metadata") == _json_safe(metadata),
    }
    if isinstance(metadata_config, dict):
        metadata_checks.update({
            "config_hash_self_consistent": metadata.get("config_hash")
            == _config_hash(metadata_config),
            "algorithm_hash_self_consistent": metadata.get("algorithm_config_hash")
            == _algorithm_config_hash(metadata_config),
            "evaluation_hash_self_consistent": metadata.get("evaluation_config_hash")
            == _evaluation_config_hash(metadata_config),
        })
    bad_metadata = [key for key, passed in metadata_checks.items() if not passed]
    if bad_metadata:
        raise RuntimeError(f"{method} v3 checkpoint metadata mismatch: {bad_metadata}")

    resume_path = Path(str(summary.get("resume_state_path", "")))
    if not resume_path.is_file() or resume_path.resolve().parent != method_dir.resolve():
        raise RuntimeError(f"{method} resume state is missing or misplaced")
    resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_checks = {
        "episode": int(resume_state.get("episode", -1)) == int(expected_episodes),
        "method": resume_state.get("method") == method,
        "protocol": resume_state.get("protocol") == CH3_EFFICIENCY_V3_SCREEN,
        "seed": int(resume_state.get("seed", -1)) == int(seed),
        "max_steps": int(resume_state.get("max_steps", -1)) == int(expected_max_steps),
        "algorithm_config_hash": resume_state.get("algorithm_config_hash")
        == expected_algorithm_hash,
        "algorithm_source_fingerprint": resume_state.get(
            "algorithm_source_fingerprint"
        ) == current_algorithm_source,
        "scenario_manifest_id": resume_state.get("scenario_manifest_id")
        == manifest.get("manifest_id"),
        "scenario_manifest_sha256": resume_state.get("scenario_manifest_sha256")
        == manifest.get("manifest_sha256"),
    }
    bad_resume = [key for key, passed in resume_checks.items() if not passed]
    if bad_resume:
        raise RuntimeError(f"{method} v3 resume-state mismatch: {bad_resume}")

    return summary, training, evaluation, {
        "algorithm_source_fingerprint": summary["algorithm_source_fingerprint"],
        "repository_source_fingerprint": summary["repository_source_fingerprint"],
        "repository_source_matches_current": summary[
            "repository_source_fingerprint"
        ] == current_repository_source,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }


def rank_candidates(rows):
    def key(row):
        return (
            -float(row["success_rate"]),
            float(row["mean_penalized_completion_step"]),
            -float(row["found_rate"]),
            float(row["mean_penalized_found_step"]),
            float(row["energy_cost"]),
            float(row["minimum_separation_violation_rate"]),
            str(row["method"]),
        )

    ordered = sorted(rows, key=key)
    for rank, row in enumerate(ordered, 1):
        row["screening_rank"] = rank
    return ordered


def pareto_relations(rows):
    def dominates(left, right):
        lv = (
            float(left["success_rate"]),
            -float(left["mean_penalized_completion_step"]),
            -float(left["energy_cost"]),
        )
        rv = (
            float(right["success_rate"]),
            -float(right["mean_penalized_completion_step"]),
            -float(right["energy_cost"]),
        )
        return all(a >= b for a, b in zip(lv, rv)) and any(
            a > b for a, b in zip(lv, rv)
        )

    for row in rows:
        row["dominated_by"] = [
            other["method"]
            for other in rows
            if other is not row and dominates(other, row)
        ]
        row["dominates"] = [
            other["method"]
            for other in rows
            if other is not row and dominates(row, other)
        ]
        row["pareto_front_member"] = not row["dominated_by"]
    return rows


def apply_shortlist(rows, *, complete, provenance_verified):
    if not complete or not provenance_verified:
        for row in rows:
            row["shortlist_status"] = "insufficient_single_seed_evidence"
        return rows
    by_method = {row["method"]: row for row in rows}
    full = by_method["ch3_v3_full_reference"]
    no_belief = by_method["ch3_v3_no_belief_reference"]
    best_success = max(float(row["success_rate"]) for row in rows)
    for row in rows:
        balanced = (
            float(row["success_rate"]) >= best_success - 0.02
            and float(row["mean_penalized_completion_step"])
            <= 1.05 * float(no_belief["mean_penalized_completion_step"])
            and float(row["energy_cost"]) <= 1.08 * float(no_belief["energy_cost"])
        )
        alternative = (
            float(row["mean_penalized_completion_step"])
            <= 0.97 * float(full["mean_penalized_completion_step"])
            and float(row["success_rate"]) >= float(full["success_rate"])
        )
        row["shortlist_status"] = (
            "shortlisted"
            if balanced and (row["pareto_front_member"] or alternative)
            else "not_shortlisted"
        )
    return rows


def _bootstrap_ci(values, seed=73001, samples=2000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = values[
        rng.integers(0, values.size, size=(int(samples), values.size))
    ].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def paired_comparisons(rows_by_method, *, bootstrap_samples=2000):
    pairs = []
    full = "ch3_v3_full_reference"
    no_belief = "ch3_v3_no_belief_reference"
    pairs.extend(
        (full, method)
        for method in CH3_EFFICIENCY_V3_SCREEN_METHODS
        if method != full
    )
    pairs.extend(
        (no_belief, method)
        for method in CH3_EFFICIENCY_V3_SCREEN_METHODS
        if method not in {full, no_belief}
    )
    output = []
    for pair_index, (left, right) in enumerate(pairs):
        left_map = {row["scenario_id"]: row for row in rows_by_method[left]}
        right_map = {row["scenario_id"]: row for row in rows_by_method[right]}
        ids = sorted(set(left_map) & set(right_map))
        if ids != sorted(left_map) or ids != sorted(right_map):
            raise ValueError(f"paired scenario mismatch for {left} vs {right}")
        for metric_index, metric in enumerate(PAIR_METRICS):
            diffs = [
                float(left_map[sid][metric]) - float(right_map[sid][metric])
                for sid in ids
            ]
            low, high = _bootstrap_ci(
                diffs,
                seed=73001 + pair_index * 20 + metric_index,
                samples=bootstrap_samples,
            )
            delta = float(np.mean(diffs))
            direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
            if abs(delta) <= 1e-12:
                directional = "no clear difference"
            else:
                better = delta > 0 if direction == "higher" else delta < 0
                directional = "positive trend" if better else "control better"
            output.append({
                "left_method": left,
                "right_method": right,
                "metric": metric,
                "paired_scenario_count": len(ids),
                "scenario_ids": "|".join(ids),
                "mean_paired_difference_left_minus_right": delta,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "better_direction": direction,
                "directional_result": directional,
            })
    return output


def aggregate_v3_screen(
    runs_root,
    output_root,
    *,
    seed=1,
    expected_episodes=200,
    expected_max_steps=400,
    expected_scenarios=50,
    scenario_manifest=DEFAULT_VALIDATION_MANIFEST,
    expected_role="validation",
    expected_replay_size=None,
    bootstrap_samples=2000,
    allow_partial=False,
):
    runs_root, output_root = Path(runs_root), Path(output_root)
    current_algorithm = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository = repository_source_fingerprint(PROJECT_ROOT)
    manifest, selected_scenarios = _expected_manifest(
        scenario_manifest,
        expected_role=expected_role,
        expected_scenarios=expected_scenarios,
    )
    expected_ids = [str(row["scenario_id"]) for row in selected_scenarios]

    available = [
        method
        for method in CH3_EFFICIENCY_V3_SCREEN_METHODS
        if (runs_root / method / f"seed_{int(seed)}" / "training_summary.json").is_file()
    ]
    missing = [m for m in CH3_EFFICIENCY_V3_SCREEN_METHODS if m not in available]
    if missing and not allow_partial:
        raise ValueError(
            f"complete aggregate requires all six candidates; missing={missing}"
        )
    methods = available if allow_partial else list(CH3_EFFICIENCY_V3_SCREEN_METHODS)
    if not methods:
        raise FileNotFoundError("no efficiency-v3 screening runs are available")

    summaries = {}
    eval_rows = {}
    audits = {}
    for method in methods:
        method_dir = runs_root / method / f"seed_{int(seed)}"
        summary, _, rows, audit = validate_v3_run(
            method,
            method_dir,
            seed=seed,
            expected_episodes=expected_episodes,
            expected_max_steps=expected_max_steps,
            expected_scenarios=expected_scenarios,
            scenario_manifest=scenario_manifest,
            expected_role=expected_role,
            expected_replay_size=expected_replay_size,
            current_algorithm_source=current_algorithm,
            current_repository_source=current_repository,
        )
        summaries[method], eval_rows[method], audits[method] = summary, rows, audit

    identities = {
        (
            summaries[m].get("scenario_manifest_id"),
            summaries[m].get("scenario_manifest_sha256"),
            tuple(summaries[m].get("scenario_ids", [])),
            summaries[m].get("algorithm_source_fingerprint"),
        )
        for m in methods
    }
    if len(identities) != 1:
        raise ValueError("candidate manifest/order/provenance identities differ")
    if tuple(next(iter(identities))[2]) != tuple(expected_ids):
        raise ValueError("candidate scenario order differs from the requested manifest")

    method_rows = []
    for method in methods:
        metrics = summarize_evaluation_rows(eval_rows[method])
        entry = get_ch3_efficiency_v3_candidate(method)
        method_rows.append({
            "method": method,
            "base_method": entry["base_method"],
            "changed_mechanisms": "|".join(entry["changed_mechanisms"]),
            "trained_episodes": int(summaries[method]["episodes"]),
            "evaluation_scenarios": len(eval_rows[method]),
            **{
                key: metrics.get(key)
                for key in (
                    "success_rate",
                    "mean_success_step",
                    "mean_penalized_completion_step",
                    "mean_normalized_penalized_completion",
                    "found_rate",
                    "mean_found_step",
                    "mean_penalized_found_step",
                    "success_given_found",
                    "mean_execution_delay",
                    "energy_cost",
                    "total_agent_distance",
                    "minimum_separation_violation_rate",
                    "gated_belief_effective_weight",
                    "gated_belief_uniform_mix",
                    "standby_update_accept_count",
                    "residual_contribution_ratio",
                    "completion_failure_rate",
                    "search_failure_rate",
                )
            },
            "checkpoint_path": audits[method]["checkpoint_path"],
            "checkpoint_sha256": audits[method]["checkpoint_sha256"],
            "algorithm_config_hash": summaries[method]["algorithm_config_hash"],
            "algorithm_source_fingerprint": audits[method][
                "algorithm_source_fingerprint"
            ],
            "repository_source_fingerprint": audits[method][
                "repository_source_fingerprint"
            ],
            "repository_source_matches_current": audits[method][
                "repository_source_matches_current"
            ],
        })

    complete = not missing
    provenance_verified = len(identities) == 1 and all(
        summaries[m]["algorithm_source_fingerprint"] == current_algorithm
        for m in methods
    )
    ranked = rank_candidates(method_rows)
    pareto_relations(ranked)
    apply_shortlist(
        ranked, complete=complete, provenance_verified=provenance_verified
    )
    paired = (
        paired_comparisons(eval_rows, bootstrap_samples=bootstrap_samples)
        if complete
        else []
    )

    repository_sources = {
        method: audits[method]["repository_source_fingerprint"] for method in methods
    }
    repository_values = list(repository_sources.values())
    repository_sources_all_equal = len(set(repository_values)) <= 1
    repository_mismatch = (
        not repository_sources_all_equal
        or any(value != current_repository for value in repository_values)
    )
    repository_warning = (
        ""
        if not repository_mismatch
        else "Non-algorithm repository files differed across runs. "
        "Algorithm source identity remained identical."
    )

    status = "complete" if complete else "partial"
    prefix = f"efficiency_v3_screen_seed_{int(seed)}_{status}"
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "aggregate_json": output_root / f"{prefix}_aggregate.json",
        "method_summary_csv": output_root / f"{prefix}_method_summary.csv",
        "paired_comparisons_csv": output_root / f"{prefix}_paired_comparisons.csv",
        "screening_decision_csv": output_root / f"{prefix}_screening_decision.csv",
        "findings_md": output_root / f"{prefix}_findings.md",
    }
    aggregate = {
        "protocol": CH3_EFFICIENCY_V3_SCREEN,
        "seed": int(seed),
        "expected_episodes": int(expected_episodes),
        "expected_max_steps": int(expected_max_steps),
        "expected_scenarios": int(expected_scenarios),
        "expected_role": expected_role,
        "scenario_manifest": str(Path(scenario_manifest)),
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "scenario_ids": expected_ids,
        "aggregate_status": status,
        "complete_method_set": complete,
        "provenance_verified": provenance_verified,
        "algorithm_source_fingerprint": current_algorithm,
        "current_repository_source_fingerprint": current_repository,
        "method_repository_source_fingerprints": repository_sources,
        "repository_sources_all_equal": repository_sources_all_equal,
        "repository_source_mismatch_warning": repository_warning,
        "missing_methods": missing,
        "ranking_rule": [
            "success_rate desc",
            "mean_penalized_completion_step asc",
            "found_rate desc",
            "mean_penalized_found_step asc",
            "energy_cost asc",
            "minimum_separation_violation_rate asc",
        ],
        "pareto_metrics": [
            "success_rate",
            "mean_penalized_completion_step",
            "energy_cost",
        ],
        "methods": ranked,
        "paired_comparisons": paired,
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }

    serial_rows = [
        {
            **row,
            "dominated_by": "|".join(row["dominated_by"]),
            "dominates": "|".join(row["dominates"]),
        }
        for row in ranked
    ]
    _write_csv(output_paths["method_summary_csv"], serial_rows)
    _write_csv(output_paths["paired_comparisons_csv"], paired)
    decision_fields = [
        "method",
        "screening_rank",
        "pareto_front_member",
        "dominated_by",
        "dominates",
        "shortlist_status",
    ]
    _write_csv(
        output_paths["screening_decision_csv"], serial_rows, decision_fields
    )

    findings = [
        "# Chapter-3 Efficiency v3 mechanism screening",
        "",
    ]
    if not complete:
        findings.extend([
            "**DEBUG PARTIAL REPORT — cannot be used for thesis conclusions.**",
            "",
        ])
    findings.extend([
        "This is a single-seed screening result, not formal statistical evidence.",
        "It cannot replace a multi-seed evaluation; a shortlist only admits a candidate to the next stage.",
        "v3 and historical v2 results are not samples from one shared training distribution.",
        "Unconditional penalized time includes the registered failure penalty.",
        "",
        f"- protocol: `{CH3_EFFICIENCY_V3_SCREEN}`",
        f"- seed: `{int(seed)}`",
        f"- episodes: `{int(expected_episodes)}`",
        f"- max_steps: `{int(expected_max_steps)}`",
        f"- scenarios: `{int(expected_scenarios)}`",
        f"- manifest: `{manifest.get('manifest_id')}`",
        f"- algorithm source: `{current_algorithm}`",
        f"- complete: `{str(complete).lower()}`",
        "",
        "## Screening order",
        "",
    ])
    findings.extend(
        f"{row['screening_rank']}. `{row['method']}` — {row['shortlist_status']}"
        for row in ranked
    )
    if repository_warning:
        findings.extend(["", repository_warning])
    output_paths["findings_md"].write_text(
        "\n".join(findings) + "\n", encoding="utf-8"
    )
    output_paths["aggregate_json"].write_text(
        json.dumps(
            _json_safe(aggregate),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=V3_ROOT / "runs")
    parser.add_argument("--output-root", type=Path, default=V3_ROOT / "summaries")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected-episodes", type=int, default=200)
    parser.add_argument("--expected-max-steps", type=int, default=400)
    parser.add_argument("--expected-scenarios", type=int, default=50)
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--expected-role", choices=("validation", "smoke"), default="validation")
    parser.add_argument("--expected-replay-size", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    result = aggregate_v3_screen(
        args.runs_root,
        args.output_root,
        seed=args.seed,
        expected_episodes=args.expected_episodes,
        expected_max_steps=args.expected_max_steps,
        expected_scenarios=args.expected_scenarios,
        scenario_manifest=args.scenario_manifest,
        expected_role=args.expected_role,
        expected_replay_size=args.expected_replay_size,
        bootstrap_samples=args.bootstrap_samples,
        allow_partial=args.allow_partial,
    )
    print(
        json.dumps({
            "status": result["aggregate_status"],
            "methods": len(result["methods"]),
            "output_files": result["output_files"],
        })
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
