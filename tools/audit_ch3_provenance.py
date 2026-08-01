"""Read-only provenance audit for every Chapter-3 artifact protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ch3_constants import (
    CH3_MISSION_V1,
    CH3_ROOT,
    CH3_UNKNOWN_MAP_V1,
    CH3_UNKNOWN_ROOT,
)
from registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
)
from train import _algorithm_config_hash, _evaluation_config_hash
from training import validate_dataset_isolation
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_fingerprint,
    base_algorithm_source_fingerprint,
    file_sha256,
    json_file_sha256,
    mission_algorithm_source_fingerprint,
    repository_source_fingerprint,
    unknown_map_algorithm_source_fingerprint,
)


CLASSIFICATIONS = (
    "verified",
    "legacy_schema",
    "missing_summary",
    "missing_checkpoint",
    "checkpoint_hash_mismatch",
    "base_algorithm_mismatch",
    "mission_algorithm_mismatch",
    "unknown_map_source_mismatch",
    "source_mismatch",
    "config_mismatch",
    "manifest_mismatch",
    "training_evaluation_leakage",
    "evaluation_manifest_mismatch",
    "evaluation_csv_mismatch",
    "replay_config_mismatch",
    "csv_mismatch",
    "episode_csv_mismatch",
    "resume_mismatch",
    "malformed_metadata",
    "legacy_pre_mission_provenance",
    "legacy_pre_dataset_isolation",
)

_CONFIG_KEYS = (
    "protocol",
    "base_candidate",
    "scenario_profile",
    "seed",
    "episodes",
    "max_steps",
    "replay_size",
    "algorithm_config_hash",
    "evaluation_config_hash",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is missing")
    return metadata


def _csv_episodes(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        episodes = [int(row["episode"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("episode CSV has invalid episode values") from exc
    return rows, episodes


def _same_fields(left, right, keys):
    return all(left.get(key) == right.get(key) for key in keys)


def _manifest_rows(manifest):
    scenarios = manifest.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("manifest scenarios is not a list")
    return scenarios


def _expected_evaluated_ids(evaluation_ids, limit):
    if limit is None:
        return list(evaluation_ids)
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_limit is invalid") from exc
    if limit < 0:
        raise ValueError("evaluation_limit cannot be negative")
    return list(evaluation_ids[:limit])


def _classify_run(
    directory,
    current_base,
    current_mission,
    current_unknown,
):
    summary_path = directory / "training_summary.json"
    if not summary_path.is_file():
        return "missing_summary", "training_summary.json is absent"
    try:
        summary = _read_json(summary_path)
    except (OSError, ValueError, TypeError) as exc:
        return "malformed_metadata", str(exc)

    protocol = summary.get("protocol")
    if (
        summary.get("provenance_schema_version")
        != PROVENANCE_SCHEMA_VERSION
    ):
        return "legacy_schema", "artifact provenance schema is not current"

    if (
        "mission_algorithm_source_fingerprint" not in summary
        and "algorithm_source_fingerprint" in summary
    ):
        return (
            "legacy_pre_mission_provenance",
            "artifact has only the legacy algorithm_source_fingerprint",
        )
    if (
        "mission_algorithm_source_fingerprint" not in summary
        or "base_algorithm_source_fingerprint" not in summary
    ):
        return (
            "legacy_pre_mission_provenance",
            "artifact predates explicit base and mission source identities",
        )

    dataset_fields = (
        "training_manifest",
        "training_manifest_id",
        "training_manifest_sha256",
        "training_scenario_ids",
        "evaluation_manifest",
        "evaluation_manifest_id",
        "evaluation_manifest_sha256",
        "evaluation_scenario_ids",
        "evaluated_scenario_ids",
        "evaluation_limit",
    )
    if any(key not in summary for key in dataset_fields):
        return (
            "legacy_pre_dataset_isolation",
            "artifact predates explicit training/evaluation dataset identity",
        )

    checkpoint_text = summary.get("checkpoint_path")
    checkpoint = Path(checkpoint_text) if isinstance(checkpoint_text, str) else None
    if checkpoint is None or not checkpoint.is_file():
        fallback = directory / "model_final.pt"
        if fallback.is_file():
            checkpoint = fallback
        else:
            return "missing_checkpoint", str(checkpoint_text)
    expected_sha = summary.get("checkpoint_sha256")
    if not isinstance(expected_sha, str) or file_sha256(checkpoint) != expected_sha:
        return "checkpoint_hash_mismatch", "checkpoint SHA256 differs from summary"
    try:
        metadata = _checkpoint_metadata(checkpoint)
    except Exception as exc:
        return "malformed_metadata", str(exc)

    summary_base = summary.get("base_algorithm_source_fingerprint")
    checkpoint_base = metadata.get("base_algorithm_source_fingerprint")
    if summary_base != current_base or checkpoint_base != current_base:
        return (
            "base_algorithm_mismatch",
            "summary/checkpoint base source does not match current base source",
        )
    if summary.get("mission_algorithm_source_fingerprint") != current_mission:
        return "mission_algorithm_mismatch", "summary does not match current mission source"
    if metadata.get("mission_algorithm_source_fingerprint") != current_mission:
        return "mission_algorithm_mismatch", "checkpoint does not match current mission source"
    if protocol == CH3_UNKNOWN_MAP_V1:
        if (
            summary.get("unknown_map_algorithm_source_fingerprint")
            != current_unknown
            or metadata.get("unknown_map_algorithm_source_fingerprint")
            != current_unknown
        ):
            return (
                "unknown_map_source_mismatch",
                "summary/checkpoint unknown source does not match current source",
            )

    if not _same_fields(summary, metadata, _CONFIG_KEYS):
        return "config_mismatch", "summary/checkpoint configuration identity differs"
    if protocol not in {CH3_MISSION_V1, CH3_UNKNOWN_MAP_V1}:
        return "config_mismatch", "artifact protocol is not a unified S/M protocol"
    resolved_config = summary.get("resolved_config")
    if not isinstance(resolved_config, dict):
        return "config_mismatch", "resolved_config is missing"
    if (
        summary.get("algorithm_config_hash") != _algorithm_config_hash(resolved_config)
        or summary.get("evaluation_config_hash")
        != _evaluation_config_hash(resolved_config)
    ):
        return "config_mismatch", "resolved_config hashes are not self-consistent"

    try:
        replay_size = int(summary.get("replay_size"))
    except (TypeError, ValueError):
        return "replay_config_mismatch", "summary replay_size is invalid"
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, dict):
        return "replay_config_mismatch", "checkpoint config is missing"
    if (
        resolved_config.get("replay_size") != replay_size
        or metadata.get("replay_size") != replay_size
        or checkpoint_config.get("replay_size") != replay_size
    ):
        return (
            "replay_config_mismatch",
            "summary, resolved config, and checkpoint replay_size differ",
        )

    if summary.get("observation_dims") != [28] * 4:
        return "config_mismatch", "summary observation dimensions differ"
    if metadata.get("observation_dims") != [28] * 4:
        return "config_mismatch", "checkpoint observation dimensions differ"
    if summary.get("action_dims") != [3] * 4:
        return "config_mismatch", "summary action dimensions differ"
    if metadata.get("action_dims") != [3] * 4:
        return "config_mismatch", "checkpoint action dimensions differ"

    manifest_keys = ("training_manifest_id", "training_manifest_sha256")
    if not _same_fields(summary, metadata, manifest_keys):
        return "manifest_mismatch", "summary/checkpoint manifest identity differs"
    training_ids = summary.get("training_scenario_ids")
    if (
        not isinstance(training_ids, list)
        or training_ids != metadata.get("training_scenario_ids")
    ):
        return "manifest_mismatch", "training scenario IDs/order differ"
    try:
        training_manifest_path = Path(summary["training_manifest"])
        training_manifest = _read_json(training_manifest_path)
        training_scenarios = _manifest_rows(training_manifest)
    except (OSError, ValueError, TypeError) as exc:
        return "manifest_mismatch", f"training manifest: {exc}"
    if (
        json_file_sha256(training_manifest_path)
        != summary["training_manifest_sha256"]
        or training_manifest.get("manifest_id")
        != summary["training_manifest_id"]
        or [str(item.get("scenario_id")) for item in training_scenarios]
        != training_ids
        or training_manifest.get("scenario_role") not in {"train", "smoke_train"}
        or training_manifest.get("scenario_split")
        != training_manifest.get("scenario_role")
    ):
        return "manifest_mismatch", "training manifest content identity differs"
    training_manifest = dict(training_manifest)
    training_manifest["manifest_sha256"] = json_file_sha256(training_manifest_path)

    evaluation_manifest = None
    evaluation_scenarios = []
    evaluation_ids = summary.get("evaluation_scenario_ids")
    if summary.get("evaluation_manifest") is None:
        if (
            summary.get("evaluation_manifest_id") is not None
            or summary.get("evaluation_manifest_sha256") is not None
            or evaluation_ids != []
        ):
            return (
                "evaluation_manifest_mismatch",
                "disabled evaluation has inconsistent identity fields",
            )
    else:
        try:
            evaluation_manifest_path = Path(summary["evaluation_manifest"])
            evaluation_manifest = _read_json(evaluation_manifest_path)
            evaluation_scenarios = _manifest_rows(evaluation_manifest)
        except (OSError, ValueError, TypeError) as exc:
            return "evaluation_manifest_mismatch", str(exc)
        evaluation_ids_from_manifest = [
            str(item.get("scenario_id")) for item in evaluation_scenarios
        ]
        if (
            json_file_sha256(evaluation_manifest_path)
            != summary["evaluation_manifest_sha256"]
            or evaluation_manifest.get("manifest_id")
            != summary["evaluation_manifest_id"]
            or evaluation_ids_from_manifest != evaluation_ids
            or evaluation_manifest.get("scenario_role")
            not in {"validation", "smoke_validation"}
            or evaluation_manifest.get("scenario_split")
            != evaluation_manifest.get("scenario_role")
        ):
            return (
                "evaluation_manifest_mismatch",
                "evaluation manifest content identity differs",
            )
        evaluation_manifest = dict(evaluation_manifest)
        evaluation_manifest["manifest_sha256"] = json_file_sha256(
            evaluation_manifest_path
        )

    try:
        validate_dataset_isolation(
            training_manifest,
            training_scenarios,
            evaluation_manifest,
            evaluation_scenarios,
        )
    except RuntimeError as exc:
        return "training_evaluation_leakage", str(exc)

    try:
        expected_evaluated_ids = _expected_evaluated_ids(
            list(evaluation_ids or []), summary.get("evaluation_limit")
        )
    except ValueError as exc:
        return "evaluation_manifest_mismatch", str(exc)
    if (
        summary.get("evaluated_scenario_ids") != expected_evaluated_ids
        or int(summary.get("evaluation_count", -1))
        != len(expected_evaluated_ids)
    ):
        return (
            "evaluation_manifest_mismatch",
            "evaluated scenario IDs do not match manifest order/evaluation_limit",
        )

    metrics_path = directory / "episode_metrics.csv"
    try:
        rows, csv_episodes = _csv_episodes(metrics_path)
    except (OSError, ValueError) as exc:
        return "episode_csv_mismatch", str(exc)
    completed = int(summary.get("episodes", -1))
    if (
        len(rows) != completed
        or csv_episodes != list(range(1, completed + 1))
        or len(csv_episodes) != len(set(csv_episodes))
    ):
        return (
            "episode_csv_mismatch",
            "episode rows are not unique and contiguous",
        )

    evaluation_path = directory / "evaluation_metrics.csv"
    try:
        with evaluation_path.open("r", newline="", encoding="utf-8") as handle:
            evaluation_rows = list(csv.DictReader(handle))
    except OSError as exc:
        return "evaluation_csv_mismatch", str(exc)
    if (
        len(evaluation_rows) != len(expected_evaluated_ids)
        or [str(row.get("scenario_id")) for row in evaluation_rows]
        != expected_evaluated_ids
    ):
        return (
            "evaluation_csv_mismatch",
            "evaluation CSV rows/IDs differ from manifest selection",
        )

    resume_path_text = summary.get("resume_state_path")
    resume_path = (
        Path(resume_path_text)
        if isinstance(resume_path_text, str)
        else directory / "resume_state.pt"
    )
    if not resume_path.is_file():
        return "resume_mismatch", "resume_state.pt is absent"
    try:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return "resume_mismatch", str(exc)
    if not isinstance(resume, dict):
        return "resume_mismatch", "resume state is not a mapping"
    if resume.get("replay_size") != replay_size:
        return "replay_config_mismatch", "resume replay_size differs"
    resume_keys = [
        "protocol",
        "base_candidate",
        "scenario_profile",
        "seed",
        "max_steps",
        "replay_size",
        "algorithm_config_hash",
        "evaluation_config_hash",
        "base_algorithm_source_fingerprint",
        "mission_algorithm_source_fingerprint",
        "training_manifest_id",
        "training_manifest_sha256",
        "training_scenario_ids",
    ]
    if protocol == CH3_UNKNOWN_MAP_V1:
        resume_keys.extend([
            "unknown_map_algorithm_source_fingerprint",
            "obstacle_knowledge_mode",
            "planner_mode",
            "target_motion_known",
        ])
    else:
        resume_keys.extend([
            "target_motion_mode",
            "obstacle_layout_identity",
        ])
    if (
        int(resume.get("episode", -1)) != completed
        or not _same_fields(summary, resume, resume_keys)
        or resume.get("base_algorithm_source_fingerprint") != current_base
        or resume.get("mission_algorithm_source_fingerprint") != current_mission
        or resume.get("protocol") != protocol
    ):
        return "resume_mismatch", "resume identity does not match summary"
    if int(metadata.get("checkpoint_episode", completed)) != completed:
        return "config_mismatch", "final checkpoint episode differs"
    return "verified", None


def _audit_structured_runs(runs_root):
    runs_root = Path(runs_root)
    current_base = base_algorithm_source_fingerprint(PROJECT_ROOT)
    current_mission = mission_algorithm_source_fingerprint(PROJECT_ROOT)
    current_unknown = unknown_map_algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository = repository_source_fingerprint(PROJECT_ROOT)
    candidates = (
        sorted(path for path in runs_root.rglob("seed_*") if path.is_dir())
        if runs_root.is_dir()
        else []
    )
    rows = []
    protocols = set()
    for directory in candidates:
        summary_path = directory / "training_summary.json"
        if summary_path.is_file():
            try:
                protocol = _read_json(summary_path).get("protocol")
            except (OSError, ValueError, TypeError):
                protocol = None
            if protocol in {CH3_MISSION_V1, CH3_UNKNOWN_MAP_V1}:
                protocols.add(protocol)
        status, detail = _classify_run(
            directory,
            current_base=current_base,
            current_mission=current_mission,
            current_unknown=current_unknown,
        )
        row = {"run_directory": str(directory), "status": status}
        if detail:
            row["detail"] = detail
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    return {
        "protocols": sorted(protocols),
        "base_algorithm_source_fingerprint": current_base,
        "mission_algorithm_source_fingerprint": current_mission,
        "unknown_map_algorithm_source_fingerprint": current_unknown,
        "repository_source_fingerprint": current_repository,
        "run_count": len(rows),
        "verified_count": counts["verified"],
        "all_verified": len(rows) == counts["verified"],
        "classification_counts": {
            status: counts[status] for status in CLASSIFICATIONS
        },
        "runs": rows,
    }


def _legacy_checkpoint_path(value, method_dir):
    if value in {None, "", "N/A"}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.is_file():
        return project_path
    return Path(method_dir) / path.name


def _legacy_entry(method_dir, current_algorithm, current_repository):
    method_dir = Path(method_dir)
    method = method_dir.parent.name
    seed_text = method_dir.name.removeprefix("seed_")
    try:
        seed = int(seed_text)
    except ValueError:
        seed = seed_text
    summary_path = method_dir / "training_summary.json"
    result = {
        "method": method,
        "seed": seed,
        "summary_path": str(summary_path),
        "checkpoint_path": None,
        "stored_algorithm_source_fingerprint": None,
        "stored_repository_source_fingerprint": None,
        "current_algorithm_source_fingerprint": current_algorithm,
        "current_repository_source_fingerprint": current_repository,
        "classification": None,
        "formal_aggregation_eligible": False,
        "reasons": [],
    }
    if not summary_path.is_file():
        result["classification"] = "missing_summary"
        result["reasons"].append("training_summary.json is missing")
        return result
    try:
        summary = _read_json(summary_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["classification"] = "malformed_metadata"
        result["reasons"].append(f"summary read failed: {type(exc).__name__}: {exc}")
        return result
    stored_algorithm = summary.get("algorithm_source_fingerprint")
    stored_repository = summary.get("repository_source_fingerprint")
    result["stored_algorithm_source_fingerprint"] = stored_algorithm
    result["stored_repository_source_fingerprint"] = stored_repository
    checkpoint = _legacy_checkpoint_path(
        summary.get("checkpoint_path"), method_dir
    )
    result["checkpoint_path"] = (
        None if checkpoint is None else str(checkpoint)
    )
    if (
        summary.get("provenance_schema_version")
        != PROVENANCE_SCHEMA_VERSION
        or stored_algorithm is None
        or stored_repository is None
    ):
        result["classification"] = "legacy_repository_only"
        result["reasons"].append(
            "legacy provenance cannot establish algorithm identity"
        )
        return result
    if method in CONTROLLER_ONLY_METHODS:
        if checkpoint is not None:
            result["classification"] = "malformed_metadata"
            result["reasons"].append(
                "controller-only run unexpectedly names a checkpoint"
            )
            return result
    else:
        if checkpoint is None or not checkpoint.is_file():
            result["classification"] = "missing_checkpoint"
            result["reasons"].append("final checkpoint is missing")
            return result
        try:
            metadata = _checkpoint_metadata(checkpoint)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            result["classification"] = "malformed_metadata"
            result["reasons"].append(
                "checkpoint metadata read failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return result
        if (
            metadata.get("provenance_schema_version")
            != PROVENANCE_SCHEMA_VERSION
        ):
            result["classification"] = "legacy_repository_only"
            result["reasons"].append("checkpoint provenance schema is legacy")
            return result
        if metadata.get("repository_source_fingerprint") != stored_repository:
            result["classification"] = "malformed_metadata"
            result["reasons"].append(
                "checkpoint and summary repository provenance are inconsistent"
            )
            return result
        if metadata.get("algorithm_source_fingerprint") != stored_algorithm:
            result["classification"] = "algorithm_mismatch"
            result["reasons"].append(
                "checkpoint and summary algorithm fingerprints differ"
            )
            return result
    if stored_algorithm != current_algorithm:
        result["classification"] = "algorithm_mismatch"
        result["reasons"].append(
            "stored algorithm fingerprint differs from current"
        )
        return result
    result["formal_aggregation_eligible"] = True
    if stored_repository != current_repository:
        result["classification"] = "repository_only_mismatch"
        result["reasons"].append(
            "repository source differs while algorithm source remains identical"
        )
    else:
        result["classification"] = "verified_v3"
    return result


def _audit_efficiency_runs(runs_root):
    runs_root = Path(runs_root)
    current_algorithm = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository = repository_source_fingerprint(PROJECT_ROOT)
    method_dirs = (
        sorted(
            path
            for path in runs_root.rglob("seed_*")
            if path.is_dir()
            and path.parent.name in ACTIVE_CH3_FINAL_EXPERIMENT_MODES
        )
        if runs_root.is_dir()
        else []
    )
    entries = [
        _legacy_entry(path, current_algorithm, current_repository)
        for path in method_dirs
    ]
    names = (
        "verified_v3",
        "legacy_repository_only",
        "missing_summary",
        "missing_checkpoint",
        "algorithm_mismatch",
        "repository_only_mismatch",
        "malformed_metadata",
    )
    counts = {
        name: sum(row["classification"] == name for row in entries)
        for name in names
    }
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "current_algorithm_source_fingerprint": current_algorithm,
        "current_repository_source_fingerprint": current_repository,
        "existing_method_count": len({row["method"] for row in entries}),
        "method_seed_count": len(entries),
        "classification_counts": counts,
        "formal_aggregation_eligible_count": sum(
            bool(row["formal_aggregation_eligible"]) for row in entries
        ),
        "entries": entries,
    }


def audit_runs(runs_root=CH3_ROOT / "runs"):
    """Audit one runs root, auto-detecting legacy, mission, and unknown artifacts."""

    root = Path(runs_root)
    candidates = (
        sorted(path for path in root.rglob("seed_*") if path.is_dir())
        if root.is_dir()
        else []
    )
    structured = False
    for directory in candidates:
        summary_path = directory / "training_summary.json"
        if summary_path.is_file():
            try:
                protocol = _read_json(summary_path).get("protocol")
            except (OSError, ValueError, TypeError):
                protocol = None
            if protocol in {CH3_MISSION_V1, CH3_UNKNOWN_MAP_V1}:
                structured = True
                break
        if directory.parent.name not in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
            structured = True
    if structured:
        return _audit_structured_runs(root)
    return _audit_efficiency_runs(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        default=None,
        help="runs root; repeat to audit multiple roots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CH3_ROOT / "provenance" / "provenance_audit.json",
    )
    args = parser.parse_args(argv)
    roots = args.runs_root or [CH3_ROOT / "runs"]
    reports = [audit_runs(root) for root in roots]
    if len(reports) == 1:
        report = reports[0]
    else:
        report = {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "all_verified": all(
                item.get("all_verified", True) for item in reports
            ),
            "reports": reports,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CH3 audit] roots={len(roots)} "
        f"all_verified={report.get('all_verified', True)} "
        f"output={args.output}"
    )
    return 0 if report.get("all_verified", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
