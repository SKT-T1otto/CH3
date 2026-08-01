"""One-entry runner for Chapter-3 task-efficiency protocol v2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_pse import evaluate
from registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
)
from tools.aggregate_ch3_efficiency_v2 import main as aggregate_main
from tools.build_ch3_efficiency_scenarios import (
    MANIFEST_ROOT,
    MANIFEST_SPECS,
    write_official_manifests,
)
from train import (
    CH3_CHECKPOINT_INTERVALS,
    CH3_EFFICIENCY_V2,
    CH3_TRAINING_BUDGETS,
    DEFAULT_EFFICIENCY_V2_OUTPUT_DIR,
    _algorithm_config_hash,
    _config_hash,
    _evaluation_config_hash,
    _json_safe,
    get_ch3_method_config,
    load_scenario_manifest,
    train_and_evaluate_method,
)
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_fingerprint,
    file_sha256,
    repository_source_fingerprint,
)


V2_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v2"


def _device(value):
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _manifest(kind):
    return MANIFEST_ROOT / MANIFEST_SPECS[kind]["filename"]


def _require_below(path, root, *, label):
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be below {root}, got {resolved}") from exc
    return resolved


def _require_runs_root(path, *, label):
    """Accept the default runs tree or a fingerprint-scoped runs tree."""
    resolved = Path(path).resolve()
    allowed_roots = (
        (V2_ROOT / "runs").resolve(),
        (V2_ROOT / "runs_by_algorithm").resolve(),
    )
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"{label} must be below one of {allowed_roots}, got {resolved}"
    )


def _validate_training_manifest(path):
    manifest, scenarios = load_scenario_manifest(path)
    checks = {
        "protocol": manifest.get("protocol") == CH3_EFFICIENCY_V2,
        "scenario_role": manifest.get("scenario_role") == "validation",
        "use_obstacles": manifest.get("use_obstacles") is False,
        "obstacle_layout_id": manifest.get("obstacle_layout_id") == "none",
        "scenario_obstacles": all(not bool(row.get("use_obstacles")) for row in scenarios),
        "flow_phases": all(
            float(row.get("flow_phase_x", 0.0)) == 0.0
            and float(row.get("flow_phase_y", 0.0)) == 0.0
            for row in scenarios
        ),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"training requires an efficiency-v2 validation manifest: {failed}")
    return manifest, scenarios


def _run_checked(name, command, timeout=None):
    print(f"[CH3 efficiency v2] {name}: {' '.join(map(str, command))}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")


def _expected_run_identity(args, episodes, manifest_path):
    manifest, scenarios = load_scenario_manifest(manifest_path)
    selected = scenarios[
        : None if args.evaluation_limit is None else max(0, int(args.evaluation_limit))
    ]
    config = get_ch3_method_config(args.method, protocol=CH3_EFFICIENCY_V2)
    config["max_steps"] = int(args.max_steps)
    if args.replay_size is not None:
        config["replay_size"] = int(args.replay_size)
    expected_run_type = (
        "controller_only" if args.method in CONTROLLER_ONLY_METHODS else "learning"
    )
    expected_episodes = 0 if expected_run_type == "controller_only" else int(episodes)
    return {
        "method": args.method,
        "protocol": CH3_EFFICIENCY_V2,
        "seed": int(args.seed),
        "episodes": expected_episodes,
        "max_steps": int(args.max_steps),
        "run_type": expected_run_type,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": algorithm_source_fingerprint(PROJECT_ROOT),
        "current_repository_source_fingerprint": repository_source_fingerprint(
            PROJECT_ROOT
        ),
        "algorithm_config_hash": _algorithm_config_hash(config),
        "evaluation_config_hash": _evaluation_config_hash(config),
        "reward_profile": config["reward_profile"],
        "scenario_manifest_id": manifest.get("manifest_id"),
        "scenario_manifest_sha256": manifest["manifest_sha256"],
        "evaluation_scenarios": len(selected),
        "scenario_ids": [str(item["scenario_id"]) for item in selected],
    }


def _read_csv(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _completed_summary_matches(summary, expected, method_dir=None):
    """Return every reason a completed run is unsafe to skip."""
    strict_keys = (
        "method", "protocol", "seed", "episodes", "max_steps", "run_type",
        "provenance_schema_version", "algorithm_source_fingerprint",
        "algorithm_config_hash", "evaluation_config_hash", "reward_profile",
        "scenario_manifest_id", "scenario_manifest_sha256",
        "evaluation_scenarios", "scenario_ids",
    )
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key in strict_keys
        for value in (expected[key],)
        if summary.get(key) != value
    }
    if not isinstance(summary.get("repository_source_fingerprint"), str):
        mismatches["repository_source_fingerprint"] = {
            "expected": "stored repository audit fingerprint",
            "actual": summary.get("repository_source_fingerprint"),
        }

    method_dir = Path(method_dir or summary.get("run_directory", ""))
    try:
        if Path(str(summary.get("run_directory"))).resolve() != method_dir.resolve():
            mismatches["run_directory"] = {
                "expected": str(method_dir),
                "actual": summary.get("run_directory"),
            }
    except (OSError, TypeError, ValueError):
        mismatches["run_directory"] = {
            "expected": str(method_dir),
            "actual": summary.get("run_directory"),
        }

    training = _read_csv(method_dir / "episode_metrics.csv")
    evaluation = _read_csv(method_dir / "evaluation_metrics.csv")
    expected_episodes = int(expected["episodes"])
    expected_ids = list(expected["scenario_ids"])
    recorded_episodes = []
    try:
        recorded_episodes = [int(row["episode"]) for row in training]
    except (KeyError, TypeError, ValueError):
        pass
    csv_checks = {
        "training_row_count": len(training) == expected_episodes,
        "training_episode_sequence": recorded_episodes
        == list(range(1, expected_episodes + 1)),
        "training_method": all(row.get("method") == expected["method"] for row in training),
        "training_seed": _rows_match_int(training, "seed", expected["seed"]),
        "evaluation_row_count": len(evaluation) == expected["evaluation_scenarios"],
        "evaluation_scenario_order": [row.get("scenario_id") for row in evaluation]
        == expected_ids,
        "evaluation_method": all(row.get("method") == expected["method"] for row in evaluation),
        "evaluation_seed": _rows_match_int(evaluation, "seed", expected["seed"]),
    }
    for key, passed in csv_checks.items():
        if not passed:
            mismatches[key] = {"expected": True, "actual": False}

    resolved_config = summary.get("resolved_config")
    config_checks = {
        "resolved_config": isinstance(resolved_config, dict),
    }
    if isinstance(resolved_config, dict):
        config_checks.update({
            "config_hash": summary.get("config_hash") == _config_hash(resolved_config),
            "algorithm_config_hash": summary.get("algorithm_config_hash")
            == _algorithm_config_hash(resolved_config),
            "evaluation_config_hash": summary.get("evaluation_config_hash")
            == _evaluation_config_hash(resolved_config),
        })
    for key, passed in config_checks.items():
        if not passed:
            mismatches[f"summary.{key}"] = {"expected": True, "actual": False}

    checkpoint_path = summary.get("checkpoint_path")
    if expected["run_type"] == "controller_only":
        controller_checks = {
            "checkpoint_path": checkpoint_path == "N/A",
            "checkpoint_sha256": summary.get("checkpoint_sha256") is None,
            "checkpoint_paths": summary.get("checkpoint_paths") == [],
            "checkpoint_metadata": summary.get("checkpoint_metadata") is None,
            "resume_state_path": summary.get("resume_state_path") in {None, "N/A"},
            "no_checkpoint_files": not any(method_dir.glob("*.pt")),
        }
        for key, passed in controller_checks.items():
            if not passed:
                mismatches[key] = {"expected": True, "actual": False}
        return mismatches

    checkpoint = Path(str(checkpoint_path))
    if not checkpoint.is_file():
        mismatches["checkpoint_path"] = {
            "expected": "existing file",
            "actual": checkpoint_path,
        }
        return mismatches
    if checkpoint.resolve().parent != method_dir.resolve():
        mismatches["checkpoint_directory"] = {
            "expected": str(method_dir.resolve()),
            "actual": str(checkpoint.resolve().parent),
        }
    listed_checkpoints = {
        str(Path(str(path)).resolve()) for path in summary.get("checkpoint_paths", [])
    }
    if str(checkpoint.resolve()) not in listed_checkpoints:
        mismatches["checkpoint_paths"] = {
            "expected": f"contains {checkpoint.resolve()}",
            "actual": summary.get("checkpoint_paths"),
        }
    actual_sha = file_sha256(checkpoint)
    if summary.get("checkpoint_sha256") != actual_sha:
        mismatches["checkpoint_sha256"] = {
            "expected": actual_sha,
            "actual": summary.get("checkpoint_sha256"),
        }
    try:
        metadata = _load_checkpoint_metadata(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        mismatches["checkpoint_metadata_load"] = {
            "expected": "valid metadata",
            "actual": f"{type(exc).__name__}: {exc}",
        }
        return mismatches

    metadata_config = metadata.get("config")
    metadata_checks = {
        "schema_version": metadata.get("schema_version") == 2,
        "algorithm": metadata.get("algorithm") == "residual_maddpg_twin_critic_v1",
        "method": metadata.get("method") == expected["method"],
        "protocol": metadata.get("protocol") == expected["protocol"],
        "run_type": metadata.get("run_type") == expected["run_type"],
        "seed": int(metadata.get("seed", -1)) == expected["seed"],
        "requested_episodes": int(metadata.get("requested_episodes", -1)) == expected_episodes,
        "checkpoint_episode": int(metadata.get("checkpoint_episode", -1)) == expected_episodes,
        "checkpoint_kind": metadata.get("checkpoint_kind") == "final",
        "max_steps": int(metadata.get("max_steps", -1)) == expected["max_steps"],
        "provenance_schema_version": metadata.get("provenance_schema_version")
        == PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": metadata.get("algorithm_source_fingerprint")
        == expected["algorithm_source_fingerprint"],
        "algorithm_source_matches_summary": metadata.get(
            "algorithm_source_fingerprint"
        ) == summary.get("algorithm_source_fingerprint"),
        "repository_source_matches_summary": metadata.get(
            "repository_source_fingerprint"
        ) == summary.get("repository_source_fingerprint"),
        "source_fingerprint_legacy_alias": metadata.get("source_fingerprint")
        == metadata.get("repository_source_fingerprint"),
        "algorithm_config_hash": metadata.get("algorithm_config_hash")
        == expected["algorithm_config_hash"],
        "evaluation_config_hash": metadata.get("evaluation_config_hash")
        == expected["evaluation_config_hash"],
        "run_config_hash": metadata.get("run_config_hash") == summary.get("run_config_hash"),
        "reward_profile": metadata.get("reward_profile") == expected["reward_profile"],
        "scenario_manifest_id": metadata.get("scenario_manifest_id")
        == expected["scenario_manifest_id"],
        "scenario_manifest_sha256": metadata.get("scenario_manifest_sha256")
        == expected["scenario_manifest_sha256"],
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
    for key, passed in metadata_checks.items():
        if not passed:
            mismatches[f"checkpoint_metadata.{key}"] = {
                "expected": True,
                "actual": False,
            }

    resume_path = Path(str(summary.get("resume_state_path", "")))
    if not resume_path.is_file() or resume_path.resolve().parent != method_dir.resolve():
        mismatches["resume_state_path"] = {
            "expected": f"existing file below {method_dir}",
            "actual": summary.get("resume_state_path"),
        }
    return mismatches


def run_training(args):
    episodes = int(
        args.episodes
        if args.episodes is not None
        else CH3_TRAINING_BUDGETS[args.budget]
    )
    if args.method in CONTROLLER_ONLY_METHODS:
        episodes = 0
    interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else CH3_CHECKPOINT_INTERVALS[args.budget]
    )
    if getattr(args, "force_new_run_directory", False):
        base = (
            Path(args.output_dir) / "runs_by_algorithm"
            if args.output_dir is not None
            else V2_ROOT / "runs_by_algorithm"
        )
        output_dir = base / algorithm_source_fingerprint(PROJECT_ROOT)[:12]
        _require_below(output_dir, base, label="forced training output_dir")
        print(
            f"[CH3 efficiency v2] forced new run directory: {output_dir}",
            flush=True,
        )
    else:
        output_dir = Path(args.output_dir or DEFAULT_EFFICIENCY_V2_OUTPUT_DIR)
        _require_below(output_dir, V2_ROOT / "runs", label="training output_dir")
    manifest_path = Path(args.scenario_manifest or _manifest("validation"))
    _validate_training_manifest(manifest_path)
    method_dir = output_dir / args.method / f"seed_{args.seed}"
    summary_path = method_dir / "training_summary.json"
    if (
        method_dir.is_dir()
        and not summary_path.is_file()
        and not args.resume
        and any(method_dir.iterdir())
    ):
        raise RuntimeError(
            "existing run directory has artifacts but no training summary and "
            "will not be overwritten; use a new output directory or archive it first"
        )
    if summary_path.is_file() and not args.resume:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = _expected_run_identity(args, episodes, manifest_path)
        mismatches = _completed_summary_matches(summary, expected, method_dir)
        if not mismatches:
            repository_matches = summary.get(
                "repository_source_fingerprint"
            ) == expected["current_repository_source_fingerprint"]
            result = dict(summary)
            result["audit"] = {
                "repository_source_matches_current": repository_matches,
                "stored_repository_source_fingerprint": summary.get(
                    "repository_source_fingerprint"
                ),
                "current_repository_source_fingerprint": expected[
                    "current_repository_source_fingerprint"
                ],
            }
            if not repository_matches:
                print(
                    "[CH3 efficiency v2] repository source differs, algorithm "
                    "source matches; completed training remains compatible",
                    flush=True,
                )
            print(f"[CH3 efficiency v2] skip completed method: {args.method}")
            return result
        raise RuntimeError(
            "existing run is incompatible and will not be overwritten; use a "
            "new output directory or archive the old run first; "
            f"mismatches={mismatches}"
        )
    return train_and_evaluate_method(
        args.method,
        seed=args.seed,
        episodes=episodes,
        max_steps=args.max_steps,
        device=_device(args.device),
        output_dir=output_dir,
        pilot=args.budget == "pilot",
        scenario_manifest=manifest_path,
        protocol=CH3_EFFICIENCY_V2,
        resume=args.resume,
        checkpoint_interval=interval,
        evaluation_limit=args.evaluation_limit,
        replay_size=args.replay_size,
    )[0]


def _completed_model_path(args):
    method_dir = (
        Path(args.output_dir or DEFAULT_EFFICIENCY_V2_OUTPUT_DIR)
        / args.method
        / f"seed_{args.seed}"
    )
    summary_path = method_dir / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing completed training summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_config = summary.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ValueError("training summary has no resolved configuration")
    validation_args = argparse.Namespace(**vars(args))
    validation_args.evaluation_limit = int(summary.get("evaluation_scenarios", -1))
    validation_args.replay_size = int(resolved_config["replay_size"])
    expected = _expected_run_identity(
        validation_args,
        int(summary.get("episodes", -1)),
        _manifest("validation"),
    )
    mismatches = _completed_summary_matches(summary, expected, method_dir)
    if mismatches:
        raise ValueError(f"completed training identity mismatch: {mismatches}")
    if args.method in CONTROLLER_ONLY_METHODS:
        return None
    checkpoint = Path(summary["checkpoint_path"])
    return checkpoint


def run_named_evaluation(args, kind):
    result_root = V2_ROOT / ("test" if kind == "test" else "obstacle_test")
    return evaluate(
        args.method,
        model_path=_completed_model_path(args),
        episodes=int(args.evaluation_limit or 0),
        seed=args.seed,
        max_steps=args.max_steps,
        device=_device(args.device),
        result_dir=result_root,
        scenario_manifest=_manifest(kind),
        protocol=CH3_EFFICIENCY_V2,
    )[0]


def run_acceptance_smoke(device):
    # The acceptance gate already executes compileall, collection, pytest,
    # validator and isolation smoke.  Do not duplicate those expensive steps.
    _run_checked(
        "acceptance",
        [
            sys.executable,
            "tools/run_ch3.py",
            "--phase",
            "acceptance",
            "--profiles",
            "all",
            "--base-candidate",
            "ch3_v3_full_reference",
            "--seed",
            "1",
            "--episodes",
            "3",
            "--resume-split",
            "1",
            "--max-steps",
            "20",
            "--replay-size",
            "32",
            "--checkpoint-interval",
            "1",
            "--evaluation-limit",
            "1",
            "--device",
            str(device),
            "--restart",
        ],
        timeout=900,
    )
    write_official_manifests()
    smoke_root = V2_ROOT / "validation" / "acceptance_smoke_runs"
    common = dict(
        method="ch3_pse_rmaddpg",
        seed=1,
        max_steps=40,
        device=_device(device),
        output_dir=smoke_root,
        pilot=False,
        scenario_manifest=_manifest("validation"),
        protocol=CH3_EFFICIENCY_V2,
        checkpoint_interval=0,
        evaluation_limit=2,
        replay_size=2048,
    )
    train_and_evaluate_method(episodes=2, resume=False, **common)
    train_and_evaluate_method(episodes=4, resume=True, **common)
    obstacle_root = V2_ROOT / "obstacle_test" / "smoke_runs"
    for method in ("ch3_pheromone_prior", "ch3_pse_rmaddpg"):
        train_and_evaluate_method(
            method,
            seed=1,
            episodes=0 if method in CONTROLLER_ONLY_METHODS else 1,
            max_steps=40,
            device=_device(device),
            output_dir=obstacle_root,
            pilot=False,
            scenario_manifest=_manifest("obstacle"),
            protocol=CH3_EFFICIENCY_V2,
            checkpoint_interval=0,
            evaluation_limit=1,
            replay_size=2048,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "generate",
            "train",
            "aggregate",
            "test",
            "obstacle",
            "acceptance-smoke",
            "provenance-audit",
        ),
        required=True,
    )
    parser.add_argument("--method", choices=ACTIVE_CH3_FINAL_EXPERIMENT_MODES)
    parser.add_argument("--budget", choices=tuple(CH3_TRAINING_BUDGETS), default="pilot")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-legacy-provenance", action="store_true")
    parser.add_argument("--force-new-run-directory", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)

    if args.phase == "generate":
        write_official_manifests()
        return 0
    if args.phase == "aggregate":
        runs_root = Path(args.output_dir or DEFAULT_EFFICIENCY_V2_OUTPUT_DIR)
        manifest_path = Path(args.scenario_manifest or _manifest("validation"))
        _require_runs_root(runs_root, label="aggregate runs root")
        _validate_training_manifest(manifest_path)
        aggregate_args = [
            "--runs-root", str(runs_root),
            "--seed", str(args.seed),
            "--expected-episodes", str(
                args.episodes if args.episodes is not None
                else CH3_TRAINING_BUDGETS[args.budget]
            ),
            "--expected-max-steps", str(args.max_steps),
            "--expected-scenarios", str(
                50 if args.evaluation_limit is None else int(args.evaluation_limit)
            ),
            "--scenario-manifest", str(manifest_path),
        ]
        if args.replay_size is not None:
            aggregate_args.extend(["--expected-replay-size", str(args.replay_size)])
        if args.allow_partial:
            aggregate_args.append("--allow-partial")
        if args.allow_legacy_provenance:
            aggregate_args.append("--allow-legacy-provenance")
        return aggregate_main(aggregate_args)
    if args.phase == "provenance-audit":
        from tools.audit_ch3_provenance import main as provenance_audit_main

        audit_args = [
            "--runs-root",
            str(Path(args.output_dir or DEFAULT_EFFICIENCY_V2_OUTPUT_DIR)),
        ]
        if args.audit_output is not None:
            audit_args.extend(["--output", str(args.audit_output)])
        return provenance_audit_main(audit_args)
    if args.phase == "acceptance-smoke":
        run_acceptance_smoke(args.device)
        return 0
    if args.method is None:
        parser.error(f"--method is required for phase={args.phase}")
    if args.phase == "train":
        result = run_training(args)
    elif args.phase == "test":
        result = run_named_evaluation(args, "test")
    else:
        result = run_named_evaluation(args, "obstacle")
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
