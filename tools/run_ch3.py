"""Single safe generate/validate/train/acceptance entry for Chapter 3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ch3_config import build_ch3_config
from ch3_constants import (
    ALL_SCENARIO_PROFILES,
    CH3_MISSION_V1,
    CH3_ROOT,
    CH3_UNKNOWN_MAP_V1,
    SCENARIO_PROFILES,
    UNKNOWN_MANIFEST_ROOT,
    UNKNOWN_MAP_PROFILES,
)
from registry.ch3_efficiency_v3_registry import CH3_EFFICIENCY_V3_SCREEN_METHODS
from runtime import build_runtime
from tools.audit_ch3_provenance import audit_runs, main as audit_main
from tools.build_ch3_scenarios import (
    MANIFEST_ROOT,
    PROFILE_CODES,
    write_scenario_manifests,
)
from tools.validate_ch3 import main as validate_main, validate
from train import _algorithm_config_hash, _evaluation_config_hash, load_scenario_manifest
from training import train_and_evaluate, validate_dataset_isolation
from utils.provenance import (
    base_algorithm_source_fingerprint,
    file_sha256,
    mission_algorithm_source_fingerprint,
    repository_source_fingerprint,
    unknown_map_algorithm_source_fingerprint,
)


def _device(value):
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _below_root(path, label):
    path = Path(path).resolve()
    try:
        path.relative_to(CH3_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be below {CH3_ROOT}") from exc
    return path


def _manifest(profile, split="train"):
    if split not in {"train", "validation", "smoke_train", "smoke_validation"}:
        raise ValueError(f"unsupported scenario split={split!r}")
    code = PROFILE_CODES[profile]
    if profile in UNKNOWN_MAP_PROFILES:
        return UNKNOWN_MANIFEST_ROOT / f"unknown_{split}_{code}.json"
    short_code = code.split("_", 1)[0]
    return MANIFEST_ROOT / f"mission_{split}_{short_code}.json"


def resolve_runtime_config(
    base_candidate, scenario_profile, max_steps, replay_size=None
):
    """Resolve strict identity without allocating an environment or replay."""

    config = build_ch3_config(base_candidate, scenario_profile)
    config["max_steps"] = int(max_steps)
    if replay_size is not None:
        config["replay_size"] = int(replay_size)
    return config


def _load_checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("checkpoint metadata is missing or malformed")
    return payload["metadata"]


def _csv_is_contiguous(path, completed):
    if not Path(path).is_file():
        return False
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        episode_values = [int(row["episode"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return episode_values == list(range(1, int(completed) + 1))


def _validate_manifest_role(manifest, *, profile, roles, label):
    expected_protocol = (
        CH3_UNKNOWN_MAP_V1
        if profile in UNKNOWN_MAP_PROFILES
        else CH3_MISSION_V1
    )
    if manifest.get("protocol") != expected_protocol:
        raise ValueError(f"{label} manifest must use {expected_protocol}")
    if manifest.get("scenario_profile") != profile:
        raise ValueError(f"{label} manifest scenario profile mismatch")
    role = manifest.get("scenario_role")
    if role not in roles or manifest.get("scenario_split") != role:
        raise ValueError(f"{label} manifest has an invalid role/split")


def _artifact_protocol(profile):
    return (
        CH3_UNKNOWN_MAP_V1
        if profile in UNKNOWN_MAP_PROFILES
        else CH3_MISSION_V1
    )


def _strict_skip(
    output_dir,
    candidate,
    profile,
    seed,
    episodes,
    max_steps,
    *,
    requested_replay_size=None,
    training_manifest_path=None,
    evaluation_manifest_path=None,
    evaluation_limit=None,
    checkpoint_interval=0,
):
    """Verify a completed run without allocating a runtime or replay buffer."""

    if evaluation_limit is not None and int(evaluation_limit) < 0:
        raise ValueError("evaluation_limit cannot be negative")
    if profile not in ALL_SCENARIO_PROFILES:
        raise ValueError(f"unsupported scenario profile={profile!r}")

    run_dir = Path(output_dir) / candidate / profile / f"seed_{int(seed)}"
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    training_manifest_path = Path(
        training_manifest_path or _manifest(profile, "train")
    )
    training_manifest, training_scenarios = load_scenario_manifest(
        training_manifest_path
    )
    _validate_manifest_role(
        training_manifest,
        profile=profile,
        roles={"train", "smoke_train"},
        label="strict skip training",
    )

    if evaluation_limit == 0 and evaluation_manifest_path is None:
        evaluation_manifest = None
        evaluation_scenarios = []
    else:
        evaluation_manifest_path = Path(
            evaluation_manifest_path or _manifest(profile, "validation")
        )
        evaluation_manifest, evaluation_scenarios = load_scenario_manifest(
            evaluation_manifest_path
        )
        _validate_manifest_role(
            evaluation_manifest,
            profile=profile,
            roles={"validation", "smoke_validation"},
            label="strict skip evaluation",
        )

    validate_dataset_isolation(
        training_manifest,
        training_scenarios,
        evaluation_manifest,
        evaluation_scenarios,
    )

    resolved_config = resolve_runtime_config(
        candidate,
        profile,
        max_steps,
        replay_size=requested_replay_size,
    )
    resolved_config["checkpoint_interval"] = max(0, int(checkpoint_interval))
    selected_evaluation = (
        evaluation_scenarios
        if evaluation_limit is None
        else evaluation_scenarios[: int(evaluation_limit)]
    )
    training_ids = [
        str(scenario["scenario_id"]) for scenario in training_scenarios
    ]
    evaluation_ids = [
        str(scenario["scenario_id"]) for scenario in evaluation_scenarios
    ]
    evaluated_ids = [
        str(scenario["scenario_id"]) for scenario in selected_evaluation
    ]
    obstacle_layout_identity = "|".join(
        sorted(
            str(scenario.get("obstacle_layout_id", "none"))
            for scenario in training_scenarios
        )
    )

    protocol = _artifact_protocol(profile)
    current_base = base_algorithm_source_fingerprint(PROJECT_ROOT)
    current_mission = mission_algorithm_source_fingerprint(PROJECT_ROOT)
    current_unknown = unknown_map_algorithm_source_fingerprint(PROJECT_ROOT)

    summary_expected = {
        "protocol": protocol,
        "base_candidate": candidate,
        "scenario_profile": profile,
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "base_algorithm_source_fingerprint": current_base,
        "mission_algorithm_source_fingerprint": current_mission,
        "replay_size": int(resolved_config["replay_size"]),
        "algorithm_config_hash": _algorithm_config_hash(resolved_config),
        "evaluation_config_hash": _evaluation_config_hash(resolved_config),
        "training_manifest_id": training_manifest["manifest_id"],
        "training_manifest_sha256": training_manifest["manifest_sha256"],
        "training_scenario_ids": training_ids,
        "evaluation_manifest_id": (
            None
            if evaluation_manifest is None
            else evaluation_manifest["manifest_id"]
        ),
        "evaluation_manifest_sha256": (
            None
            if evaluation_manifest is None
            else evaluation_manifest["manifest_sha256"]
        ),
        "evaluation_scenario_ids": evaluation_ids,
        "evaluated_scenario_ids": evaluated_ids,
        "evaluation_count": len(evaluated_ids),
        "evaluation_limit": evaluation_limit,
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
    }
    if profile in UNKNOWN_MAP_PROFILES:
        summary_expected.update(
            {
                "base_runtime_protocol": resolved_config["protocol"],
                "unknown_map_algorithm_source_fingerprint": current_unknown,
                "target_motion_known": True,
                "obstacle_knowledge_mode": resolved_config[
                    "obstacle_knowledge_mode"
                ],
                "planner_mode": resolved_config["planner_mode"],
                "unknown_map_schema": resolved_config["unknown_map_schema"],
                "target_belief_schema": resolved_config[
                    "target_belief_schema"
                ],
            }
        )
    else:
        summary_expected.update(
            {
                "target_motion_mode": resolved_config["target_motion_mode"],
                "obstacle_layout_identity": obstacle_layout_identity,
            }
        )

    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in summary_expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "incompatible occupied run directory; archive it or select a new "
            f"--output-dir: {mismatches}"
        )

    canonical_config = json.loads(json.dumps(resolved_config, sort_keys=True))
    if summary.get("resolved_config") != canonical_config:
        raise ValueError(
            "resolved config mismatch; archive the run or select a new "
            "--output-dir"
        )

    checkpoint = run_dir / "model_final.pt"
    resume_path = run_dir / "resume_state.pt"
    if not checkpoint.is_file() or not resume_path.is_file():
        raise ValueError(
            "occupied run is incomplete; archive it or select a new "
            "--output-dir"
        )
    if summary.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("checkpoint SHA256 mismatch; refusing strict skip")

    metadata = _load_checkpoint_metadata(checkpoint)
    metadata_expected = {
        "protocol": protocol,
        "base_candidate": candidate,
        "scenario_profile": profile,
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "replay_size": int(resolved_config["replay_size"]),
        "base_algorithm_source_fingerprint": current_base,
        "mission_algorithm_source_fingerprint": current_mission,
        "algorithm_config_hash": _algorithm_config_hash(resolved_config),
        "evaluation_config_hash": _evaluation_config_hash(resolved_config),
        "training_manifest_id": training_manifest["manifest_id"],
        "training_manifest_sha256": training_manifest["manifest_sha256"],
        "training_scenario_ids": training_ids,
        "target_motion_mode": resolved_config["target_motion_mode"],
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
    }
    if profile in UNKNOWN_MAP_PROFILES:
        metadata_expected.update(
            {
                "base_runtime_protocol": resolved_config["protocol"],
                "unknown_map_algorithm_source_fingerprint": current_unknown,
                "target_motion_known": True,
                "obstacle_knowledge_mode": resolved_config[
                    "obstacle_knowledge_mode"
                ],
                "planner_mode": resolved_config["planner_mode"],
                "unknown_map_schema": resolved_config["unknown_map_schema"],
                "target_belief_schema": resolved_config[
                    "target_belief_schema"
                ],
                "map_sharing_mode": resolved_config["map_sharing_mode"],
            }
        )
    else:
        metadata_expected["obstacle_layout_identity"] = (
            obstacle_layout_identity
        )

    checkpoint_mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in metadata_expected.items()
        if metadata.get(key) != value
    }
    checkpoint_episode = metadata.get("checkpoint_episode")
    if (
        checkpoint_episode is not None
        and int(checkpoint_episode) != int(episodes)
    ):
        checkpoint_mismatches["checkpoint_episode"] = {
            "expected": int(episodes),
            "actual": checkpoint_episode,
        }
    if metadata.get("config") != canonical_config:
        checkpoint_mismatches["config"] = {
            "expected": canonical_config,
            "actual": metadata.get("config"),
        }
    if checkpoint_mismatches:
        raise ValueError(
            "checkpoint metadata mismatch; refusing strict skip: "
            f"{checkpoint_mismatches}"
        )

    if not _csv_is_contiguous(run_dir / "episode_metrics.csv", episodes):
        raise ValueError(
            "episode CSV is not unique and contiguous; refusing strict skip"
        )
    evaluation_csv = run_dir / "evaluation_metrics.csv"
    if not evaluation_csv.is_file():
        raise ValueError("evaluation CSV is missing; refusing strict skip")
    with evaluation_csv.open("r", newline="", encoding="utf-8") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    if [str(row.get("scenario_id")) for row in evaluation_rows] != evaluated_ids:
        raise ValueError(
            "evaluation CSV identity mismatch; refusing strict skip"
        )

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_expected = {
        "protocol": protocol,
        "base_candidate": candidate,
        "scenario_profile": profile,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "replay_size": int(resolved_config["replay_size"]),
        "base_algorithm_source_fingerprint": current_base,
        "mission_algorithm_source_fingerprint": current_mission,
        "algorithm_config_hash": _algorithm_config_hash(resolved_config),
        "evaluation_config_hash": _evaluation_config_hash(resolved_config),
        "training_manifest_id": training_manifest["manifest_id"],
        "training_manifest_sha256": training_manifest["manifest_sha256"],
        "training_scenario_ids": training_ids,
    }
    if profile in UNKNOWN_MAP_PROFILES:
        resume_expected.update(
            {
                "base_runtime_protocol": resolved_config["protocol"],
                "unknown_map_algorithm_source_fingerprint": current_unknown,
                "target_motion_known": True,
                "obstacle_knowledge_mode": resolved_config[
                    "obstacle_knowledge_mode"
                ],
                "planner_mode": resolved_config["planner_mode"],
            }
        )
    else:
        resume_expected.update(
            {
                "target_motion_mode": resolved_config["target_motion_mode"],
                "obstacle_layout_identity": obstacle_layout_identity,
            }
        )

    resume_mismatches = {}
    if not isinstance(resume, dict):
        resume_mismatches["payload"] = "resume state is not a mapping"
    else:
        if int(resume.get("episode", -1)) != int(episodes):
            resume_mismatches["episode"] = {
                "expected": int(episodes),
                "actual": resume.get("episode"),
            }
        resume_mismatches.update(
            {
                key: {"expected": value, "actual": resume.get(key)}
                for key, value in resume_expected.items()
                if resume.get(key) != value
            }
        )
    if resume_mismatches:
        raise ValueError(
            "resume identity mismatch; refusing strict skip: "
            f"{resume_mismatches}"
        )

    print(f"[CH3] strict skip: {summary_path.parent}")
    return True


def _train(
    args,
    profile=None,
    training_manifest=None,
    evaluation_manifest=None,
    output_dir=None,
    max_steps=None,
):
    profile = profile or args.scenario_profile
    if profile is None:
        raise ValueError("--scenario-profile is required")
    episodes = int(args.episodes)
    max_steps = int(args.max_steps if max_steps is None else max_steps)
    output_dir = _below_root(
        output_dir or args.output_dir or CH3_ROOT / "runs", "output"
    )
    if args.manifest is not None and args.training_manifest is not None:
        raise ValueError(
            "--manifest and --training-manifest cannot both be provided"
        )
    training_manifest = Path(
        training_manifest
        or args.training_manifest
        or args.manifest
        or _manifest(profile, "train")
    )
    requested_evaluation_manifest = (
        evaluation_manifest or args.evaluation_manifest
    )
    if requested_evaluation_manifest is None and args.evaluation_limit != 0:
        requested_evaluation_manifest = _manifest(profile, "validation")
    evaluation_manifest = (
        None
        if requested_evaluation_manifest is None
        else Path(requested_evaluation_manifest)
    )
    if not args.resume and _strict_skip(
        output_dir,
        args.base_candidate,
        profile,
        args.seed,
        episodes,
        max_steps,
        requested_replay_size=args.replay_size,
        training_manifest_path=training_manifest,
        evaluation_manifest_path=evaluation_manifest,
        evaluation_limit=args.evaluation_limit,
        checkpoint_interval=args.checkpoint_interval,
    ):
        return
    return train_and_evaluate(
        args.base_candidate,
        profile,
        seed=args.seed,
        episodes=episodes,
        max_steps=max_steps,
        device=_device(args.device),
        output_dir=output_dir,
        training_manifest=training_manifest,
        evaluation_manifest=evaluation_manifest,
        resume=args.resume,
        evaluation_limit=args.evaluation_limit,
        checkpoint_interval=args.checkpoint_interval,
        replay_size=args.replay_size,
    )


def _select_profiles(value, explicit=None):
    if explicit:
        return tuple(dict.fromkeys(explicit))
    if value == "all":
        return tuple(ALL_SCENARIO_PROFILES)
    if value == "mission":
        return tuple(SCENARIO_PROFILES)
    if value == "unknown":
        return tuple(UNKNOWN_MAP_PROFILES)
    raise ValueError(f"unsupported profile selection={value!r}")


def _all_tensors_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_preflight_stage(name, command, *, timeout_seconds):
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
            env=environment,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        returncode = int(result.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        output = (stdout + "\n" + stderr).strip()
        returncode = 124
        timed_out = True
    return {
        "name": name,
        "command": command,
        "timeout_seconds": float(timeout_seconds),
        "runtime_seconds": float(time.perf_counter() - started),
        "returncode": returncode,
        "timed_out": timed_out,
        "passed": returncode == 0,
        "output": output,
    }


def _parse_count(output, pattern):
    matches = re.findall(pattern, output)
    return int(matches[-1]) if matches else None


def _run_acceptance_preflight():
    python = sys.executable
    repository_before = repository_source_fingerprint(PROJECT_ROOT)
    base_before = base_algorithm_source_fingerprint(PROJECT_ROOT)
    mission_before = mission_algorithm_source_fingerprint(PROJECT_ROOT)
    unknown_before = unknown_map_algorithm_source_fingerprint(PROJECT_ROOT)

    stages = [
        _run_preflight_stage(
            "compileall",
            [python, "-m", "compileall", ".", "-q"],
            timeout_seconds=120,
        ),
        _run_preflight_stage(
            "pytest_collect",
            [python, "-m", "pytest", "tests", "--collect-only", "-q"],
            timeout_seconds=240,
        ),
        _run_preflight_stage(
            "pytest",
            [python, "-m", "pytest", "tests", "-q"],
            timeout_seconds=900,
        ),
        _run_preflight_stage(
            "config_validator",
            [python, "tools/validate_ch3_config.py"],
            timeout_seconds=240,
        ),
        _run_preflight_stage(
            "pretraining_smoke",
            [python, "tools/smoke_ch3_isolation.py"],
            timeout_seconds=360,
        ),
    ]
    collected = _parse_count(
        stages[1]["output"], r"(\d+)\s+(?:test|tests)\s+collected"
    )
    passed = _parse_count(stages[2]["output"], r"(\d+)\s+passed")
    skipped = _parse_count(stages[2]["output"], r"(\d+)\s+skipped") or 0
    xfailed = _parse_count(stages[2]["output"], r"(\d+)\s+xfailed") or 0

    source_after = {
        "repository": repository_source_fingerprint(PROJECT_ROOT),
        "base": base_algorithm_source_fingerprint(PROJECT_ROOT),
        "mission": mission_algorithm_source_fingerprint(PROJECT_ROOT),
        "unknown": unknown_map_algorithm_source_fingerprint(PROJECT_ROOT),
    }
    source_before = {
        "repository": repository_before,
        "base": base_before,
        "mission": mission_before,
        "unknown": unknown_before,
    }
    source_unchanged = source_before == source_after
    counts_valid = (
        collected is not None
        and collected > 0
        and passed == collected
        and skipped == 0
        and xfailed == 0
    )
    return {
        "passed": (
            all(stage["passed"] for stage in stages)
            and counts_valid
            and source_unchanged
        ),
        "stages": stages,
        "pytest_collected_count": collected,
        "pytest_passed_count": passed,
        "pytest_skipped_count": skipped,
        "pytest_xfailed_count": xfailed,
        "pytest_counts_verified": counts_valid,
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_unchanged,
    }


def _write_acceptance_report(report):
    output = CH3_ROOT / "validation" / "ch3_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CH3 acceptance] all_passed={report['all_passed']} "
        f"passed={report.get('passed_profile_count', 0)}/8 "
        f"output={output}"
    )
    return output


def _acceptance(args):
    profiles = _select_profiles(args.profiles, args.explicit_profiles)
    if set(profiles) != set(ALL_SCENARIO_PROFILES):
        raise ValueError("final acceptance requires --profiles all")

    preflight = _run_acceptance_preflight()
    if not preflight["passed"]:
        report = {
            "all_passed": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "passed_profile_count": 0,
            "failed_profile_count": len(profiles),
            "profiles": {},
            "preflight": preflight,
            "failures": ["preflight"],
        }
        _write_acceptance_report(report)
        return 1

    gate_root = CH3_ROOT / "acceptance"
    if gate_root.exists():
        if not args.restart:
            raise FileExistsError(
                f"{gate_root} exists; pass --restart to replace bounded "
                "artifacts"
            )
        shutil.rmtree(gate_root)
    gate_root.mkdir(parents=True)

    manifests = write_scenario_manifests(
        kind="smoke",
        profiles=profiles,
        output_root=gate_root / "manifests",
    )
    runs_root = gate_root / "runs"
    profile_results = {}
    resume_profiles = {
        "S10_MOVING_CLEAR",
        "M10_MOVING_UNKNOWN_SINGLE",
    }

    for profile in profiles:
        expected_protocol = _artifact_protocol(profile)
        result = {
            "passed": False,
            "protocol": expected_protocol,
            "resumed": profile in resume_profiles,
            "errors": [],
        }
        try:
            local_args = argparse.Namespace(**vars(args))
            local_args.scenario_profile = profile
            local_args.output_dir = runs_root
            local_args.training_manifest = None
            local_args.evaluation_manifest = None
            local_args.manifest = None
            local_args.resume = False
            train_manifest = manifests[f"smoke_train_{profile}"]
            eval_manifest = manifests[f"smoke_validation_{profile}"]

            if profile in resume_profiles:
                local_args.episodes = int(args.resume_split)
                _train(
                    local_args,
                    profile=profile,
                    training_manifest=train_manifest,
                    evaluation_manifest=eval_manifest,
                    output_dir=runs_root,
                )
                local_args.resume = True

            local_args.episodes = int(args.episodes)
            summary, _, evaluation_rows = _train(
                local_args,
                profile=profile,
                training_manifest=train_manifest,
                evaluation_manifest=eval_manifest,
                output_dir=runs_root,
            )

            run_dir = (
                runs_root
                / args.base_candidate
                / profile
                / f"seed_{int(args.seed)}"
            )
            checkpoint = run_dir / "model_final.pt"
            resume_state_path = run_dir / "resume_state.pt"
            episode_csv = run_dir / "episode_metrics.csv"
            evaluation_csv = run_dir / "evaluation_metrics.csv"
            expected_intermediate = [
                run_dir / f"checkpoint_ep{episode:06d}.pt"
                for episode in range(
                    int(args.checkpoint_interval),
                    int(args.episodes) + 1,
                    int(args.checkpoint_interval),
                )
            ]

            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            metadata = payload.get("metadata", {})
            resume_state = torch.load(
                resume_state_path, map_location="cpu", weights_only=False
            )
            evaluation_csv_rows = _read_csv_rows(evaluation_csv)
            training_ids = set(summary.get("training_scenario_ids", []))
            evaluation_ids = set(
                summary.get("evaluation_scenario_ids", [])
            )

            current_base = base_algorithm_source_fingerprint(PROJECT_ROOT)
            current_mission = mission_algorithm_source_fingerprint(
                PROJECT_ROOT
            )
            current_unknown = unknown_map_algorithm_source_fingerprint(
                PROJECT_ROOT
            )
            source_ok = (
                summary.get("base_algorithm_source_fingerprint")
                == current_base
                and metadata.get("base_algorithm_source_fingerprint")
                == current_base
                and summary.get("mission_algorithm_source_fingerprint")
                == current_mission
                and metadata.get("mission_algorithm_source_fingerprint")
                == current_mission
            )
            if profile in UNKNOWN_MAP_PROFILES:
                source_ok = (
                    source_ok
                    and summary.get(
                        "unknown_map_algorithm_source_fingerprint"
                    )
                    == current_unknown
                    and metadata.get(
                        "unknown_map_algorithm_source_fingerprint"
                    )
                    == current_unknown
                )

            checks = {
                "episode_csv_contiguous": _csv_is_contiguous(
                    episode_csv, args.episodes
                ),
                "evaluation_csv": (
                    len(evaluation_csv_rows)
                    == int(args.evaluation_limit)
                    and len(evaluation_rows)
                    == int(args.evaluation_limit)
                ),
                "intermediate_checkpoints": (
                    bool(expected_intermediate)
                    and all(path.is_file() for path in expected_intermediate)
                ),
                "final_checkpoint": checkpoint.is_file(),
                "checkpoint_sha256": (
                    summary.get("checkpoint_sha256")
                    == file_sha256(checkpoint)
                ),
                "resume_state": resume_state_path.is_file(),
                "summary": (run_dir / "training_summary.json").is_file(),
                "finite_checkpoint_tensors": _all_tensors_finite(payload),
                "finite_resume_tensors": _all_tensors_finite(resume_state),
                "observation_dims": (
                    summary.get("observation_dims") == [28] * 4
                    and metadata.get("observation_dims") == [28] * 4
                ),
                "action_dims": (
                    summary.get("action_dims") == [3] * 4
                    and metadata.get("action_dims") == [3] * 4
                ),
                "protocol": (
                    summary.get("protocol") == expected_protocol
                    and metadata.get("protocol") == expected_protocol
                    and resume_state.get("protocol") == expected_protocol
                ),
                "config_hash": (
                    summary.get("algorithm_config_hash")
                    == metadata.get("algorithm_config_hash")
                    == resume_state.get("algorithm_config_hash")
                    and summary.get("evaluation_config_hash")
                    == metadata.get("evaluation_config_hash")
                    == resume_state.get("evaluation_config_hash")
                ),
                "checkpoint_episode": (
                    int(metadata.get("checkpoint_episode", -1))
                    == int(args.episodes)
                ),
                "resume_episode": (
                    int(resume_state.get("episode", -1))
                    == int(args.episodes)
                ),
                "resume_path_exercised": (
                    profile not in resume_profiles
                    or isinstance(summary.get("resume_audit"), dict)
                ),
                "manifest_isolation": not training_ids & evaluation_ids,
                "source_fingerprints": source_ok,
            }
            result.update(
                {
                    "checks": checks,
                    "passed": all(checks.values()),
                    "run_directory": str(run_dir),
                    "algorithm_config_hash": summary.get(
                        "algorithm_config_hash"
                    ),
                    "evaluation_config_hash": summary.get(
                        "evaluation_config_hash"
                    ),
                    "intermediate_checkpoints": [
                        str(path) for path in expected_intermediate
                    ],
                }
            )
        except Exception as exc:
            result["errors"].append(repr(exc))
        profile_results[profile] = result

    audit_report = audit_runs(runs_root)
    audit_passed = (
        audit_report.get("all_verified") is True
        and audit_report.get("run_count") == len(profiles)
        and audit_report.get("verified_count") == len(profiles)
    )
    validation_report = validate(profiles)
    passed_count = sum(
        bool(item["passed"]) for item in profile_results.values()
    )
    failures = [
        profile
        for profile, item in profile_results.items()
        if not item["passed"]
    ]
    report = {
        "all_passed": (
            preflight["passed"]
            and passed_count == len(profiles)
            and audit_passed
            and validation_report["passed"]
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "passed_profile_count": passed_count,
        "failed_profile_count": len(profiles) - passed_count,
        "profiles": profile_results,
        "resume": {
            profile: profile_results[profile]["passed"]
            for profile in sorted(resume_profiles)
        },
        "checkpoint_audit": {
            profile: profile_results[profile].get("checks", {})
            for profile in profiles
        },
        "preflight": preflight,
        "provenance_audit": audit_report,
        "validator": {
            "passed": validation_report["passed"],
            "failed_checks": validation_report["failed_checks"],
        },
        "config_hashes": {
            profile: {
                "algorithm_config_hash": profile_results[profile].get(
                    "algorithm_config_hash"
                ),
                "evaluation_config_hash": profile_results[profile].get(
                    "evaluation_config_hash"
                ),
            }
            for profile in profiles
        },
        "source_fingerprints": {
            "base": base_algorithm_source_fingerprint(PROJECT_ROOT),
            "mission": mission_algorithm_source_fingerprint(PROJECT_ROOT),
            "unknown": unknown_map_algorithm_source_fingerprint(
                PROJECT_ROOT
            ),
            "repository": repository_source_fingerprint(PROJECT_ROOT),
        },
        "parameters": {
            "base_candidate": args.base_candidate,
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "resume_split": int(args.resume_split),
            "max_steps": int(args.max_steps),
            "replay_size": int(args.replay_size),
            "checkpoint_interval": int(args.checkpoint_interval),
            "evaluation_limit": int(args.evaluation_limit),
            "device": _device(args.device),
        },
        "failures": failures,
    }
    _write_acceptance_report(report)
    return 0 if report["all_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "generate",
            "validate",
            "train",
            "acceptance",
            "provenance-audit",
        ),
    )
    parser.add_argument(
        "--base-candidate",
        choices=CH3_EFFICIENCY_V3_SCREEN_METHODS,
        default="ch3_v3_full_reference",
    )
    parser.add_argument(
        "--scenario-profile", choices=ALL_SCENARIO_PROFILES
    )
    parser.add_argument(
        "--profiles",
        choices=("all", "mission", "unknown"),
        default="all",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=ALL_SCENARIO_PROFILES,
        dest="explicit_profiles",
    )
    parser.add_argument(
        "--kind",
        choices=("all", "train", "validation", "smoke"),
        default="all",
    )
    parser.add_argument(
        "--manifest", type=Path, help="deprecated phase-specific alias"
    )
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-limit", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-split", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    args = parser.parse_args(argv)

    if args.evaluation_limit is not None and args.evaluation_limit < 0:
        parser.error("--evaluation-limit cannot be negative")
    if args.episodes > 4 and not args.allow_long_run:
        parser.error("episodes > 4 requires --allow-long-run")
    if args.resume_split < 1 or args.resume_split >= args.episodes:
        if args.phase == "acceptance":
            parser.error("--resume-split must be within [1, episodes)")

    if args.phase == "generate":
        write_scenario_manifests(
            kind=args.kind,
            profiles=_select_profiles(
                args.profiles, args.explicit_profiles
            ),
        )
    elif args.phase == "validate":
        validate_args = ["--profiles", args.profiles]
        for profile in args.explicit_profiles or ():
            validate_args.extend(["--profile", profile])
        return validate_main(validate_args)
    elif args.phase == "provenance-audit":
        audit_args = []
        if args.output_dir is not None:
            audit_args.extend(["--runs-root", str(args.output_dir)])
        return audit_main(audit_args)
    elif args.phase == "train":
        if args.scenario_profile is None:
            parser.error("--scenario-profile is required for train")
        _train(args)
    else:
        if (
            args.episodes != 3
            or args.max_steps != 20
            or args.replay_size != 32
            or args.checkpoint_interval != 1
            or args.evaluation_limit != 1
        ):
            parser.error(
                "acceptance requires episodes=3, max_steps=20, "
                "replay-size=32, checkpoint-interval=1, "
                "evaluation-limit=1"
            )
        return _acceptance(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
