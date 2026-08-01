"""Standalone evaluator for the pure Chapter-3 experiment methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
    assert_ch3_method,
)
from train import (
    CHECKPOINT_SCHEMA_VERSION,
    CH3Runtime,
    PROJECT_ROOT,
    _algorithm_config_hash,
    _build_train_env,
    _config_hash,
    _evaluation_config_hash,
    _resolve_device,
    _run_episode,
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    CH3_PROTOCOL_CONFIGS,
    get_ch3_method_config,
    load_scenario_manifest,
    set_ch3_determinism,
    summarize_evaluation_rows,
)
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_fingerprint,
    file_sha256,
    repository_source_fingerprint,
    runtime_versions,
)


DEFAULT_V1_RESULT_DIR = PROJECT_ROOT / "data" / "chapter3_final" / "evaluations"
DEFAULT_V2_RESULT_DIR = PROJECT_ROOT / "data" / "chapter3_efficiency_v2" / "validation"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(rows: list[Dict[str, Any]], key: str, *, found_only: bool = False) -> float:
    values = []
    for row in rows:
        if found_only and float(row.get("found", 0.0)) < 0.5:
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _validate_checkpoint_metadata(
    metadata: dict,
    *,
    method: str,
    config: dict,
    env,
    max_steps: int,
    allow_source_mismatch: bool = False,
    allow_legacy_provenance: bool = False,
    protocol: str = CH3_PILOT_V1,
) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError(
            "checkpoint has no Chapter-3 identity metadata; retrain it with the current train.py"
        )
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "residual_maddpg_twin_critic_v1",
        "method": method,
        "run_type": str(config["run_type"]),
        "max_steps": int(max_steps),
        "reward_profile": config.get("reward_profile", CH3_PILOT_V1),
        "residual_action_reg": float(config.get("residual_action_reg", 1e-2)),
        "observation_dims": [
            int(env.observation_space[f"agent_{i}"].shape[0]) for i in range(env.num_agents)
        ],
        "action_dims": [
            int(env.action_space[f"agent_{i}"].shape[0]) for i in range(env.num_agents)
        ],
    }
    # Validate the stored training identity for self-consistency, then compare
    # only evaluation-relevant configuration with the current code.  Replay
    # capacity and checkpoint cadence affect training/resume but must not make a
    # completed model impossible to evaluate.
    checkpoint_config = metadata.get("config")
    if metadata.get("algorithm_config_hash") is not None:
        if not isinstance(checkpoint_config, dict):
            expected["checkpoint_config"] = "dictionary"
        elif _algorithm_config_hash(checkpoint_config) != metadata.get("algorithm_config_hash"):
            expected["algorithm_config_hash_self_consistent"] = True
        elif _evaluation_config_hash(checkpoint_config) != metadata.get("evaluation_config_hash"):
            expected["evaluation_config_hash_self_consistent"] = True
        expected["evaluation_config_hash"] = _evaluation_config_hash(config)
    elif protocol == CH3_PILOT_V1:
        expected["config_hash"] = _config_hash(config)
    else:
        expected["evaluation_config_hash"] = _evaluation_config_hash(config)
    mismatches = {}
    for key, value in expected.items():
        if key == "checkpoint_config":
            actual = metadata.get("config")
        elif key == "algorithm_config_hash_self_consistent":
            actual = False
        elif key == "evaluation_config_hash_self_consistent":
            actual = False
        else:
            actual = metadata.get(key)
        if actual != value:
            mismatches[key] = {"expected": value, "checkpoint": actual}
    checkpoint_protocol = metadata.get("protocol", CH3_PILOT_V1)
    if checkpoint_protocol != protocol:
        mismatches["protocol"] = {
            "expected": protocol,
            "checkpoint": checkpoint_protocol,
        }
    for key in ("seed", "requested_episodes", "checkpoint_episode", "global_step", "update_step"):
        if not isinstance(metadata.get(key), int):
            mismatches[key] = {
                "expected": "integer",
                "checkpoint": metadata.get(key),
            }
    if metadata.get("checkpoint_kind") not in {"periodic", "final"}:
        mismatches["checkpoint_kind"] = {
            "expected": "periodic or final",
            "checkpoint": metadata.get("checkpoint_kind"),
        }
    current_algorithm = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository = repository_source_fingerprint(PROJECT_ROOT)
    checkpoint_algorithm = metadata.get("algorithm_source_fingerprint")
    checkpoint_repository = metadata.get("repository_source_fingerprint")
    legacy = (
        metadata.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION
        or checkpoint_algorithm is None
        or checkpoint_repository is None
    )
    legacy_allowed = bool(allow_legacy_provenance or allow_source_mismatch)
    if legacy:
        if not legacy_allowed:
            mismatches["legacy_provenance"] = {
                "expected": "verified algorithm provenance",
                "checkpoint": "legacy provenance cannot establish algorithm identity",
            }
    elif checkpoint_algorithm != current_algorithm:
        mismatches["algorithm_source_fingerprint"] = {
            "expected": current_algorithm,
            "checkpoint": checkpoint_algorithm,
        }
    if mismatches:
        raise ValueError(f"checkpoint identity mismatch for {method}: {mismatches}")
    return {
        "algorithm_source_fingerprint": current_algorithm,
        "repository_source_fingerprint": current_repository,
        "checkpoint_algorithm_source_fingerprint": checkpoint_algorithm,
        "checkpoint_repository_source_fingerprint": checkpoint_repository,
        "algorithm_source_matches_checkpoint": (
            None if legacy else checkpoint_algorithm == current_algorithm
        ),
        "repository_source_matches_checkpoint": (
            None if checkpoint_repository is None
            else checkpoint_repository == current_repository
        ),
        "legacy_provenance_unverified": legacy,
    }


def _build_evaluation_runtime(
    method: str,
    *,
    model_path: Path | None,
    seed: int,
    max_steps: int,
    device: str | torch.device,
    allow_source_mismatch: bool = False,
    allow_legacy_provenance: bool = False,
    protocol: str = CH3_PILOT_V1,
) -> CH3Runtime:
    del seed  # evaluation seed affects scenarios, not checkpoint construction
    method = assert_ch3_method(method)
    config = get_ch3_method_config(method, protocol=protocol)
    config["max_steps"] = int(max_steps)
    train_device = _resolve_device(device)
    env, _ = _build_train_env(torch.device("cpu"), int(max_steps), config)
    run_type = str(config["run_type"])

    if method in CONTROLLER_ONLY_METHODS:
        if model_path is not None:
            raise ValueError(f"{method} is controller-only and must not receive --model-path")
        return CH3Runtime(method, config, env, None, None, run_type, train_device)

    if model_path is None:
        raise ValueError(f"{method} is learned and requires --model-path")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {model_path}")
    maddpg = MADDPG.init_from_save(str(model_path), device=train_device)
    audit = _validate_checkpoint_metadata(
        maddpg.checkpoint_metadata,
        method=method,
        config=config,
        env=env,
        max_steps=max_steps,
        allow_source_mismatch=allow_source_mismatch,
        allow_legacy_provenance=allow_legacy_provenance,
        protocol=protocol,
    )
    maddpg.checkpoint_metadata = dict(maddpg.checkpoint_metadata)
    maddpg.checkpoint_metadata["evaluation_source_audit"] = audit
    maddpg.prep_rollouts(device=train_device)
    return CH3Runtime(method, config, env, maddpg, None, run_type, train_device)


def evaluate(
    method: str,
    *,
    model_path: Path | None,
    episodes: int,
    seed: int,
    max_steps: int,
    device: str | torch.device,
    result_dir: Path | None,
    scenario_manifest: Path | None = None,
    allow_source_mismatch: bool = False,
    allow_legacy_provenance: bool = False,
    protocol: str = CH3_PILOT_V1,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    set_ch3_determinism(seed)
    manifest_id = manifest_sha256 = None
    manifest_role = None
    if scenario_manifest is None:
        if int(episodes) <= 0:
            raise ValueError("episodes must be positive when no scenario manifest is supplied")
        scenarios = [None] * int(episodes)
    else:
        manifest, loaded = load_scenario_manifest(scenario_manifest)
        manifest_id = manifest.get("manifest_id")
        manifest_sha256 = manifest["manifest_sha256"]
        manifest_role = manifest.get("scenario_role")
        scenarios = loaded if episodes <= 0 else loaded[: int(episodes)]
    if not scenarios:
        raise ValueError("evaluation requires at least one episode or scenario")
    legacy_debug = bool(allow_legacy_provenance or allow_source_mismatch)
    if legacy_debug and manifest_role in {"test", "obstacle"}:
        raise ValueError(
            "legacy provenance debug mode is forbidden for test or obstacle reports"
        )
    runtime = _build_evaluation_runtime(
        method,
        model_path=model_path,
        seed=seed,
        max_steps=max_steps,
        device=device,
        allow_source_mismatch=allow_source_mismatch,
        allow_legacy_provenance=allow_legacy_provenance,
        protocol=protocol,
    )

    rows: list[Dict[str, Any]] = []
    for episode_index, scenario in enumerate(scenarios, start=1):
        row, _, _ = _run_episode(runtime, explore=False, scenario=scenario)
        row.update(method=method, seed=int(seed), episode=episode_index)
        rows.append(row)
        print(
            f"episode {episode_index}/{len(scenarios)} "
            f"found={row['found']} success={row['success']} steps={row['steps']}"
        )

    checkpoint_metadata = None if runtime.maddpg is None else runtime.maddpg.checkpoint_metadata
    current_algorithm_source = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository_source = repository_source_fingerprint(PROJECT_ROOT)
    source_audit = (
        {
            "algorithm_source_fingerprint": current_algorithm_source,
            "repository_source_fingerprint": current_repository_source,
            "checkpoint_algorithm_source_fingerprint": None,
            "checkpoint_repository_source_fingerprint": None,
            "algorithm_source_matches_checkpoint": True,
            "repository_source_matches_checkpoint": True,
            "legacy_provenance_unverified": False,
        }
        if runtime.maddpg is None
        else checkpoint_metadata["evaluation_source_audit"]
    )
    checkpoint_sha256 = None if model_path is None else file_sha256(model_path)
    summary = {
        "method": method,
        "protocol": protocol,
        "run_type": runtime.run_type,
        "eval_seed": int(seed),
        "episodes": len(rows),
        "max_steps": int(max_steps),
        "resolved_device": str(runtime.train_device),
        "model_path": "N/A" if model_path is None else str(model_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": checkpoint_metadata,
        "scenario_manifest": None if scenario_manifest is None else str(scenario_manifest),
        "scenario_manifest_id": manifest_id,
        "scenario_manifest_sha256": manifest_sha256,
        "scenario_ids": [row["scenario_id"] for row in rows],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **source_audit,
        "source_fingerprint": current_repository_source,
        "source_fingerprint_semantics": "legacy_repository_alias",
        "runtime_versions": runtime_versions(),
        "found_rate": _finite_mean(rows, "found"),
        "success_rate": _finite_mean(rows, "success"),
        "collision_rate": _finite_mean(rows, "collision"),
        "mean_reward": _finite_mean(rows, "reward"),
        "mean_energy_cost": _finite_mean(rows, "energy_cost"),
        "mean_found_step": _finite_mean(rows, "found_step", found_only=True),
        "mean_exec_delay": _finite_mean(rows, "exec_delay", found_only=True),
        "mean_handoff_delay": _finite_mean(rows, "handoff_delay", found_only=True),
        "communication_model": runtime.env.communication_model_id,
    }
    summary.update(summarize_evaluation_rows(rows))

    if runtime.maddpg is None:
        checkpoint_scope = "controller_only"
    else:
        metadata = runtime.maddpg.checkpoint_metadata
        checkpoint_scope = (
            f"train_seed_{metadata['seed']}__"
            f"ep_{metadata['checkpoint_episode']:06d}__"
            f"sha_{checkpoint_sha256[:12]}"
        )
    if result_dir is None:
        result_dir = (
            DEFAULT_V2_RESULT_DIR if protocol == CH3_EFFICIENCY_V2
            else DEFAULT_V1_RESULT_DIR
        )
    output_dir = Path(result_dir) / method / checkpoint_scope / f"eval_seed_{int(seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["result_directory"] = str(output_dir)
    _write_csv(output_dir / "evaluation_metrics.csv", rows)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(
            _json_safe(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return summary, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pure Chapter-3 checkpoint evaluation")
    parser.add_argument("--method", required=True, choices=ACTIVE_CH3_FINAL_EXPERIMENT_MODES)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--protocol", choices=tuple(CH3_PROTOCOL_CONFIGS), default=CH3_PILOT_V1)
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="deprecated alias for --allow-legacy-provenance (debug only)",
    )
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help="debug-only evaluation of legacy provenance; never bypasses algorithm mismatch",
    )
    args = parser.parse_args(argv)

    summary, _ = evaluate(
        args.method,
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
        result_dir=args.result_dir,
        scenario_manifest=args.scenario_manifest,
        allow_source_mismatch=args.allow_source_mismatch,
        allow_legacy_provenance=args.allow_legacy_provenance,
        protocol=args.protocol,
    )
    print(
        json.dumps(
            _json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
