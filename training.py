"""Unified Chapter-3 training, evaluation, checkpoint, and resume workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ch3_constants import (
    CH3_MISSION_V1,
    CH3_UNKNOWN_MAP_V1,
    MISSION_SCENARIO_PROFILES,
    UNKNOWN_MAP_PROFILES,
)
from metrics import augment_episode_metrics
from runtime import build_runtime
from train import (
    _algorithm_config_hash,
    _config_hash,
    _evaluation_config_hash,
    _json_safe,
    _restore_rng_state,
    _rng_state_dict,
    _run_episode,
    load_scenario_manifest,
    summarize_evaluation_rows,
)
from utils.provenance import (
    assert_mission_algorithm_source_unchanged,
    assert_unknown_map_source_unchanged,
    capture_mission_provenance_snapshot,
    capture_unknown_map_provenance_snapshot,
    file_sha256,
    runtime_versions,
)


@dataclass(frozen=True)
class TrainingProtocolSpec:
    artifact_protocol: str
    profiles: tuple[str, ...]
    provenance_snapshot: Callable
    source_assertion: Callable
    require_moving_target: bool


_MISSION_SPEC = TrainingProtocolSpec(
    artifact_protocol=CH3_MISSION_V1,
    profiles=tuple(MISSION_SCENARIO_PROFILES),
    provenance_snapshot=capture_mission_provenance_snapshot,
    source_assertion=assert_mission_algorithm_source_unchanged,
    require_moving_target=False,
)
_UNKNOWN_SPEC = TrainingProtocolSpec(
    artifact_protocol=CH3_UNKNOWN_MAP_V1,
    profiles=tuple(UNKNOWN_MAP_PROFILES),
    provenance_snapshot=capture_unknown_map_provenance_snapshot,
    source_assertion=assert_unknown_map_source_unchanged,
    require_moving_target=True,
)


def _protocol_spec(scenario_profile):
    for spec in (_MISSION_SPEC, _UNKNOWN_SPEC):
        if scenario_profile in spec.profiles:
            return spec
    raise ValueError(f"unsupported Chapter-3 scenario profile={scenario_profile!r}")


def _write_csv(path, rows):
    rows = [
        {
            key: (
                None
                if isinstance(value, (float, np.floating))
                and not np.isfinite(float(value))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_episode(
    runtime,
    *,
    scenario,
    explore=False,
    train_updates=False,
    global_step=0,
    update_step=0,
):
    row, global_step, update_step = _run_episode(
        runtime,
        explore=explore,
        scenario=scenario,
        train_updates=train_updates,
        global_step=global_step,
        update_step=update_step,
    )
    return (
        augment_episode_metrics(row, runtime.env, scenario),
        global_step,
        update_step,
    )


_IDENTITY_ONLY_SCENARIO_FIELDS = frozenset({
    "scenario_id",
    "pair_group_id",
    "scenario_seed",
    "planner_seed",
    "target_motion_seed",
    "scenario_role",
    "scenario_split",
})


def _scenario_content_hash(scenario):
    payload = {
        key: value
        for key, value in dict(scenario).items()
        if key not in _IDENTITY_ONLY_SCENARIO_FIELDS
    }
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _present_values(scenarios, key, *, stringify=False):
    values = set()
    for scenario in scenarios:
        value = scenario.get(key)
        if value is None or value == "":
            continue
        values.add(str(value) if stringify else value)
    return values


def validate_dataset_isolation(
    training_manifest,
    training_scenarios,
    evaluation_manifest,
    evaluation_scenarios,
):
    """Validate physical and identity isolation between dataset splits."""

    if evaluation_manifest is None:
        if evaluation_scenarios:
            raise RuntimeError("training/evaluation scenario leakage detected")
        return {
            "scenario_id_overlap_count": 0,
            "scenario_seed_overlap_count": 0,
            "pair_group_overlap_count": 0,
            "physical_content_overlap_count": 0,
        }

    training_scenarios = list(training_scenarios)
    evaluation_scenarios = list(evaluation_scenarios)
    training_ids = _present_values(
        training_scenarios, "scenario_id", stringify=True
    )
    evaluation_ids = _present_values(
        evaluation_scenarios, "scenario_id", stringify=True
    )
    training_seeds = _present_values(training_scenarios, "scenario_seed")
    evaluation_seeds = _present_values(evaluation_scenarios, "scenario_seed")
    training_pairs = _present_values(
        training_scenarios, "pair_group_id", stringify=True
    )
    evaluation_pairs = _present_values(
        evaluation_scenarios, "pair_group_id", stringify=True
    )
    training_content = {
        _scenario_content_hash(item) for item in training_scenarios
    }
    evaluation_content = {
        _scenario_content_hash(item) for item in evaluation_scenarios
    }
    diagnostics = {
        "scenario_id_overlap_count": len(training_ids & evaluation_ids),
        "scenario_seed_overlap_count": len(training_seeds & evaluation_seeds),
        "pair_group_overlap_count": len(training_pairs & evaluation_pairs),
        "physical_content_overlap_count": len(
            training_content & evaluation_content
        ),
    }
    same_manifest_hash = (
        training_manifest.get("manifest_sha256") is not None
        and training_manifest.get("manifest_sha256")
        == evaluation_manifest.get("manifest_sha256")
    )
    same_generator_seed = (
        training_manifest.get("generator_seed") is not None
        and training_manifest.get("generator_seed")
        == evaluation_manifest.get("generator_seed")
    )
    if any(diagnostics.values()) or same_manifest_hash or same_generator_seed:
        raise RuntimeError("training/evaluation scenario leakage detected")
    return diagnostics


_validate_dataset_isolation = validate_dataset_isolation


def _load_manifest(path, *, profile, allowed_roles, label, spec):
    manifest, scenarios = load_scenario_manifest(path)
    if manifest.get("protocol") != spec.artifact_protocol:
        raise ValueError(
            f"{label} manifest must use {spec.artifact_protocol}"
        )
    if manifest.get("scenario_profile") != profile:
        raise ValueError(f"{label} manifest scenario profile mismatch")
    role = manifest.get("scenario_role")
    split = manifest.get("scenario_split")
    if role not in allowed_roles or split != role:
        raise ValueError(
            f"{label} manifest role must be one of {sorted(allowed_roles)}"
        )
    if spec.require_moving_target and any(
        scenario.get("target_motion_mode") != "constant_velocity_reflect_v1"
        for scenario in scenarios
    ):
        raise ValueError("every unknown-map scenario must use a moving target")
    return manifest, scenarios


def _identity(
    runtime,
    *,
    base_candidate,
    scenario_profile,
    seed,
    max_steps,
    manifest,
    scenario_ids,
    obstacle_layout_identity,
    snapshot,
    spec,
):
    common = {
        "protocol": spec.artifact_protocol,
        "base_candidate": base_candidate,
        "scenario_profile": scenario_profile,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "replay_size": int(runtime.config["replay_size"]),
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "evaluation_config_hash": _evaluation_config_hash(runtime.config),
        "training_manifest_id": manifest["manifest_id"],
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_scenario_ids": list(scenario_ids),
    }
    if spec is _UNKNOWN_SPEC:
        common.update({
            "base_runtime_protocol": runtime.config["protocol"],
            "unknown_map_algorithm_source_fingerprint": snapshot[
                "unknown_map_algorithm_source_fingerprint"
            ],
            "base_algorithm_source_fingerprint": snapshot[
                "base_algorithm_source_fingerprint"
            ],
            "mission_algorithm_source_fingerprint": snapshot[
                "mission_algorithm_source_fingerprint"
            ],
            "obstacle_knowledge_mode": runtime.config[
                "obstacle_knowledge_mode"
            ],
            "planner_mode": runtime.config["planner_mode"],
            "target_motion_known": True,
        })
    else:
        common.update({
            "mission_algorithm_source_fingerprint": snapshot[
                "mission_algorithm_source_fingerprint"
            ],
            "target_motion_mode": runtime.config["target_motion_mode"],
            "obstacle_layout_identity": obstacle_layout_identity,
            "scenario_manifest_id": manifest["manifest_id"],
            "scenario_manifest_sha256": manifest["manifest_sha256"],
        })
    return common


def _save_resume(
    path,
    runtime,
    *,
    episode,
    global_step,
    update_step,
    sigma,
    identity,
    snapshot,
):
    state = {
        "schema_version": 1,
        "episode": int(episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "sigma": float(sigma),
        **identity,
        **snapshot,
        "maddpg": runtime.maddpg.training_state_dict(),
        "replay_buffer": runtime.replay_buffer.state_dict(),
        **_rng_state_dict(),
    }
    torch.save(state, path)


def _load_resume(path, runtime, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in identity.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume identity mismatch: {mismatches}")
    if state.get("protocol") != identity["protocol"]:
        raise ValueError(
            f"resume state is not a {identity['protocol']} artifact"
        )
    runtime.maddpg.load_training_state_dict(state["maddpg"])
    runtime.replay_buffer.load_state_dict(state["replay_buffer"])
    runtime.maddpg.prep_rollouts(device=runtime.train_device)
    _restore_rng_state(state)
    return state


def _checkpoint_metadata(
    runtime,
    *,
    base_candidate,
    scenario_profile,
    seed,
    episodes,
    max_steps,
    manifest,
    snapshot,
    episode,
    obstacle_layout_identity,
    spec,
):
    env = runtime.env
    scenario_ids = [
        str(scenario["scenario_id"]) for scenario in manifest["scenarios"]
    ]
    common = {
        "schema_version": 1,
        "protocol": spec.artifact_protocol,
        "base_candidate": base_candidate,
        "scenario_profile": scenario_profile,
        "seed": int(seed),
        "episodes": int(episodes),
        "checkpoint_episode": int(episode),
        "max_steps": int(max_steps),
        "replay_size": int(runtime.config["replay_size"]),
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "evaluation_config_hash": _evaluation_config_hash(runtime.config),
        "training_manifest_id": manifest["manifest_id"],
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_scenario_ids": scenario_ids,
        "target_motion_mode": env.target_motion_mode,
        "handoff_payload_schema": env.handoff_payload_schema,
        "handoff_event_delay_steps": 1,
        "travel_cost_mode": env.travel_cost_mode,
        "navigation_path_mode": env.navigation_path_mode,
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
        "config": _json_safe(runtime.config),
        **snapshot,
    }
    if spec is _UNKNOWN_SPEC:
        common.update({
            "base_runtime_protocol": runtime.config["protocol"],
            "target_motion_known": True,
            "obstacle_knowledge_mode": env.obstacle_knowledge_mode,
            "planner_mode": env.planner_mode,
            "unknown_map_schema": env.unknown_map_schema,
            "target_belief_schema": env.target_belief_schema,
            "map_sharing_mode": env.map_sharing_mode,
        })
    else:
        common.update({
            "obstacle_layout_id": env.obstacle_layout_id,
            "obstacle_layout_identity": obstacle_layout_identity,
            "handoff_delivery_phase": "pre_transition",
            "handoff_physical_age_at_delivery_steps": 0,
            "scenario_manifest_id": manifest["manifest_id"],
            "scenario_manifest_sha256": manifest["manifest_sha256"],
            "scenario_ids": scenario_ids,
            "scenario_manifest_semantics":
                "legacy_alias_for_training_manifest",
        })
    return common


def _summary(
    *,
    runtime,
    spec,
    base_candidate,
    scenario_profile,
    seed,
    episodes,
    max_steps,
    method_dir,
    checkpoint,
    resume_path,
    started,
    global_step,
    update_step,
    training_manifest_path,
    manifest,
    training_scenarios,
    training_scenario_ids,
    evaluation_manifest_path,
    evaluation_manifest_data,
    evaluation_scenarios,
    evaluation_limit,
    selected,
    obstacle_layout_identity,
    isolation,
    snapshot,
    resume_audit,
    metadata,
    evaluation_rows,
):
    summary = {
        "protocol": spec.artifact_protocol,
        "base_candidate": base_candidate,
        "method": base_candidate,
        "scenario_profile": scenario_profile,
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "replay_size": int(runtime.config["replay_size"]),
        "run_directory": str(method_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "resume_state_path": str(resume_path),
        "training_time": float(time.perf_counter() - started),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "resolved_config": _json_safe(runtime.config),
        "config_hash": _config_hash(runtime.config),
        "algorithm_config_hash": _algorithm_config_hash(runtime.config),
        "evaluation_config_hash": _evaluation_config_hash(runtime.config),
        "observation_dims": [28] * 4,
        "action_dims": [3] * 4,
        "training_manifest": str(training_manifest_path),
        "training_manifest_id": manifest["manifest_id"],
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_scenario_ids": training_scenario_ids,
        "training_scenario_count": len(training_scenarios),
        "evaluation_manifest": (
            None
            if evaluation_manifest_path is None
            else str(evaluation_manifest_path)
        ),
        "evaluation_manifest_id": (
            None
            if evaluation_manifest_data is None
            else evaluation_manifest_data["manifest_id"]
        ),
        "evaluation_manifest_sha256": (
            None
            if evaluation_manifest_data is None
            else evaluation_manifest_data["manifest_sha256"]
        ),
        "evaluation_scenario_ids": [
            str(scenario["scenario_id"])
            for scenario in evaluation_scenarios
        ],
        "evaluation_limit": evaluation_limit,
        "evaluated_scenario_ids": [
            str(scenario["scenario_id"]) for scenario in selected
        ],
        "evaluation_count": len(selected),
        **snapshot,
        "runtime_versions": runtime_versions(),
        "resume_audit": resume_audit,
        "checkpoint_metadata": metadata,
    }
    if spec is _UNKNOWN_SPEC:
        summary.update({
            "base_runtime_protocol": runtime.config["protocol"],
            "target_motion_known": True,
            "obstacle_knowledge_mode": runtime.config[
                "obstacle_knowledge_mode"
            ],
            "planner_mode": runtime.config["planner_mode"],
            "unknown_map_schema": runtime.config["unknown_map_schema"],
            "target_belief_schema": runtime.config["target_belief_schema"],
            "dataset_isolation": isolation,
        })
    else:
        summary.update({
            "target_motion_mode": runtime.config["target_motion_mode"],
            "obstacle_layout_identity": obstacle_layout_identity,
            "evaluation_scenario_count": len(evaluation_scenarios),
            "scenario_manifest": str(training_manifest_path),
            "scenario_manifest_id": manifest["manifest_id"],
            "scenario_manifest_sha256": manifest["manifest_sha256"],
            "scenario_manifest_semantics":
                "legacy_alias_for_training_manifest",
            "scenario_ids": [
                str(scenario["scenario_id"]) for scenario in selected
            ],
            "evaluation_scenarios": len(selected),
        })
    summary.update(summarize_evaluation_rows(evaluation_rows))
    return summary


def train_and_evaluate(
    base_candidate,
    scenario_profile,
    *,
    seed,
    episodes,
    max_steps,
    device,
    output_dir,
    training_manifest=None,
    evaluation_manifest=None,
    scenario_manifest=None,
    resume=False,
    evaluation_limit=None,
    checkpoint_interval=0,
    replay_size=None,
):
    """Train and evaluate any registered S- or M-profile."""

    spec = _protocol_spec(scenario_profile)
    project_root = Path(__file__).resolve().parent
    snapshot = spec.provenance_snapshot(project_root)
    if training_manifest is not None and scenario_manifest is not None:
        raise ValueError(
            "training_manifest and deprecated scenario_manifest cannot both be set"
        )
    if training_manifest is None:
        if scenario_manifest is None:
            raise ValueError("training_manifest is required")
        if evaluation_limit != 0:
            raise ValueError(
                "deprecated scenario_manifest is training-only and requires "
                "evaluation_limit=0"
            )
        training_manifest = scenario_manifest
    if evaluation_limit is not None and int(evaluation_limit) < 0:
        raise ValueError("evaluation_limit cannot be negative")
    if evaluation_limit != 0 and evaluation_manifest is None:
        raise ValueError(
            "evaluation_manifest is required when evaluation is enabled"
        )

    training_manifest_path = Path(training_manifest)
    manifest, training_scenarios = _load_manifest(
        training_manifest_path,
        profile=scenario_profile,
        allowed_roles={"train", "smoke_train"},
        label="training",
        spec=spec,
    )
    evaluation_manifest_data = None
    evaluation_scenarios = []
    evaluation_manifest_path = None
    if evaluation_manifest is not None:
        evaluation_manifest_path = Path(evaluation_manifest)
        evaluation_manifest_data, evaluation_scenarios = _load_manifest(
            evaluation_manifest_path,
            profile=scenario_profile,
            allowed_roles={"validation", "smoke_validation"},
            label="evaluation",
            spec=spec,
        )
    isolation = validate_dataset_isolation(
        manifest,
        training_scenarios,
        evaluation_manifest_data,
        evaluation_scenarios,
    )

    runtime = build_runtime(
        base_candidate,
        scenario_profile,
        seed=seed,
        max_steps=max_steps,
        device=device,
        replay_size=replay_size,
    )
    artifact_protocol = runtime.config.get(
        "artifact_protocol", CH3_MISSION_V1
    )
    if artifact_protocol != spec.artifact_protocol:
        raise ValueError("runtime artifact protocol does not match profile")
    runtime.config["checkpoint_interval"] = max(0, int(checkpoint_interval))

    method_dir = (
        Path(output_dir)
        / base_candidate
        / scenario_profile
        / f"seed_{int(seed)}"
    )
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = method_dir / "episode_metrics.csv"
    resume_path = method_dir / "resume_state.pt"
    obstacle_layout_identity = "|".join(sorted({
        str(scenario.get("obstacle_layout_id", "none"))
        for scenario in training_scenarios
    }))
    training_scenario_ids = [
        str(scenario["scenario_id"]) for scenario in training_scenarios
    ]
    identity = _identity(
        runtime,
        base_candidate=base_candidate,
        scenario_profile=scenario_profile,
        seed=seed,
        max_steps=max_steps,
        manifest=manifest,
        scenario_ids=training_scenario_ids,
        obstacle_layout_identity=obstacle_layout_identity,
        snapshot=snapshot,
        spec=spec,
    )

    rows = []
    global_step = 0
    update_step = 0
    start_episode = 1
    sigma = float(runtime.config["initial_sigma"])
    resume_audit = None
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        state = _load_resume(resume_path, runtime, identity)
        rows = _read_csv(metrics_path)
        completed = int(state["episode"])
        requested = int(episodes)
        if requested < completed:
            raise ValueError(
                f"requested episodes={requested} is below completed episode={completed}"
            )
        episodes_in_csv = [int(row["episode"]) for row in rows]
        if (
            len(episodes_in_csv) != len(set(episodes_in_csv))
            or episodes_in_csv != list(range(1, completed + 1))
        ):
            raise ValueError("episode CSV is not contiguous with resume state")
        if not rows or int(rows[-1]["episode"]) != completed:
            raise ValueError("resume state episode does not match final CSV episode")
        if requested == completed:
            summary_path = method_dir / "training_summary.json"
            evaluation_path = method_dir / "evaluation_metrics.csv"
            if not summary_path.is_file():
                raise ValueError("completed run has no training_summary.json")
            previous_summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            requested_evaluation_sha = (
                None
                if evaluation_manifest_data is None
                else evaluation_manifest_data["manifest_sha256"]
            )
            if (
                previous_summary.get("evaluation_manifest_sha256")
                != requested_evaluation_sha
                or previous_summary.get("evaluation_limit")
                != evaluation_limit
            ):
                raise ValueError(
                    "completed run evaluation identity differs; use the "
                    "independent evaluate phase"
                )
            print(
                f"[CH3] already complete at episode={completed}; "
                "artifacts unchanged",
                flush=True,
            )
            return previous_summary, rows, _read_csv(evaluation_path)
        global_step = int(state["global_step"])
        update_step = int(state["update_step"])
        sigma = float(state["sigma"])
        start_episode = completed + 1
        if spec is _UNKNOWN_SPEC:
            resume_audit = {
                "resumed_from_episode": completed,
                "repository_source_changed_since_resume": (
                    state.get("repository_source_fingerprint")
                    != snapshot["repository_source_fingerprint"]
                ),
            }
        else:
            resume_audit = {
                "resumed_from_repository_source_fingerprint": state.get(
                    "repository_source_fingerprint"
                ),
                "current_repository_source_fingerprint": snapshot[
                    "repository_source_fingerprint"
                ],
                "repository_changed": (
                    state.get("repository_source_fingerprint")
                    != snapshot["repository_source_fingerprint"]
                ),
            }
    elif any(method_dir.iterdir()):
        raise ValueError(
            f"refusing to overwrite occupied run directory: {method_dir}"
        )

    started = time.perf_counter()
    for episode in range(start_episode, int(episodes) + 1):
        runtime.maddpg.scale_noise(sigma, multiply=False)
        scenario = training_scenarios[
            (episode - 1) % len(training_scenarios)
        ]
        row, global_step, update_step = run_episode(
            runtime,
            scenario=scenario,
            explore=True,
            train_updates=True,
            global_step=global_step,
            update_step=update_step,
        )
        row.update(
            method=base_candidate,
            base_candidate=base_candidate,
            seed=int(seed),
            episode=int(episode),
        )
        rows.append(row)
        if episode > int(runtime.config["sigma_hold_episodes"]):
            sigma = max(
                float(runtime.config["min_sigma"]),
                sigma * float(runtime.config["sigma_decay"]),
            )
        spec.source_assertion(project_root, snapshot)
        _save_resume(
            resume_path,
            runtime,
            episode=episode,
            global_step=global_step,
            update_step=update_step,
            sigma=sigma,
            identity=identity,
            snapshot=snapshot,
        )
        _write_csv(metrics_path, rows)
        if checkpoint_interval and episode % int(checkpoint_interval) == 0:
            metadata = _checkpoint_metadata(
                runtime,
                base_candidate=base_candidate,
                scenario_profile=scenario_profile,
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
                manifest=manifest,
                snapshot=snapshot,
                episode=episode,
                obstacle_layout_identity=obstacle_layout_identity,
                spec=spec,
            )
            runtime.maddpg.save(
                str(method_dir / f"checkpoint_ep{episode:06d}.pt"),
                metadata=metadata,
            )
        print(
            f"[CH3] {base_candidate} {scenario_profile} "
            f"episode={episode}/{episodes}",
            flush=True,
        )

    checkpoint = method_dir / "model_final.pt"
    metadata = _checkpoint_metadata(
        runtime,
        base_candidate=base_candidate,
        scenario_profile=scenario_profile,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        manifest=manifest,
        snapshot=snapshot,
        episode=episodes,
        obstacle_layout_identity=obstacle_layout_identity,
        spec=spec,
    )
    spec.source_assertion(project_root, snapshot)
    runtime.maddpg.save(str(checkpoint), metadata=metadata)

    selected = (
        evaluation_scenarios
        if evaluation_limit is None
        else evaluation_scenarios[: int(evaluation_limit)]
    )
    evaluation_rows = []
    for scenario in selected:
        row, _, _ = run_episode(
            runtime,
            scenario=scenario,
            explore=False,
        )
        row.update(
            method=base_candidate,
            base_candidate=base_candidate,
            seed=int(seed),
        )
        evaluation_rows.append(row)
    _write_csv(method_dir / "evaluation_metrics.csv", evaluation_rows)

    summary = _summary(
        runtime=runtime,
        spec=spec,
        base_candidate=base_candidate,
        scenario_profile=scenario_profile,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        method_dir=method_dir,
        checkpoint=checkpoint,
        resume_path=resume_path,
        started=started,
        global_step=global_step,
        update_step=update_step,
        training_manifest_path=training_manifest_path,
        manifest=manifest,
        training_scenarios=training_scenarios,
        training_scenario_ids=training_scenario_ids,
        evaluation_manifest_path=evaluation_manifest_path,
        evaluation_manifest_data=evaluation_manifest_data,
        evaluation_scenarios=evaluation_scenarios,
        evaluation_limit=evaluation_limit,
        selected=selected,
        obstacle_layout_identity=obstacle_layout_identity,
        isolation=isolation,
        snapshot=snapshot,
        resume_audit=resume_audit,
        metadata=metadata,
        evaluation_rows=evaluation_rows,
    )
    (method_dir / "training_summary.json").write_text(
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, rows, evaluation_rows
