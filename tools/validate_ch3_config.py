"""Validate source and runtime invariants of both Chapter-3 protocols."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comm.basic_communication import FixedReliableHandoff  # noqa: E402
from env import UAVEnv  # noqa: E402
from registry.experiment_registry import (  # noqa: E402
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    CONTROLLER_ONLY_METHODS,
)
from registry.ch3_efficiency_v3_registry import (  # noqa: E402
    CH3_EFFICIENCY_V3_SCREEN,
    CH3_EFFICIENCY_V3_SCREEN_METHODS,
    validate_v3_candidate_registry,
)
from train import (  # noqa: E402
    CH3_EFFICIENCY_V2,
    CH3_EFFICIENCY_V2_METHOD_CONFIGS,
    CH3_METHOD_CONFIGS,
    CH3_PILOT_V1,
    CH3_PROTOCOL_CONFIGS,
    _algorithm_config_hash,
    build_ch3_runtime,
    build_ch3_runtime_from_resolved_config,
)
from utils.ch3_buffer import CH3ReplayBuffer  # noqa: E402
from utils.networks import MLPNetwork  # noqa: E402
from utils.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_files,
    algorithm_source_fingerprint,
    repository_source_fingerprint,
    runtime_versions,
)

REMOVED_SYMBOLS = (
    "legacy_" + "ch5", "Dynamic" + "CommGraph", "Semantic" + "MessageEncoder",
    "VOI" + "Selector", "RoleAware" + "Topology", "Teammate" + "Predictor",
    "Belief" + "Fusion", "ChannelAware" + "Actor",
)
REMOVED_FRAGMENTS = (
    "comm_" + "graph", "semantic_" + "message", "voi_" + "selector",
    "role_" + "topology", "teammate_" + "predictor", "belief_" + "fusion",
    "normal_" + "comm", "weak_" + "comm", "severe_" + "comm",
    "robust_" + "full", "comm_" + "loss_prob", "upper_" + "comm_base_loss",
    "payload_" + "loss_scale", "critical_" + "message_loss_prob",
    "reconnect_" + "island_timeout_steps", "use_" + "robust_" + "disturbance",
    "apply_" + "uploaded_update", "acoustic " + "channel",
    "compressed local " + "update", "CHAPTER_" + "CONFIGS",
    "get_" + "ablation_config",
)
EXPECTED_COMM_SOURCES = ("__init__.py", "basic_communication.py")
FORBIDDEN_CONSTRUCTOR_FRAGMENTS = (
    "comm", "snr", "semantic", "voi", "topology", "reconnect", "prediction",
    "reliability", "quarantine", "channel", "tail", "disturbance", "noise",
    "actuator_lag",
)


def _source_texts():
    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if "__pycache__" in relative.parts:
            continue
        if relative.parts and relative.parts[0] == "data":
            continue
        yield relative, path.read_text(encoding="utf-8")


def _buffer_forbidden_metadata(buffer):
    forbidden = ("tail", "comm", "scenario", "graph", "message", "belief", "quarantine")
    return sorted(
        name for name in vars(buffer)
        if any(token in name.lower() for token in forbidden)
    )


def validate_ch3_configs():
    errors = []

    def check(scope, key, expected, actual):
        if actual != expected:
            errors.append({
                "scope": str(scope), "key": key, "expected": expected, "actual": actual,
            })

    comm_sources = tuple(sorted(path.name for path in (PROJECT_ROOT / "comm").glob("*.py")))
    check("filesystem", "comm_package_sources", EXPECTED_COMM_SOURCES, comm_sources)
    for relative, source in _source_texts():
        for symbol in REMOVED_SYMBOLS:
            check(relative, f"symbol_absent:{symbol}", False, symbol in source)
        for fragment in REMOVED_FRAGMENTS:
            check(relative, f"source_fragment_absent:{fragment}", False, fragment in source)

    try:
        importlib.import_module("evaluate_pse")
        evaluator_import_error = None
    except Exception as exc:
        evaluator_import_error = f"{type(exc).__name__}: {exc}"
    check("evaluate_pse", "import_error", None, evaluator_import_error)

    signature = inspect.signature(UAVEnv.__init__)
    forbidden_parameters = sorted(
        name for name in signature.parameters
        if any(fragment in name.lower() for fragment in FORBIDDEN_CONSTRUCTOR_FRAGMENTS)
    )
    check("UAVEnv", "forbidden_constructor_parameters", [], forbidden_parameters)
    check("registry", "methods", tuple(ACTIVE_CH3_FINAL_EXPERIMENT_MODES), tuple(CH3_METHOD_CONFIGS))
    check(
        "efficiency_v2_registry",
        "methods",
        tuple(ACTIVE_CH3_FINAL_EXPERIMENT_MODES),
        tuple(CH3_EFFICIENCY_V2_METHOD_CONFIGS),
    )
    check(
        "protocol_registry",
        "protocols",
        (CH3_PILOT_V1, CH3_EFFICIENCY_V2),
        tuple(CH3_PROTOCOL_CONFIGS),
    )

    actor_types = {}
    residual_scales = {}
    algorithm_hashes = {}
    for protocol in (CH3_PILOT_V1, CH3_EFFICIENCY_V2):
        actor_types[protocol] = {}
        residual_scales[protocol] = {}
        algorithm_hashes[protocol] = {}
        for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
            scope = f"{protocol}:{method}"
            runtime = build_ch3_runtime(
                method,
                seed=1,
                max_steps=4,
                device="cpu",
                replay_size=8,
                protocol=protocol,
            )
            env = runtime.env
            check(scope, "obs_dim", 28, env.obs_dim)
            check(scope, "action_dim", (3,), env.action_space["agent_0"].shape)
            check(scope, "communication_mode", "ch3_fixed_reliable", env.communication_mode)
            check(scope, "communication_model", FixedReliableHandoff.model_id, env.communication_model_id)
            check(scope, "handoff_service_type", FixedReliableHandoff, type(env.fixed_reliable_handoff))
            check(scope, "energy_field_absent", False, hasattr(env, "last_" + "comm_energy"))
            check(scope, "observation_layout_end", 28, env.get_observation_layout()[-1]["end"])
            residual_scales[protocol][method] = {
                "search": env.residual_scale_search,
                "executor": env.residual_scale_executor,
            }
            algorithm_hashes[protocol][method] = _algorithm_config_hash(runtime.config)
            if method in CONTROLLER_ONLY_METHODS:
                check(scope, "maddpg", None, runtime.maddpg)
                check(scope, "replay_buffer", None, runtime.replay_buffer)
                actor_types[protocol][method] = "N/A"
            else:
                check(scope, "actor_plain_mlp", True, all(
                    type(agent.policy) is MLPNetwork for agent in runtime.maddpg.agents
                ))
                actor_types[protocol][method] = type(runtime.maddpg.agents[0].policy).__name__

    v2_full = CH3_EFFICIENCY_V2_METHOD_CONFIGS["ch3_pse_rmaddpg"]
    check("efficiency_v2", "reward_scale", 400.0, float(v2_full["reward_scale"]))
    check("efficiency_v2", "reward_profile", "task_efficiency_v2", v2_full["reward_profile"])
    check("efficiency_v2", "residual_scale_search", 0.20, float(v2_full["residual_scale_search"]))
    check("efficiency_v2", "residual_scale_executor", 0.15, float(v2_full["residual_scale_executor"]))
    check("efficiency_v2", "residual_action_reg", 0.05, float(v2_full["residual_action_reg"]))
    check("efficiency_v2", "belief_weight", 0.60, float(v2_full["pse_belief_weight"]))
    check("efficiency_v2", "standby_start", 80, int(v2_full["pse_standby_start_step"]))
    check("efficiency_v2", "standby_interval", 10, int(v2_full["pse_standby_update_interval"]))
    check(
        "protocol_hashes",
        "v1_v2_full_differ",
        False,
        algorithm_hashes[CH3_PILOT_V1]["ch3_pse_rmaddpg"]
        == algorithm_hashes[CH3_EFFICIENCY_V2]["ch3_pse_rmaddpg"],
    )

    screening_actor_types = {}
    try:
        screening_configs = validate_v3_candidate_registry()
    except Exception as exc:
        screening_configs = {}
        errors.append({
            "scope": "efficiency_v3_registry",
            "key": "registry_validation",
            "expected": "valid",
            "actual": f"{type(exc).__name__}: {exc}",
        })
    check(
        "efficiency_v3_registry",
        "methods",
        tuple(CH3_EFFICIENCY_V3_SCREEN_METHODS),
        tuple(screening_configs),
    )
    for index, method in enumerate(CH3_EFFICIENCY_V3_SCREEN_METHODS):
        if method not in screening_configs:
            continue
        scope = f"{CH3_EFFICIENCY_V3_SCREEN}:{method}"
        runtime = build_ch3_runtime_from_resolved_config(
            method,
            screening_configs[method],
            seed=300 + index,
            max_steps=4,
            device="cpu",
            replay_size=8,
        )
        env = runtime.env
        check(scope, "run_type", "learning", runtime.run_type)
        check(scope, "obs_dim", 28, env.obs_dim)
        check(scope, "action_dim", (3,), env.action_space["agent_0"].shape)
        check(scope, "protocol", CH3_EFFICIENCY_V3_SCREEN, runtime.config.get("protocol"))
        check(
            scope,
            "exec_cost_reference_mode",
            "fixed_initial_wait_point",
            env.pse_exec_cost_reference_mode,
        )
        check(scope, "communication_model", FixedReliableHandoff.model_id, env.communication_model_id)
        check(scope, "maddpg_present", True, runtime.maddpg is not None)
        check(scope, "replay_present", True, runtime.replay_buffer is not None)
        check(scope, "actor_plain_mlp", True, all(
            type(agent.policy) is MLPNetwork for agent in runtime.maddpg.agents
        ))
        screening_actor_types[method] = type(runtime.maddpg.agents[0].policy).__name__

    buffer = CH3ReplayBuffer(4, 4, (28,) * 4, (3,) * 4)
    check("CH3ReplayBuffer", "forbidden_metadata", [], _buffer_forbidden_metadata(buffer))
    repository_fingerprint = repository_source_fingerprint(PROJECT_ROOT)
    algorithm_fingerprint = algorithm_source_fingerprint(PROJECT_ROOT)
    summary = {
        "pure_chapter3_project": not errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "algorithm_source_fingerprint": algorithm_fingerprint,
        "repository_source_fingerprint": repository_fingerprint,
        "algorithm_source_files": algorithm_source_files(PROJECT_ROOT),
        "source_fingerprint": repository_fingerprint,
        "runtime_versions": runtime_versions(),
        "comm_package_is_basic_only": comm_sources == EXPECTED_COMM_SOURCES,
        "comm_package_sources": list(comm_sources),
        "protocols": [CH3_PILOT_V1, CH3_EFFICIENCY_V2],
        "screening_protocol": CH3_EFFICIENCY_V3_SCREEN,
        "screening_methods": list(CH3_EFFICIENCY_V3_SCREEN_METHODS),
        "screening_actor_types": screening_actor_types,
        "methods": list(ACTIVE_CH3_FINAL_EXPERIMENT_MODES),
        "controller_only_methods": list(CONTROLLER_ONLY_METHODS),
        "communication_model": FixedReliableHandoff.model_id,
        "observation_dim": 28,
        "action_dim": 3,
        "actor_types": actor_types,
        "residual_scales": residual_scales,
        "algorithm_config_hashes": algorithm_hashes,
        "environment_constructor_parameters": list(signature.parameters),
        "evaluator_import_error": evaluator_import_error,
        "replay_buffer": {
            "type": "CH3ReplayBuffer",
            "sample_items": 8,
            "forbidden_metadata": _buffer_forbidden_metadata(buffer),
        },
        "errors": errors,
    }
    return errors, summary


def main():
    errors, summary = validate_ch3_configs()
    output = PROJECT_ROOT / "data" / "chapter3_final" / "manifests" / "ch3_config_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        print(f"[CH3 config] FAILED with {len(errors)} error(s)")
        for error in errors:
            print(error)
        return 1
    print(
        f"[CH3 config] PASS protocols={len(summary['protocols'])} "
        f"methods={len(summary['methods'])} screening={len(summary['screening_methods'])} "
        f"obs={summary['observation_dim']} "
        f"action={summary['action_dim']} source={summary['source_fingerprint'][:12]} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
