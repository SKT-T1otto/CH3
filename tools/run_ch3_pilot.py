"""Run or aggregate the gated Chapter-3 single-seed pilot protocol."""

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

from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES  # noqa: E402
from train import CONTROLLER_ONLY_METHODS, _json_safe, train_and_evaluate_method  # noqa: E402
from utils.provenance import source_fingerprint  # noqa: E402

PILOT_ROOT = PROJECT_ROOT / "data" / "chapter3_final" / "pilot"
DEFAULT_MANIFEST = PILOT_ROOT / "pilot_scenarios.json"


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean(rows, key, *, found_only=False):
    values = []
    for row in rows:
        if found_only and _float(row, "found") < 0.5:
            continue
        value = _float(row, key)
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


COMPARISONS = (
    ("A_residual_control", "ch3_pheromone_rmaddpg", "ch3_pheromone_prior",
     ("success_rate", "collision_rate", "energy_cost", "reward", "residual_norm")),
    ("B_full_pse_vs_pheromone", "ch3_pse_rmaddpg", "ch3_pheromone_rmaddpg",
     ("found_rate", "success_rate", "succ_if_found", "found_step", "exec_delay")),
    ("C_belief", "ch3_pse_rmaddpg", "ch3_pse_no_belief",
     ("found_rate", "found_step", "coverage_at_found", "claim_overlap", "success_rate")),
    ("D_search_execution_coupling", "ch3_pse_rmaddpg", "ch3_pse_no_exec_cost",
     ("succ_if_found", "exec_delay", "executor_path_after_found", "success_rate")),
    ("E_dynamic_standby", "ch3_pse_rmaddpg", "ch3_pse_no_standby",
     ("standby_to_target_dist_at_found", "executor_path_after_found", "exec_delay", "success_rate")),
    ("F_residual", "ch3_pse_rmaddpg", "ch3_pse_no_residual",
     ("success_rate", "collision_rate", "energy_cost", "reward", "residual_norm")),
)
LOWER_IS_BETTER = {
    "collision_rate", "energy_cost", "found_step", "exec_delay", "claim_overlap",
    "executor_path_after_found", "standby_to_target_dist_at_found",
}
NEUTRAL_METRICS = {"residual_norm"}


def _metric_value(rows, metric):
    aliases = {"success_rate": "success", "collision_rate": "collision", "found_rate": "found"}
    key = aliases.get(metric, metric)
    return _mean(rows, key, found_only=metric in {
        "succ_if_found", "found_step", "coverage_at_found", "claim_overlap",
        "exec_delay", "executor_path_after_found",
        "standby_to_target_dist_at_found",
    })


def _ablation_rows(evaluation_by_method):
    output = []
    for comparison, full_method, control_method, metrics in COMPARISONS:
        provisional, signs = [], []
        for metric in metrics:
            full_value = _metric_value(evaluation_by_method[full_method], metric)
            control_value = _metric_value(evaluation_by_method[control_method], metric)
            delta = full_value - control_value
            if not (math.isfinite(full_value) and math.isfinite(control_value)):
                direction, sign = "unavailable", None
            elif abs(delta) <= 1e-12:
                direction, sign = "equal", 0
            else:
                direction = "full_higher" if delta > 0 else "full_lower"
                if metric in NEUTRAL_METRICS:
                    sign = None
                elif metric in LOWER_IS_BETTER:
                    sign = 1 if delta < 0 else -1
                else:
                    sign = 1 if delta > 0 else -1
            if sign is not None:
                signs.append(sign)
            provisional.append({
                "comparison": comparison, "full_method": full_method,
                "control_method": control_method, "metric": metric,
                "full_value": full_value, "control_value": control_value,
                "delta": delta, "direction": direction,
            })
        if not signs:
            interpretation = "insufficient_pilot_training"
        elif any(sign > 0 for sign in signs) and any(sign < 0 for sign in signs):
            interpretation = "mixed"
        elif any(sign > 0 for sign in signs):
            interpretation = "directionally_supportive"
        elif any(sign < 0 for sign in signs):
            interpretation = "control_better"
        else:
            interpretation = "no_observed_benefit"
        for row in provisional:
            row["pilot_interpretation"] = interpretation
            output.append(row)
    return output


def _load_current_gate_reports(current_source):
    validation_path = PROJECT_ROOT / "data" / "chapter3_final" / "manifests" / "ch3_config_validation.json"
    gate_path = PROJECT_ROOT / "data" / "chapter3_final" / "manifests" / "ch3_acceptance_gate.json"
    if not validation_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError(
            "run tools/run_ch3.py --phase acceptance before pilot aggregation"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if validation.get("source_fingerprint") != current_source:
        raise RuntimeError("configuration validation report belongs to different source code")
    if gate.get("source_fingerprint") != current_source:
        raise RuntimeError("acceptance report belongs to different source code")
    if not validation.get("pure_chapter3_project") or validation.get("errors"):
        raise RuntimeError("configuration validation report is not accepted")
    if not gate.get("all_passed"):
        raise RuntimeError("acceptance gate did not pass")
    return validation, gate


def aggregate_pilot(
    run_root=PILOT_ROOT / "runs",
    *,
    expected_seed=1,
    expected_episodes=200,
    expected_max_steps=400,
    expected_scenarios=50,
):
    current_source = source_fingerprint(PROJECT_ROOT)
    validation, acceptance_gate = _load_current_gate_reports(current_source)
    summaries, all_training, all_evaluation = [], [], []
    evaluation_by_method = {}
    reference_ids = reference_manifest_id = reference_manifest_sha = None

    for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
        method_root = Path(run_root) / method
        seed_dirs = sorted(path for path in method_root.glob("seed_*") if path.is_dir())
        expected_dir = method_root / f"seed_{int(expected_seed)}"
        if seed_dirs != [expected_dir]:
            raise RuntimeError(
                f"{method_root} must contain only {expected_dir.name}; found {seed_dirs}"
            )
        method_dir = expected_dir
        summary_path = method_dir / "training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing pilot summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        training = _read_csv(method_dir / "episode_metrics.csv")
        evaluation = _read_csv(method_dir / "evaluation_metrics.csv")

        expected_run_type = "controller_only" if method in CONTROLLER_ONLY_METHODS else "learning"
        expected_trained = 0 if method in CONTROLLER_ONLY_METHODS else int(expected_episodes)
        checks = {
            "method": summary.get("method") == method,
            "seed": int(summary.get("seed", -1)) == int(expected_seed),
            "pilot": summary.get("pilot") is True,
            "run_type": summary.get("run_type") == expected_run_type,
            "episodes": int(summary.get("episodes", -1)) == expected_trained,
            "max_steps": int(summary.get("max_steps", -1)) == int(expected_max_steps),
            "source": summary.get("source_fingerprint") == current_source,
            "evaluation_count": len(evaluation) == int(expected_scenarios),
            "training_count": len(training) == expected_trained,
            "communication": summary.get("communication_model") == "fixed_reliable_one_step_v1",
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"{method} pilot protocol mismatch: {failed}")

        checkpoint_path = summary.get("checkpoint_path")
        if method in CONTROLLER_ONLY_METHODS:
            if checkpoint_path != "N/A":
                raise RuntimeError(f"{method} controller-only run created a checkpoint")
        else:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{method} checkpoint missing: {checkpoint}")

        scenario_ids = [row.get("scenario_id") for row in evaluation]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise RuntimeError(f"{method} evaluation contains duplicate scenario IDs")
        manifest_id = summary.get("scenario_manifest_id")
        manifest_sha = summary.get("scenario_manifest_sha256")
        if reference_ids is None:
            reference_ids = scenario_ids
            reference_manifest_id = manifest_id
            reference_manifest_sha = manifest_sha
        elif scenario_ids != reference_ids:
            raise RuntimeError(f"{method} scenario IDs/order differ from paired reference")
        elif manifest_id != reference_manifest_id or manifest_sha != reference_manifest_sha:
            raise RuntimeError(f"{method} scenario manifest identity differs from paired reference")

        evaluation_by_method[method] = evaluation
        all_training.extend(training)
        all_evaluation.extend(evaluation)
        summaries.append({
            "method": method, "seed": summary["seed"], "run_type": summary["run_type"],
            "trained_episodes": summary["episodes"], "max_steps": summary["max_steps"],
            "training_time": summary["training_time"], "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "actor_runtime_ms": summary["actor_runtime_ms"],
            "communication_model": summary["communication_model"],
            "scenario_manifest_id": manifest_id,
            "scenario_manifest_sha256": manifest_sha,
            "evaluation_scenarios": len(evaluation),
            "found_rate": _mean(evaluation, "found"),
            "success_rate": _mean(evaluation, "success"),
            "collision_rate": _mean(evaluation, "collision"),
            "mean_reward": _mean(evaluation, "reward"),
            "mean_energy_cost": _mean(evaluation, "energy_cost"),
        })

    ablations = _ablation_rows(evaluation_by_method)
    _write_csv(PILOT_ROOT / "pilot_training_summary.csv", summaries)
    _write_csv(PILOT_ROOT / "pilot_episode_metrics.csv", all_training)
    _write_csv(PILOT_ROOT / "pilot_evaluation_metrics.csv", all_evaluation)
    _write_csv(PILOT_ROOT / "pilot_ablation_matrix.csv", ablations)

    finite_handoff_delays = sorted({
        _float(row, "handoff_delay") for row in all_training + all_evaluation
        if math.isfinite(_float(row, "handoff_delay"))
    })
    acceptance_checks = {
        "source_current": True,
        "pure_chapter3_project": bool(validation["pure_chapter3_project"]),
        "comm_package_is_basic_only": bool(validation["comm_package_is_basic_only"]),
        "local_observation_dim_28": validation["observation_dim"] == 28,
        "pure_ch3_replay_buffer": not validation["replay_buffer"]["forbidden_metadata"],
        "fixed_handoff_delay_one": finite_handoff_delays in ([], [1.0]),
        "same_seed": all(row["seed"] == int(expected_seed) for row in summaries),
        "same_manifest": all(
            row["scenario_manifest_sha256"] == reference_manifest_sha for row in summaries
        ),
        "same_scenario_order": True,
        "compile_tests_validator_smoke_passed": bool(acceptance_gate["all_passed"]),
    }
    isolation_report = {
        "source_fingerprint": current_source,
        "communication_mode": "ch3_fixed_reliable",
        "communication_model_id": "fixed_reliable_one_step_v1",
        "seed": int(expected_seed),
        "learning_episodes": int(expected_episodes),
        "max_steps": int(expected_max_steps),
        "scenario_count": int(expected_scenarios),
        "scenario_manifest_id": reference_manifest_id,
        "scenario_manifest_sha256": reference_manifest_sha,
        "scenario_ids": reference_ids,
        "methods": list(ACTIVE_CH3_FINAL_EXPERIMENT_MODES),
        "controller_only_methods": list(CONTROLLER_ONLY_METHODS),
        "observed_handoff_delays": finite_handoff_delays,
        "acceptance_gate": acceptance_gate,
        "acceptance_checks": acceptance_checks,
        "isolation_accepted": all(acceptance_checks.values()),
    }
    (PILOT_ROOT / "communication_isolation_report.json").write_text(
        json.dumps(isolation_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    interpretations = {row["comparison"]: row["pilot_interpretation"] for row in ablations}
    unsupported = [
        name for name, value in interpretations.items()
        if value in {"mixed", "no_observed_benefit", "control_better", "insufficient_pilot_training"}
    ]
    lines = [
        f"# 第三章单 seed pilot 观察（seed={expected_seed}）", "",
        f"本文件报告学习方法 {expected_episodes} episode、{expected_scenarios} 个严格配对场景的方向性观察，不构成论文正式证据。", "",
        "## 六组比较", "",
    ]
    lines.extend(f"- `{name}`：`{value}`" for name, value in interpretations.items())
    lines.extend(["", "## 未获得明确方向性支持或训练不足的比较", ""])
    lines.extend(f"- `{name}`" for name in unsupported) if unsupported else lines.append("- 无；但这仍不是统计显著性结论。")
    lines.extend([
        "", "## 边界", "",
        f"所有方法使用相同 seed={expected_seed}、相同 manifest `{reference_manifest_id}` 和相同有序场景。",
        "只允许一次固定可靠的一步目标交接，不包含物理信道、丢包、带宽或通信能耗。",
        "未执行多随机种子正式训练、独立 validation/test 或显著性检验。",
    ])
    (PILOT_ROOT / "pilot_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[CH3 pilot aggregate] methods={len(summaries)} seed={expected_seed} "
        f"scenarios={expected_scenarios} isolation={isolation_report['isolation_accepted']}"
    )
    return isolation_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("short", "pilot"), required=True)
    parser.add_argument("--method", choices=ACTIVE_CH3_FINAL_EXPERIMENT_MODES)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-scenarios", type=int, default=50)
    args = parser.parse_args()

    if args.phase == "short":
        if args.aggregate_only:
            raise ValueError("short phase does not use --aggregate-only")
        if args.method is None:
            raise ValueError("short phase requires --method")
        if args.method in CONTROLLER_ONLY_METHODS:
            raise ValueError("controller-only methods do not perform short training")
        episodes = 20 if args.episodes is None else args.episodes
        output_dir = PILOT_ROOT / "short_training"
        scenario_manifest = None
    else:
        episodes = 200 if args.episodes is None else args.episodes
        if args.aggregate_only:
            aggregate_pilot(
                PILOT_ROOT / "runs",
                expected_seed=args.seed,
                expected_episodes=episodes,
                expected_max_steps=args.max_steps,
                expected_scenarios=args.expected_scenarios,
            )
            return 0
        if args.method is None:
            raise ValueError("pilot phase requires --method unless --aggregate-only is used")
        output_dir = PILOT_ROOT / "runs"
        scenario_manifest = args.scenario_manifest

    actual_episodes = 0 if args.method in CONTROLLER_ONLY_METHODS else episodes
    summary, _, _ = train_and_evaluate_method(
        args.method, seed=args.seed, episodes=actual_episodes, max_steps=args.max_steps,
        device=args.device, output_dir=output_dir, pilot=True,
        scenario_manifest=scenario_manifest,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
