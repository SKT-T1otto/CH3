"""Fast non-training smoke checks for every pure Chapter-3 method."""

from __future__ import annotations

import json
import sys
import time
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

from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES  # noqa: E402
from train import build_ch3_runtime  # noqa: E402
from utils.provenance import source_fingerprint  # noqa: E402

OUTPUT = PROJECT_ROOT / "data" / "chapter3_final" / "pilot" / "pretraining_smoke_report.json"


def _finite_transition(observations, rewards):
    return (
        all(torch.isfinite(torch.as_tensor(item)).all() for item in observations)
        and bool(torch.isfinite(torch.as_tensor(rewards)).all())
    )


def _handoff_scenario(method_index):
    return {
        "scenario_id": f"smoke_handoff_{method_index}",
        "scenario_seed": 9000 + method_index,
        "initial_agent_positions": [[2.0, 2.0, 1.0], [8.0, 2.0, 2.0], [2.0, 8.0, 3.0], [16.0, 16.0, 4.0]],
        "target_position": [2.0, 2.0, 1.0],
        "flow_phase_x": 0.0,
        "flow_phase_y": 0.0,
        "initial_executor_wait_point": [10.0, 10.0, 4.0],
        "planner_seed": 12000 + method_index,
    }


def main():
    started = time.perf_counter()
    checkpoints_before = {path.resolve() for path in PROJECT_ROOT.rglob("*.pt")}
    rows = []
    for method_index, method in enumerate(ACTIVE_CH3_FINAL_EXPERIMENT_MODES):
        runtime = build_ch3_runtime(method, seed=100 + method_index, max_steps=12, device="cpu", replay_size=32)
        env = runtime.env
        update_count_before = 0 if runtime.maddpg is None else int(runtime.maddpg.niter)
        observations = env.reset()
        zero_finite = True
        action_shape_ok = True
        for _ in range(4):
            actions = torch.zeros((env.num_agents, 3))
            action_shape_ok &= tuple(actions.shape) == (env.num_agents, 3)
            observations, rewards, _ = env.step(actions)
            zero_finite &= _finite_transition(observations, rewards)
        observations = env.reset()
        random_finite = True
        for _ in range(6):
            actions = torch.zeros((env.num_agents, 3)) if runtime.run_type == "controller_only" else 2.0 * torch.rand((env.num_agents, 3)) - 1.0
            action_shape_ok &= tuple(actions.shape) == (env.num_agents, 3)
            observations, rewards, dones = env.step(actions)
            random_finite &= _finite_transition(observations, rewards)
            if all(dones):
                break
        env.reset(scenario=_handoff_scenario(method_index))
        _, discovery_rewards, _ = env.step(torch.zeros((env.num_agents, 3)))
        after_discovery = env.get_ch3_communication_metrics()
        env.step(torch.zeros((env.num_agents, 3)))
        after_delivery = env.get_ch3_communication_metrics()
        env.step(torch.zeros((env.num_agents, 3)))
        after_extra = env.get_ch3_communication_metrics()
        discovery_reward_ok = bool(torch.as_tensor(discovery_rewards)[env.finder_idx] > 0.0)
        handoff_ok = (
            bool(env.task_found) and after_discovery["handoff_count"] == 0
            and after_delivery["handoff_count"] == 1
            and after_delivery["handoff_delay"] == 1.0
            and after_extra["handoff_count"] == 1
        )
        update_count_after = 0 if runtime.maddpg is None else int(runtime.maddpg.niter)
        rows.append({
            "method": method, "run_type": runtime.run_type,
            "zero_action_finite": bool(zero_finite),
            "random_action_finite": bool(random_finite),
            "action_shape_valid": bool(action_shape_ok),
            "discovery_reward_preserved": discovery_reward_ok,
            "fixed_one_step_handoff": handoff_ok,
            "training_updates": update_count_after - update_count_before,
        })
    failed = [
        row["method"] for row in rows
        if not all(row[key] for key in (
            "zero_action_finite", "random_action_finite", "action_shape_valid",
            "discovery_reward_preserved", "fixed_one_step_handoff",
        )) or row["training_updates"] != 0
    ]
    checkpoints_after = {path.resolve() for path in PROJECT_ROOT.rglob("*.pt")}
    created = sorted(str(path) for path in checkpoints_after - checkpoints_before)
    if created:
        failed.append("checkpoint_created_by_smoke")
    report = {
        "source_fingerprint": source_fingerprint(PROJECT_ROOT),
        "training_updates": sum(row["training_updates"] for row in rows),
        "checkpoint_files_created": len(created),
        "created_checkpoint_paths": created,
        "runtime_seconds": time.perf_counter() - started,
        "methods": rows,
        "failed_methods": failed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed:
        print(f"[CH3 smoke] FAILED: {failed}")
        return 1
    print(f"[CH3 smoke] PASS methods={len(rows)} seconds={report['runtime_seconds']:.2f} report={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
