"""Build the fixed paired-scenario manifest used by Chapter-3 pilot evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "chapter3_final" / "pilot" / "pilot_scenarios.json"


def _sample_positions(rng, *, n_agents=4, min_distance=2.8):
    points = []
    while len(points) < n_agents:
        point = np.array([
            rng.uniform(1.0, 19.0),
            rng.uniform(1.0, 19.0),
            rng.uniform(0.5, 7.5),
        ])
        if all(np.linalg.norm(point - old) >= min_distance for old in points):
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def build_pilot_scenarios(count=50, seed=31001):
    root_rng = np.random.default_rng(int(seed))
    scenarios = []
    for index in range(int(count)):
        scenario_seed = int(root_rng.integers(1, 2**31 - 1))
        rng = np.random.default_rng(scenario_seed)
        positions = _sample_positions(rng)
        target = np.array([
            rng.uniform(1.0, 19.0),
            rng.uniform(1.0, 19.0),
            rng.uniform(0.5, 7.5),
        ])
        scenarios.append({
            "scenario_id": f"ch3_pilot_{index:03d}",
            "scenario_seed": scenario_seed,
            "initial_agent_positions": positions.round(8).tolist(),
            "target_position": target.round(8).tolist(),
            "flow_phase_x": 0.0,
            "flow_phase_y": 0.0,
            "initial_executor_wait_point": [10.0, 10.0, 4.0],
            "planner_seed": int(root_rng.integers(1, 2**31 - 1)),
        })
    return {
        "manifest_id": "ch3_pilot_paired_scenarios_v1",
        "scenario_count": int(count),
        "generator_seed": int(seed),
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_pilot_scenarios(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[CH3 pilot scenarios] wrote {args.count} scenarios to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
