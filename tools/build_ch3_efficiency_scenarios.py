"""Build the three isolated scenario manifests for Chapter-3 efficiency v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v2"
MANIFEST_ROOT = V2_ROOT / "manifests"
SPACE_SIZE = np.asarray([20.0, 20.0, 8.0], dtype=np.float64)
Z_RANGE = (0.5, 7.5)
MIN_AGENT_DISTANCE = 2.8
DEFAULT_FIXED_V1 = (
    (np.asarray([5.0, 5.0, 2.0]), np.asarray([2.5, 2.5, 2.0])),
    (np.asarray([11.0, 10.0, 4.0]), np.asarray([3.0, 3.0, 2.5])),
    (np.asarray([15.5, 6.0, 5.5]), np.asarray([2.0, 3.0, 2.0])),
)

MANIFEST_SPECS = {
    "validation": {
        "filename": "efficiency_v2_validation_scenarios.json",
        "count": 50,
        "generator_seed": 41001,
        "use_obstacles": False,
        "obstacle_layout_id": "none",
    },
    "test": {
        "filename": "efficiency_v2_test_scenarios.json",
        "count": 100,
        "generator_seed": 51001,
        "use_obstacles": False,
        "obstacle_layout_id": "none",
    },
    "obstacle": {
        "filename": "efficiency_v2_obstacle_scenarios.json",
        "count": 20,
        "generator_seed": 61001,
        "use_obstacles": True,
        "obstacle_layout_id": "default_fixed_v1",
    },
}


def point_inside_obstacle(point, layout_id):
    if layout_id in {None, "none"}:
        return False
    if layout_id != "default_fixed_v1":
        raise ValueError(f"unsupported obstacle layout {layout_id!r}")
    point = np.asarray(point, dtype=np.float64)
    return any(
        bool(np.all(point >= center - size / 2.0) and np.all(point <= center + size / 2.0))
        for center, size in DEFAULT_FIXED_V1
    )


def _validate_point(point, *, layout_id, name):
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {point.shape}")
    if np.any(point < 0.0) or np.any(point > SPACE_SIZE):
        raise ValueError(f"{name} is outside the environment boundary")
    if point_inside_obstacle(point, layout_id):
        raise ValueError(f"{name} lies inside obstacle layout {layout_id}")


def _sample_free_point(rng, *, layout_id, margin=1.0):
    for _ in range(4096):
        point = np.asarray(
            [
                rng.uniform(margin, SPACE_SIZE[0] - margin),
                rng.uniform(margin, SPACE_SIZE[1] - margin),
                rng.uniform(*Z_RANGE),
            ],
            dtype=np.float64,
        )
        if not point_inside_obstacle(point, layout_id):
            return point
    raise RuntimeError(f"unable to sample a free point for layout {layout_id}")


def _sample_agent_positions(rng, *, layout_id):
    points = []
    for agent_index in range(4):
        for _ in range(4096):
            point = _sample_free_point(rng, layout_id=layout_id)
            if all(np.linalg.norm(point - previous) >= MIN_AGENT_DISTANCE for previous in points):
                points.append(point)
                break
        else:
            raise RuntimeError(f"unable to place agent {agent_index} with minimum separation")
    result = np.stack(points)
    distances = np.linalg.norm(result[:, None, :] - result[None, :, :], axis=-1)
    distances += np.eye(4) * 1e9
    if float(distances.min()) < MIN_AGENT_DISTANCE:
        raise AssertionError("generated agent positions violate minimum separation")
    return result


def build_efficiency_manifest(
    kind,
    *,
    count=None,
    generator_seed=None,
    use_obstacles=None,
    obstacle_layout_id=None,
):
    kind = str(kind)
    if kind not in MANIFEST_SPECS:
        raise ValueError(f"unknown manifest kind {kind!r}")
    spec = dict(MANIFEST_SPECS[kind])
    count = int(spec["count"] if count is None else count)
    generator_seed = int(spec["generator_seed"] if generator_seed is None else generator_seed)
    use_obstacles = bool(spec["use_obstacles"] if use_obstacles is None else use_obstacles)
    layout_id = str(
        spec["obstacle_layout_id"] if obstacle_layout_id is None else obstacle_layout_id
    )
    if use_obstacles and layout_id != "default_fixed_v1":
        raise ValueError("obstacle scenarios require obstacle_layout_id=default_fixed_v1")
    if not use_obstacles:
        layout_id = "none"
    rng = np.random.default_rng(generator_seed)
    scenarios = []
    for index in range(count):
        positions = _sample_agent_positions(rng, layout_id=layout_id)
        target = _sample_free_point(rng, layout_id=layout_id)
        wait_point = _sample_free_point(rng, layout_id=layout_id)
        for agent_index, point in enumerate(positions):
            _validate_point(point, layout_id=layout_id, name=f"agent[{agent_index}]")
        _validate_point(target, layout_id=layout_id, name="target")
        _validate_point(wait_point, layout_id=layout_id, name="executor wait point")
        scenarios.append(
            {
                "scenario_id": f"efficiency_v2_{kind}_{index + 1:04d}",
                "scenario_seed": generator_seed + index,
                "planner_seed": int(rng.integers(1, 2**31 - 1)),
                "use_obstacles": use_obstacles,
                "obstacle_layout_id": layout_id,
                "initial_agent_positions": positions.round(7).tolist(),
                "target_position": target.round(7).tolist(),
                "initial_executor_wait_point": wait_point.round(7).tolist(),
                # Main efficiency and obstacle-generalization protocols isolate
                # task coordination from flow-phase robustness.  A separate
                # robustness manifest can introduce non-zero phases later.
                "flow_phase_x": 0.0,
                "flow_phase_y": 0.0,
            }
        )
    return {
        "protocol": "ch3_efficiency_v2",
        "manifest_id": f"ch3_efficiency_v2_{kind}_scenarios_v1",
        "scenario_role": kind,
        "scenario_count": count,
        "generator_seed": generator_seed,
        "use_obstacles": use_obstacles,
        "obstacle_layout_id": layout_id,
        "scenarios": scenarios,
    }


def write_official_manifests(kinds=("validation", "test", "obstacle")):
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("validation", "test", "obstacle_test", "runs", "summaries"):
        (V2_ROOT / directory).mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind in kinds:
        manifest = build_efficiency_manifest(kind)
        path = MANIFEST_ROOT / MANIFEST_SPECS[kind]["filename"]
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        outputs[kind] = path
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", *MANIFEST_SPECS), default="all")
    args = parser.parse_args(argv)
    kinds = tuple(MANIFEST_SPECS) if args.kind == "all" else (args.kind,)
    for kind, path in write_official_manifests(kinds).items():
        print(f"[CH3 efficiency v2] wrote {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
