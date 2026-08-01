"""Generate strictly paired Chapter-3 manifests and bounded smoke manifests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ch3_constants import (
    ALL_SCENARIO_PROFILES,
    CH3_MISSION_V1,
    CH3_UNKNOWN_MAP_V1,
    CH3_ROOT,
    DEFAULT_SMOKE_SCENARIO_COUNT,
    DEFAULT_TRAIN_SCENARIO_COUNT,
    DEFAULT_VALIDATION_SCENARIO_COUNT,
    SCENARIO_PROFILES,
    SCENARIO_SPLITS,
    SMOKE_TRAIN_GENERATOR_SEED,
    SMOKE_VALIDATION_GENERATOR_SEED,
    TRAIN_GENERATOR_SEED,
    UNKNOWN_DEFAULT_SMOKE_SCENARIO_COUNT,
    UNKNOWN_DEFAULT_TRAIN_SCENARIO_COUNT,
    UNKNOWN_DEFAULT_VALIDATION_SCENARIO_COUNT,
    UNKNOWN_MANIFEST_ROOT,
    UNKNOWN_MAP_PROFILES,
    UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED,
    UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED,
    UNKNOWN_TRAIN_GENERATOR_SEED,
    UNKNOWN_VALIDATION_GENERATOR_SEED,
    VALIDATION_GENERATOR_SEED,
)
from target_motion import TargetState, simulate_target_trajectory
from map.path_planner import ObstacleAwareTaskMapPlanner


MANIFEST_ROOT = CH3_ROOT / "manifests"
SPACE_SIZE = np.asarray([20.0, 20.0, 8.0], dtype=np.float64)
GENERATOR_SEED = TRAIN_GENERATOR_SEED
PLANNER_GRID_SIZE = (10, 10, 8)
PLANNER_OBSTACLE_CLEARANCE = 0.40
TARGET_OBSTACLE_CLEARANCE = 0.20
TARGET_CAPTURE_RADIUS = 0.80
SCENARIO_SCHEMA_VERSION = 3
MAX_SCENARIO_SAMPLE_ATTEMPTS = 64
PROFILE_CODES = {
    "S00_STATIC_CLEAR": "s00_static_clear",
    "S10_MOVING_CLEAR": "s10_moving_clear",
    "S01_STATIC_OBSTACLE": "s01_static_obstacle",
    "S11_MOVING_OBSTACLE": "s11_moving_obstacle",
    "M00_MOVING_CLEAR": "m00",
    "M10_MOVING_UNKNOWN_SINGLE": "m10",
    "M20_MOVING_UNKNOWN_MULTI": "m20",
    "M90_MOVING_KNOWN_ORACLE": "m90",
}
_BUILD_CACHE = {}

_UNKNOWN_PROFILE_KNOWLEDGE = {
    "M00_MOVING_CLEAR": "online_unknown",
    "M10_MOVING_UNKNOWN_SINGLE": "online_unknown",
    "M20_MOVING_UNKNOWN_MULTI": "online_unknown",
    "M90_MOVING_KNOWN_ORACLE": "oracle",
}
_UNKNOWN_PROFILE_FAMILY = {
    "M00_MOVING_CLEAR": "clear",
    "M10_MOVING_UNKNOWN_SINGLE": "random_single_aabb_v1",
    "M20_MOVING_UNKNOWN_MULTI": "random_multi_aabb_v1",
    "M90_MOVING_KNOWN_ORACLE": "random_multi_aabb_v1",
}


def _canonical_sha(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _inside(point, obstacles, clearance=0.0):
    point = np.asarray(point, dtype=np.float64)
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"], dtype=np.float64)
        half = np.asarray(obstacle["size"], dtype=np.float64) / 2.0 + clearance
        if np.all(point >= center - half) and np.all(point <= center + half):
            return True
    return False


def _sample_point(rng, obstacles, margin=1.0, clearance=0.5):
    for _ in range(4096):
        point = np.asarray([
            rng.uniform(margin, SPACE_SIZE[0] - margin),
            rng.uniform(margin, SPACE_SIZE[1] - margin),
            rng.uniform(0.8, 7.2),
        ])
        if not _inside(point, obstacles, clearance):
            return point
    raise RuntimeError("unable to sample a legal mission point")


def _sample_agents(rng, obstacles):
    points = []
    for _ in range(4):
        for _ in range(4096):
            point = _sample_point(rng, obstacles)
            if all(np.linalg.norm(point - previous) >= 2.8 for previous in points):
                points.append(point)
                break
        else:
            raise RuntimeError("unable to sample separated mission agents")
    return np.stack(points)


def _paired_obstacles(index):
    x_shift = ((index % 3) - 1) * 0.35
    return [
        {
            "center": [10.0 + x_shift, 10.0, 4.0],
            "size": [2.4, 6.0, 5.2],
        }
    ]


def _trajectory_record(position, velocity, mode, obstacles):
    state = TargetState(
        position, velocity, 0, mode,
        state_schema="moving_target_state_v1",
        obstacle_layout_id="custom_aabb_v1" if obstacles else "none",
    )
    trajectory = simulate_target_trajectory(
        state, 400, 0.2, SPACE_SIZE, obstacles,
        clearance=TARGET_OBSTACLE_CLEARANCE, max_reflections=4,
        max_prediction_steps=400,
    )
    points = np.stack([item.position for item in trajectory])
    if not np.all(np.isfinite(points)):
        raise RuntimeError("non-finite target trajectory")
    if np.any(points < -1e-8) or np.any(points > SPACE_SIZE + 1e-8):
        raise RuntimeError("target trajectory left world bounds")
    if any(
        _inside(point, obstacles, TARGET_OBSTACLE_CLEARANCE - 1e-8)
        for point in points
    ):
        raise RuntimeError("target trajectory entered an expanded obstacle")
    distance = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if mode == "static" and distance != 0.0:
        raise RuntimeError("static target moved")
    if mode != "static" and distance <= 10.0:
        raise RuntimeError("moving target trajectory is too short")
    rounded = points.round(8).tolist()
    layout_hash = _canonical_sha(obstacles)
    identity = {
        "dt": 0.2,
        "max_steps": 400,
        "target_obstacle_clearance": TARGET_OBSTACLE_CLEARANCE,
        "motion_mode": mode,
        "obstacle_layout_hash": layout_hash,
        "points": rounded,
    }
    return _canonical_sha(identity), distance, points


def _planner(obstacles, agent_positions):
    planner = ObstacleAwareTaskMapPlanner(
        space_size=SPACE_SIZE,
        grid_size=PLANNER_GRID_SIZE,
        z_range=(0.8, 7.2),
        planner_obstacle_clearance=PLANNER_OBSTACLE_CLEARANCE,
        device="cpu",
        dtype=torch.float32,
    )
    planner.reset(
        agent_positions=torch.as_tensor(agent_positions, dtype=torch.float32),
        obstacles=obstacles,
    )
    return planner


def validate_scenario_reachability(
    agent_positions, wait_point, target_point, trajectory_points, obstacles
):
    """Validate endpoint connectivity with the runtime planner implementation."""

    planner = _planner(obstacles, agent_positions)
    labels = planner._component_labels()

    def connector_map(point):
        vector = np.asarray(point, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise RuntimeError("scenario endpoint is not a finite 3-vector")
        if np.any(vector < 0.0) or np.any(vector > SPACE_SIZE):
            raise RuntimeError("scenario endpoint is outside world bounds")
        connectors = {}
        for cost, cell in planner._connector_candidates(point):
            component = labels.get(cell)
            if component is not None:
                connectors[int(component)] = (float(cost), cell)
        return connectors

    point_connectors = [connector_map(point) for point in agent_positions]
    point_connectors.append(connector_map(wait_point))
    point_connectors.append(connector_map(target_point))
    point_components = [set(item) for item in point_connectors]
    if any(not item for item in point_components):
        raise RuntimeError("scenario endpoint cannot connect to a valid planner cell")
    common = set.intersection(*point_components)
    if not common:
        raise RuntimeError("scenario endpoints are not in one reachable component")
    component = min(common)
    reference = np.asarray(agent_positions[0], dtype=np.float64)
    exact_points = [
        *[np.asarray(point, dtype=np.float64) for point in agent_positions],
        np.asarray(wait_point, dtype=np.float64),
        np.asarray(target_point, dtype=np.float64),
    ]
    for point in exact_points:
        path = planner.grid_astar_path(reference, point, role="executor")
        if (
            not path["reachable"]
            or path.get("resolved_component_id") != component
            or not path.get("start_connector_valid")
            or not path.get("goal_connector_valid")
            or not path.get("exact_goal_reachable")
        ):
            raise RuntimeError("scenario endpoint is not exactly reachable")
        if any(
            not planner.segment_is_free(path["points"][index], path["points"][index + 1])
            for index in range(len(path["points"]) - 1)
        ):
            raise RuntimeError("scenario endpoint path has a blocked continuous edge")
    max_connector_distance = 0.0
    connector_failures = 0
    if not obstacles:
        trajectory = np.asarray(trajectory_points, dtype=np.float64)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 3
            or not np.all(np.isfinite(trajectory))
            or np.any(trajectory < 0.0)
            or np.any(trajectory > SPACE_SIZE)
        ):
            raise RuntimeError("target trajectory contains an invalid endpoint")
        distances = np.linalg.norm(
            trajectory[:, None, :] - planner._flat_xyz_centers_np[None, :, :],
            axis=2,
        )
        max_connector_distance = float(np.min(distances, axis=1).max())
        component_id = _canonical_sha({
            "layout": _canonical_sha(obstacles),
            "component": int(component),
            "grid_size": PLANNER_GRID_SIZE,
            "clearance": PLANNER_OBSTACLE_CLEARANCE,
        })[:16]
        return {
            "connectivity_component_id": component_id,
            "target_trajectory_reachable": True,
            "target_trajectory_exact_endpoint_reachable": True,
            "target_trajectory_max_connector_distance": max_connector_distance,
            "target_trajectory_connector_failure_count": 0,
        }
    for point in trajectory_points:
        if _inside(point, obstacles, TARGET_OBSTACLE_CLEARANCE - 1e-8):
            raise RuntimeError("target trajectory violates target clearance")
        connectors = connector_map(point)
        if component not in connectors:
            connector_failures += 1
            raise RuntimeError("target trajectory enters an unreachable region")
        connector_cost, connector_cell = connectors[component]
        max_connector_distance = max(
            max_connector_distance,
            float(
                np.linalg.norm(
                    np.asarray(point, dtype=np.float64)
                    - planner._xyz_centers_np[connector_cell]
                )
            ),
        )
        if not planner.segment_is_free(
            planner.xyz_centers[connector_cell], point
        ):
            connector_failures += 1
            raise RuntimeError("target trajectory final connector is blocked")
    component_id = _canonical_sha({
        "layout": _canonical_sha(obstacles),
        "component": int(component),
        "grid_size": PLANNER_GRID_SIZE,
        "clearance": PLANNER_OBSTACLE_CLEARANCE,
    })[:16]
    return {
        "connectivity_component_id": component_id,
        "target_trajectory_reachable": True,
        "target_trajectory_exact_endpoint_reachable": True,
        "target_trajectory_max_connector_distance": float(max_connector_distance),
        "target_trajectory_connector_failure_count": int(connector_failures),
    }


def _build_mission_profile_manifests(
    count=DEFAULT_VALIDATION_SCENARIO_COUNT,
    generator_seed=GENERATOR_SEED,
    split="train",
):
    count = int(count)
    split = str(split)
    if split not in SCENARIO_SPLITS:
        raise ValueError(
            f"unsupported scenario split={split!r}; expected {SCENARIO_SPLITS}"
        )
    if count <= 0:
        raise ValueError("scenario count must be positive")
    cache_key = (count, int(generator_seed), split)
    if cache_key in _BUILD_CACHE:
        return deepcopy(_BUILD_CACHE[cache_key])
    rng = np.random.default_rng(int(generator_seed))
    manifests = {
        profile: {
            "protocol": CH3_MISSION_V1,
            "manifest_id": (
                f"ch3_mission_v1_{split}_{PROFILE_CODES[profile]}_scenarios_v1"
            ),
            "scenario_profile": profile,
            "scenario_role": split,
            "scenario_split": split,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_count": count,
            "generator_seed": int(generator_seed),
            "max_steps": 400,
            "scenarios": [],
        }
        for profile in SCENARIO_PROFILES
    }
    for index in range(count):
        obstacles = _paired_obstacles(index)
        records = None
        for attempt in range(MAX_SCENARIO_SAMPLE_ATTEMPTS):
            positions = _sample_agents(rng, obstacles)
            target = _sample_point(rng, obstacles, clearance=0.30)
            wait = _sample_point(rng, obstacles, clearance=0.50)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            speed = rng.uniform(0.25, 0.55)
            velocity = direction * speed
            planner_seed = int(rng.integers(1, 2**31 - 1))
            candidate_records = {}
            try:
                for profile in SCENARIO_PROFILES:
                    moving = profile.startswith("S10") or profile.startswith("S11")
                    obstacle = profile.startswith("S01") or profile.startswith("S11")
                    profile_obstacles = obstacles if obstacle else []
                    initial_velocity = velocity if moving else np.zeros(3)
                    mode = (
                        "constant_velocity_reflect_v1" if moving else "static"
                    )
                    trajectory_sha, trajectory_distance, trajectory_points = (
                        _trajectory_record(
                            target, initial_velocity, mode, profile_obstacles
                        )
                    )
                    reachability = validate_scenario_reachability(
                        positions,
                        wait,
                        target,
                        trajectory_points,
                        profile_obstacles,
                    )
                    candidate_records[profile] = (
                        initial_velocity,
                        mode,
                        trajectory_sha,
                        trajectory_distance,
                        reachability,
                    )
            except RuntimeError:
                continue
            records = candidate_records
            break
        if records is None:
            raise RuntimeError(
                f"unable to sample reachable paired scenario index={index} "
                f"after {MAX_SCENARIO_SAMPLE_ATTEMPTS} attempts"
            )
        scenario_seed = int(generator_seed + index)
        pair_group = f"mission_{split}_pair_{index + 1:04d}"
        obstacle_sha = _canonical_sha(obstacles)
        for profile in SCENARIO_PROFILES:
            moving = profile.startswith("S10") or profile.startswith("S11")
            obstacle = profile.startswith("S01") or profile.startswith("S11")
            profile_obstacles = obstacles if obstacle else []
            (
                initial_velocity,
                mode,
                trajectory_sha,
                trajectory_distance,
                reachability,
            ) = records[profile]
            code = PROFILE_CODES[profile]
            manifests[profile]["scenarios"].append({
                "scenario_id": f"mission_{split}_{code}_{index + 1:04d}",
                "pair_group_id": pair_group,
                "scenario_profile": profile,
                "scenario_role": split,
                "scenario_split": split,
                "protocol": CH3_MISSION_V1,
                "scenario_seed": scenario_seed,
                "planner_seed": planner_seed,
                "max_steps": 400,
                "use_obstacles": obstacle,
                "obstacle_layout_id": "custom_aabb_v1" if obstacle else "none",
                "obstacle_layout_sha256": obstacle_sha if obstacle else _canonical_sha([]),
                "obstacles": json.loads(json.dumps(profile_obstacles)),
                "initial_agent_positions": positions.round(7).tolist(),
                "initial_executor_wait_point": wait.round(7).tolist(),
                "target_position": target.round(7).tolist(),
                "target_initial_position": target.round(7).tolist(),
                "target_initial_velocity": initial_velocity.round(7).tolist(),
                "target_motion_mode": mode,
                "target_state_schema": "moving_target_state_v1",
                "target_motion_seed": int(generator_seed * 10 + index),
                "target_trajectory_sha256": trajectory_sha,
                "target_trajectory_distance": trajectory_distance,
                "planner_grid_size": list(PLANNER_GRID_SIZE),
                "planner_obstacle_clearance": PLANNER_OBSTACLE_CLEARANCE,
                "target_obstacle_clearance": TARGET_OBSTACLE_CLEARANCE,
                "target_capture_radius": TARGET_CAPTURE_RADIUS,
                **reachability,
                "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
                "flow_phase_x": 0.0,
                "flow_phase_y": 0.0,
            })
    _BUILD_CACHE[cache_key] = deepcopy(manifests)
    return deepcopy(manifests)


def _boxes_overlap(left, right, margin=0.35):
    left_center = np.asarray(left["center"], dtype=np.float64)
    left_half = np.asarray(left["size"], dtype=np.float64) / 2.0 + margin
    right_center = np.asarray(right["center"], dtype=np.float64)
    right_half = np.asarray(right["size"], dtype=np.float64) / 2.0 + margin
    return bool(
        np.all(np.abs(left_center - right_center) <= left_half + right_half)
    )


def _sample_unknown_agents(rng):
    points = []
    for _ in range(4):
        for _ in range(4096):
            point = _sample_point(rng, ())
            if all(
                np.linalg.norm(point - previous) >= 2.8
                for previous in points
            ):
                points.append(point)
                break
        else:
            raise RuntimeError("unable to sample separated agent positions")
    return np.stack(points)


def _sample_unknown_obstacles(rng, count, protected_points):
    obstacles = []
    protected = [
        np.asarray(point, dtype=np.float64)
        for point in protected_points
    ]
    for _ in range(int(count)):
        accepted = None
        for _ in range(4096):
            size = np.asarray([
                rng.uniform(1.4, 4.0),
                rng.uniform(1.4, 5.0),
                rng.uniform(1.2, 4.8),
            ])
            half = size / 2.0
            center = np.asarray([
                rng.uniform(
                    half[0] + 0.8,
                    SPACE_SIZE[0] - half[0] - 0.8,
                ),
                rng.uniform(
                    half[1] + 0.8,
                    SPACE_SIZE[1] - half[1] - 0.8,
                ),
                rng.uniform(
                    half[2] + 0.5,
                    SPACE_SIZE[2] - half[2] - 0.5,
                ),
            ])
            candidate = {
                "center": center.round(7).tolist(),
                "size": size.round(7).tolist(),
            }
            if any(
                _inside(point, [candidate], clearance=1.0)
                for point in protected
            ):
                continue
            if any(
                _boxes_overlap(candidate, previous)
                for previous in obstacles
            ):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError("unable to sample a non-overlapping obstacle")
        obstacles.append(accepted)
    return obstacles


def _unknown_trajectory_record(position, velocity, obstacles):
    state = TargetState(
        position,
        velocity,
        0,
        "constant_velocity_reflect_v1",
        state_schema="moving_target_state_v1",
        obstacle_layout_id="custom_aabb_v1" if obstacles else "none",
    )
    trajectory = simulate_target_trajectory(
        state,
        400,
        0.2,
        SPACE_SIZE,
        obstacles,
        clearance=TARGET_OBSTACLE_CLEARANCE,
        max_reflections=4,
        max_prediction_steps=400,
    )
    points = np.stack([item.position for item in trajectory])
    if not np.all(np.isfinite(points)):
        raise RuntimeError("target trajectory is non-finite")
    if np.any(points < -1e-8) or np.any(points > SPACE_SIZE + 1e-8):
        raise RuntimeError("target trajectory left the world")
    if any(
        _inside(
            point,
            obstacles,
            clearance=TARGET_OBSTACLE_CLEARANCE - 1e-8,
        )
        for point in points
    ):
        raise RuntimeError("target trajectory entered an expanded obstacle")
    distance = float(
        np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    )
    if distance <= 10.0:
        raise RuntimeError("moving target trajectory is too short")
    identity = {
        "dt": 0.2,
        "max_steps": 400,
        "target_obstacle_clearance": TARGET_OBSTACLE_CLEARANCE,
        "motion_mode": "constant_velocity_reflect_v1",
        "obstacle_layout_hash": _canonical_sha(obstacles),
        "points": points.round(8).tolist(),
    }
    return _canonical_sha(identity), distance, points


def _validate_unknown_truth_reachability(
    agent_positions,
    wait_point,
    target_point,
    trajectory_points,
    obstacles,
):
    """Use obstacle truth only for dataset feasibility construction."""

    planner = _planner(obstacles, agent_positions)
    labels = planner._component_labels()

    def connector_map(point):
        connectors = {}
        for cost, cell in planner._connector_candidates(point):
            component = labels.get(cell)
            if component is not None:
                connectors[int(component)] = (float(cost), cell)
        return connectors

    endpoints = [
        *[
            np.asarray(point, dtype=np.float64)
            for point in agent_positions
        ],
        np.asarray(wait_point, dtype=np.float64),
        np.asarray(target_point, dtype=np.float64),
    ]
    endpoint_connectors = [connector_map(point) for point in endpoints]
    if any(not item for item in endpoint_connectors):
        raise RuntimeError("endpoint has no oracle connector")
    common = set.intersection(
        *[set(item) for item in endpoint_connectors]
    )
    if not common:
        raise RuntimeError("endpoints have no common oracle component")
    component = min(common)
    reference = endpoints[0]
    for point in endpoints:
        path = planner.grid_astar_path(
            reference, point, role="executor"
        )
        if (
            not path["reachable"]
            or path.get("resolved_component_id", component) != component
            or not path.get("start_connector_valid", True)
            or not path.get("goal_connector_valid", True)
            or not path.get("exact_goal_reachable", True)
        ):
            raise RuntimeError("endpoint is not exactly oracle-reachable")

    max_connector_distance = 0.0
    for point in trajectory_points:
        connectors = connector_map(point)
        if component not in connectors:
            raise RuntimeError(
                "target trajectory enters an oracle-unreachable component"
            )
        _, cell = connectors[component]
        center = planner._xyz_centers_np[cell]
        max_connector_distance = max(
            max_connector_distance,
            float(np.linalg.norm(point - center)),
        )
        if not planner.segment_is_free(
            planner.xyz_centers[cell], point
        ):
            raise RuntimeError(
                "target trajectory has a blocked final connector"
            )
    component_id = _canonical_sha({
        "layout": _canonical_sha(obstacles),
        "component": int(component),
        "grid_size": PLANNER_GRID_SIZE,
        "clearance": PLANNER_OBSTACLE_CLEARANCE,
    })[:16]
    return {
        "connectivity_component_id": component_id,
        "target_trajectory_reachable": True,
        "target_trajectory_exact_endpoint_reachable": True,
        "target_trajectory_max_connector_distance":
            float(max_connector_distance),
        "target_trajectory_connector_failure_count": 0,
    }


def _unknown_profile_obstacles(profile, single, multi):
    if profile == "M00_MOVING_CLEAR":
        return []
    if profile == "M10_MOVING_UNKNOWN_SINGLE":
        return single
    return multi


def _build_unknown_profile_manifests(count, generator_seed, split):
    count = int(count)
    generator_seed = int(generator_seed)
    split = str(split)
    if split not in SCENARIO_SPLITS:
        raise ValueError(
            f"unsupported split={split!r}; expected {SCENARIO_SPLITS}"
        )
    if count <= 0:
        raise ValueError("scenario count must be positive")
    cache_key = ("unknown", count, generator_seed, split)
    if cache_key in _BUILD_CACHE:
        return deepcopy(_BUILD_CACHE[cache_key])

    rng = np.random.default_rng(generator_seed)
    manifests = {
        profile: {
            "protocol": CH3_UNKNOWN_MAP_V1,
            "manifest_id": (
                f"{CH3_UNKNOWN_MAP_V1}_{split}_"
                f"{PROFILE_CODES[profile]}_scenarios_v1"
            ),
            "scenario_profile": profile,
            "scenario_role": split,
            "scenario_split": split,
            "scenario_schema_version": 1,
            "scenario_count": count,
            "generator_seed": generator_seed,
            "max_steps": 400,
            "target_motion_known": True,
            "scenarios": [],
        }
        for profile in UNKNOWN_MAP_PROFILES
    }
    for index in range(count):
        records = None
        for _attempt in range(96):
            agents = _sample_unknown_agents(rng)
            target = _sample_point(rng, ())
            wait = _sample_point(rng, ())
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm
            velocity = direction * rng.uniform(0.25, 0.55)
            protected = [*agents, target, wait]
            try:
                single = _sample_unknown_obstacles(
                    rng, 1, protected
                )
                multi = _sample_unknown_obstacles(
                    rng,
                    int(rng.integers(2, 5)),
                    protected,
                )
                profile_records = {}
                for profile in UNKNOWN_MAP_PROFILES:
                    obstacles = _unknown_profile_obstacles(
                        profile, single, multi
                    )
                    (
                        trajectory_sha,
                        trajectory_distance,
                        trajectory_points,
                    ) = _unknown_trajectory_record(
                        target, velocity, obstacles
                    )
                    reachability = (
                        _validate_unknown_truth_reachability(
                            agents,
                            wait,
                            target,
                            trajectory_points,
                            obstacles,
                        )
                    )
                    profile_records[profile] = {
                        "obstacles": deepcopy(obstacles),
                        "trajectory_sha": trajectory_sha,
                        "trajectory_distance": trajectory_distance,
                        "reachability": reachability,
                    }
            except (RuntimeError, ValueError):
                continue
            records = {
                "agents": agents,
                "target": target,
                "wait": wait,
                "velocity": velocity,
                "profiles": profile_records,
                "planner_seed": int(rng.integers(1, 2**31 - 1)),
            }
            break
        if records is None:
            raise RuntimeError(
                f"unable to sample unknown-map scenario index={index}"
            )

        scenario_seed = generator_seed + index
        pair_group = f"unknown_{split}_pair_{index + 1:04d}"
        for profile in UNKNOWN_MAP_PROFILES:
            profile_record = records["profiles"][profile]
            obstacles = profile_record["obstacles"]
            knowledge = _UNKNOWN_PROFILE_KNOWLEDGE[profile]
            family = _UNKNOWN_PROFILE_FAMILY[profile]
            code = PROFILE_CODES[profile]
            manifests[profile]["scenarios"].append({
                "scenario_id": (
                    f"unknown_{split}_{code}_{index + 1:04d}"
                ),
                "pair_group_id": pair_group,
                "scenario_profile": profile,
                "scenario_role": split,
                "scenario_split": split,
                "protocol": CH3_UNKNOWN_MAP_V1,
                "scenario_seed": int(scenario_seed),
                "planner_seed": int(records["planner_seed"]),
                "max_steps": 400,
                "target_motion_known": True,
                "target_motion_mode":
                    "constant_velocity_reflect_v1",
                "target_state_schema": "moving_target_state_v1",
                "target_motion_seed": int(
                    generator_seed * 10 + index
                ),
                "initial_agent_positions":
                    records["agents"].round(7).tolist(),
                "initial_executor_wait_point":
                    records["wait"].round(7).tolist(),
                "target_position":
                    records["target"].round(7).tolist(),
                "target_initial_position":
                    records["target"].round(7).tolist(),
                "target_initial_velocity":
                    records["velocity"].round(7).tolist(),
                "use_obstacles": bool(obstacles),
                "obstacle_layout_id":
                    "custom_aabb_v1" if obstacles else "none",
                "obstacle_family": family,
                "obstacle_knowledge_mode": knowledge,
                "obstacle_layout_sha256":
                    _canonical_sha(obstacles),
                "obstacle_count": len(obstacles),
                "obstacles": deepcopy(obstacles),
                "planner_grid_size": list(PLANNER_GRID_SIZE),
                "planner_obstacle_clearance":
                    PLANNER_OBSTACLE_CLEARANCE,
                "target_obstacle_clearance":
                    TARGET_OBSTACLE_CLEARANCE,
                "target_capture_radius": TARGET_CAPTURE_RADIUS,
                "target_trajectory_sha256":
                    profile_record["trajectory_sha"],
                "target_trajectory_distance":
                    profile_record["trajectory_distance"],
                "unknown_map_schema":
                    "shared_logodds_occupancy_v1",
                "map_sharing_mode":
                    "central_shared_deterministic_v1",
                "flow_phase_x": 0.0,
                "flow_phase_y": 0.0,
                "scenario_schema_version": 1,
                **profile_record["reachability"],
            })
    _BUILD_CACHE[cache_key] = deepcopy(manifests)
    return deepcopy(manifests)


def _write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_profiles(profiles):
    if profiles is None:
        return tuple(ALL_SCENARIO_PROFILES)
    if isinstance(profiles, str):
        profiles = (profiles,)
    selected = []
    for value in profiles:
        if value == "all":
            selected.extend(ALL_SCENARIO_PROFILES)
        elif value == "mission":
            selected.extend(SCENARIO_PROFILES)
        elif value == "unknown":
            selected.extend(UNKNOWN_MAP_PROFILES)
        elif value in ALL_SCENARIO_PROFILES:
            selected.append(value)
        else:
            raise ValueError(f"unsupported scenario profile selector={value!r}")
    return tuple(dict.fromkeys(selected))


_DEFAULT_MISSION_GENERATOR_SEEDS = {
    "train": TRAIN_GENERATOR_SEED,
    "validation": VALIDATION_GENERATOR_SEED,
    "smoke_train": SMOKE_TRAIN_GENERATOR_SEED,
    "smoke_validation": SMOKE_VALIDATION_GENERATOR_SEED,
}
_DEFAULT_UNKNOWN_GENERATOR_SEEDS = {
    "train": UNKNOWN_TRAIN_GENERATOR_SEED,
    "validation": UNKNOWN_VALIDATION_GENERATOR_SEED,
    "smoke_train": UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED,
    "smoke_validation": UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED,
}


def _default_generator_seed(split, *, unknown):
    split = str(split)
    table = (
        _DEFAULT_UNKNOWN_GENERATOR_SEEDS
        if unknown
        else _DEFAULT_MISSION_GENERATOR_SEEDS
    )
    try:
        return int(table[split])
    except KeyError as exc:
        raise ValueError(
            f"unsupported scenario split={split!r}; expected "
            f"{tuple(_DEFAULT_MISSION_GENERATOR_SEEDS)}"
        ) from exc


def build_scenario_manifests(
    count=DEFAULT_VALIDATION_SCENARIO_COUNT,
    generator_seed=None,
    split="train",
    profiles="mission",
):
    """Build selected S/M manifests without writing them.

    When ``generator_seed`` is omitted, mission and unknown-map families use
    their own registered seed for the requested split.  An explicit seed is
    intentionally applied to every selected family.
    """

    selected = _resolve_profiles(profiles)
    manifests = {}
    explicit_seed = (
        None if generator_seed is None else int(generator_seed)
    )

    if any(profile in SCENARIO_PROFILES for profile in selected):
        mission_seed = (
            explicit_seed
            if explicit_seed is not None
            else _default_generator_seed(split, unknown=False)
        )
        manifests.update({
            profile: manifest
            for profile, manifest in _build_mission_profile_manifests(
                count, mission_seed, split
            ).items()
            if profile in selected
        })

    if any(profile in UNKNOWN_MAP_PROFILES for profile in selected):
        unknown_seed = (
            explicit_seed
            if explicit_seed is not None
            else _default_generator_seed(split, unknown=True)
        )
        manifests.update({
            profile: manifest
            for profile, manifest in _build_unknown_profile_manifests(
                count, unknown_seed, split
            ).items()
            if profile in selected
        })

    return manifests


def write_scenario_manifests(kind="all", profiles=None, output_root=None):
    """Write manifests for any selected S/M profiles through one dispatcher."""

    selected = _resolve_profiles(profiles)
    if kind not in ("train", "validation", "smoke", "all"):
        raise ValueError(f"unsupported manifest kind={kind!r}")
    outputs = {}
    mission_requests = []
    if kind in ("train", "all"):
        mission_requests.append(
            ("train", DEFAULT_TRAIN_SCENARIO_COUNT, TRAIN_GENERATOR_SEED)
        )
    if kind in ("validation", "all"):
        mission_requests.append(
            (
                "validation",
                DEFAULT_VALIDATION_SCENARIO_COUNT,
                VALIDATION_GENERATOR_SEED,
            )
        )
    if kind in ("smoke", "all"):
        mission_requests.extend(
            (
                (
                    "smoke_train",
                    DEFAULT_SMOKE_SCENARIO_COUNT,
                    SMOKE_TRAIN_GENERATOR_SEED,
                ),
                (
                    "smoke_validation",
                    DEFAULT_SMOKE_SCENARIO_COUNT,
                    SMOKE_VALIDATION_GENERATOR_SEED,
                ),
            )
        )
    mission_root = (
        Path(output_root) if output_root is not None else MANIFEST_ROOT
    )
    if any(profile in SCENARIO_PROFILES for profile in selected):
        for split, count, seed in mission_requests:
            generated = _build_mission_profile_manifests(
                count, seed, split
            )
            for profile, manifest in generated.items():
                if profile not in selected:
                    continue
                short_code = PROFILE_CODES[profile].split("_", 1)[0]
                path = mission_root / f"mission_{split}_{short_code}.json"
                _write_manifest(path, manifest)
                outputs[f"{split}_{profile}"] = path

    unknown_requests = []
    if kind in ("train", "all"):
        unknown_requests.append((
            "train",
            UNKNOWN_DEFAULT_TRAIN_SCENARIO_COUNT,
            UNKNOWN_TRAIN_GENERATOR_SEED,
        ))
    if kind in ("validation", "all"):
        unknown_requests.append((
            "validation",
            UNKNOWN_DEFAULT_VALIDATION_SCENARIO_COUNT,
            UNKNOWN_VALIDATION_GENERATOR_SEED,
        ))
    if kind in ("smoke", "all"):
        unknown_requests.extend((
            (
                "smoke_train",
                UNKNOWN_DEFAULT_SMOKE_SCENARIO_COUNT,
                UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED,
            ),
            (
                "smoke_validation",
                UNKNOWN_DEFAULT_SMOKE_SCENARIO_COUNT,
                UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED,
            ),
        ))
    unknown_root = (
        Path(output_root)
        if output_root is not None
        else UNKNOWN_MANIFEST_ROOT
    )
    if any(profile in UNKNOWN_MAP_PROFILES for profile in selected):
        for split, count, seed in unknown_requests:
            generated = _build_unknown_profile_manifests(
                count, seed, split
            )
            for profile, manifest in generated.items():
                if profile not in selected:
                    continue
                code = PROFILE_CODES[profile]
                path = unknown_root / f"unknown_{split}_{code}.json"
                _write_manifest(path, manifest)
                outputs[f"{split}_{profile}"] = path
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("all", "train", "validation", "smoke"),
        default="all",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="all, mission, unknown, or one or more explicit profiles",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=ALL_SCENARIO_PROFILES,
        help="select one profile; may be repeated",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    selectors = (
        args.profile
        if args.profile
        else args.profiles
    )
    for label, path in write_scenario_manifests(
        args.kind,
        profiles=selectors,
        output_root=args.output_root,
    ).items():
        print(f"[CH3] wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
