"""Reproduce and capture S01 illegal planner endpoints without changing production code."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import UAVEnv
from map.path_planner import ObstacleAwareTaskMapPlanner
from target_motion import segment_aabb_first_hit
from tools.run_ch3 import main as run_main
import training


DIAGNOSTIC_ROOT = (
    PROJECT_ROOT
    / "data"
    / "chapter3"
    / "diagnostics"
    / "s01_illegal_endpoint_repro"
)
OUTPUT = DIAGNOSTIC_ROOT / "illegal_endpoint_diagnostic.json"
RUN_DIR = (
    DIAGNOSTIC_ROOT
    / "ch3_v3_full_reference"
    / "S01_STATIC_OBSTACLE"
    / "seed_1"
)

_context = {
    "episode": 33,
    "env": None,
    "agent_id": None,
    "previous_positions": None,
    "previous_collision_flags": None,
    "actions": None,
    "dynamics_history": [],
}


def _plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _aabb_state(point, obstacle, clearance):
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(obstacle["center"], dtype=np.float64)
    size = np.asarray(obstacle["size"], dtype=np.float64)
    lower = center - size / 2.0
    upper = center + size / 2.0
    expanded_lower = lower - clearance
    expanded_upper = upper + clearance
    outside_delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    if np.all(point >= lower) and np.all(point <= upper):
        surface_distance = -float(
            np.min(np.minimum(point - lower, upper - point))
        )
    else:
        surface_distance = float(np.linalg.norm(outside_delta))
    return {
        "center": center,
        "size": size,
        "lower": lower,
        "upper": upper,
        "expanded_lower": expanded_lower,
        "expanded_upper": expanded_upper,
        "inside_true_aabb": bool(
            np.all(point >= lower) and np.all(point <= upper)
        ),
        "inside_expanded_aabb": bool(
            np.all(point >= expanded_lower)
            and np.all(point <= expanded_upper)
        ),
        "signed_distance_to_true_aabb_surface": surface_distance,
    }


def _blocked_by(planner, start, end):
    blocked = []
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    for index, (lower, upper) in enumerate(planner._obstacle_boxes_np):
        expanded_lower = lower - planner.planner_obstacle_clearance
        expanded_upper = upper + planner.planner_obstacle_clearance
        endpoint_inside = bool(
            (
                np.all(start >= expanded_lower)
                and np.all(start <= expanded_upper)
            )
            or (
                np.all(end >= expanded_lower)
                and np.all(end <= expanded_upper)
            )
        )
        hit = segment_aabb_first_hit(
            start, end, expanded_lower, expanded_upper
        )
        if endpoint_inside or (
            hit is not None and -planner.eps <= hit[0] <= 1.0 + planner.eps
        ):
            blocked.append({
                "obstacle_index": index,
                "endpoint_inside": endpoint_inside,
                "hit": hit,
            })
    return blocked


def _planner_snapshot(planner, point, env, agent_id):
    point_tensor = planner._as_points(point).reshape(3)
    point_np = point_tensor.detach().cpu().numpy().astype(np.float64)
    valid = bool(planner._point_is_valid(point_tensor))
    labels = planner._component_labels()
    component_ids = sorted(set(labels.values()))
    before_keys = list(planner._geodesic_cache)
    connectors_before = planner._connector_candidates(
        point_tensor, role="searcher"
    )

    deltas = planner._flat_xyz_centers_np - point_np
    valid_flats = planner._valid_flats_np
    nearest_flat = int(
        valid_flats[
            int(np.argmin(np.sum(deltas[valid_flats] ** 2, axis=1)))
        ]
    )
    nearest_cell = planner._unflatten(nearest_flat)
    nearest_center = planner._flat_xyz_centers_np[nearest_flat]

    connector_rows = []
    for cost, cell in connectors_before:
        connector_rows.append({
            "cost": float(cost),
            "cell": cell,
            "component_id": labels.get(tuple(cell)),
            "center": planner.xyz_centers[tuple(cell)],
        })

    planner._geodesic_cache.clear()
    connectors_after_clear = planner._connector_candidates(
        point_tensor, role="searcher"
    )
    labels_after = planner._component_labels()

    nearest_index = nearest_cell
    neighborhood = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (
                    nearest_index[0] + dx,
                    nearest_index[1] + dy,
                    nearest_index[2] + dz,
                )
                if all(
                    0 <= cell[axis] < planner.grid_size[axis]
                    for axis in range(3)
                ):
                    neighborhood.append({
                        "cell": cell,
                        "center": planner.xyz_centers[cell],
                        "valid": bool(planner.valid_mask[cell].item()),
                        "component_id": labels_after.get(cell),
                    })

    waypoint = env._search_waypoints[agent_id]
    waypoint_valid = bool(planner._point_is_valid(waypoint))
    waypoint_connectors = planner._connector_candidates(
        waypoint, role="searcher"
    )
    nearest_is_free = bool(
        planner.segment_is_free(point_tensor, nearest_center)
    )
    exact_center = bool(
        np.min(np.sum(deltas ** 2, axis=1)) <= planner.eps ** 2
    )
    return {
        "point_is_valid": valid,
        "failure_class": (
            "A_POINT_INVALID"
            if not valid
            else (
                "B_VALID_POINT_NO_CONNECTORS"
                if not connectors_before
                else "OTHER"
            )
        ),
        "connector_count_before": len(connectors_before),
        "connectors_before": connector_rows,
        "connector_count_after_cache_clear": len(connectors_after_clear),
        "connectors_after_cache_clear": connectors_after_clear,
        "cache_changed_result": (
            connectors_before != connectors_after_clear
        ),
        "cache_key_count_before": len(before_keys),
        "endpoint_cache_keys_before": [
            key for key in before_keys
            if isinstance(key, tuple)
            and key
            and key[0] == "endpoint_connectors"
        ],
        "nearest_valid_grid_cell": nearest_cell,
        "nearest_valid_grid_center": nearest_center,
        "distance_to_nearest_valid_grid_center": float(
            np.linalg.norm(point_np - nearest_center)
        ),
        "segment_to_nearest_center_is_free": nearest_is_free,
        "segment_to_nearest_center_blocked_by": _blocked_by(
            planner, point_np, nearest_center
        ),
        "projected_endpoint_connector_count": len(
            planner._connector_candidates(nearest_center, role="searcher")
        ),
        "grid_revision": int(planner.grid_revision),
        "obstacle_layout_hash": planner.obstacle_layout_hash,
        "connected_component_count": len(component_ids),
        "connected_component_ids": component_ids,
        "nearby_grid_cells_3x3x3": neighborhood,
        "single_component_fast_branch": {
            "single_component": len(component_ids) == 1,
            "current_pos_equals_grid_center": exact_center,
            "ordinary_search_executed": (
                len(component_ids) != 1
                or not exact_center
            ),
        },
        "search_waypoint_valid": waypoint_valid,
        "search_waypoint_connector_count": len(waypoint_connectors),
    }


def _capture(planner, current_pos, agent_id, error):
    env = _context["env"]
    point = np.asarray(_plain(current_pos), dtype=np.float64)
    obstacles = list(env.obstacles)
    clearance = float(planner.planner_obstacle_clearance)
    obstacle_states = [
        _aabb_state(point, obstacle, clearance)
        for obstacle in obstacles
    ]
    bounds = env.space_size.detach().cpu().numpy()
    path = env._navigation_paths[agent_id]
    path_index = env._navigation_path_indices[agent_id]
    waypoint = env._search_waypoints[agent_id]
    payload = {
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "episode": int(_context["episode"]),
        "step_count": int(env.step_count),
        "agent_id": int(agent_id),
        "current_pos": point,
        "previous_positions": _context["previous_positions"],
        "previous_agent_pos": (
            None
            if _context["previous_positions"] is None
            else _context["previous_positions"][agent_id]
        ),
        "current_velocity": env._agent_vel[agent_id],
        "current_acceleration": env._agent_acc[agent_id],
        "last_prior_acceleration": env._last_prior_acc[agent_id],
        "last_residual_acceleration": env._last_residual_acc[agent_id],
        "input_action": (
            None
            if _context["actions"] is None
            else _context["actions"][agent_id]
        ),
        "dynamics_history": _context["dynamics_history"],
        "current_search_waypoint": waypoint,
        "current_nav_target": env._nav_targets[agent_id],
        "distance_to_search_waypoint": float(
            torch.norm(env._agent_pos[agent_id] - waypoint).item()
        ),
        "search_arrive_eps": float(env.search_arrive_eps),
        "current_target_arrived": bool(
            env.current_target_arrived[agent_id].item()
        ),
        "just_reached_waypoint": bool(
            env.just_reached_waypoint[agent_id].item()
        ),
        "waypoint_reached_counts": env.waypoint_reached_counts,
        "navigation_path": path,
        "navigation_path_index": int(path_index),
        "navigation_path_final_target": env._path_final_targets[agent_id],
        "collision_flag": bool(env._collision_flags[agent_id].item()),
        "previous_step_collision_flag": (
            None
            if _context["previous_collision_flags"] is None
            else bool(_context["previous_collision_flags"][agent_id])
        ),
        "obstacles": obstacles,
        "obstacle_layout_id": env.obstacle_layout_id,
        "planner_obstacle_clearance": clearance,
        "obstacle_states": obstacle_states,
        "inside_any_true_obstacle": any(
            item["inside_true_aabb"] for item in obstacle_states
        ),
        "inside_any_expanded_obstacle": any(
            item["inside_expanded_aabb"] for item in obstacle_states
        ),
        "out_of_bounds": bool(
            np.any(point < 0.0) or np.any(point > bounds)
        ),
        "finite": bool(np.all(np.isfinite(point))),
        "planner": _planner_snapshot(
            planner, current_pos, env, agent_id
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_original_reset = UAVEnv.reset
_original_dynamics = UAVEnv._apply_agent_dynamics
_original_choose = UAVEnv._choose_next_search_waypoint
_original_sample = ObstacleAwareTaskMapPlanner.sample_next_waypoint


def _reset(self, scenario=None):
    _context["env"] = self
    return _original_reset(self, scenario=scenario)


def _dynamics(self, actions):
    _context["env"] = self
    _context["previous_positions"] = self._agent_pos.detach().cpu().clone()
    _context["previous_collision_flags"] = (
        self._collision_flags.detach().cpu().clone()
    )
    _context["actions"] = torch.as_tensor(actions).detach().cpu().clone()
    result = _original_dynamics(self, actions)
    _context["dynamics_history"].append({
        "step_before_increment": int(self.step_count),
        "positions_before": _context["previous_positions"],
        "positions_after": self._agent_pos.detach().cpu().clone(),
        "velocities_after": self._agent_vel.detach().cpu().clone(),
        "accelerations_after": self._agent_acc.detach().cpu().clone(),
        "prior_after": self._last_prior_acc.detach().cpu().clone(),
        "residual_after": self._last_residual_acc.detach().cpu().clone(),
        "collision_flags_after":
            self._collision_flags.detach().cpu().clone(),
        "planner_point_valid_after": [
            bool(self.map_module._point_is_valid(self._agent_pos[index]))
            for index in range(4)
        ],
    })
    if len(_context["dynamics_history"]) > 400:
        _context["dynamics_history"] = _context["dynamics_history"][-400:]
    return result


def _choose(self, agent_id, reserved_positions=None):
    _context["env"] = self
    _context["agent_id"] = int(agent_id)
    return _original_choose(
        self, agent_id, reserved_positions=reserved_positions
    )


def _sample(
    self, agent_id, current_pos, reserved_positions=None, anchor=None
):
    try:
        return _original_sample(
            self,
            agent_id,
            current_pos,
            reserved_positions=reserved_positions,
            anchor=anchor,
        )
    except RuntimeError as error:
        if str(error) == "current_pos is not a legal reachable planner point":
            _capture(self, current_pos, int(agent_id), error)
        raise


_original_run_episode = training.run_episode


def _run_episode(runtime, *args, **kwargs):
    if bool(kwargs.get("explore", False)):
        _context["episode"] += 1
    return _original_run_episode(runtime, *args, **kwargs)


def main():
    if not (RUN_DIR / "resume_state.pt").is_file():
        raise FileNotFoundError(RUN_DIR / "resume_state.pt")
    state = torch.load(
        RUN_DIR / "resume_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(state.get("episode", -1)) != 33:
        raise RuntimeError(
            f"expected resume episode 33, got {state.get('episode')}"
        )
    _context["episode"] = int(state["episode"])
    _context["dynamics_history"] = []
    UAVEnv.reset = _reset
    UAVEnv._apply_agent_dynamics = _dynamics
    UAVEnv._choose_next_search_waypoint = _choose
    ObstacleAwareTaskMapPlanner.sample_next_waypoint = _sample
    training.run_episode = _run_episode
    return run_main([
        "--phase", "train",
        "--base-candidate", "ch3_v3_full_reference",
        "--scenario-profile", "S01_STATIC_OBSTACLE",
        "--seed", "1",
        "--episodes", "200",
        "--max-steps", "400",
        "--replay-size", "500000",
        "--checkpoint-interval", "50",
        "--evaluation-limit", "10",
        "--device", "auto",
        "--output-dir", str(DIAGNOSTIC_ROOT),
        "--allow-long-run",
        "--resume",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
