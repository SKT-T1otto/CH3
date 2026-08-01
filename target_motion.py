"""Deterministic target state, reflection, detection, and interception."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Iterable

import numpy as np


EPS = 1e-9
TARGET_STATE_SCHEMA = "moving_target_state_v1"
TARGET_PAYLOAD_RESERVED_FIELDS = frozenset({
    "position",
    "position_at_detection",
    "velocity",
    "velocity_at_detection",
    "sample_step",
    "motion_mode",
    "reflection_count",
    "obstacle_layout_id",
    "state_schema",
    "metadata",
})


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validate_finite_numbers(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{name}[{index}]")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite numbers")
    elif isinstance(value, Real) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} contains non-finite numbers")


def _vec3(value, name):
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result.copy()


def _expanded_boxes(obstacles: Iterable[dict], clearance: float):
    boxes = []
    clearance = float(clearance)
    for obstacle in obstacles or ():
        center = _vec3(obstacle["center"], "obstacle.center")
        size = _vec3(obstacle["size"], "obstacle.size")
        if np.any(size <= 0):
            raise ValueError("obstacle size must be strictly positive")
        half = size / 2.0 + clearance
        boxes.append((center - half, center + half))
    return boxes


def segment_aabb_first_hit(start, end, lower, upper):
    """Return the earliest segment/AABB entry as ``(tau, normal)`` or ``None``."""
    start, end = _vec3(start, "start"), _vec3(end, "end")
    lower, upper = _vec3(lower, "lower"), _vec3(upper, "upper")
    direction = end - start
    t_enter, t_exit = 0.0, 1.0
    entering = []
    for axis in range(3):
        if abs(direction[axis]) <= EPS:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        t_low = (lower[axis] - start[axis]) / direction[axis]
        t_high = (upper[axis] - start[axis]) / direction[axis]
        low_normal = np.zeros(3, dtype=np.float64)
        high_normal = np.zeros(3, dtype=np.float64)
        low_normal[axis] = -1.0
        high_normal[axis] = 1.0
        if t_low > t_high:
            t_low, t_high = t_high, t_low
            low_normal, high_normal = high_normal, low_normal
        if t_low > t_enter + EPS:
            t_enter, entering = t_low, [low_normal]
        elif abs(t_low - t_enter) <= EPS:
            entering.append(low_normal)
        t_exit = min(t_exit, t_high)
        if t_enter - t_exit > EPS:
            return None
    if not entering or t_enter < -EPS or t_enter > 1.0 + EPS:
        return None
    normal = np.sum(entering, axis=0)
    return float(max(0.0, t_enter)), normal


def _boundary_first_hit(start, end, bounds):
    bounds = _vec3(bounds, "bounds")
    direction = end - start
    hits = []
    for axis in range(3):
        if direction[axis] < -EPS and end[axis] < 0.0:
            tau, sign = (0.0 - start[axis]) / direction[axis], 1.0
        elif direction[axis] > EPS and end[axis] > bounds[axis]:
            tau, sign = (bounds[axis] - start[axis]) / direction[axis], -1.0
        else:
            continue
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        hits.append((float(tau), axis, normal))
    if not hits:
        return None
    earliest = min(item[0] for item in hits)
    normal = np.sum(
        [item[2] for item in hits if abs(item[0] - earliest) <= EPS], axis=0
    )
    return earliest, normal


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    sample_step: int
    motion_mode: str
    reflection_count: int = 0
    state_schema: str = TARGET_STATE_SCHEMA
    obstacle_layout_id: str = "none"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.position = _vec3(self.position, "position")
        self.velocity = _vec3(self.velocity, "velocity")
        self.sample_step = _nonnegative_int(self.sample_step, "sample_step")
        self.reflection_count = _nonnegative_int(
            self.reflection_count, "reflection_count"
        )
        self.motion_mode = str(self.motion_mode)
        self.state_schema = str(self.state_schema)
        if self.state_schema != TARGET_STATE_SCHEMA:
            raise ValueError(
                f"state_schema must be {TARGET_STATE_SCHEMA!r}, "
                f"got {self.state_schema!r}"
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        conflicts = sorted(TARGET_PAYLOAD_RESERVED_FIELDS & set(self.metadata))
        if conflicts:
            raise ValueError(
                "metadata contains reserved payload fields: " + ", ".join(conflicts)
            )
        self.metadata = deepcopy(self.metadata)
        _validate_finite_numbers(self.metadata, "metadata")

    def copy(self):
        return TargetState(
            self.position.copy(), self.velocity.copy(), self.sample_step,
            self.motion_mode, self.reflection_count, self.state_schema,
            self.obstacle_layout_id, deepcopy(self.metadata),
        )

    def to_payload(self):
        payload = {
            "position": self.position.tolist(),
            "position_at_detection": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "velocity_at_detection": self.velocity.tolist(),
            "sample_step": self.sample_step,
            "motion_mode": self.motion_mode,
            "reflection_count": self.reflection_count,
            "obstacle_layout_id": self.obstacle_layout_id,
            "state_schema": self.state_schema,
            "metadata": deepcopy(self.metadata),
        }
        _validate_finite_numbers(payload, "payload")
        return payload

    @classmethod
    def from_payload(cls, payload):
        if isinstance(payload, cls):
            return payload.copy()
        if not isinstance(payload, dict):
            raise ValueError("target payload must be a mapping")
        if "position" in payload:
            position = payload["position"]
        elif "position_at_detection" in payload:
            position = payload["position_at_detection"]
        else:
            raise ValueError("target payload has no position field")
        if "velocity" in payload:
            velocity = payload["velocity"]
        elif "velocity_at_detection" in payload:
            velocity = payload["velocity_at_detection"]
        else:
            raise ValueError("target payload has no velocity field")
        nested = payload.get("metadata", {})
        if not isinstance(nested, dict):
            raise ValueError("target payload metadata must be a mapping")
        legacy = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in TARGET_PAYLOAD_RESERVED_FIELDS
        }
        overlap = sorted(set(legacy) & set(nested))
        if overlap:
            raise ValueError(
                "duplicate nested and legacy metadata keys: " + ", ".join(overlap)
            )
        metadata = deepcopy(nested)
        metadata.update(legacy)
        return cls(
            position,
            velocity,
            payload["sample_step"], payload["motion_mode"],
            payload.get("reflection_count", 0),
            payload.get("state_schema", TARGET_STATE_SCHEMA),
            payload.get("obstacle_layout_id", "none"),
            metadata,
        )

    def advance(self, dt, bounds, obstacles=(), clearance=0.0, max_reflections=4):
        return advance_target_state(
            self, dt, bounds, obstacles, clearance=clearance,
            max_reflections=max_reflections,
        )

    def predict(
        self, steps, dt, bounds, obstacles=(), clearance=0.0,
        max_reflections=4, max_prediction_steps=500,
    ):
        return predict_target_state(
            self, steps, dt, bounds, obstacles, clearance=clearance,
            max_reflections=max_reflections,
            max_prediction_steps=max_prediction_steps,
        )


def advance_target_state(
    state: TargetState, dt, bounds, obstacles=(), *, clearance=0.0,
    max_reflections=4,
):
    state = state.copy()
    dt = float(dt)
    bounds = _vec3(bounds, "bounds")
    if dt < 0:
        raise ValueError("dt must be nonnegative")
    if state.motion_mode == "static" or dt == 0:
        state.sample_step += 1
        return state
    if state.motion_mode != "constant_velocity_reflect_v1":
        raise ValueError(f"unsupported target motion mode={state.motion_mode!r}")
    boxes = _expanded_boxes(obstacles, clearance)
    if np.any(state.position < 0) or np.any(state.position > bounds):
        raise ValueError("target starts outside world bounds")
    if any(np.all(state.position > lo) and np.all(state.position < hi) for lo, hi in boxes):
        raise ValueError("target starts inside an expanded obstacle")
    position, velocity = state.position.copy(), state.velocity.copy()
    remaining = dt
    reflected = 0
    while remaining > EPS:
        candidate = position + velocity * remaining
        hits = []
        boundary = _boundary_first_hit(position, candidate, bounds)
        if boundary is not None:
            hits.append((boundary[0], "world_boundary", boundary[1]))
        for index, (lower, upper) in enumerate(boxes):
            hit = segment_aabb_first_hit(position, candidate, lower, upper)
            if hit is not None:
                hits.append((hit[0], f"obstacle[{index}]", hit[1]))
        if not hits:
            position = candidate
            remaining = 0.0
            break
        tau = min(item[0] for item in hits)
        simultaneous = [
            item for item in hits if abs(item[0] - tau) <= EPS
        ]
        normal = np.sum([item[2] for item in simultaneous], axis=0)
        hit_objects = [item[1] for item in simultaneous]
        position = position + velocity * remaining * max(0.0, tau)
        remaining *= max(0.0, 1.0 - tau)
        axes = np.flatnonzero(np.abs(normal) > EPS)
        if axes.size == 0:
            position = candidate
            remaining = 0.0
            break
        if reflected >= int(max_reflections):
            raise RuntimeError(
                "target reflection limit exceeded: "
                f"step={state.sample_step}, position={position.tolist()}, "
                f"velocity={velocity.tolist()}, remaining={remaining}, "
                f"hit_objects={hit_objects}"
            )
        velocity[axes] *= -1.0
        reflected += 1
        position = position + velocity * min(remaining, 1e-8)
        remaining = max(0.0, remaining - min(remaining, 1e-8))
    position = np.clip(position, 0.0, bounds)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise RuntimeError("target integration produced non-finite state")
    if any(np.all(position > lo) and np.all(position < hi) for lo, hi in boxes):
        raise RuntimeError("target integration ended inside an obstacle")
    state.position = position.copy()
    state.velocity = velocity.copy()
    state.sample_step += 1
    state.reflection_count += reflected
    return state


def predict_target_state(
    state, steps, dt, bounds, obstacles=(), *, clearance=0.0,
    max_reflections=4, max_prediction_steps=500,
):
    steps = _nonnegative_int(steps, "steps")
    maximum = _nonnegative_int(max_prediction_steps, "max_prediction_steps")
    if steps > maximum:
        raise ValueError(
            f"prediction steps={steps} exceeds max_prediction_steps={maximum}"
        )
    predicted = TargetState.from_payload(
        state.to_payload() if isinstance(state, TargetState) else state
    )
    for _ in range(steps):
        predicted = advance_target_state(
            predicted, dt, bounds, obstacles, clearance=clearance,
            max_reflections=max_reflections,
        )
    return predicted


def simulate_target_trajectory(
    state, steps, dt, bounds, obstacles=(), *, clearance=0.0,
    max_reflections=4, max_prediction_steps=10000,
):
    steps = _nonnegative_int(steps, "steps")
    maximum = _nonnegative_int(max_prediction_steps, "max_prediction_steps")
    if steps > maximum:
        raise ValueError(
            f"trajectory steps={steps} exceeds max_prediction_steps={maximum}"
        )
    current = state.copy()
    trajectory = [current.copy()]
    for _ in range(steps):
        current = advance_target_state(
            current, dt, bounds, obstacles, clearance=clearance,
            max_reflections=max_reflections,
        )
        trajectory.append(current.copy())
    return trajectory


def swept_relative_min_distance(agent_start, agent_end, target_start, target_end):
    a0, a1 = _vec3(agent_start, "agent_start"), _vec3(agent_end, "agent_end")
    t0, t1 = _vec3(target_start, "target_start"), _vec3(target_end, "target_end")
    r0 = a0 - t0
    delta = (a1 - a0) - (t1 - t0)
    denominator = float(np.dot(delta, delta))
    tau = 0.0 if denominator <= EPS else float(np.clip(-np.dot(r0, delta) / denominator, 0.0, 1.0))
    distance = float(np.linalg.norm(r0 + tau * delta))
    return distance, tau


def solve_intercept_point(
    executor_position,
    delivered_target_state: TargetState,
    current_step: int,
    travel_time_function: Callable,
    *,
    dt=0.2,
    bounds=(20.0, 20.0, 8.0),
    obstacles=(),
    clearance=0.0,
    max_iterations=4,
    max_reflections=4,
    max_prediction_steps=500,
):
    delivered = delivered_target_state.copy()
    age = max(0, int(current_step) - delivered.sample_step)
    if age > int(max_prediction_steps):
        return {"reachable": False, "position": None, "travel_time": None}
    current = predict_target_state(
        delivered, age, dt, bounds, obstacles, clearance=clearance,
        max_reflections=max_reflections,
        max_prediction_steps=max_prediction_steps,
    )
    intercept = current.position.copy()
    for _ in range(int(max_iterations)):
        travel_time = float(travel_time_function(executor_position, intercept))
        if not np.isfinite(travel_time):
            return {"reachable": False, "position": None, "travel_time": None}
        future_steps = max(0, int(np.ceil(travel_time / float(dt))))
        if future_steps > int(max_prediction_steps):
            return {"reachable": False, "position": None, "travel_time": None}
        future = predict_target_state(
            current, future_steps, dt, bounds, obstacles,
            clearance=clearance, max_reflections=max_reflections,
            max_prediction_steps=max_prediction_steps,
        )
        intercept = future.position.copy()
    final_travel_time = float(travel_time_function(executor_position, intercept))
    if not np.isfinite(final_travel_time):
        return {"reachable": False, "position": None, "travel_time": None}
    return {
        "reachable": True,
        "position": intercept.copy(),
        "travel_time": final_travel_time,
    }
