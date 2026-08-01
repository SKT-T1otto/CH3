"""Unified episode metrics with finite/blank serialization semantics."""

from __future__ import annotations

import json
import math


def _vector_text(value):
    if value is None:
        return None
    return json.dumps([float(item) for item in value], separators=(",", ":"))


def _finite_or_none(row):
    return {
        key: (
            None
            if isinstance(value, float) and not math.isfinite(value)
            else value
        )
        for key, value in row.items()
    }


def augment_episode_metrics(row, env, scenario=None):
    row = dict(row)
    scenario = dict(scenario or {})
    failure = int(env.max_steps + env.failure_penalty_steps)
    found_step = env.found_step
    success_step = env.success_step
    steps = max(1, int(env.step_count))
    row.update({
        "scenario_profile": scenario.get("scenario_profile", env.scenario_profile),
        "pair_group_id": scenario.get("pair_group_id"),
        "target_motion_mode": env.target_state.motion_mode,
        "target_initial_speed": float(
            sum(float(x) ** 2 for x in env.initial_target_state.velocity) ** 0.5
        ),
        "target_mean_speed": float(env.target_distance_travelled / (steps * env.dt)),
        "target_distance_travelled": float(env.target_distance_travelled),
        "target_reflection_count": int(env.target_state.reflection_count),
        "target_position_at_found": _vector_text(env.target_position_at_found),
        "target_velocity_at_found": _vector_text(env.target_velocity_at_found),
        "handoff_payload_sample_step": env.handoff_payload_sample_step,
        "handoff_payload_delivery_step": env.handoff_payload_delivery_step,
        "handoff_payload_age_steps": env.handoff_payload_age_steps,
        "handoff_delivery_phase": env.handoff_delivery_phase,
        "handoff_event_delay_steps": env.handoff_event_delay_steps,
        "handoff_physical_age_at_delivery_steps": (
            env.handoff_physical_age_at_delivery_steps
        ),
        "predicted_target_position_at_delivery": _vector_text(
            env.predicted_target_position_at_delivery
        ),
        "target_prediction_error_at_delivery": env.target_prediction_error_at_delivery,
        "mean_target_prediction_error": env.mean_target_prediction_error,
        "predicted_intercept_position": _vector_text(env.predicted_intercept_position),
        "target_position_at_capture": _vector_text(env.target_position_at_capture),
        "capture_position_error": env.capture_position_error,
        "capture_swept_min_distance": env.capture_swept_min_distance,
        "capture_contact_step_count": int(env.capture_contact_step_count),
        "capture_full_hold_step_count": int(env.capture_full_hold_step_count),
        "capture_hold_counter_max": int(env.capture_hold_counter_max),
        "path_replan_count": int(env.path_replan_count),
        "path_unreachable_count": int(
            env.path_unreachable_count
            + getattr(env.map_module, "waypoint_unreachable_event_count", 0)
        ),
        "planned_geodesic_distance": float(env.planned_geodesic_distance),
        "executed_path_distance": float(env.executed_path_distance),
        "subgoal_count": int(env.subgoal_count),
        "obstacle_collision_count": int(env.obstacle_collision_count),
        "waypoint_endpoint_guard_reject_count": int(
            getattr(env, "waypoint_endpoint_guard_reject_count", 0)
        ),
        "waypoint_endpoint_point_invalid_count": int(
            getattr(env, "waypoint_endpoint_point_invalid_count", 0)
        ),
        "waypoint_endpoint_no_connector_count": int(
            getattr(env, "waypoint_endpoint_no_connector_count", 0)
        ),
        "waypoint_endpoint_guard_recovery_count": int(
            getattr(env, "waypoint_endpoint_guard_recovery_count", 0)
        ),
        "waypoint_endpoint_guard_max_streak": int(
            getattr(env, "waypoint_endpoint_guard_max_streak", 0)
        ),
        "path_replan_deferred_invalid_endpoint_count": int(
            getattr(
                env,
                "path_replan_deferred_invalid_endpoint_count",
                0,
            )
        ),
        "path_subgoal_advance_deferred_invalid_endpoint_count": int(
            getattr(
                env,
                "path_subgoal_advance_deferred_invalid_endpoint_count",
                0,
            )
        ),
        "penalized_found_step": int(found_step) if found_step is not None else failure,
        "penalized_completion_step": (
            int(success_step) if success_step is not None else failure
        ),
        "normalized_penalized_completion": (
            (int(success_step) if success_step is not None else failure)
            / float(env.max_steps)
        ),
        "completion_failure": int(success_step is None),
        "search_failure": int(found_step is None),
    })
    if getattr(env, "artifact_protocol", None) == "ch3_unknown_map_v1":
        row.update(env.get_unknown_map_metrics())
        row.update({
            "artifact_protocol": "ch3_unknown_map_v1",
            "target_motion_known": bool(env.target_motion_known),
            "unknown_map_schema": env.unknown_map_schema,
            "target_belief_schema": env.target_belief_schema,
            "map_sharing_mode": env.map_sharing_mode,
            "ground_truth_obstacle_count": len(env.ground_truth_obstacles),
        })
    return _finite_or_none(row)
