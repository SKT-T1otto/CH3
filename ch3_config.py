"""Unified configuration builder for known- and unknown-map Chapter-3 profiles."""

from __future__ import annotations

from copy import deepcopy

from ch3_constants import (
    ALL_SCENARIO_PROFILES,
    CH3_MISSION_V1,
    CH3_UNKNOWN_MAP_V1,
    MISSION_SCENARIO_PROFILES,
    UNKNOWN_MAP_PROFILES,
)
from registry.ch3_efficiency_v3_registry import (
    CH3_EFFICIENCY_V3_SCREEN_METHODS,
    resolve_ch3_efficiency_v3_config,
)


_COMMON = {
    "protocol": CH3_MISSION_V1,
    "target_continues_after_detection": True,
    "target_state_schema": "moving_target_state_v1",
    "target_capture_radius": 0.80,
    "target_capture_hold_steps": 5,
    "target_max_reflections_per_step": 4,
    "handoff_payload_schema": "moving_target_position_velocity_timestamp_v1",
    "executor_intercept_iterations": 4,
    "planner_obstacle_clearance": 0.40,
    "target_obstacle_clearance": 0.20,
    "path_subgoal_radius": 0.75,
    "path_replan_interval": 10,
    "failure_penalty_steps": 100,
}

_MISSION_SETTINGS = {
    "S00_STATIC_CLEAR": {
        "target_motion_mode": "static",
        "target_belief_transition_mode": "static",
        "target_belief_diffusion_rate": 0.0,
        "use_obstacles": False,
    },
    "S10_MOVING_CLEAR": {
        "target_motion_mode": "constant_velocity_reflect_v1",
        "target_belief_transition_mode": "isotropic_diffusion_v1",
        "target_belief_diffusion_rate": 0.12,
        "use_obstacles": False,
    },
    "S01_STATIC_OBSTACLE": {
        "target_motion_mode": "static",
        "target_belief_transition_mode": "static",
        "target_belief_diffusion_rate": 0.0,
        "use_obstacles": True,
    },
    "S11_MOVING_OBSTACLE": {
        "target_motion_mode": "constant_velocity_reflect_v1",
        "target_belief_transition_mode": "isotropic_diffusion_v1",
        "target_belief_diffusion_rate": 0.12,
        "use_obstacles": True,
    },
}

_UNKNOWN_COMMON = {
    "artifact_protocol": CH3_UNKNOWN_MAP_V1,
    "environment_class": "env.UAVEnv",
    "target_motion_known": True,
    "target_motion_mode": "constant_velocity_reflect_v1",
    "target_belief_transition_mode": "occupancy_constrained_diffusion_v1",
    "target_belief_diffusion_rate": 0.12,
    "target_negative_observation_strength": 0.90,
    "target_negative_likelihood_floor": 0.05,
    "target_revisit_half_life_steps": 30.0,
    "target_recency_penalty_weight": 0.15,
    "obstacle_information_gain_weight": 0.10,
    "executor_intercept_mode": "known-map-conditional-reflect-fixed-point_v1",
    "travel_cost_mode": "online_grid_geodesic_v1",
    "navigation_path_mode": "online_grid_astar_subgoals_v1",
    "obstacle_sensor_range": 4.5,
    "obstacle_sensor_ray_mode": "neighbor26_v1",
    "obstacle_sensor_noise_std": 0.0,
    "occupancy_free_logodds": -0.85,
    "occupancy_occupied_logodds": 1.70,
    "occupancy_logodds_clip": 6.0,
    "occupancy_free_threshold": 0.30,
    "occupancy_occupied_threshold": 0.70,
    "occupancy_unknown_cost_weight": 0.35,
    "occupancy_risk_cost_weight": 1.25,
    "occupancy_replan_probability_delta": 0.08,
    "reservation_decay": 0.985,
    "replan_on_map_change": True,
    "unknown_map_schema": "shared_logodds_occupancy_v1",
    "target_belief_schema": "moving_target_bayes_filter_v1",
    "map_sharing_mode": "central_shared_deterministic_v1",
}

_UNKNOWN_SETTINGS = {
    "M00_MOVING_CLEAR": {
        "use_obstacles": False,
        "obstacle_knowledge_mode": "online_unknown",
        "obstacle_family": "clear",
        "planner_mode": "online_astar_v1",
    },
    "M10_MOVING_UNKNOWN_SINGLE": {
        "use_obstacles": True,
        "obstacle_knowledge_mode": "online_unknown",
        "obstacle_family": "random_single_aabb_v1",
        "planner_mode": "online_astar_v1",
    },
    "M20_MOVING_UNKNOWN_MULTI": {
        "use_obstacles": True,
        "obstacle_knowledge_mode": "online_unknown",
        "obstacle_family": "random_multi_aabb_v1",
        "planner_mode": "online_astar_v1",
    },
    "M90_MOVING_KNOWN_ORACLE": {
        "use_obstacles": True,
        "obstacle_knowledge_mode": "oracle",
        "obstacle_family": "random_multi_aabb_v1",
        "planner_mode": "oracle_astar_v1",
        "target_belief_transition_mode": "isotropic_diffusion_v1",
    },
}


def build_ch3_config(base_candidate: str, scenario_profile: str) -> dict:
    if base_candidate not in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        raise ValueError(f"unsupported Chapter-3 base candidate={base_candidate!r}")
    if scenario_profile not in ALL_SCENARIO_PROFILES:
        raise ValueError(
            f"unsupported scenario profile={scenario_profile!r}; "
            f"expected {ALL_SCENARIO_PROFILES}"
        )
    config = deepcopy(resolve_ch3_efficiency_v3_config(base_candidate))
    config.update(_COMMON)
    if scenario_profile in MISSION_SCENARIO_PROFILES:
        config.update(
            executor_intercept_mode="constant_velocity_reflect_fixed_point_v1",
            travel_cost_mode="grid_geodesic_v1",
            navigation_path_mode="grid_astar_subgoals_v1",
        )
        config.update(_MISSION_SETTINGS[scenario_profile])
    else:
        config.update(_UNKNOWN_COMMON)
        config.update(_UNKNOWN_SETTINGS[scenario_profile])
    config.update(
        base_candidate=base_candidate,
        scenario_profile=scenario_profile,
    )
    return config


def build_mission_config(base_candidate: str, scenario_profile: str) -> dict:
    if scenario_profile not in MISSION_SCENARIO_PROFILES:
        raise ValueError(f"unsupported mission scenario profile={scenario_profile!r}")
    return build_ch3_config(base_candidate, scenario_profile)


def build_unknown_map_config(base_candidate: str, scenario_profile: str) -> dict:
    if scenario_profile not in UNKNOWN_MAP_PROFILES:
        raise ValueError(f"unsupported unknown-map profile={scenario_profile!r}")
    return build_ch3_config(base_candidate, scenario_profile)


def profile_diff(left: dict, right: dict) -> set[str]:
    return {
        key for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


scenario_profile_diff = profile_diff
unknown_profile_diff = profile_diff
