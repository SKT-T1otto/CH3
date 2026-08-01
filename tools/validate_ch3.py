"""Validate Chapter-3 routing, mission semantics, planning, and bounded stepping."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_env import UAVEnv as BaseUAVEnv
from ch3_config import build_mission_config, build_unknown_map_config
from ch3_constants import (
    ALL_SCENARIO_PROFILES,
    CH3_MISSION_V1,
    CH3_UNKNOWN_MAP_V1,
    CH3_ROOT,
    SCENARIO_PROFILES,
    SMOKE_TRAIN_GENERATOR_SEED,
    SMOKE_VALIDATION_GENERATOR_SEED,
    TRAIN_GENERATOR_SEED,
    UNKNOWN_MAP_PROFILES,
    VALIDATION_GENERATOR_SEED,
)
from env import UAVEnv as MissionUAVEnv
from map.path_planner import OnlineUnknownMapTaskPlanner
from runtime import build_runtime
from registry.ch3_efficiency_v3_registry import resolve_ch3_efficiency_v3_config
from tools.build_ch3_scenarios import (
    build_scenario_manifests,
    validate_scenario_reachability,
)
from train import (
    CH3_EFFICIENCY_V2,
    CH3_EFFICIENCY_V3_SCREEN,
    CH3_PILOT_V1,
    build_ch3_runtime,
    build_ch3_runtime_from_resolved_config,
)
from utils.provenance import (
    base_algorithm_source_fingerprint,
    mission_algorithm_source_fingerprint,
    repository_source_fingerprint,
    unknown_map_algorithm_source_fingerprint,
)


def _path_edges_are_free(planner, path):
    points = [np.asarray(point, dtype=np.float64) for point in path["points"]]
    return all(
        planner.segment_is_free(left, right)
        for left, right in zip(points, points[1:])
    )


def _validate_handoff(env, scenario):
    env.reset(scenario)
    env._publish_detection(0)
    sample_position = env.target_state.position.copy()
    env.step(torch.zeros((4, 3), dtype=torch.float32))
    return {
        "delay": env.last_handoff_delay == 1.0,
        "phase": env.handoff_delivery_phase == "pre_transition",
        "physical_age": env.handoff_physical_age_at_delivery_steps == 0,
        "prediction": (
            env.target_prediction_error_at_delivery is not None
            and env.target_prediction_error_at_delivery <= 1e-8
            and np.allclose(env.predicted_target_position_at_delivery, sample_position)
        ),
        "next_age": env.handoff_payload_age_steps == 1,
        "count": env.ch3_handoff_count == 1,
        "payload": (
            env.handoff_payload_sample_step == 0
            and env.executor_delivered_target_state is not None
            and env.executor_delivered_target_state.position.shape == (3,)
            and env.executor_delivered_target_state.velocity.shape == (3,)
        ),
    }


def _validate_swept_detection(env, scenario):
    env.reset(scenario)
    agent_start = env._agent_pos.detach().cpu().numpy().copy()
    agent_start[0] = np.asarray([5.0, 5.0, 4.0])
    env._agent_pos[0].copy_(env._vec([5.0, 5.0, 4.0]))
    target_start = np.asarray([0.0, 5.0, 4.0])
    env.target_state.position = np.asarray([10.0, 5.0, 4.0])
    env._task_target.copy_(env._vec(env.target_state.position))
    env._maybe_detect_swept(agent_start, target_start)
    return env.task_found and env.finder_idx == 0


def _validate_capture_hold(env, scenario):
    env.reset(scenario)
    env.task_found = True
    env.executor_target_assigned = True
    point = np.asarray([8.0, 8.0, 4.0])
    env._agent_pos[env.executor_idx].copy_(env._vec(point))
    env.target_state.position = point.copy()
    starts = env._agent_pos.detach().cpu().numpy().copy()
    for _ in range(env.target_capture_hold_steps):
        env._update_capture(starts, point)
    full_hold_success = (
        env.mission_complete
        and env.capture_full_hold_step_count == env.target_capture_hold_steps
    )

    env.reset(scenario)
    env.task_found = True
    env.executor_target_assigned = True
    env._agent_pos[env.executor_idx].copy_(env._vec([10.0, 5.0, 4.0]))
    crossing_starts = env._agent_pos.detach().cpu().numpy().copy()
    crossing_starts[env.executor_idx] = np.asarray([0.0, 5.0, 4.0])
    env.target_state.position = np.asarray([5.0, 5.0, 4.0])
    env._update_capture(crossing_starts, [5.0, 5.0, 4.0])
    crossing_rejected = (
        env.capture_contact_step_count == 1
        and env._capture_hold_counter == 0
        and not env.mission_complete
    )
    return full_hold_success and crossing_rejected


def _validate_waypoint_endpoint_guard(check, scenario):
    env = build_runtime(
        "ch3_v3_full_reference",
        "S01_STATIC_OBSTACLE",
        seed=1,
        max_steps=400,
        device="cpu",
        replay_size=8,
    ).env
    env.reset(scenario)
    invalid = env._vec(
        [8.526883125305176, 10.9552640914917, 6.954549312591553]
    )
    waypoint = env._vec([9.0, 11.0, 7.5])
    old_path = [
        env._vec([7.0, 11.0, 6.5]),
        waypoint.clone(),
    ]
    env.task_found = False
    env.agent_finished[:3] = False
    env.just_reached_waypoint.zero_()
    env._agent_pos[0].copy_(invalid)
    env._search_waypoints[0].copy_(waypoint)
    env._search_waypoints[1].copy_(env._vec([1.0, 1.0, 1.0]))
    env._search_waypoints[2].copy_(env._vec([19.0, 19.0, 7.0]))
    env._navigation_paths[0] = [point.clone() for point in old_path]
    env._navigation_path_indices[0] = 1
    env._nav_targets[0].copy_(waypoint)
    env._path_final_targets[0].copy_(waypoint)
    env._path_last_replan_steps[0] = 350
    env.step_count = 364
    env.diverse_fallback_prob = 0.0

    endpoint = env.map_module.endpoint_status(invalid)
    check(
        "waypoint_endpoint_guard:point_invalid",
        endpoint == {
            "point_valid": False,
            "connector_count": 0,
            "reachable": False,
            "failure_reason": "point_invalid",
        },
    )

    sample_called = []
    original_sample = env.map_module.sample_next_waypoint

    def forbidden_sample(*args, **kwargs):
        sample_called.append(True)
        raise AssertionError(
            "sample_next_waypoint called from an invalid endpoint"
        )

    before = {
        "waypoint": env._search_waypoints[0].clone(),
        "reached": int(env.waypoint_reached_counts[0].item()),
        "total": int(env.total_waypoints_per_agent[0].item()),
        "arrived": bool(env.current_target_arrived[0].item()),
        "visited": env.map_module.coverage.clone(),
        "claims": env.map_module.claim_count.clone(),
    }
    env.map_module.sample_next_waypoint = forbidden_sample
    try:
        env._update_search_path_events()
    finally:
        env.map_module.sample_next_waypoint = original_sample
    check(
        "waypoint_endpoint_guard:no_false_completion",
        not sample_called
        and torch.equal(env._search_waypoints[0], before["waypoint"])
        and int(env.waypoint_reached_counts[0].item())
        == before["reached"]
        and int(env.total_waypoints_per_agent[0].item())
        == before["total"]
        and bool(env.current_target_arrived[0].item())
        == before["arrived"]
        and not bool(env.just_reached_waypoint[0].item())
        and torch.equal(env.map_module.coverage, before["visited"])
        and torch.equal(env.map_module.claim_count, before["claims"])
        and env.waypoint_endpoint_guard_reject_count == 1
        and env.waypoint_endpoint_point_invalid_count == 1,
    )

    original_astar = env.map_module.grid_astar_path
    invalid_astar_calls = []

    def guarded_astar(start, goal, role="searcher"):
        if torch.allclose(torch.as_tensor(start), invalid):
            invalid_astar_calls.append(True)
            raise AssertionError("A* called from an invalid endpoint")
        return original_astar(start, goal, role=role)

    nav_before = {
        "path": [point.clone() for point in env._navigation_paths[0]],
        "index": env._navigation_path_indices[0],
        "target": env._nav_targets[0].clone(),
        "final": env._path_final_targets[0].clone(),
        "last": env._path_last_replan_steps[0],
        "deferred": env.path_replan_deferred_invalid_endpoint_count,
    }
    env.map_module.grid_astar_path = guarded_astar
    try:
        env._update_nav_targets(force=True)
    finally:
        env.map_module.grid_astar_path = original_astar
    check(
        "waypoint_endpoint_guard:replan_deferred",
        not invalid_astar_calls
        and len(nav_before["path"]) == len(env._navigation_paths[0])
        and all(
            torch.equal(left, right)
            for left, right in zip(
                nav_before["path"], env._navigation_paths[0]
            )
        )
        and env._navigation_path_indices[0] == nav_before["index"]
        and torch.equal(env._nav_targets[0], nav_before["target"])
        and torch.equal(
            env._path_final_targets[0], nav_before["final"]
        )
        and env._path_last_replan_steps[0] == nav_before["last"]
        and env.path_replan_deferred_invalid_endpoint_count
        == nav_before["deferred"] + 1,
    )

    replacement = env._vec([3.0, 3.0, 3.0])
    recovery_calls = []

    def recovery_sample(*args, **kwargs):
        recovery_calls.append(int(kwargs["agent_id"]))
        env.map_module.register_waypoint_claim(replacement)
        return replacement.clone()

    reached_before = int(env.waypoint_reached_counts[0].item())
    env._agent_pos[0].copy_(waypoint)
    env.map_module.sample_next_waypoint = recovery_sample
    try:
        env._update_search_path_events()
        env._update_search_path_events()
    finally:
        env.map_module.sample_next_waypoint = original_sample
    check(
        "waypoint_endpoint_guard:recovery",
        recovery_calls == [0]
        and env.waypoint_endpoint_guard_recovery_count == 1
        and int(env._waypoint_endpoint_guard_streak[0].item()) == 0
        and int(env.waypoint_reached_counts[0].item())
        == reached_before + 1
        and torch.equal(env._search_waypoints[0], replacement),
    )


def _unknown_validation_scenario(profile):
    single = [
        {"center": [10.0, 10.0, 4.0], "size": [2.0, 4.0, 4.0]}
    ]
    multi = [
        {"center": [8.0, 10.0, 3.0], "size": [2.0, 3.0, 3.0]},
        {"center": [13.0, 10.0, 5.0], "size": [2.5, 3.0, 3.0]},
    ]
    if profile == "M00_MOVING_CLEAR":
        obstacles = []
        knowledge = "online_unknown"
    elif profile == "M10_MOVING_UNKNOWN_SINGLE":
        obstacles = single
        knowledge = "online_unknown"
    elif profile == "M20_MOVING_UNKNOWN_MULTI":
        obstacles = multi
        knowledge = "online_unknown"
    else:
        obstacles = multi
        knowledge = "oracle"
    return {
        "scenario_id": f"validator_{profile}",
        "pair_group_id": "validator_pair",
        "scenario_seed": 91001,
        "planner_seed": 92001,
        "scenario_profile": profile,
        "scenario_role": "smoke_train",
        "scenario_split": "smoke_train",
        "protocol": CH3_UNKNOWN_MAP_V1,
        "initial_agent_positions": [
            [2.0, 2.0, 1.0],
            [2.0, 18.0, 2.0],
            [18.0, 2.0, 3.0],
            [18.0, 18.0, 4.0],
        ],
        "initial_executor_wait_point": [15.0, 15.0, 4.0],
        "target_position": [4.0, 10.0, 4.0],
        "target_initial_position": [4.0, 10.0, 4.0],
        "target_initial_velocity": [0.35, 0.20, 0.10],
        "target_motion_mode": "constant_velocity_reflect_v1",
        "target_motion_known": True,
        "target_state_schema": "moving_target_state_v1",
        "use_obstacles": bool(obstacles),
        "obstacle_layout_id":
            "custom_aabb_v1" if obstacles else "none",
        "obstacle_knowledge_mode": knowledge,
        "obstacles": obstacles,
        "flow_phase_x": 0.0,
        "flow_phase_y": 0.0,
    }


def _validate_unknown_profiles(selected, check, errors, profile_rows):
    for profile in UNKNOWN_MAP_PROFILES:
        if profile not in selected:
            continue
        try:
            config = build_unknown_map_config(
                "ch3_v3_full_reference", profile
            )
            check(
                f"{profile}:moving_known",
                config["target_motion_known"] is True
                and config["target_motion_mode"]
                == "constant_velocity_reflect_v1",
            )
            check(
                f"{profile}:knowledge_identity",
                config["obstacle_knowledge_mode"]
                == (
                    "oracle"
                    if profile == "M90_MOVING_KNOWN_ORACLE"
                    else "online_unknown"
                ),
            )
            profile_rows.setdefault(profile, {
                "artifact_protocol": CH3_UNKNOWN_MAP_V1,
                "target_motion_mode": config["target_motion_mode"],
                "obstacle_knowledge_mode":
                    config["obstacle_knowledge_mode"],
                "planner_mode": config["planner_mode"],
            })
        except Exception as exc:
            errors.append({
                "check": f"{profile}:config",
                "detail": repr(exc),
            })

    for profile in (
        "M10_MOVING_UNKNOWN_SINGLE",
        "M90_MOVING_KNOWN_ORACLE",
    ):
        if profile not in selected:
            continue
        try:
            runtime = build_runtime(
                "ch3_v3_full_reference",
                profile,
                seed=1,
                max_steps=8,
                device="cpu",
                replay_size=32,
            )
            env = runtime.env
            scenario = _unknown_validation_scenario(profile)
            observations = env.reset(scenario)
            check(
                f"{profile}:obs28",
                all(
                    torch.as_tensor(item).numel() == 28
                    for item in observations
                ),
            )
            check(
                f"{profile}:action3",
                all(
                    env.action_space[f"agent_{index}"].shape == (3,)
                    for index in range(4)
                ),
            )
            check(
                f"{profile}:target_moving",
                env.target_state.motion_mode
                == "constant_velocity_reflect_v1",
            )
            if profile != "M90_MOVING_KNOWN_ORACLE":
                check(
                    f"{profile}:no_truth_leak",
                    len(env.ground_truth_obstacles)
                    == len(scenario["obstacles"])
                    and env.map_module.obstacles == [],
                )
                stats = env.map_module.map_statistics()
                check(
                    f"{profile}:partial_map",
                    0.0 < stats["map_known_fraction"] < 1.0,
                )
            else:
                check(
                    f"{profile}:oracle_map",
                    len(env.map_module.obstacles)
                    == len(env.ground_truth_obstacles),
                )

            finite = True
            for _ in range(3):
                observations, rewards, dones = env.step(
                    torch.zeros((4, 3), dtype=torch.float32)
                )
                finite &= all(
                    bool(torch.isfinite(torch.as_tensor(item)).all())
                    for item in observations
                )
                finite &= bool(
                    torch.isfinite(torch.as_tensor(rewards)).all()
                )
                if all(dones):
                    break
            check(f"{profile}:finite3", finite)
            profile_rows[profile] = {
                "ground_truth_obstacle_count":
                    len(env.ground_truth_obstacles),
                **env.get_unknown_map_metrics(),
            }
        except Exception as exc:
            errors.append({
                "check": f"{profile}:runtime",
                "detail": repr(exc),
            })

    if any(profile in selected for profile in UNKNOWN_MAP_PROFILES):
        try:
            planner = OnlineUnknownMapTaskPlanner(
                space_size=(20, 20, 8),
                grid_size=(5, 5, 4),
                z_range=(0.5, 7.5),
                device="cpu",
                target_belief_transition_mode=
                    "occupancy_constrained_diffusion_v1",
                target_belief_diffusion_rate=0.12,
            )
            planner.reset(None, [])
            cell = planner._grid_index_from_point(
                torch.tensor([10.0, 10.0, 4.0])
            )
            before = float(planner.belief_map[cell].item())
            planner.set_runtime_context(
                executor_pos=torch.tensor([18.0, 18.0, 4.0]),
                executor_wait_point=torch.tensor([15.0, 15.0, 4.0]),
                step=4,
            )
            planner.update_belief_negative(
                torch.tensor([[10.0, 10.0, 4.0]]),
                sensor_ranges=torch.tensor([4.0]),
            )
            suppressed = float(planner.belief_map[cell].item())
            for _ in range(12):
                planner.predict_belief_motion()
            recovered = float(planner.belief_map[cell].item())
            check(
                "target_negative_observation_suppresses",
                suppressed < before,
            )
            check(
                "target_probability_not_permanently_zero",
                suppressed > 0.0 and recovered > 0.0,
            )
            check(
                "target_probability_can_recover",
                recovered > suppressed,
            )
            check(
                "target_belief_normalized",
                math.isclose(
                    float(planner.belief_map.sum().item()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ),
            )
        except Exception as exc:
            errors.append({
                "check": "belief_semantics",
                "detail": repr(exc),
            })


def _selected_profiles(profiles):
    if profiles is None or profiles == "all":
        return tuple(ALL_SCENARIO_PROFILES)
    if profiles == "mission":
        return tuple(SCENARIO_PROFILES)
    if profiles == "unknown":
        return tuple(UNKNOWN_MAP_PROFILES)
    if isinstance(profiles, str):
        profiles = (profiles,)
    selected = tuple(dict.fromkeys(profiles))
    invalid = [
        profile
        for profile in selected
        if profile not in ALL_SCENARIO_PROFILES
    ]
    if invalid:
        raise ValueError(f"unsupported profiles={invalid}")
    return selected


def validate(profiles=None):
    selected = _selected_profiles(profiles)
    checks, errors = {}, []

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if not condition:
            errors.append({"check": name, "detail": detail})

    base_fp = base_algorithm_source_fingerprint(PROJECT_ROOT)
    mission_fp = mission_algorithm_source_fingerprint(PROJECT_ROOT)
    unknown_fp = unknown_map_algorithm_source_fingerprint(PROJECT_ROOT)
    repository_fp = repository_source_fingerprint(PROJECT_ROOT)
    check("base_fingerprint_exists", bool(base_fp))
    check("mission_fingerprint_exists", bool(mission_fp))
    check("unknown_fingerprint_exists", bool(unknown_fp))
    check("base_mission_fingerprints_differ", base_fp != mission_fp)
    check("mission_unknown_fingerprints_differ", mission_fp != unknown_fp)
    check(
        "output_root_isolated",
        CH3_ROOT.resolve() == (PROJECT_ROOT / "data" / "chapter3").resolve(),
    )

    routed = {}
    for label, protocol in (
        ("v1", CH3_PILOT_V1),
        ("v2", CH3_EFFICIENCY_V2),
    ):
        runtime = build_ch3_runtime(
            "ch3_pse_rmaddpg",
            seed=901,
            max_steps=2,
            device="cpu",
            replay_size=8,
            protocol=protocol,
        )
        routed[label] = type(runtime.env).__name__
        check(f"routing:{label}:base", type(runtime.env) is BaseUAVEnv)
    v3_runtime = build_ch3_runtime_from_resolved_config(
        "ch3_v3_full_reference",
        resolve_ch3_efficiency_v3_config("ch3_v3_full_reference"),
        seed=901,
        max_steps=2,
        device="cpu",
        replay_size=8,
    )
    routed["v3"] = type(v3_runtime.env).__name__
    check("routing:v3:base", type(v3_runtime.env) is BaseUAVEnv)
    mission_route = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=901,
        max_steps=2,
        device="cpu",
        replay_size=8,
    )
    routed["mission"] = type(mission_route.env).__name__
    check("routing:mission:mission_env", type(mission_route.env) is MissionUAVEnv)

    generated = build_scenario_manifests(
        1, TRAIN_GENERATOR_SEED, "train"
    )
    validation = build_scenario_manifests(
        1, VALIDATION_GENERATOR_SEED, "validation"
    )
    smoke_train = build_scenario_manifests(
        1, SMOKE_TRAIN_GENERATOR_SEED, "smoke_train"
    )
    smoke_validation = build_scenario_manifests(
        1, SMOKE_VALIDATION_GENERATOR_SEED, "smoke_validation"
    )
    if any(profile in selected for profile in SCENARIO_PROFILES):
        check("profiles_exact", tuple(generated) == tuple(SCENARIO_PROFILES))
    for label, manifests in (
        ("train", generated),
        ("validation", validation),
        ("smoke_train", smoke_train),
        ("smoke_validation", smoke_validation),
    ):
        pair_groups = [
            [
                row["pair_group_id"]
                for row in manifests[profile]["scenarios"]
            ]
            for profile in SCENARIO_PROFILES
        ]
        check(
            f"{label}:strict_pairing",
            all(values == pair_groups[0] for values in pair_groups[1:]),
        )
    train_ids = {
        row["scenario_id"]
        for manifest in generated.values()
        for row in manifest["scenarios"]
    }
    validation_ids = {
        row["scenario_id"]
        for manifest in validation.values()
        for row in manifest["scenarios"]
    }
    smoke_train_ids = {
        row["scenario_id"]
        for manifest in smoke_train.values()
        for row in manifest["scenarios"]
    }
    smoke_validation_ids = {
        row["scenario_id"]
        for manifest in smoke_validation.values()
        for row in manifest["scenarios"]
    }
    check("train_validation_ids_disjoint", not train_ids & validation_ids)
    check(
        "smoke_train_validation_ids_disjoint",
        not smoke_train_ids & smoke_validation_ids,
    )
    check(
        "split_generator_seeds_distinct",
        len({
            TRAIN_GENERATOR_SEED,
            VALIDATION_GENERATOR_SEED,
            SMOKE_TRAIN_GENERATOR_SEED,
            SMOKE_VALIDATION_GENERATOR_SEED,
        }) == 4,
    )
    check(
        "strict_skip_default_replay_500000",
        build_mission_config(
            "ch3_v3_full_reference", "S00_STATIC_CLEAR"
        )["replay_size"] == 500000,
    )
    check(
        "strict_skip_smoke_replay_32",
        dict(
            build_mission_config(
                "ch3_v3_full_reference", "S00_STATIC_CLEAR"
            ),
            replay_size=32,
        )["replay_size"] == 32,
    )
    smoke_rows = {}
    for profile in SCENARIO_PROFILES:
        if profile not in selected:
            continue
        config = build_mission_config("ch3_v3_full_reference", profile)
        runtime = build_runtime(
            "ch3_v3_full_reference",
            profile,
            seed=9001,
            max_steps=20,
            device="cpu",
            replay_size=32,
            resolved_config=config,
        )
        env = runtime.env
        scenario = generated[profile]["scenarios"][0]
        obs = env.reset(scenario)
        check(
            f"{profile}:obs28",
            len(obs) == 4 and all(item.numel() == 28 for item in obs),
        )
        check(f"{profile}:action3", env.action_space["agent_0"].shape == (3,))
        check(
            f"{profile}:executor_target_hidden",
            not bool(env._agent_task_known[env.executor_idx].item()),
        )

        path = env.map_module.grid_astar_path(
            scenario["initial_agent_positions"][0],
            scenario["initial_executor_wait_point"],
        )
        check(f"{profile}:path_reachable", path["reachable"], path["failure_reason"])
        check(f"{profile}:path_edges_free", _path_edges_are_free(env.map_module, path))
        check(
            f"{profile}:connectors_free",
            env.map_module.segment_is_free(path["points"][0], path["points"][1])
            and env.map_module.segment_is_free(path["points"][-2], path["points"][-1]),
        )
        unreachable = env.map_module.grid_astar_path([-1, 0, 0], [1, 1, 1])
        check(
            f"{profile}:unreachable_inf",
            not unreachable["reachable"] and math.isinf(unreachable["cost"]),
        )
        check(
            f"{profile}:trajectory_reachable_flag",
            scenario["target_trajectory_reachable"] is True,
        )
        try:
            reachability = validate_scenario_reachability(
                scenario["initial_agent_positions"],
                scenario["initial_executor_wait_point"],
                scenario["target_initial_position"],
                [scenario["target_initial_position"]],
                scenario["obstacles"],
            )
            connected = (
                reachability["connectivity_component_id"]
                == scenario["connectivity_component_id"]
                and reachability[
                    "target_trajectory_exact_endpoint_reachable"
                ]
                is True
            )
        except RuntimeError:
            connected = False
        check(f"{profile}:connected_component", connected)

        finite = True
        for _ in range(20):
            obs, rewards, _ = env.step(torch.zeros((4, 3)))
            finite &= all(bool(torch.isfinite(item).all()) for item in obs)
            finite &= bool(torch.isfinite(torch.as_tensor(rewards)).all())
            finite &= bool(torch.isfinite(env._agent_pos).all())
            finite &= bool(np.all(np.isfinite(env.target_state.position)))
        check(f"{profile}:finite20", finite)
        check(
            f"{profile}:target_bounds",
            np.all(env.target_state.position >= 0)
            and np.all(env.target_state.position <= np.asarray([20, 20, 8])),
        )
        check(
            f"{profile}:target_outside_obstacles",
            not any(
                np.all(
                    env.target_state.position
                    >= np.asarray(obstacle["center"])
                    - np.asarray(obstacle["size"]) / 2
                )
                and np.all(
                    env.target_state.position
                    <= np.asarray(obstacle["center"])
                    + np.asarray(obstacle["size"]) / 2
                )
                for obstacle in env.obstacles
            ),
        )
        check(f"{profile}:handoff_at_most_once", env.ch3_handoff_count <= 1)
        smoke_rows[profile] = {
            "target_distance": env.target_distance_travelled,
            "obstacles": len(env.obstacles),
            "finite": finite,
        }

    if "S10_MOVING_CLEAR" in selected:
        moving_env = build_runtime(
            "ch3_v3_full_reference",
            "S10_MOVING_CLEAR",
            seed=33,
            max_steps=20,
            device="cpu",
            replay_size=16,
        ).env
        moving_scenario = deepcopy(
            generated["S10_MOVING_CLEAR"]["scenarios"][0]
        )
        handoff = _validate_handoff(moving_env, moving_scenario)
        for name, passed in handoff.items():
            check(f"handoff:{name}", passed)
        check(
            "swept_detection",
            _validate_swept_detection(moving_env, moving_scenario),
        )
        check(
            "capture_hold_semantics",
            _validate_capture_hold(moving_env, moving_scenario),
        )
        fallback_planner = moving_env.map_module
        current = moving_env._agent_pos[0].clone()
        original_candidates = fallback_planner._candidate_points
        fallback_planner._candidate_points = lambda *args, **kwargs: (
            fallback_planner.flat_xyz_centers[:2],
            torch.full(
                (2,),
                -torch.inf,
                dtype=fallback_planner.dtype,
                device=fallback_planner.device,
            ),
        )
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        claims = fallback_planner.claim_count.clone()
        fallback = fallback_planner.sample_next_waypoint(0, current)
        fallback_planner._candidate_points = original_candidates
        check("waypoint_all_unreachable_holds", torch.equal(fallback, current))
        check(
            "waypoint_all_unreachable_no_claim",
            torch.equal(fallback_planner.claim_count, claims),
        )
        check(
            "waypoint_all_unreachable_no_torch_rng",
            torch.equal(torch.random.get_rng_state(), torch_state),
        )
        current_numpy_state = np.random.get_state()
        check(
            "waypoint_all_unreachable_no_numpy_rng",
            numpy_state[0] == current_numpy_state[0]
            and np.array_equal(numpy_state[1], current_numpy_state[1])
            and numpy_state[2:] == current_numpy_state[2:],
        )

    if "S01_STATIC_OBSTACLE" in selected:
        _validate_waypoint_endpoint_guard(
            check,
            generated["S01_STATIC_OBSTACLE"]["scenarios"][0],
        )

    _validate_unknown_profiles(selected, check, errors, smoke_rows)
    failed_checks = [
        key for key, value in checks.items() if value is not True
    ]
    protocols = []
    if any(profile in selected for profile in SCENARIO_PROFILES):
        protocols.append(CH3_MISSION_V1)
    if any(profile in selected for profile in UNKNOWN_MAP_PROFILES):
        protocols.append(CH3_UNKNOWN_MAP_V1)

    return {
        "passed": not errors and not failed_checks,
        "protocols": protocols,
        "checks": checks,
        "failed_checks": failed_checks,
        "errors": errors,
        "profiles": smoke_rows,
        "routing": routed,
        "base_algorithm_source_fingerprint": base_fp,
        "mission_algorithm_source_fingerprint": mission_fp,
        "unknown_map_algorithm_source_fingerprint": unknown_fp,
        "repository_source_fingerprint": repository_fp,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CH3_ROOT / "validation" / "validation.json",
    )
    parser.add_argument(
        "--profiles",
        choices=("all", "mission", "unknown"),
        default="all",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=ALL_SCENARIO_PROFILES,
        dest="explicit_profiles",
    )
    args = parser.parse_args(argv)
    report = validate(args.explicit_profiles or args.profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CH3] {'PASS' if report['passed'] else 'FAIL'} "
        f"checks={len(report['checks'])} output={args.output}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
