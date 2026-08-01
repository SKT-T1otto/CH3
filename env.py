"""Unified Chapter-3 environment: base mechanics plus mission extensions."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json

import numpy as np
import torch

from comm.basic_communication import FixedReliableHandoff
from map.map_module import PheromoneWaypointPlanner, ProbabilisticTaskMapPlanner
from map.path_planner import (
    ObstacleAwareTaskMapPlanner,
    OnlineUnknownMapTaskPlanner,
)
from ch3_constants import CH3_MISSION_V1, CH3_UNKNOWN_MAP_V1
from target_motion import (
    TargetState,
    advance_target_state,
    predict_target_state,
    segment_aabb_first_hit,
    solve_intercept_point,
    swept_relative_min_distance,
)


class _BaseUAVEnv:
    """Four-agent Chapter-3 environment with deterministic target handoff."""

    communication_mode = "ch3_fixed_reliable"
    communication_model_id = FixedReliableHandoff.model_id
    base_obs_dim = 28

    def __init__(
        self,
        n_agents=4,
        space_size=(20, 20, 8),
        dt=0.2,
        max_steps=400,
        random_z_range=(0.5, 7.5),
        random_search_waypoint_count=(2, 4),
        random_executor_waypoint_count=(1, 1),
        use_obstacles=False,
        planner_grid_size=(10, 10, 8),
        planner_visit_radius=1,
        planner_pheromone_decay=0.990,
        planner_suppression=0.54,
        planner_min_waypoint_separation=4.8,
        planner_step_update_interval=2,
        planner_step_update_suppress_only=False,
        hold_steps=5,
        search_hold_steps=0,
        executor_hold_steps=None,
        hold_speed_thresh=0.20,
        device=None,
        return_numpy=True,
        diverse_fallback_prob=0.25,
        diverse_fallback_tries=64,
        search_spread_reward_gain=0.75,
        planner_coverage_weight=1.10,
        planner_claim_weight=0.80,
        planner_stochastic_topk=14,
        planner_stochastic_eps=0.26,
        prior_kv_xy=1.10,
        prior_kv_z=1.00,
        prior_slow_radius_xy=2.40,
        prior_slow_radius_z=1.20,
        prior_strength_search=1.00,
        prior_strength_executor=1.00,
        residual_scale_search=0.45,
        residual_scale_executor=0.35,
        use_residual_prior=True,
        protocol="ch3_pilot_v1",
        reward_profile="ch3_pilot_v1",
        reward_scale=100.0,
        team_find_bonus=20.0,
        finder_extra_bonus=40.0,
        mission_complete_bonus=300.0,
        time_penalty=0.03,
        lambda_a=0.010,
        lambda_da=0.040,
        search_detect_bonuses=(120.0, 140.0, 160.0),
        early_find_bonus_gain=0.0,
        early_success_bonus_gain=0.0,
        use_pse_planner=False,
        pse_use_belief=True,
        pse_use_exec_cost=True,
        pse_use_standby=True,
        pse_fixed_standby_mode="search_centroid",
        pse_belief_detect_prob=0.75,
        pse_belief_miss_decay=0.20,
        pse_detect_sigma=1.20,
        pse_belief_topk=48,
        pse_belief_weight=1.20,
        pse_use_gated_belief=False,
        pse_belief_weight_max=0.25,
        pse_belief_gate_start_step=80,
        pse_belief_gate_full_step=160,
        pse_belief_entropy_high=0.90,
        pse_belief_entropy_low=0.65,
        pse_belief_uniform_mix_high=0.75,
        pse_belief_uniform_mix_low=0.35,
        pse_exec_cost_weight=0.18,
        pse_exec_cost_reference_mode="physical_position",
        pse_use_exec_cost_schedule=False,
        pse_exec_cost_weight_min=0.10,
        pse_exec_cost_weight_max=0.24,
        pse_exec_cost_schedule_warmup_steps=120,
        pse_exec_cost_entropy_low=0.45,
        pse_exec_cost_entropy_high=0.95,
        pse_search_cost_weight=0.08,
        pse_base_score_weight=0.25,
        pse_standby_topk=48,
        pse_standby_candidates=64,
        pse_standby_move_weight=0.25,
        pse_standby_hysteresis_weight=0.10,
        pse_standby_safe_weight=0.05,
        pse_lazy_standby=False,
        pse_standby_entropy_gate=0.75,
        pse_standby_min_step=80,
        pse_standby_update_interval_lazy=4,
        pse_standby_move_weight_lazy=0.40,
        pse_standby_hysteresis_weight_lazy=0.25,
        pse_standby_start_step=0,
        pse_standby_update_interval=2,
        pse_standby_min_relative_gain=0.0,
        pse_standby_max_target_shift=float("inf"),
    ):
        if int(n_agents) != 4:
            raise ValueError("UAVEnv requires exactly 4 agents: 3 searchers and 1 executor")
        self.device = self._resolve_device(device)
        self.return_numpy = bool(return_numpy)
        self.dtype = torch.float32
        self.eps = 1e-6
        self.n_agents = self.num_agents = 4
        self.n_search = 3
        self.executor_idx = 3
        self.space_size = self._vec(space_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.task_mode = "mission"
        self.use_obstacles = bool(use_obstacles)
        self._default_use_obstacles = bool(use_obstacles)
        self.random_z_range = tuple(float(x) for x in random_z_range)
        self.random_search_waypoint_count = tuple(int(x) for x in random_search_waypoint_count)
        self.random_executor_waypoint_count = tuple(int(x) for x in random_executor_waypoint_count)
        self.hold_steps = int(hold_steps)
        self.search_hold_steps = max(0, int(search_hold_steps))
        self.executor_hold_steps = int(hold_steps if executor_hold_steps is None else executor_hold_steps)
        self.hold_speed_thresh = float(hold_speed_thresh)
        self.planner_min_waypoint_separation = float(planner_min_waypoint_separation)
        self.planner_step_update_interval = max(1, int(planner_step_update_interval))
        self.planner_step_update_suppress_only = bool(planner_step_update_suppress_only)

        self.role_names = ["search_fast", "search_balanced", "search_precise", "executor"]
        self.role_onehots = torch.eye(4, dtype=self.dtype, device=self.device)
        self._zero3 = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._lower_bound = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._sampling_margin_min = self._vec([0.0, 0.0, 0.5])

        self.diverse_fallback_prob = float(np.clip(diverse_fallback_prob, 0.0, 1.0))
        self.diverse_fallback_tries = max(4, int(diverse_fallback_tries))
        self.search_spread_reward_gain = float(search_spread_reward_gain)
        self.planner_coverage_weight = float(planner_coverage_weight)
        self.planner_claim_weight = float(planner_claim_weight)
        self.planner_stochastic_topk = max(1, int(planner_stochastic_topk))
        self.planner_stochastic_eps = float(np.clip(planner_stochastic_eps, 0.0, 1.0))

        self.use_residual_prior = bool(use_residual_prior)
        self.prior_kv_xy = float(prior_kv_xy)
        self.prior_kv_z = float(prior_kv_z)
        self.prior_slow_radius_xy = float(max(prior_slow_radius_xy, self.eps))
        self.prior_slow_radius_z = float(max(prior_slow_radius_z, self.eps))
        self.prior_strength_search = float(prior_strength_search)
        self.prior_strength_executor = float(prior_strength_executor)
        self.residual_scale_search = float(residual_scale_search)
        self.residual_scale_executor = float(residual_scale_executor)
        self.protocol = str(protocol)
        self.reward_profile = str(reward_profile)
        self.efficiency_protocol_v2 = self.protocol in (
            "ch3_efficiency_v2", "ch3_efficiency_v3_screen"
        )

        self.use_pse_planner = bool(use_pse_planner)
        self.pse_use_belief = bool(pse_use_belief)
        self.pse_use_exec_cost = bool(pse_use_exec_cost)
        self.pse_use_standby = bool(pse_use_standby)
        self.pse_fixed_standby_mode = str(pse_fixed_standby_mode)
        if self.pse_fixed_standby_mode not in ("search_centroid", "space_center"):
            raise ValueError(f"Unknown pse_fixed_standby_mode={self.pse_fixed_standby_mode}")
        self.pse_belief_detect_prob = float(pse_belief_detect_prob)
        self.pse_belief_miss_decay = float(pse_belief_miss_decay)
        self.pse_detect_sigma = float(pse_detect_sigma)
        self.pse_belief_topk = max(1, int(pse_belief_topk))
        self.pse_belief_weight = float(pse_belief_weight)
        self.pse_use_gated_belief = bool(pse_use_gated_belief)
        self.pse_belief_weight_max = float(pse_belief_weight_max)
        self.pse_belief_gate_start_step = int(pse_belief_gate_start_step)
        self.pse_belief_gate_full_step = int(pse_belief_gate_full_step)
        self.pse_belief_entropy_high = float(pse_belief_entropy_high)
        self.pse_belief_entropy_low = float(pse_belief_entropy_low)
        self.pse_belief_uniform_mix_high = float(pse_belief_uniform_mix_high)
        self.pse_belief_uniform_mix_low = float(pse_belief_uniform_mix_low)
        self.pse_exec_cost_weight = float(pse_exec_cost_weight)
        self.pse_exec_cost_reference_mode = str(pse_exec_cost_reference_mode)
        if self.pse_exec_cost_reference_mode not in (
            "physical_position", "fixed_initial_wait_point"
        ):
            raise ValueError(
                f"Unknown pse_exec_cost_reference_mode={self.pse_exec_cost_reference_mode}"
            )
        self.pse_use_exec_cost_schedule = bool(pse_use_exec_cost_schedule)
        self.pse_exec_cost_weight_min = float(pse_exec_cost_weight_min)
        self.pse_exec_cost_weight_max = float(pse_exec_cost_weight_max)
        self.pse_exec_cost_schedule_warmup_steps = max(1, int(pse_exec_cost_schedule_warmup_steps))
        self.pse_exec_cost_entropy_low = float(pse_exec_cost_entropy_low)
        self.pse_exec_cost_entropy_high = float(pse_exec_cost_entropy_high)
        self.pse_search_cost_weight = float(pse_search_cost_weight)
        self.pse_base_score_weight = float(pse_base_score_weight)
        self.pse_standby_topk = max(1, int(pse_standby_topk))
        self.pse_standby_candidates = max(1, int(pse_standby_candidates))
        self.pse_standby_move_weight = float(pse_standby_move_weight)
        self.pse_standby_hysteresis_weight = float(pse_standby_hysteresis_weight)
        self.pse_standby_safe_weight = float(pse_standby_safe_weight)
        self.pse_lazy_standby = bool(pse_lazy_standby)
        self.pse_standby_entropy_gate = float(pse_standby_entropy_gate)
        self.pse_standby_min_step = max(0, int(pse_standby_min_step))
        self.pse_standby_update_interval_lazy = max(1, int(pse_standby_update_interval_lazy))
        self.pse_standby_move_weight_lazy = float(pse_standby_move_weight_lazy)
        self.pse_standby_hysteresis_weight_lazy = float(pse_standby_hysteresis_weight_lazy)
        self.pse_standby_start_step = max(0, int(pse_standby_start_step))
        self.pse_standby_update_interval = max(1, int(pse_standby_update_interval))
        self.pse_standby_min_relative_gain = float(max(0.0, pse_standby_min_relative_gain))
        self.pse_standby_max_target_shift = float(max(0.0, pse_standby_max_target_shift))

        self.agent_specs = [
            dict(name="search_fast", type="search", a_xy_max=1.30, a_z_max=0.75, v_xy_max=2.80, v_z_max=1.20, drag_xy=0.10, drag_z=0.16, buoyancy_bias=0.00, sensor_range=2.00, energy_coeff=1.15, progress_gain=20.0, waypoint_bonus=12.0, detect_bonus=float(search_detect_bonuses[0])),
            dict(name="search_balanced", type="search", a_xy_max=1.00, a_z_max=0.70, v_xy_max=2.20, v_z_max=1.00, drag_xy=0.12, drag_z=0.18, buoyancy_bias=0.00, sensor_range=2.35, energy_coeff=1.00, progress_gain=18.0, waypoint_bonus=10.0, detect_bonus=float(search_detect_bonuses[1])),
            dict(name="search_precise", type="search", a_xy_max=0.90, a_z_max=0.65, v_xy_max=1.80, v_z_max=0.90, drag_xy=0.14, drag_z=0.20, buoyancy_bias=0.00, sensor_range=2.75, energy_coeff=0.90, progress_gain=16.0, waypoint_bonus=9.0, detect_bonus=float(search_detect_bonuses[2])),
            dict(name="executor", type="execute", a_xy_max=0.80, a_z_max=0.55, v_xy_max=1.50, v_z_max=0.75, drag_xy=0.16, drag_z=0.24, buoyancy_bias=0.02, sensor_range=0.0, energy_coeff=1.05, progress_gain=28.0, waypoint_bonus=0.0, detect_bonus=0.0),
        ]
        self._build_agent_spec_tensors()

        self.safe_dist = 1.6
        self.collision_penalty = -60.0
        self.sep_penalty_k = 1.2
        self.time_penalty = float(time_penalty)
        self.lambda_a = float(lambda_a)
        self.lambda_da = float(lambda_da)
        self.reward_scale = float(reward_scale)
        self.search_arrive_eps = 0.9
        self.detect_eps_bias = 0.10
        self.executor_arrive_eps = 1.0
        self.executor_hold_radius = 0.8
        self.executor_hold_bonus = 1.2
        self.mission_complete_bonus = float(mission_complete_bonus)
        self.team_find_bonus = float(team_find_bonus)
        self.finder_extra_bonus = float(finder_extra_bonus)
        self.early_find_bonus_gain = float(early_find_bonus_gain)
        self.early_success_bonus_gain = float(early_success_bonus_gain)
        self.coverage_reward_gain = 100.0
        self.flow_gain = 0.18
        self.flow_z_gain = 0.0
        self._flow_phase_x = 0.0
        self._flow_phase_y = 0.0
        self.current_scenario_seed = None
        self.obstacle_layout_id = "default_fixed_v1" if self.use_obstacles else "none"

        self.default_obstacles = [
            {"center": np.array([5.0, 5.0, 2.0], dtype=np.float32), "size": np.array([2.5, 2.5, 2.0], dtype=np.float32)},
            {"center": np.array([11.0, 10.0, 4.0], dtype=np.float32), "size": np.array([3.0, 3.0, 2.5], dtype=np.float32)},
            {"center": np.array([15.5, 6.0, 5.5], dtype=np.float32), "size": np.array([2.0, 3.0, 2.0], dtype=np.float32)},
        ]
        self.obstacles = self.default_obstacles if self.use_obstacles else []
        self._build_obstacle_tensors()

        self._agent_pos = torch.zeros((4, 3), dtype=self.dtype, device=self.device)
        self._agent_vel = torch.zeros_like(self._agent_pos)
        self._agent_acc = torch.zeros_like(self._agent_pos)
        self._prev_acc = torch.zeros_like(self._agent_pos)
        self._nav_targets = torch.zeros_like(self._agent_pos)
        self._targets = torch.zeros_like(self._agent_pos)
        self._task_target = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._search_waypoints = torch.zeros((3, 3), dtype=self.dtype, device=self.device)
        self._executor_wait_point = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._collision_flags = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._agent_task_known = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._agent_task_est = torch.zeros((4, 3), dtype=self.dtype, device=self.device)
        self.waypoint_reached_counts = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.hold_success_counts = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.total_waypoints_per_agent = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.agent_finished = torch.zeros(4, dtype=torch.bool, device=self.device)
        self.just_reached_waypoint = torch.zeros(4, dtype=torch.bool, device=self.device)
        self.just_held_target = torch.zeros(4, dtype=torch.bool, device=self.device)
        self.hold_counters = torch.zeros(4, dtype=torch.int32, device=self.device)
        self.current_target_arrived = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._last_residual_acc = torch.zeros_like(self._agent_pos)
        self._last_prior_acc = torch.zeros_like(self._agent_pos)

        planner_cls = ProbabilisticTaskMapPlanner if self.use_pse_planner else PheromoneWaypointPlanner
        planner_kwargs = dict(
            space_size=self.space_size,
            n_agents=4,
            n_search=3,
            executor_idx=3,
            grid_size=planner_grid_size,
            z_range=self.random_z_range,
            search_count_range=self.random_search_waypoint_count,
            executor_count_range=self.random_executor_waypoint_count,
            visit_radius=planner_visit_radius,
            pheromone_decay=planner_pheromone_decay,
            suppression=planner_suppression,
            min_waypoint_separation=planner_min_waypoint_separation,
            coverage_weight=self.planner_coverage_weight,
            claim_weight=self.planner_claim_weight,
            stochastic_topk=self.planner_stochastic_topk,
            stochastic_eps=self.planner_stochastic_eps,
            device=self.device,
            dtype=self.dtype,
        )
        if self.use_pse_planner:
            planner_kwargs.update(
                pse_belief_detect_prob=self.pse_belief_detect_prob,
                pse_belief_miss_decay=self.pse_belief_miss_decay,
                pse_detect_sigma=self.pse_detect_sigma,
                pse_belief_topk=self.pse_belief_topk,
                pse_belief_weight=self.pse_belief_weight,
                pse_use_gated_belief=self.pse_use_gated_belief,
                pse_belief_weight_max=self.pse_belief_weight_max,
                pse_belief_gate_start_step=self.pse_belief_gate_start_step,
                pse_belief_gate_full_step=self.pse_belief_gate_full_step,
                pse_belief_entropy_high=self.pse_belief_entropy_high,
                pse_belief_entropy_low=self.pse_belief_entropy_low,
                pse_belief_uniform_mix_high=self.pse_belief_uniform_mix_high,
                pse_belief_uniform_mix_low=self.pse_belief_uniform_mix_low,
                pse_exec_cost_weight=self.pse_exec_cost_weight,
                pse_search_cost_weight=self.pse_search_cost_weight,
                pse_base_score_weight=self.pse_base_score_weight,
                pse_standby_topk=self.pse_standby_topk,
                pse_standby_candidates=self.pse_standby_candidates,
                pse_standby_move_weight=self.pse_standby_move_weight,
                pse_standby_hysteresis_weight=self.pse_standby_hysteresis_weight,
                pse_standby_safe_weight=self.pse_standby_safe_weight,
            )
        self.map_module = planner_cls(**planner_kwargs)
        self.fixed_reliable_handoff = FixedReliableHandoff(delay_steps=1)
        self.reset()

    @staticmethod
    def _resolve_device(device):
        if device is None:
            return torch.device("cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved

    def _vec(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _scalar(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _to_public(self, value):
        return value.detach().cpu().numpy().copy() if self.return_numpy else value.detach().clone()

    def _obs_to_public(self, observations):
        return [self._to_public(item) for item in observations]

    def _rewards_to_public(self, rewards):
        return rewards.detach().cpu().numpy().copy() if self.return_numpy else rewards.detach().clone()

    def _actions_to_tensor(self, actions):
        if torch.is_tensor(actions):
            tensor = actions.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(np.asarray(actions), dtype=self.dtype, device=self.device)
        if tuple(tensor.shape) != (4, 3):
            raise ValueError(f"actions must have shape (4, 3), got {tuple(tensor.shape)}")
        return tensor

    def _build_agent_spec_tensors(self):
        self._a_xy_max = self._vec([s["a_xy_max"] for s in self.agent_specs])
        self._a_z_max = self._vec([s["a_z_max"] for s in self.agent_specs])
        self._v_xy_max = self._vec([s["v_xy_max"] for s in self.agent_specs])
        self._v_z_max = self._vec([s["v_z_max"] for s in self.agent_specs])
        self._drag_xy = self._vec([s["drag_xy"] for s in self.agent_specs])
        self._drag_z = self._vec([s["drag_z"] for s in self.agent_specs])
        self._buoyancy_bias = self._vec([s["buoyancy_bias"] for s in self.agent_specs])
        self._sensor_range = self._vec([s["sensor_range"] for s in self.agent_specs])
        self._energy_coeff = self._vec([s["energy_coeff"] for s in self.agent_specs])
        self._progress_gain = self._vec([s["progress_gain"] for s in self.agent_specs])
        self._waypoint_bonus = self._vec([s["waypoint_bonus"] for s in self.agent_specs])
        self._detect_bonus = self._vec([s["detect_bonus"] for s in self.agent_specs])
        self._prior_strength = self._vec([
            self.prior_strength_search,
            self.prior_strength_search,
            self.prior_strength_search,
            self.prior_strength_executor,
        ])
        self._residual_scale = self._vec([
            self.residual_scale_search,
            self.residual_scale_search,
            self.residual_scale_search,
            self.residual_scale_executor,
        ])

    @property
    def agent_pos(self): return self._to_public(self._agent_pos)
    @property
    def agent_vel(self): return self._to_public(self._agent_vel)
    @property
    def agent_acc(self): return self._to_public(self._agent_acc)
    @property
    def prev_acc(self): return self._to_public(self._prev_acc)
    @property
    def nav_targets(self): return self._to_public(self._nav_targets)
    @property
    def targets(self): return self._to_public(self._targets)
    @property
    def collision_flags(self): return self._to_public(self._collision_flags)
    @property
    def task_target(self): return self._to_public(self._task_target)

    def _build_obstacle_tensors(self):
        if not self.obstacles:
            self._obstacle_lower = None
            self._obstacle_upper = None
            return
        centers = torch.stack([self._vec(o["center"]) for o in self.obstacles])
        sizes = torch.stack([self._vec(o["size"]) for o in self.obstacles])
        self._obstacle_lower = centers - sizes / 2
        self._obstacle_upper = centers + sizes / 2

    def _apply_scenario_obstacles(self, scenario):
        """Apply a manifest's obstacle identity before positions are validated."""
        if scenario is None:
            requested_use = self._default_use_obstacles
            requested_layout = "default_fixed_v1" if requested_use else "none"
        else:
            requested_use = bool(scenario.get("use_obstacles", self.use_obstacles))
            requested_layout = str(
                scenario.get(
                    "obstacle_layout_id",
                    "default_fixed_v1" if requested_use else "none",
                )
            )
        if requested_use and requested_layout != "default_fixed_v1":
            raise ValueError(f"unsupported obstacle_layout_id={requested_layout!r}")
        self.use_obstacles = requested_use
        self.obstacle_layout_id = requested_layout if requested_use else "none"
        self.obstacles = self.default_obstacles if requested_use else []
        self._build_obstacle_tensors()

    def _validate_scenario_point(self, point, *, name):
        point = self._vec(point).reshape(3)
        if bool(torch.any(point < self._lower_bound)) or bool(torch.any(point > self.space_size)):
            raise ValueError(f"scenario {name} is outside the environment boundary: {point.tolist()}")
        if self.is_inside_obstacle(point):
            raise ValueError(f"scenario {name} is inside obstacle layout {self.obstacle_layout_id}")
        return point

    def _sample_free_point(self, margin=0.8):
        for _ in range(512):
            point = self._vec([
                margin + torch.rand(()).item() * max(0.1, float(self.space_size[0]) - 2 * margin),
                margin + torch.rand(()).item() * max(0.1, float(self.space_size[1]) - 2 * margin),
                self.random_z_range[0] + torch.rand(()).item() * (self.random_z_range[1] - self.random_z_range[0]),
            ])
            if not self.is_inside_obstacle(point):
                return point
        raise RuntimeError("unable to sample a free point")

    def _sample_initial_positions(self, min_dist=2.8):
        positions = []
        for _ in range(self.n_agents):
            for _ in range(512):
                point = self._sample_free_point(margin=1.0)
                if all(float(torch.norm(point - old)) >= min_dist for old in positions):
                    positions.append(point)
                    break
            else:
                raise RuntimeError("unable to sample separated agent positions")
        return torch.stack(positions)

    def _flow_at(self, pos):
        x, y = pos[..., 0], pos[..., 1]
        return torch.stack([
            self.flow_gain * torch.sin(0.25 * y + self._flow_phase_x),
            self.flow_gain * torch.cos(0.22 * x + self._flow_phase_y),
            torch.zeros_like(x),
        ], dim=-1)

    def _current_coverage_ratio_internal(self):
        return float((self.map_module.coverage > 1e-6).float().mean().item())

    def _current_search_task_min_dist(self):
        return float(torch.min(torch.norm(self._agent_pos[:3] - self._task_target, dim=1)).item())

    def _set_pse_planner_context(self):
        if not self.use_pse_planner:
            return
        effective_weight, schedule_factor = self._compute_pse_exec_cost_weight()
        physical_position = self._agent_pos[self.executor_idx]
        reference_position = (
            self._initial_executor_wait_point
            if self.pse_exec_cost_reference_mode == "fixed_initial_wait_point"
            and hasattr(self, "_initial_executor_wait_point")
            else physical_position
        )
        self.exec_cost_reference_position = reference_position.clone()
        self.physical_executor_position = physical_position.clone()
        self.exec_cost_reference_to_executor_distance = float(
            torch.norm(reference_position - physical_position).item()
        )
        self.map_module.set_runtime_context(
            executor_pos=reference_position,
            executor_wait_point=self._executor_wait_point,
            use_belief=self.pse_use_belief,
            use_exec_cost=self.pse_use_exec_cost,
            use_standby=self.pse_use_standby,
            pse_exec_cost_weight_effective=effective_weight,
            pse_exec_cost_schedule_factor=schedule_factor,
            step=self.step_count,
        )

    def _compute_pse_exec_cost_weight(self):
        if not self.pse_use_exec_cost:
            self.last_pse_exec_cost_weight_effective = 0.0
            self.last_pse_exec_cost_schedule_factor = 0.0
            return 0.0, 0.0
        if not self.pse_use_exec_cost_schedule:
            self.last_pse_exec_cost_weight_effective = self.pse_exec_cost_weight
            self.last_pse_exec_cost_schedule_factor = 1.0
            return self.pse_exec_cost_weight, 1.0
        time_factor = float(np.clip(self.step_count / self.pse_exec_cost_schedule_warmup_steps, 0.0, 1.0))
        denom = max(self.pse_exec_cost_entropy_high - self.pse_exec_cost_entropy_low, self.eps)
        entropy_factor = float(np.clip(
            (self.pse_exec_cost_entropy_high - self.last_belief_entropy_normalized) / denom,
            0.0,
            1.0,
        ))
        factor = max(time_factor, entropy_factor)
        effective = self.pse_exec_cost_weight_min + factor * (
            self.pse_exec_cost_weight_max - self.pse_exec_cost_weight_min
        )
        self.last_pse_exec_cost_weight_effective = float(effective)
        self.last_pse_exec_cost_schedule_factor = float(factor)
        return float(effective), float(factor)

    def _update_pse_belief(self, force_detection=False):
        if not (self.use_pse_planner and self.pse_use_belief):
            return
        if force_detection or self.task_found:
            self.map_module.update_belief_detection(self._task_target)
        else:
            self.map_module.update_belief_negative(
                self._agent_pos[: self.n_search],
                sensor_ranges=self._sensor_range[: self.n_search],
            )
        self._sync_pse_diagnostics()

    def _update_pse_executor_standby(self, force=False):
        if self.efficiency_protocol_v2:
            return self._update_pse_executor_standby_v2()
        self.last_pse_standby_update_used = 0.0
        self.last_pse_lazy_standby_active = 0.0
        self.last_pse_standby_update_allowed = 1.0
        self.last_pse_standby_update_skipped_by_lazy_gate = 0.0
        interval = self.planner_step_update_interval
        lazy = self.pse_lazy_standby and not self.task_found
        if lazy:
            interval = self.pse_standby_update_interval_lazy
            self.last_pse_lazy_standby_active = 1.0
        self.last_pse_standby_update_interval = float(interval)
        if not (self.use_pse_planner and self.pse_use_standby and not self.task_found):
            return
        if self.step_count == self._last_pse_standby_update_step:
            return
        if not force and self.step_count % interval != 0:
            return
        if lazy and not force:
            allowed = (
                self.step_count >= self.pse_standby_min_step
                or self.last_belief_entropy_normalized <= self.pse_standby_entropy_gate
            )
            self.last_pse_standby_update_allowed = float(allowed)
            if not allowed:
                self.last_pse_standby_update_skipped_by_lazy_gate = 1.0
                return
        self._set_pse_planner_context()
        standby = self.map_module.plan_executor_standby(
            self._agent_pos[self.executor_idx],
            prev_standby=self._executor_wait_point,
            move_weight=self.pse_standby_move_weight_lazy if lazy else None,
            hysteresis_weight=self.pse_standby_hysteresis_weight_lazy if lazy else None,
        )
        standby = self._vec(standby).reshape(3)
        standby = torch.clamp(standby, min=self._sampling_margin_min, max=self.space_size - 0.5)
        if not self.is_inside_obstacle(standby):
            self._executor_wait_point.copy_(standby)
            self._last_pse_standby_update_step = self.step_count
            self.last_pse_standby_update_used = 1.0
            self.executor_wait_held = False
            self.current_target_arrived[self.executor_idx] = False

    def _nearest_valid_standby_point(self, point, old_point):
        point = torch.clamp(
            self._vec(point).reshape(3),
            min=self._sampling_margin_min,
            max=self.space_size - 0.5,
        )
        if not self.is_inside_obstacle(point):
            return point
        valid_points = getattr(self.map_module, "flat_valid_points", None)
        if valid_points is None or valid_points.numel() == 0:
            raise RuntimeError("standby target is obstructed and planner has no valid points")
        shifts = torch.norm(valid_points - old_point.reshape(1, 3), dim=1)
        eligible = shifts <= self.pse_standby_max_target_shift + self.eps
        if not torch.any(eligible):
            raise RuntimeError("standby target is obstructed and no valid point satisfies max shift")
        candidates = valid_points[eligible]
        index = int(torch.argmin(torch.norm(candidates - point.reshape(1, 3), dim=1)).item())
        return candidates[index].clone()

    def _update_pse_executor_standby_v2(self):
        self.last_pse_standby_update_used = 0.0
        self.last_pse_standby_update_allowed = 0.0
        self.last_pse_standby_update_interval = float(self.pse_standby_update_interval)
        if not (self.use_pse_planner and self.pse_use_standby and not self.task_found):
            return
        if self.step_count < self.pse_standby_start_step:
            return
        if (
            self._last_pse_standby_update_step >= 0
            and self.step_count - self._last_pse_standby_update_step
            < self.pse_standby_update_interval
        ):
            return
        self.last_pse_standby_update_allowed = 1.0
        self.standby_update_attempt_count += 1
        self._last_pse_standby_update_step = int(self.step_count)
        old = self._executor_wait_point.clone()
        try:
            self._set_pse_planner_context()
            current_cost = float(self.map_module.expected_response_cost(old))
            candidate = self.map_module.plan_executor_standby(
                self._agent_pos[self.executor_idx],
                prev_standby=old,
                move_weight=self.pse_standby_move_weight,
                hysteresis_weight=self.pse_standby_hysteresis_weight,
            )
            candidate = self._vec(candidate).reshape(3)
            delta = candidate - old
            shift = float(torch.norm(delta).item())
            if shift > self.pse_standby_max_target_shift:
                candidate = old + delta * (self.pse_standby_max_target_shift / max(shift, self.eps))
            candidate = self._nearest_valid_standby_point(candidate, old)
            actual_shift = float(torch.norm(candidate - old).item())
            if actual_shift <= self.eps:
                self.standby_update_reject_count += 1
                return
            # Judge the point that will actually be installed. The raw planner
            # proposal can be changed by the shift cap or obstacle projection;
            # scoring the pre-projection point could accept a worse wait point.
            candidate_cost = float(self.map_module.expected_response_cost(candidate))
            relative_gain = (current_cost - candidate_cost) / max(current_cost, self.eps)
            self.standby_current_response_cost = current_cost
            self.standby_candidate_response_cost = candidate_cost
            self.standby_relative_gain = float(relative_gain)
            if relative_gain + self.eps < self.pse_standby_min_relative_gain:
                self.standby_update_reject_count += 1
                return
            self._executor_wait_point.copy_(candidate)
            self.standby_update_accept_count += 1
            self.standby_total_target_shift += actual_shift
            self._standby_accepted_gain_sum += float(relative_gain)
            self.standby_mean_accepted_gain = (
                self._standby_accepted_gain_sum / self.standby_update_accept_count
            )
            self.last_pse_standby_update_used = 1.0
            self.executor_wait_held = False
            self.current_target_arrived[self.executor_idx] = False
        except (RuntimeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "v2 executor standby update failed "
                f"at step={self.step_count}, layout={self.obstacle_layout_id}, "
                f"wait_point={old.detach().cpu().tolist()}"
            ) from exc

    def _sync_pse_diagnostics(self):
        if not self.use_pse_planner:
            return
        planner = self.map_module
        self.last_belief_entropy = float(getattr(planner, "last_belief_entropy", 0.0))
        self.last_belief_entropy_normalized = float(getattr(planner, "last_belief_entropy_normalized", 0.0))
        self.last_belief_peak_probability = float(getattr(planner, "last_belief_peak_probability", 0.0))
        self.last_valid_belief_cell_count = int(getattr(planner, "last_valid_belief_cell_count", 0))
        self.last_exec_response_cost = float(getattr(planner, "last_exec_response_cost", 0.0))
        self.last_search_score_mean = float(getattr(planner, "last_search_score_mean", 0.0))
        self.last_pse_claim_overlap = float(getattr(planner, "last_claim_overlap", 0.0))
        for name in (
            "gated_belief_time_factor", "gated_belief_entropy_confidence",
            "gated_belief_total_confidence", "gated_belief_uniform_mix",
            "gated_belief_effective_weight", "gated_belief_mix_entropy",
            "gated_belief_mix_peak_probability",
        ):
            setattr(self, f"last_{name}", float(getattr(planner, f"last_{name}", 0.0)))
        standby = getattr(planner, "last_executor_standby", None)
        standby = self._executor_wait_point if standby is None else self._vec(standby)
        self.last_standby_to_target_dist = float(torch.norm(standby - self._task_target).item())
        if self.mission_complete and self.found_step is not None:
            self.last_success_step_minus_found_step = float(self.step_count - self.found_step)

    def _reset_pse_diagnostics(self):
        self.last_belief_entropy = float(getattr(self.map_module, "last_belief_entropy", 0.0))
        self.last_belief_entropy_normalized = float(getattr(self.map_module, "last_belief_entropy_normalized", 0.0))
        self.last_belief_peak_probability = float(getattr(self.map_module, "last_belief_peak_probability", 0.0))
        self.last_valid_belief_cell_count = int(getattr(self.map_module, "last_valid_belief_cell_count", 0))
        self.last_standby_to_target_dist = 0.0
        self.last_exec_response_cost = 0.0
        self.last_search_score_mean = 0.0
        self.last_pse_claim_overlap = 0.0
        self.last_pse_exec_cost_weight_effective = self.pse_exec_cost_weight if self.pse_use_exec_cost else 0.0
        self.last_pse_exec_cost_schedule_factor = 0.0
        self.last_pse_lazy_standby_active = 0.0
        self.last_pse_standby_update_allowed = 1.0
        self.last_pse_standby_update_skipped_by_lazy_gate = 0.0
        self.last_pse_standby_update_used = 0.0
        self.last_pse_standby_update_interval = float(self.planner_step_update_interval)
        self.last_success_step_minus_found_step = float("nan")
        self._last_pse_standby_update_step = -1
        self.standby_update_attempt_count = 0
        self.standby_update_accept_count = 0
        self.standby_update_reject_count = 0
        self.standby_total_target_shift = 0.0
        self.standby_executor_travel_distance = 0.0
        self.standby_current_response_cost = float("nan")
        self.standby_candidate_response_cost = float("nan")
        self.standby_relative_gain = float("nan")
        self.standby_mean_accepted_gain = float("nan")
        self._standby_accepted_gain_sum = 0.0
        for name in (
            "gated_belief_time_factor", "gated_belief_entropy_confidence",
            "gated_belief_total_confidence", "gated_belief_uniform_mix",
            "gated_belief_effective_weight", "gated_belief_mix_entropy",
            "gated_belief_mix_peak_probability",
        ):
            setattr(self, f"last_{name}", 0.0)
        self.exec_cost_reference_position = self._agent_pos[self.executor_idx].clone()
        self.physical_executor_position = self._agent_pos[self.executor_idx].clone()
        self.exec_cost_reference_to_executor_distance = 0.0

    def reset(self, scenario=None):
        self.step_count = 0
        scenario = None if scenario is None else dict(scenario)
        self._apply_scenario_obstacles(scenario)
        if scenario is not None:
            torch.manual_seed(int(scenario["planner_seed"]))
            self.current_scenario_seed = int(scenario["scenario_seed"])
            self._flow_phase_x = float(scenario.get("flow_phase_x", 0.0))
            self._flow_phase_y = float(scenario.get("flow_phase_y", 0.0))
        else:
            self.current_scenario_seed = None
            self._flow_phase_x = 0.0
            self._flow_phase_y = 0.0
        self._collision_flags.zero_()
        self._agent_vel.zero_()
        self._agent_acc.zero_()
        self._prev_acc.zero_()
        self._agent_pos = (
            self._sample_initial_positions(min_dist=2.8)
            if scenario is None
            else self._vec(scenario["initial_agent_positions"]).reshape(4, 3).clone()
        )
        for index, point in enumerate(self._agent_pos):
            self._validate_scenario_point(point, name=f"initial_agent_positions[{index}]")
        pairwise_initial = torch.cdist(self._agent_pos, self._agent_pos)
        pairwise_initial.fill_diagonal_(float("inf"))
        if float(pairwise_initial.min().item()) < 2.8 - self.eps:
            raise ValueError("scenario initial_agent_positions violate minimum distance 2.8")
        if hasattr(self.map_module, "set_z_range"):
            self.map_module.set_z_range(self.random_z_range, rebuild_grid=True)
        self.map_module.reset(None, self.obstacles)
        if self.use_pse_planner:
            self.map_module.reset_belief_map()
        self._reset_pse_diagnostics()

        self.task_found = False
        self.finder_idx = -1
        self.mission_complete = False
        self.search_stage_complete = False
        self.executor_target_assigned = False
        self.executor_wait_held = False
        self._found_event = False
        self._mission_complete_event = False
        self._executor_wait_hold_event = False
        self._agent_task_known.zero_()
        self._agent_task_est.zero_()
        self.waypoint_reached_counts.zero_()
        self.hold_success_counts.zero_()
        self.total_waypoints_per_agent.zero_()
        self.agent_finished.zero_()
        self.just_reached_waypoint.zero_()
        self.just_held_target.zero_()
        self.hold_counters.zero_()
        self.current_target_arrived.zero_()
        self.found_step = None
        self.success_step = None
        self.handoff_step = None
        self.executor_received_target_step = None
        self.last_handoff_delay = float("nan")
        self.ch3_handoff_count = 0
        self.fixed_reliable_handoff.reset()
        self._reset_residual_prior_diagnostics()

        target = (
            self._sample_free_point(margin=1.0)
            if scenario is None
            else self._validate_scenario_point(scenario["target_position"], name="target_position")
        )
        self._task_target.copy_(target)
        if scenario is not None:
            self._executor_wait_point.copy_(
                self._validate_scenario_point(
                    scenario["initial_executor_wait_point"],
                    name="initial_executor_wait_point",
                )
            )
        elif self.pse_fixed_standby_mode == "space_center":
            self._executor_wait_point.copy_(self._vec([10.0, 10.0, sum(self.random_z_range) / 2]))
        else:
            self._executor_wait_point.copy_(self._agent_pos[:3].mean(dim=0))
        self._initial_executor_wait_point = self._executor_wait_point.clone()
        self._set_pse_planner_context()
        if not self.efficiency_protocol_v2:
            self._update_pse_executor_standby(force=True)
        self._search_waypoints = self.map_module.initial_search_targets(self._agent_pos[:3])
        self._nav_targets[:3] = self._search_waypoints
        self._nav_targets[3] = self._executor_wait_point
        self._targets.copy_(self._nav_targets)
        self.total_waypoints_per_agent.fill_(1)
        self._prev_nav_distances = self._compute_nav_distances()
        self._prev_coverage_ratio = self._current_coverage_ratio_internal()
        self._prev_search_task_min_dist = self._current_search_task_min_dist()
        self.belief_entropy_at_start = float(self.last_belief_entropy)
        self.last_reward_components = self._empty_reward_components()
        self.last_raw_reward = torch.zeros(4, dtype=self.dtype, device=self.device)
        return self._obs_to_public(self._get_obs())

    def _advance_fixed_handoff(self):
        event = self.fixed_reliable_handoff.advance(entering_step=self.step_count + 1)
        if event is None:
            return False
        target = self._vec(event["target"])
        self._agent_task_est[self.executor_idx].copy_(target)
        self._agent_task_known[self.executor_idx] = True
        self.executor_target_assigned = True
        self.current_target_arrived[self.executor_idx] = False
        self.executor_received_target_step = int(event["delivery_step"])
        self.last_handoff_delay = float(self.executor_received_target_step - event["found_step"])
        self.ch3_handoff_count += 1
        return True

    def _update_nav_targets(self):
        if not self.task_found:
            self._nav_targets[:3] = self._search_waypoints
            self._nav_targets[3] = self._executor_wait_point
        else:
            self._nav_targets[:3] = self._agent_pos[:3]
            self._nav_targets[3] = (
                self._agent_task_est[3] if self.executor_target_assigned else self._executor_wait_point
            )
        self._targets.copy_(self._nav_targets)

    def _maybe_detect_task(self):
        if self.task_found:
            return
        distances = torch.norm(self._agent_pos[:3] - self._task_target, dim=1)
        detected = distances <= self._sensor_range[:3] + self.detect_eps_bias
        if not torch.any(detected):
            return
        self.task_found = True
        self.finder_idx = int(torch.argmax(detected.to(torch.int32)).item())
        self.search_stage_complete = True
        self.agent_finished[:3] = True
        self._found_event = True
        self.found_step = int(self.step_count)
        self.handoff_step = int(self.step_count)
        self._agent_task_known[self.finder_idx] = True
        self._agent_task_est[self.finder_idx].copy_(self._task_target)
        published = self.fixed_reliable_handoff.publish_target(
            found_step=self.found_step,
            finder_idx=self.finder_idx,
            target=self._task_target,
        )
        if not published:
            raise RuntimeError("target handoff was already published")
        self._update_pse_belief(force_detection=True)
        self._sync_pse_diagnostics()

    def _hold_steps_for_agent(self, agent_id):
        return self.search_hold_steps if int(agent_id) < 3 else self.executor_hold_steps

    def _get_obs(self):
        observations = []
        nearest = torch.clamp(self._nearest_obstacle_distance(self._agent_pos) / 10.0, 0.0, 1.0)
        for i in range(4):
            pos = self._agent_pos[i]
            vel = self._agent_vel[i]
            relative_nav = self._nav_targets[i] - pos
            nav_norm = torch.norm(relative_nav).clamp_min(self.eps)
            nav_direction = relative_nav / nav_norm
            speed = torch.norm(vel).clamp_min(self.eps)
            known = bool(self._agent_task_known[i].item())
            known_target_delta = self._agent_task_est[i] - pos if known else self._zero3
            total = max(1, int(self.total_waypoints_per_agent[i]))
            progress = self.waypoint_reached_counts[i].to(self.dtype) / float(total)
            hold_progress = torch.clamp(
                self.hold_counters[i].to(self.dtype) / max(1.0, float(self._hold_steps_for_agent(i))),
                0.0,
                1.0,
            )
            phase = self._vec([0.0, 1.0] if known else [1.0, 0.0])
            obs = torch.cat([
                pos,
                vel,
                relative_nav,
                nav_direction,
                known_target_delta,
                torch.clamp(nav_norm / 10.0, 0.0, 1.0).reshape(1),
                torch.clamp(speed / (self._v_xy_max[i] + self.eps), 0.0, 1.0).reshape(1),
                torch.tanh((torch.dot(vel, nav_direction) / (self._v_xy_max[i] + self.eps)).reshape(1)),
                torch.stack([nearest[i], progress, self.agent_finished[i].to(self.dtype), hold_progress]),
                self.role_onehots[i],
                phase,
            ]).to(self.dtype)
            if obs.numel() != 28:
                raise RuntimeError(f"observation for agent {i} has {obs.numel()} values, expected 28")
            observations.append(obs)
        return observations

    def _compute_waypoint_prior_acc(self):
        relative = self._nav_targets - self._agent_pos
        prior = torch.zeros_like(self._agent_acc)
        dist_xy = torch.norm(relative[:, :2], dim=1, keepdim=True).clamp_min(self.eps)
        desired_xy = (
            relative[:, :2] / dist_xy
            * self._v_xy_max.unsqueeze(1)
            * torch.clamp(dist_xy / self.prior_slow_radius_xy, 0.0, 1.0)
        )
        prior[:, :2] = self.prior_kv_xy * (desired_xy - self._agent_vel[:, :2])
        desired_z = torch.clamp(
            relative[:, 2] / self.prior_slow_radius_z, -1.0, 1.0
        ) * self._v_z_max
        prior[:, 2] = self.prior_kv_z * (desired_z - self._agent_vel[:, 2])
        prior[:, :2] = torch.clamp(prior[:, :2], -self._a_xy_max[:, None], self._a_xy_max[:, None])
        prior[:, 2] = torch.clamp(prior[:, 2], -self._a_z_max, self._a_z_max)
        return prior

    def _actions_to_residual_acc(self, actions):
        residual = torch.empty_like(actions)
        residual[:, :2] = actions[:, :2] * self._a_xy_max[:, None]
        residual[:, 2] = actions[:, 2] * self._a_z_max
        return residual

    def _reset_residual_prior_diagnostics(self):
        self.last_prior_term_norm = 0.0
        self.last_residual_term_norm = 0.0
        self.last_final_acc_cmd_norm = 0.0
        self.last_residual_contribution_ratio = 0.0
        self.last_prior_contribution_ratio = 0.0
        self.last_prior_term_norm_search = 0.0
        self.last_residual_term_norm_search = 0.0
        self.last_residual_contribution_ratio_search = 0.0
        self.last_prior_term_norm_executor = 0.0
        self.last_residual_term_norm_executor = 0.0
        self.last_residual_contribution_ratio_executor = 0.0
        self.last_residual_norm = 0.0

    def _record_residual_prior_diagnostics(self, prior, residual, final):
        prior_norm = torch.norm(prior.detach(), dim=1)
        residual_norm = torch.norm(residual.detach(), dim=1)
        final_norm = torch.norm(final.detach(), dim=1)
        denominator = prior_norm + residual_norm + self.eps
        residual_ratio = residual_norm / denominator
        prior_ratio = prior_norm / denominator
        self.last_prior_term_norm = float(prior_norm.mean())
        self.last_residual_term_norm = float(residual_norm.mean())
        self.last_final_acc_cmd_norm = float(final_norm.mean())
        self.last_residual_contribution_ratio = float(residual_ratio.mean())
        self.last_prior_contribution_ratio = float(prior_ratio.mean())
        self.last_prior_term_norm_search = float(prior_norm[:3].mean())
        self.last_residual_term_norm_search = float(residual_norm[:3].mean())
        self.last_residual_contribution_ratio_search = float(residual_ratio[:3].mean())
        self.last_prior_term_norm_executor = float(prior_norm[3])
        self.last_residual_term_norm_executor = float(residual_norm[3])
        self.last_residual_contribution_ratio_executor = float(residual_ratio[3])

    def _apply_agent_dynamics(self, actions):
        actions = torch.clamp(self._actions_to_tensor(actions), -1.0, 1.0)
        self._collision_flags.zero_()
        old_position = self._agent_pos.clone()
        blocked = self.agent_finished.clone()
        actions = actions.clone()
        actions[blocked] = 0.0
        raw_residual = self._actions_to_residual_acc(actions)
        residual = self._residual_scale[:, None] * raw_residual
        prior = (
            self._prior_strength[:, None] * self._compute_waypoint_prior_acc()
            if self.use_residual_prior
            else torch.zeros_like(raw_residual)
        )
        desired_acceleration = prior + residual
        desired_acceleration[:, :2] = torch.clamp(
            desired_acceleration[:, :2], -self._a_xy_max[:, None], self._a_xy_max[:, None]
        )
        desired_acceleration[:, 2] = torch.clamp(
            desired_acceleration[:, 2], -self._a_z_max, self._a_z_max
        )
        self._last_residual_acc.copy_(residual.detach())
        self._last_prior_acc.copy_(prior.detach())
        self.last_residual_norm = float(torch.norm(residual, dim=1).mean())
        self._record_residual_prior_diagnostics(prior, residual, desired_acceleration)
        self._agent_acc.copy_(desired_acceleration)

        flow = self._flow_at(self._agent_pos)
        self._agent_vel[:, :2] += desired_acceleration[:, :2] * self.dt
        self._agent_vel[:, :2] += (
            -self._drag_xy[:, None] * self._agent_vel[:, :2] + flow[:, :2]
        ) * self.dt
        self._agent_vel[:, 2] += desired_acceleration[:, 2] * self.dt
        self._agent_vel[:, 2] += (
            -self._drag_z * self._agent_vel[:, 2] + self._buoyancy_bias
        ) * self.dt
        xy_speed = torch.norm(self._agent_vel[:, :2], dim=1).clamp_min(self.eps)
        xy_scale = torch.minimum(torch.ones_like(xy_speed), self._v_xy_max / xy_speed)
        self._agent_vel[:, :2] *= xy_scale[:, None]
        self._agent_vel[:, 2] = torch.clamp(self._agent_vel[:, 2], -self._v_z_max, self._v_z_max)
        self._agent_pos += self._agent_vel * self.dt
        clipped = torch.clamp(self._agent_pos, min=self._lower_bound, max=self.space_size)
        wall = torch.any(clipped != self._agent_pos, dim=1)
        self._agent_pos.copy_(clipped)
        self._agent_vel[wall] *= 0.4
        inside = self._points_inside_obstacles(self._agent_pos)
        if torch.any(inside):
            self._agent_pos[inside] = old_position[inside]
            self._agent_vel[inside] *= -0.2
            self._collision_flags[inside] = True
        self._agent_pos[blocked] = old_position[blocked]
        self._agent_vel[blocked] = 0.0
        self._agent_acc[blocked] = 0.0
        if not self.task_found:
            self.standby_executor_travel_distance += float(
                torch.norm(self._agent_pos[self.executor_idx] - old_position[self.executor_idx]).item()
            )
        self.step_count += 1

    def _planner_step_update(self):
        self._set_pse_planner_context()
        if self.step_count % self.planner_step_update_interval != 0:
            return
        self._update_pse_belief(force_detection=False)
        self.map_module.update_from_searcher_positions(
            self._agent_pos[:3],
            apply_decay=False,
            suppress_only=self.planner_step_update_suppress_only,
            sensor_ranges=self._sensor_range[:3],
        )

    def _sample_diverse_waypoint(self, agent_id, reserved_positions=None):
        reserved_positions = list(reserved_positions or [])
        reserved = torch.stack([self._vec(p) for p in reserved_positions]) if reserved_positions else None
        best_point = None
        best_score = None
        for _ in range(self.diverse_fallback_tries):
            point = self._sample_free_point(margin=1.0)
            travel = torch.norm(point - self._agent_pos[agent_id])
            travel_score = torch.exp(-0.5 * ((travel - 6.5) / 3.2) ** 2)
            separation_score = self._scalar(1.0)
            if reserved is not None:
                separation = torch.min(torch.norm(reserved - point, dim=1))
                if separation < self.planner_min_waypoint_separation:
                    continue
                separation_score = torch.clamp(
                    separation / self.planner_min_waypoint_separation, max=3.0
                )
            score = 0.65 * separation_score + 0.35 * travel_score
            if best_score is None or score > best_score:
                best_score, best_point = score, point
        return best_point if best_point is not None else self._sample_free_point(margin=1.0)

    def _choose_next_search_waypoint(self, agent_id, reserved_positions=None):
        reserved_positions = list(reserved_positions or [])
        self._set_pse_planner_context()
        if torch.rand(()).item() < self.diverse_fallback_prob:
            waypoint = self._sample_diverse_waypoint(agent_id, reserved_positions)
            self.map_module.register_waypoint_claim(waypoint)
            return waypoint
        return self.map_module.sample_next_waypoint(
            agent_id=agent_id,
            current_pos=self._agent_pos[agent_id],
            reserved_positions=reserved_positions,
        )

    def _search_spread_reward(self):
        distances = torch.cdist(self._agent_pos[:3], self._agent_pos[:3])
        distances.fill_diagonal_(1e6)
        minimum = torch.min(distances, dim=1).values
        return self.search_spread_reward_gain * torch.clamp(
            (minimum - self.safe_dist) / self.safe_dist, 0.0, 1.0
        )

    def _update_search_waypoint_events(self, nav_distances):
        if self.task_found:
            self.hold_counters[:3] = 0
            return False
        updated = False
        arrived = (nav_distances[:3] < self.search_arrive_eps) & (~self.current_target_arrived[:3])
        for i in torch.nonzero(arrived).flatten().tolist():
            self.just_reached_waypoint[i] = True
            self.waypoint_reached_counts[i] += 1
            self.current_target_arrived[i] = True
            self.map_module.register_visited_point(self._agent_pos[i], suppress_only=False)
            reserved = [self._search_waypoints[j] for j in range(3) if j != i]
            self._search_waypoints[i] = self._choose_next_search_waypoint(i, reserved)
            self.total_waypoints_per_agent[i] += 1
            self.current_target_arrived[i] = False
            updated = True
        return updated

    def _update_executor_hold_events(self, nav_distances, speeds):
        i = self.executor_idx
        stable = speeds[i] < self.hold_speed_thresh
        if not self.task_found or not self.executor_target_assigned:
            near = nav_distances[i] < self.executor_hold_radius
            if bool(near) and not bool(self.current_target_arrived[i]):
                self.waypoint_reached_counts[i] += 1
                self.current_target_arrived[i] = True
            if not self.executor_wait_held:
                self.hold_counters[i] = self.hold_counters[i] + 1 if bool(near and stable) else 0
            if not self.executor_wait_held and int(self.hold_counters[i]) >= self.executor_hold_steps:
                self.executor_wait_held = True
                self.just_held_target[i] = True
                self.hold_success_counts[i] += 1
                self._executor_wait_hold_event = True
            return
        near = nav_distances[i] < self.executor_arrive_eps
        if bool(near) and not bool(self.current_target_arrived[i]):
            self.waypoint_reached_counts[i] += 1
            self.current_target_arrived[i] = True
        self.hold_counters[i] = self.hold_counters[i] + 1 if bool(near and stable) else 0
        if not self.mission_complete and int(self.hold_counters[i]) >= self.executor_hold_steps:
            self.just_held_target[i] = True
            self.hold_success_counts[i] += 1
            self.mission_complete = True
            self.agent_finished[i] = True
            self._mission_complete_event = True
            self.success_step = int(self.step_count)

    @staticmethod
    def reward_component_names():
        return (
            "reward_find_event",
            "reward_early_find",
            "reward_completion_event",
            "reward_early_completion",
            "reward_progress",
            "reward_coverage",
            "reward_time_penalty",
            "reward_energy_penalty",
            "reward_smoothness_penalty",
            "reward_collision_penalty",
            "reward_separation_penalty",
        )

    def _empty_reward_components(self):
        return {
            name: torch.zeros(4, dtype=self.dtype, device=self.device)
            for name in self.reward_component_names()
        }

    def _calculate_mission_rewards(self, previous_nav_distances):
        components = self._empty_reward_components()
        pairwise = torch.cdist(self._agent_pos, self._agent_pos)
        separation = torch.clamp(self.safe_dist - pairwise, min=0.0)
        separation.fill_diagonal_(0.0)
        components["reward_separation_penalty"] = -self.sep_penalty_k * separation.sum(dim=1)
        components["reward_collision_penalty"] = (
            self._collision_flags.to(self.dtype) * self.collision_penalty
        )
        components["reward_time_penalty"].fill_(-self.time_penalty)
        components["reward_energy_penalty"] = (
            -self.lambda_a * self._energy_coeff * torch.sum(self._agent_acc ** 2, dim=1)
        )
        components["reward_smoothness_penalty"] = (
            -self.lambda_da * torch.sum((self._agent_acc - self._prev_acc) ** 2, dim=1)
        )

        current = self._compute_nav_distances()
        progress = previous_nav_distances - current
        if not self.task_found:
            components["reward_progress"][:3] += self._progress_gain[:3] * progress[:3]
            reached_indices = torch.nonzero(self.just_reached_waypoint[:3]).flatten()
            if reached_indices.numel():
                components["reward_coverage"][reached_indices] += self._waypoint_bonus[reached_indices]
            coverage_delta = max(0.0, self._current_coverage_ratio_internal() - self._prev_coverage_ratio)
            components["reward_coverage"][:3] += self.coverage_reward_gain * coverage_delta
            components["reward_coverage"][:3] += self._search_spread_reward()
            components["reward_progress"][3] += 8.0 * progress[3]
            if self._executor_wait_hold_event:
                components["reward_progress"][3] += self.executor_hold_bonus
        else:
            # Efficiency-v2 freezes the three searchers after discovery.  They
            # can no longer influence executor completion time, so continuing
            # shaping and penalties must not assign executor delay to inactive
            # agents.  Pilot-v1 intentionally retains its historical reward
            # behavior for reproducibility.
            if self.efficiency_protocol_v2:
                for name in (
                    "reward_progress",
                    "reward_coverage",
                    "reward_time_penalty",
                    "reward_energy_penalty",
                    "reward_smoothness_penalty",
                    "reward_collision_penalty",
                    "reward_separation_penalty",
                    "reward_completion_event",
                    "reward_early_completion",
                ):
                    components[name][:3] = 0.0
            if self._found_event:
                components["reward_find_event"][:3] += self.team_find_bonus
                components["reward_find_event"][self.finder_idx] += (
                    self._detect_bonus[self.finder_idx] + self.finder_extra_bonus
                )
                early_factor = float(np.clip(
                    1.0 - float(self.found_step) / max(1.0, float(self.max_steps)),
                    0.0,
                    1.0,
                ))
                components["reward_early_find"][self.finder_idx] += (
                    self.early_find_bonus_gain * early_factor
                )
            if self.executor_target_assigned:
                components["reward_progress"][3] += self._progress_gain[3] * progress[3]
                if self._mission_complete_event:
                    components["reward_completion_event"][3] += self.mission_complete_bonus
                    early_factor = float(np.clip(
                        1.0 - float(self.success_step) / max(1.0, float(self.max_steps)),
                        0.0,
                        1.0,
                    ))
                    components["reward_early_completion"][3] += (
                        self.early_success_bonus_gain * early_factor
                    )
            else:
                components["reward_progress"][3] += 4.0 * progress[3]
        components = {name: torch.nan_to_num(value) for name, value in components.items()}
        raw_reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
        self.last_reward_components = components
        self.last_raw_reward = raw_reward.detach().clone()
        return torch.tanh(raw_reward / self.reward_scale)


    def _compute_nav_distances(self):
        return torch.norm(self._agent_pos - self._nav_targets, dim=1)

    def _points_inside_obstacles(self, points):
        if self._obstacle_lower is None:
            return torch.zeros(points.shape[:-1], dtype=torch.bool, device=self.device)
        point = points.unsqueeze(-2)
        return (((point >= self._obstacle_lower) & (point <= self._obstacle_upper)).all(dim=-1)).any(dim=-1)

    def _nearest_obstacle_distance(self, points):
        if self._obstacle_lower is None:
            return torch.full(points.shape[:-1], 10.0, dtype=self.dtype, device=self.device)
        point = points.unsqueeze(-2)
        zeros = torch.zeros_like(point)
        delta = torch.maximum(zeros, self._obstacle_lower - point) + torch.maximum(
            zeros, point - self._obstacle_upper
        )
        return torch.norm(delta, dim=-1).min(dim=-1).values

    def is_inside_obstacle(self, point):
        point = point if torch.is_tensor(point) else self._vec(point)
        return bool(self._points_inside_obstacles(point).item())

    def get_ch3_communication_metrics(self):
        return {
            "communication_mode": self.communication_mode,
            "handoff_count": int(self.ch3_handoff_count),
            "found_step": self.found_step,
            "handoff_step": self.handoff_step,
            "executor_received_target_step": self.executor_received_target_step,
            "handoff_delay": float(self.last_handoff_delay),
        }

    def get_observation_layout(self):
        fields = (
            ("position", 3),
            ("velocity", 3),
            ("navigation_target_delta", 3),
            ("navigation_target_direction", 3),
            ("known_target_delta", 3),
            ("navigation_distance", 1),
            ("speed", 1),
            ("closing_speed", 1),
            ("nearest_obstacle_distance", 1),
            ("waypoint_progress", 1),
            ("agent_finished", 1),
            ("hold_progress", 1),
            ("role_onehot", 4),
            ("target_knowledge_phase", 2),
        )
        layout = []
        start = 0
        for name, dimension in fields:
            end = start + dimension
            layout.append({"name": name, "start": start, "end": end, "dim": dimension})
            start = end
        if start != 28:
            raise RuntimeError("invalid Chapter-3 observation layout")
        return layout

    @property
    def obs_dim(self):
        return 28

    @property
    def observation_space(self):
        return {f"agent_{i}": DummySpace(shape=(28,)) for i in range(4)}

    @property
    def action_space(self):
        return {f"agent_{i}": DummySpace(shape=(3,)) for i in range(4)}


class DummySpace:
    def __init__(self, shape):
        self.shape = shape


class UAVEnv(_BaseUAVEnv):
    """Primary Chapter-3 environment with moving targets and obstacle routing."""

    def __init__(
        self,
        *,
        target_motion_mode="static",
        target_continues_after_detection=True,
        target_state_schema="moving_target_state_v1",
        target_capture_radius=0.80,
        target_capture_hold_steps=5,
        target_max_reflections_per_step=4,
        target_belief_transition_mode="static",
        target_belief_diffusion_rate=0.0,
        handoff_payload_schema="moving_target_position_velocity_timestamp_v1",
        executor_intercept_mode="constant_velocity_reflect_fixed_point_v1",
        executor_intercept_iterations=4,
        travel_cost_mode="grid_geodesic_v1",
        navigation_path_mode="grid_astar_subgoals_v1",
        planner_obstacle_clearance=0.40,
        target_obstacle_clearance=0.20,
        path_subgoal_radius=0.75,
        path_replan_interval=10,
        failure_penalty_steps=100,
        scenario_profile="S00_STATIC_CLEAR",
        base_candidate="ch3_v3_full_reference",
        artifact_protocol=None,
        obstacle_knowledge_mode="known",
        planner_mode=None,
        obstacle_sensor_range=4.5,
        obstacle_sensor_ray_mode="neighbor26_v1",
        obstacle_sensor_noise_std=0.0,
        occupancy_free_logodds=-0.85,
        occupancy_occupied_logodds=1.70,
        occupancy_logodds_clip=6.0,
        occupancy_free_threshold=0.30,
        occupancy_occupied_threshold=0.70,
        occupancy_unknown_cost_weight=0.35,
        occupancy_risk_cost_weight=1.25,
        occupancy_replan_probability_delta=0.08,
        target_negative_observation_strength=0.90,
        target_negative_likelihood_floor=0.05,
        target_revisit_half_life_steps=30.0,
        target_recency_penalty_weight=0.15,
        obstacle_information_gain_weight=0.10,
        reservation_decay=0.985,
        replan_on_map_change=True,
        target_motion_known=True,
        map_sharing_mode="central_shared_deterministic_v1",
        unknown_map_schema="shared_logodds_occupancy_v1",
        target_belief_schema="moving_target_bayes_filter_v1",
        **kwargs,
    ):
        self._mission_initializing = True
        self._mission_resetting = False

        requested_protocol = str(kwargs.get("protocol", CH3_MISSION_V1))
        self.artifact_protocol = str(artifact_protocol or requested_protocol)
        self._mission_features_enabled = (
            requested_protocol == CH3_MISSION_V1
            or self.artifact_protocol == CH3_UNKNOWN_MAP_V1
        )

        self.target_motion_mode = str(target_motion_mode)
        self.target_continues_after_detection = bool(target_continues_after_detection)
        self.target_state_schema = str(target_state_schema)
        self.target_capture_radius = float(target_capture_radius)
        self.target_capture_hold_steps = int(target_capture_hold_steps)
        self.target_max_reflections_per_step = int(target_max_reflections_per_step)
        self.target_belief_transition_mode = str(target_belief_transition_mode)
        self.target_belief_diffusion_rate = float(target_belief_diffusion_rate)
        self.handoff_payload_schema = str(handoff_payload_schema)
        self.executor_intercept_mode = str(executor_intercept_mode)
        self.executor_intercept_iterations = int(executor_intercept_iterations)
        self.travel_cost_mode = str(travel_cost_mode)
        self.navigation_path_mode = str(navigation_path_mode)
        self.planner_obstacle_clearance = float(planner_obstacle_clearance)
        self.target_obstacle_clearance = float(target_obstacle_clearance)
        self.path_subgoal_radius = float(path_subgoal_radius)
        self.path_replan_interval = max(1, int(path_replan_interval))
        self.failure_penalty_steps = int(failure_penalty_steps)
        self.scenario_profile = str(scenario_profile)
        self.base_candidate = str(base_candidate)

        self.obstacle_knowledge_mode = str(obstacle_knowledge_mode)
        if self.obstacle_knowledge_mode not in {"known", "online_unknown", "oracle"}:
            raise ValueError(
                "obstacle_knowledge_mode must be 'known', 'online_unknown', or 'oracle'"
            )
        self.planner_mode = str(
            planner_mode
            or ("online_astar_v1" if self.obstacle_knowledge_mode == "online_unknown"
                else "oracle_astar_v1" if self.obstacle_knowledge_mode == "oracle"
                else "grid_astar_v1")
        )
        self.obstacle_sensor_range = float(max(obstacle_sensor_range, 0.1))
        self.obstacle_sensor_ray_mode = str(obstacle_sensor_ray_mode)
        if self.obstacle_sensor_ray_mode != "neighbor26_v1":
            raise ValueError("only obstacle_sensor_ray_mode='neighbor26_v1' is supported")
        self.obstacle_sensor_noise_std = float(max(obstacle_sensor_noise_std, 0.0))
        if self.obstacle_sensor_noise_std != 0.0:
            raise ValueError("v1 requires deterministic obstacle_sensor_noise_std=0.0")
        self.occupancy_free_logodds = float(occupancy_free_logodds)
        self.occupancy_occupied_logodds = float(occupancy_occupied_logodds)
        self.occupancy_logodds_clip = float(occupancy_logodds_clip)
        self.occupancy_free_threshold = float(occupancy_free_threshold)
        self.occupancy_occupied_threshold = float(occupancy_occupied_threshold)
        self.occupancy_unknown_cost_weight = float(occupancy_unknown_cost_weight)
        self.occupancy_risk_cost_weight = float(occupancy_risk_cost_weight)
        self.occupancy_replan_probability_delta = float(
            occupancy_replan_probability_delta
        )
        self.target_negative_observation_strength = float(
            target_negative_observation_strength
        )
        self.target_negative_likelihood_floor = float(
            target_negative_likelihood_floor
        )
        self.target_revisit_half_life_steps = float(target_revisit_half_life_steps)
        self.target_recency_penalty_weight = float(target_recency_penalty_weight)
        self.obstacle_information_gain_weight = float(obstacle_information_gain_weight)
        self.reservation_decay = float(reservation_decay)
        self.replan_on_map_change = bool(replan_on_map_change)
        self.target_motion_known = bool(target_motion_known)
        self.map_sharing_mode = str(map_sharing_mode)
        self.unknown_map_schema = str(unknown_map_schema)
        self.target_belief_schema = str(target_belief_schema)

        self.ground_truth_obstacles = []
        self.map_sensor_scan_count = 0
        self.map_triggered_replan_count = 0
        self.map_collision_count = 0
        self.target_prediction_map_fallback_count = 0
        self._last_map_change_step = None

        kwargs.setdefault("protocol", CH3_MISSION_V1)
        super().__init__(**kwargs)

        if self._mission_features_enabled:
            if self.protocol == CH3_MISSION_V1:
                self.efficiency_protocol_v2 = True
            self.map_module = self._build_obstacle_planner()
            self._initialize_path_state()

        self._mission_initializing = False
        self.reset()

    def _planner_common_kwargs(self):
        old = self.map_module
        return dict(
            space_size=self.space_size,
            n_agents=4,
            n_search=3,
            executor_idx=3,
            grid_size=old.grid_size,
            z_range=self.random_z_range,
            search_count_range=old.search_count_range,
            executor_count_range=old.executor_count_range,
            visit_radius=old.visit_radius,
            pheromone_decay=old.pheromone_decay,
            suppression=old.suppression,
            min_waypoint_separation=old.min_waypoint_separation,
            coverage_weight=old.coverage_weight,
            claim_weight=old.claim_weight,
            stochastic_topk=old.stochastic_topk,
            stochastic_eps=old.stochastic_eps,
            device=self.device,
            dtype=self.dtype,
            pse_belief_detect_prob=self.pse_belief_detect_prob,
            pse_belief_miss_decay=self.pse_belief_miss_decay,
            pse_detect_sigma=self.pse_detect_sigma,
            pse_belief_topk=self.pse_belief_topk,
            pse_belief_weight=self.pse_belief_weight,
            pse_use_gated_belief=self.pse_use_gated_belief,
            pse_belief_weight_max=self.pse_belief_weight_max,
            pse_belief_gate_start_step=self.pse_belief_gate_start_step,
            pse_belief_gate_full_step=self.pse_belief_gate_full_step,
            pse_belief_entropy_high=self.pse_belief_entropy_high,
            pse_belief_entropy_low=self.pse_belief_entropy_low,
            pse_belief_uniform_mix_high=self.pse_belief_uniform_mix_high,
            pse_belief_uniform_mix_low=self.pse_belief_uniform_mix_low,
            pse_exec_cost_weight=self.pse_exec_cost_weight,
            pse_search_cost_weight=self.pse_search_cost_weight,
            pse_base_score_weight=self.pse_base_score_weight,
            pse_standby_topk=self.pse_standby_topk,
            pse_standby_candidates=self.pse_standby_candidates,
            pse_standby_move_weight=self.pse_standby_move_weight,
            pse_standby_hysteresis_weight=self.pse_standby_hysteresis_weight,
            pse_standby_safe_weight=self.pse_standby_safe_weight,
            planner_obstacle_clearance=self.planner_obstacle_clearance,
            target_belief_transition_mode=self.target_belief_transition_mode,
            target_belief_diffusion_rate=self.target_belief_diffusion_rate,
        )

    def _build_obstacle_planner(self):
        kwargs = self._planner_common_kwargs()
        if self.obstacle_knowledge_mode == "online_unknown":
            kwargs.update(
                occupancy_free_logodds=self.occupancy_free_logodds,
                occupancy_occupied_logodds=self.occupancy_occupied_logodds,
                occupancy_logodds_clip=self.occupancy_logodds_clip,
                occupancy_free_threshold=self.occupancy_free_threshold,
                occupancy_occupied_threshold=self.occupancy_occupied_threshold,
                occupancy_unknown_cost_weight=self.occupancy_unknown_cost_weight,
                occupancy_risk_cost_weight=self.occupancy_risk_cost_weight,
                occupancy_replan_probability_delta=(
                    self.occupancy_replan_probability_delta
                ),
                target_negative_observation_strength=(
                    self.target_negative_observation_strength
                ),
                target_negative_likelihood_floor=(
                    self.target_negative_likelihood_floor
                ),
                target_revisit_half_life_steps=(
                    self.target_revisit_half_life_steps
                ),
                target_recency_penalty_weight=(
                    self.target_recency_penalty_weight
                ),
                obstacle_information_gain_weight=(
                    self.obstacle_information_gain_weight
                ),
                reservation_decay=self.reservation_decay,
            )
            return OnlineUnknownMapTaskPlanner(**kwargs)
        return ObstacleAwareTaskMapPlanner(**kwargs)

    def _initialize_path_state(self):
        self._navigation_paths = [[] for _ in range(4)]
        self._navigation_path_indices = [0] * 4
        self._path_final_targets = torch.full(
            (4, 3), float("nan"), dtype=self.dtype, device=self.device
        )
        self._path_last_replan_steps = [-10**9] * 4
        self.path_replan_count = 0
        self.path_unreachable_count = 0
        self.planned_geodesic_distance = 0.0
        self.executed_path_distance = 0.0
        self.subgoal_count = 0
        self._reset_endpoint_guard_diagnostics()

    def _reset_endpoint_guard_diagnostics(self):
        self.waypoint_endpoint_guard_reject_count = 0
        self.waypoint_endpoint_point_invalid_count = 0
        self.waypoint_endpoint_no_connector_count = 0
        self.waypoint_endpoint_guard_recovery_count = 0
        self.waypoint_endpoint_guard_max_streak = 0
        self.path_replan_deferred_invalid_endpoint_count = 0
        self.path_subgoal_advance_deferred_invalid_endpoint_count = 0
        self._waypoint_endpoint_guard_streak = torch.zeros(
            self.num_agents,
            dtype=torch.int64,
            device=self.device,
        )

    def _validate_custom_obstacles(self, obstacles):
        validated = []
        bounds = self.space_size.detach().cpu().numpy()
        for index, obstacle in enumerate(obstacles or ()):
            center = np.asarray(obstacle.get("center"), dtype=np.float64).reshape(-1)
            size = np.asarray(obstacle.get("size"), dtype=np.float64).reshape(-1)
            if center.shape != (3,) or size.shape != (3,):
                raise ValueError(f"custom obstacle {index} center/size must be 3-vectors")
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
                raise ValueError(f"custom obstacle {index} must be finite")
            if np.any(size <= 0):
                raise ValueError(f"custom obstacle {index} size must be positive")
            if np.any(center - size / 2 < 0) or np.any(center + size / 2 > bounds):
                raise ValueError(f"custom obstacle {index} is outside world bounds")
            validated.append({
                "center": center.astype(np.float32).copy(),
                "size": size.astype(np.float32).copy(),
            })
        return validated

    def _apply_scenario_obstacles(self, scenario):
        if self._mission_initializing:
            return super()._apply_scenario_obstacles(scenario)
        scenario = {} if scenario is None else dict(scenario)
        layout = str(scenario.get("obstacle_layout_id", "none"))
        if layout == "none":
            obstacles = []
        elif layout == "default_fixed_v1":
            obstacles = deepcopy(self.default_obstacles)
        elif layout == "custom_aabb_v1":
            obstacles = self._validate_custom_obstacles(scenario.get("obstacles", []))
            if not obstacles:
                raise ValueError("custom_aabb_v1 requires at least one obstacle")
        else:
            raise ValueError(f"unsupported obstacle layout={layout!r}")
        self.use_obstacles = bool(obstacles)
        self.obstacle_layout_id = layout if obstacles else "none"
        self.obstacles = deepcopy(obstacles)
        self._build_obstacle_tensors()

    def reset(self, scenario=None):
        if self._mission_initializing or not self._mission_features_enabled:
            return _BaseUAVEnv.reset(self, scenario=scenario)

        raw = {} if scenario is None else deepcopy(dict(scenario))
        declared_mode = str(
            raw.get("obstacle_knowledge_mode", self.obstacle_knowledge_mode)
        )
        if declared_mode != self.obstacle_knowledge_mode:
            raise ValueError("scenario obstacle_knowledge_mode does not match runtime")
        if raw:
            raw["target_position"] = raw.get(
                "target_initial_position", raw.get("target_position")
            )

        self._mission_resetting = True
        _BaseUAVEnv.reset(self, scenario=raw or None)
        self._mission_resetting = False

        self.ground_truth_obstacles = deepcopy(self.obstacles)
        initial_velocity = (
            np.zeros(3, dtype=np.float64)
            if not raw
            else np.asarray(
                raw.get("target_initial_velocity", [0, 0, 0]), dtype=np.float64
            )
        )
        motion_mode = str(raw.get("target_motion_mode", self.target_motion_mode))
        self.target_state = TargetState(
            self._task_target.detach().cpu().numpy(),
            initial_velocity,
            0,
            motion_mode,
            state_schema=self.target_state_schema,
            obstacle_layout_id=self.obstacle_layout_id,
        )
        self.initial_target_state = self.target_state.copy()
        self.executor_delivered_target_state = None
        self.predicted_intercept_position = None
        self._intercept_cache_signature = None
        self._intercept_cache_payload = None
        self.intercept_cache_hit_count = 0
        self.intercept_cache_miss_count = 0
        self.predicted_target_position_at_delivery = None
        self.target_prediction_error_at_delivery = None
        self._target_prediction_error_sum = 0.0
        self._target_prediction_error_count = 0
        self.target_position_at_found = None
        self.target_velocity_at_found = None
        self.target_position_at_capture = None
        self.capture_position_error = None
        self.handoff_payload_sample_step = None
        self.handoff_payload_delivery_step = None
        self.handoff_payload_age_steps = None
        self.handoff_delivery_phase = "pre_transition"
        self.handoff_event_delay_steps = 1
        self.handoff_physical_age_at_delivery_steps = None
        self.target_distance_travelled = 0.0
        self.obstacle_collision_count = 0
        self._capture_hold_counter = 0
        self.capture_swept_min_distance = None
        self.capture_contact_step_count = 0
        self.capture_full_hold_step_count = 0
        self.capture_hold_counter_max = 0
        self.map_sensor_scan_count = 0
        self.map_triggered_replan_count = 0
        self.map_collision_count = 0
        self.target_prediction_map_fallback_count = 0
        self._last_map_change_step = None

        if self.online_unknown_map_active:
            if getattr(self.map_module, "obstacles", None):
                raise RuntimeError(
                    "online planner was initialized with ground-truth obstacles"
                )
            self.map_module.set_mapping_step(0)
            self._sense_and_update_shared_map()
            self._search_waypoints = self.map_module.initial_search_targets(
                self._agent_pos[:3]
            )
            self._nav_targets[:3] = self._search_waypoints
            self._nav_targets[3] = self._executor_wait_point
            self._targets.copy_(self._nav_targets)

        self._initialize_path_state()
        self._update_nav_targets(force=True)
        return self._obs_to_public(self._get_obs())

    def _final_navigation_targets(self):
        finals = self._agent_pos.clone()
        if not self.task_found:
            finals[:3] = self._search_waypoints
            finals[3] = self._executor_wait_point
        elif self.executor_target_assigned and self.predicted_intercept_position is not None:
            finals[3] = self._vec(self.predicted_intercept_position)
        else:
            finals[3] = self._executor_wait_point
        return finals

    def _update_nav_targets(self, force=False):
        if (
            self._mission_initializing
            or self._mission_resetting
            or not self._mission_features_enabled
        ):
            return _BaseUAVEnv._update_nav_targets(self)
        finals = self._final_navigation_targets()
        for agent in range(4):
            if agent < self.n_search and (
                self.task_found or bool(self.agent_finished[agent].item())
            ):
                self._navigation_paths[agent] = []
                self._navigation_path_indices[agent] = 0
                self._nav_targets[agent] = self._agent_pos[agent]
                continue
            changed = (
                not bool(torch.isfinite(self._path_final_targets[agent]).all())
                or float(
                    torch.norm(finals[agent] - self._path_final_targets[agent]).item()
                )
                > self.path_subgoal_radius
            )
            periodic = (
                self.step_count - self._path_last_replan_steps[agent]
                >= self.path_replan_interval
            )
            replan_requested = bool(force or changed or periodic)
            path = self._navigation_paths[agent]
            index = self._navigation_path_indices[agent]
            subgoal_advance_requested = (
                index < len(path) - 1
                and float(
                    torch.norm(
                        self._agent_pos[agent] - path[index]
                    ).item()
                )
                <= self.path_subgoal_radius
            )
            role = (
                "executor"
                if agent == self.executor_idx
                else "searcher"
            )
            if replan_requested or subgoal_advance_requested:
                endpoint = self.map_module.endpoint_status(
                    self._agent_pos[agent],
                    role=role,
                )
                if not endpoint["reachable"]:
                    if replan_requested:
                        self.path_replan_deferred_invalid_endpoint_count += 1
                    if subgoal_advance_requested:
                        self.path_subgoal_advance_deferred_invalid_endpoint_count += 1
                    continue
            if replan_requested:
                result = self.map_module.grid_astar_path(
                    self._agent_pos[agent], finals[agent], role=role
                )
                self.path_replan_count += 1
                self._path_last_replan_steps[agent] = int(self.step_count)
                self._path_final_targets[agent] = finals[agent].clone()
                if result["reachable"]:
                    path = self.map_module.path_to_subgoals(result, finals[agent])
                    self._navigation_paths[agent] = path
                    self._navigation_path_indices[agent] = 0
                    previous = self._agent_pos[agent]
                    self.planned_geodesic_distance += sum(
                        float(
                            torch.norm(
                                point
                                - (previous if index == 0 else path[index - 1])
                            ).item()
                        )
                        for index, point in enumerate(path)
                    )
                    self.subgoal_count += len(path)
                else:
                    self.path_unreachable_count += 1
                    self._navigation_paths[agent] = []
                    self._navigation_path_indices[agent] = 0
            path = self._navigation_paths[agent]
            index = self._navigation_path_indices[agent]
            while (
                index < len(path) - 1
                and float(torch.norm(self._agent_pos[agent] - path[index]).item())
                <= self.path_subgoal_radius
            ):
                index += 1
            self._navigation_path_indices[agent] = index
            self._nav_targets[agent] = (
                path[index] if index < len(path) else self._agent_pos[agent]
            )
        self._targets.copy_(self._nav_targets)

    def _publish_detection(self, finder_idx):
        self.task_found = True
        self.finder_idx = int(finder_idx)
        self.search_stage_complete = True
        self.agent_finished[:3] = True
        self._found_event = True
        self.found_step = int(self.step_count)
        self.handoff_step = int(self.step_count)
        self.target_position_at_found = self.target_state.position.copy()
        self.target_velocity_at_found = self.target_state.velocity.copy()
        self._agent_task_known[finder_idx] = True
        self._agent_task_est[finder_idx].copy_(self._vec(self.target_state.position))
        payload = self.target_state.copy()
        payload.sample_step = int(self.step_count)
        payload.obstacle_layout_id = self.obstacle_layout_id
        published = self.fixed_reliable_handoff.publish_target(
            found_step=self.found_step, finder_idx=finder_idx, target=payload
        )
        if not published:
            raise RuntimeError("target handoff was already published")
        self.map_module.update_belief_detection(self._task_target)
        self._sync_pse_diagnostics()

    def _maybe_detect_swept(self, agent_start, target_start):
        if self.task_found:
            return
        target_end = self.target_state.position
        candidates = []
        for agent in range(3):
            distance, tau = swept_relative_min_distance(
                agent_start[agent],
                self._agent_pos[agent].detach().cpu().numpy(),
                target_start,
                target_end,
            )
            if distance <= float(self._sensor_range[agent].item()) + self.detect_eps_bias:
                candidates.append((distance, agent, tau))
        if candidates:
            _, finder, _ = min(candidates, key=lambda item: (item[0], item[1]))
            self._publish_detection(finder)

    def _travel_time_for_intercept(self, start, goal):
        result = self.map_module.grid_geodesic_cost(start, goal, role="executor")
        if not result["reachable"]:
            return float("inf")
        return float(result.get("travel_time", result["cost"]))

    def _known_prediction_obstacles(self):
        if self.online_unknown_map_active:
            return self.map_module.known_obstacle_aabbs()
        return self.ground_truth_obstacles

    # CH3_M00_EQUIVALENT_OPTIMIZATION_V1
    def _intercept_solver_signature(self, current_step):
        state = self.executor_delivered_target_state
        if state is None:
            return None
        executor_position = (
            self._agent_pos[self.executor_idx]
            .detach().cpu().numpy().astype(np.float64, copy=False)
        )
        payload = state.to_payload()
        return (
            int(current_step),
            tuple(float(value) for value in executor_position),
            int(getattr(self.map_module, "grid_revision", -1)),
            int(getattr(self.map_module, "map_revision", -1)),
            str(getattr(self.map_module, "obstacle_layout_hash", "")),
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _apply_cached_intercept(self, payload, *, replay_diagnostics):
        if replay_diagnostics:
            self.path_unreachable_count += int(
                payload["path_unreachable_delta"]
            )
            self.target_prediction_map_fallback_count += int(
                payload["map_fallback_delta"]
            )
        position = payload["position"]
        if position is None:
            self.predicted_intercept_position = None
            return
        self.predicted_intercept_position = np.asarray(
            position, dtype=np.float64
        ).copy()
        self._agent_task_est[self.executor_idx].copy_(
            self._vec(self.predicted_intercept_position)
        )

    def _refresh_intercept(self, current_step):
        if self.executor_delivered_target_state is None:
            return
        signature = self._intercept_solver_signature(current_step)
        if (
            signature == self._intercept_cache_signature
            and self._intercept_cache_payload is not None
        ):
            self.intercept_cache_hit_count += 1
            self._apply_cached_intercept(
                self._intercept_cache_payload,
                replay_diagnostics=True,
            )
            return

        self.intercept_cache_miss_count += 1
        unreachable_before = int(self.path_unreachable_count)
        fallback_before = int(self.target_prediction_map_fallback_count)
        known_obstacles = self._known_prediction_obstacles()
        try:
            solution = solve_intercept_point(
                self._agent_pos[self.executor_idx].detach().cpu().numpy(),
                self.executor_delivered_target_state,
                int(current_step),
                self._travel_time_for_intercept,
                dt=self.dt,
                bounds=self.space_size.detach().cpu().numpy(),
                obstacles=known_obstacles,
                clearance=self.target_obstacle_clearance,
                max_iterations=self.executor_intercept_iterations,
                max_reflections=self.target_max_reflections_per_step,
                max_prediction_steps=self.max_steps + self.failure_penalty_steps,
            )
        except (RuntimeError, ValueError):
            if not self.online_unknown_map_active:
                raise
            self.target_prediction_map_fallback_count += 1
            solution = solve_intercept_point(
                self._agent_pos[self.executor_idx].detach().cpu().numpy(),
                self.executor_delivered_target_state,
                int(current_step),
                self._travel_time_for_intercept,
                dt=self.dt,
                bounds=self.space_size.detach().cpu().numpy(),
                obstacles=(),
                clearance=0.0,
                max_iterations=self.executor_intercept_iterations,
                max_reflections=self.target_max_reflections_per_step,
                max_prediction_steps=self.max_steps + self.failure_penalty_steps,
            )
        if not solution["reachable"]:
            self.path_unreachable_count += 1
            position = None
        else:
            position = np.asarray(
                solution["position"], dtype=np.float64
            ).copy()
        payload = {
            "position": position,
            "path_unreachable_delta": (
                int(self.path_unreachable_count) - unreachable_before
            ),
            "map_fallback_delta": (
                int(self.target_prediction_map_fallback_count) - fallback_before
            ),
        }
        self._intercept_cache_signature = signature
        self._intercept_cache_payload = payload
        self._apply_cached_intercept(payload, replay_diagnostics=False)


    def _predict_delivered_target(self, physical_age):
        known_obstacles = self._known_prediction_obstacles()
        try:
            return predict_target_state(
                self.executor_delivered_target_state,
                physical_age,
                self.dt,
                self.space_size.detach().cpu().numpy(),
                known_obstacles,
                clearance=self.target_obstacle_clearance,
                max_reflections=self.target_max_reflections_per_step,
                max_prediction_steps=self.max_steps + self.failure_penalty_steps,
            )
        except (RuntimeError, ValueError):
            if not self.online_unknown_map_active:
                raise
            self.target_prediction_map_fallback_count += 1
            return predict_target_state(
                self.executor_delivered_target_state,
                physical_age,
                self.dt,
                self.space_size.detach().cpu().numpy(),
                (),
                clearance=0.0,
                max_reflections=self.target_max_reflections_per_step,
                max_prediction_steps=self.max_steps + self.failure_penalty_steps,
            )

    def _advance_fixed_handoff(self):
        if self._mission_initializing or not self._mission_features_enabled:
            return _BaseUAVEnv._advance_fixed_handoff(self)
        event = self.fixed_reliable_handoff.advance(entering_step=self.step_count + 1)
        if event is None:
            return False
        delivered = TargetState.from_payload(event["target"].to_payload())
        self.executor_delivered_target_state = delivered.copy()
        self._agent_task_known[self.executor_idx] = True
        self.executor_target_assigned = True
        self.current_target_arrived[self.executor_idx] = False
        self.executor_received_target_step = int(event["delivery_step"])
        self.last_handoff_delay = float(event["delivery_step"] - event["found_step"])
        self.ch3_handoff_count += 1
        self.handoff_payload_sample_step = int(delivered.sample_step)
        self.handoff_payload_delivery_step = int(event["delivery_step"])
        self.handoff_payload_age_steps = 0
        self.handoff_physical_age_at_delivery_steps = 0
        predicted = delivered.copy()
        self.predicted_target_position_at_delivery = predicted.position.copy()
        self.target_prediction_error_at_delivery = float(
            np.linalg.norm(predicted.position - self.target_state.position)
        )
        self._refresh_intercept(self.step_count)
        return True

    def _update_search_path_events(self):
        if self.task_found:
            return
        for agent in range(3):
            distance = float(
                torch.norm(
                    self._agent_pos[agent]
                    - self._search_waypoints[agent]
                ).item()
            )
            if distance >= self.search_arrive_eps:
                continue
            endpoint = self.map_module.endpoint_status(
                self._agent_pos[agent],
                role="searcher",
            )
            if not endpoint["reachable"]:
                self.waypoint_endpoint_guard_reject_count += 1
                self._waypoint_endpoint_guard_streak[agent] += 1
                streak = int(
                    self._waypoint_endpoint_guard_streak[agent].item()
                )
                self.waypoint_endpoint_guard_max_streak = max(
                    self.waypoint_endpoint_guard_max_streak,
                    streak,
                )
                if endpoint["failure_reason"] == "point_invalid":
                    self.waypoint_endpoint_point_invalid_count += 1
                elif endpoint["failure_reason"] == "no_connector":
                    self.waypoint_endpoint_no_connector_count += 1
                else:
                    raise RuntimeError(
                        "endpoint_status returned an invalid failure reason"
                    )
                continue
            if int(
                self._waypoint_endpoint_guard_streak[agent].item()
            ) > 0:
                self.waypoint_endpoint_guard_recovery_count += 1
                self._waypoint_endpoint_guard_streak[agent] = 0
            self.just_reached_waypoint[agent] = True
            self.waypoint_reached_counts[agent] += 1
            self.map_module.register_visited_point(self._agent_pos[agent], suppress_only=False)
            reserved = [self._search_waypoints[j] for j in range(3) if j != agent]
            self._search_waypoints[agent] = self._choose_next_search_waypoint(agent, reserved)
            self.total_waypoints_per_agent[agent] += 1
            self.current_target_arrived[agent] = False

    def _update_capture(self, agent_start, target_start):
        if not (self.task_found and self.executor_target_assigned):
            self._capture_hold_counter = 0
            self.hold_counters[self.executor_idx] = 0
            return
        executor_start = np.asarray(agent_start[self.executor_idx], dtype=np.float64)
        executor_end = (
            self._agent_pos[self.executor_idx].detach().cpu().numpy().astype(np.float64)
        )
        target_start = np.asarray(target_start, dtype=np.float64)
        target_end = self.target_state.position.astype(np.float64)
        distance_start = float(np.linalg.norm(executor_start - target_start))
        distance_end = float(np.linalg.norm(executor_end - target_end))
        swept_min, _ = swept_relative_min_distance(
            executor_start, executor_end, target_start, target_end
        )
        self.capture_swept_min_distance = float(swept_min)
        contact = swept_min <= self.target_capture_radius
        full_hold = max(distance_start, distance_end) <= self.target_capture_radius
        if contact:
            self.capture_contact_step_count += 1
        if full_hold:
            self.capture_full_hold_step_count += 1
            self._capture_hold_counter += 1
        else:
            self._capture_hold_counter = 0
        self.capture_hold_counter_max = max(
            self.capture_hold_counter_max, self._capture_hold_counter
        )
        self.hold_counters[self.executor_idx] = self._capture_hold_counter
        if not self.mission_complete and self._capture_hold_counter >= self.target_capture_hold_steps:
            self.mission_complete = True
            self.agent_finished[self.executor_idx] = True
            self._mission_complete_event = True
            self.success_step = int(self.step_count)
            self.target_position_at_capture = self.target_state.position.copy()
            self.capture_position_error = distance_end

    @property
    def mean_target_prediction_error(self):
        if self._target_prediction_error_count == 0:
            return None
        return self._target_prediction_error_sum / self._target_prediction_error_count

    @property
    def online_unknown_map_active(self):
        return (
            self._mission_features_enabled
            and self.obstacle_knowledge_mode == "online_unknown"
            and hasattr(self.map_module, "integrate_obstacle_scan")
        )

    def _ray_limit_to_world(self, origin, direction, maximum):
        limit = float(maximum)
        bounds = self.space_size.detach().cpu().numpy().astype(np.float64)
        for axis in range(3):
            value = float(direction[axis])
            if abs(value) <= 1e-12:
                continue
            boundary = bounds[axis] if value > 0.0 else 0.0
            candidate = (boundary - float(origin[axis])) / value
            if candidate >= 0.0:
                limit = min(limit, candidate)
        return max(0.0, limit)

    def _raycast_ground_truth(self, origin, direction):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return 0.0, False
        direction = direction / norm
        maximum = self._ray_limit_to_world(
            origin, direction, self.obstacle_sensor_range
        )
        end = origin + direction * maximum
        best = maximum
        hit_any = False
        for obstacle in self.ground_truth_obstacles:
            center = np.asarray(obstacle["center"], dtype=np.float64)
            size = np.asarray(obstacle["size"], dtype=np.float64)
            lower = center - size / 2.0
            upper = center + size / 2.0
            hit = segment_aabb_first_hit(origin, end, lower, upper)
            if hit is None:
                continue
            distance = float(hit[0]) * maximum
            if distance < best:
                best = distance
                hit_any = True
        return best, hit_any

    def _sense_and_update_shared_map(self):
        if not self.online_unknown_map_active:
            return False
        self.map_module.set_mapping_step(self.step_count)
        directions = self.map_module.sensor_directions.detach().cpu().numpy()
        origins = self._agent_pos.detach().cpu().numpy().astype(np.float64)
        distances = np.zeros(
            (origins.shape[0], directions.shape[0]), dtype=np.float64
        )
        hit_mask = np.zeros_like(distances, dtype=bool)
        for agent_index, origin in enumerate(origins):
            for ray_index, direction in enumerate(directions):
                distance, hit = self._raycast_ground_truth(origin, direction)
                distances[agent_index, ray_index] = distance
                hit_mask[agent_index, ray_index] = hit
        changed = self.map_module.integrate_obstacle_scan(
            origins,
            directions,
            distances,
            hit_mask,
            current_step=self.step_count,
        )
        self.map_sensor_scan_count += int(distances.size)
        if changed:
            self._last_map_change_step = int(self.step_count)
            if self.replan_on_map_change:
                self.map_triggered_replan_count += 1
        return bool(changed)

    def _nearest_obstacle_distance(self, points):
        truth = _BaseUAVEnv._nearest_obstacle_distance(self, points)
        if self.online_unknown_map_active:
            return torch.clamp(truth, max=self.obstacle_sensor_range)
        return truth

    def _truth_occupancy_mask(self):
        centers = self.map_module.xyz_centers
        mask = torch.zeros(
            self.map_module.grid_size,
            dtype=torch.bool,
            device=self.device,
        )
        for obstacle in self.ground_truth_obstacles:
            center = self._vec(obstacle["center"]).reshape(3)
            size = self._vec(obstacle["size"]).reshape(3)
            lower = center - size / 2.0
            upper = center + size / 2.0
            mask |= ((centers >= lower) & (centers <= upper)).all(dim=-1)
        return mask

    def get_unknown_map_metrics(self):
        if not self.online_unknown_map_active:
            return {
                "obstacle_knowledge_mode": self.obstacle_knowledge_mode,
                "planner_mode": self.planner_mode,
                "map_known_fraction": 1.0,
                "map_unknown_fraction": 0.0,
                "map_known_free_fraction": None,
                "map_known_occupied_fraction": None,
                "map_occupancy_entropy": 0.0,
                "obstacle_map_iou": 1.0,
                "obstacle_map_precision": 1.0,
                "obstacle_map_recall": 1.0,
                "map_revision": 0,
                "map_update_count": 0,
                "map_changed_cell_count_total": 0,
                "map_sensor_scan_count": 0,
                "map_triggered_replan_count": 0,
                "map_collision_count": int(self.map_collision_count),
                "target_prediction_map_fallback_count": int(
                    self.target_prediction_map_fallback_count
                ),
            }
        statistics = self.map_module.map_statistics()
        truth = self._truth_occupancy_mask()
        predicted = self.map_module.known_occupied_mask
        true_positive = int(torch.count_nonzero(predicted & truth).item())
        false_positive = int(torch.count_nonzero(predicted & ~truth).item())
        false_negative = int(torch.count_nonzero(~predicted & truth).item())
        union = int(torch.count_nonzero(predicted | truth).item())
        precision = (
            None
            if true_positive + false_positive == 0
            else true_positive / float(true_positive + false_positive)
        )
        recall = (
            None
            if true_positive + false_negative == 0
            else true_positive / float(true_positive + false_negative)
        )
        iou = None if union == 0 else true_positive / float(union)
        return {
            "obstacle_knowledge_mode": self.obstacle_knowledge_mode,
            "planner_mode": self.planner_mode,
            **statistics,
            "obstacle_map_iou": iou,
            "obstacle_map_precision": precision,
            "obstacle_map_recall": recall,
            "map_sensor_scan_count": int(self.map_sensor_scan_count),
            "map_triggered_replan_count": int(self.map_triggered_replan_count),
            "map_collision_count": int(self.map_collision_count),
            "target_prediction_map_fallback_count": int(
                self.target_prediction_map_fallback_count
            ),
            "last_map_change_step": self._last_map_change_step,
        }

    def step(self, actions):
        if not self._mission_features_enabled:
            self._found_event = False
            self._mission_complete_event = False
            self._executor_wait_hold_event = False
            self.just_reached_waypoint.zero_()
            self.just_held_target.zero_()
            self._advance_fixed_handoff()
            self._update_pse_executor_standby()
            self._update_nav_targets()
            previous_nav_distances = self._compute_nav_distances()
            self._apply_agent_dynamics(actions)
            self._planner_step_update()
            self._maybe_detect_task()
            self._update_nav_targets()
            nav_distances = self._compute_nav_distances()
            speeds = torch.norm(self._agent_vel, dim=1)
            if self._update_search_waypoint_events(nav_distances):
                self._update_nav_targets()
                nav_distances = self._compute_nav_distances()
            self._update_executor_hold_events(nav_distances, speeds)
            rewards = self._calculate_mission_rewards(previous_nav_distances)
            observations = self._get_obs()
            done = self.mission_complete or self.step_count >= self.max_steps
            self._prev_nav_distances = self._compute_nav_distances()
            self._prev_acc.copy_(self._agent_acc)
            self._prev_coverage_ratio = self._current_coverage_ratio_internal()
            self._prev_search_task_min_dist = self._current_search_task_min_dist()
            self._sync_pse_diagnostics()
            return (
                self._obs_to_public(observations),
                self._rewards_to_public(rewards),
                [done] * 4,
            )

        self._found_event = False
        self._mission_complete_event = False
        self._executor_wait_hold_event = False
        self.just_reached_waypoint.zero_()
        self.just_held_target.zero_()
        self._advance_fixed_handoff()
        self._update_pse_executor_standby()
        self._refresh_intercept(self.step_count)
        self._update_nav_targets()
        previous_nav_distances = self._compute_nav_distances()
        agent_start = self._agent_pos.detach().cpu().numpy().copy()
        if self.target_state.motion_mode == "static":
            injected_target = self._task_target.detach().cpu().numpy()
            if not np.array_equal(injected_target, self.target_state.position):
                self.target_state.position = injected_target.astype(
                    np.float64, copy=True
                )
        target_start = self.target_state.position.copy()

        self._apply_agent_dynamics(actions)
        movement = np.linalg.norm(
            self._agent_pos.detach().cpu().numpy() - agent_start, axis=1
        )
        self.executed_path_distance += float(movement.sum())
        new_collisions = int(self._collision_flags.sum().item())
        self.map_collision_count += new_collisions
        map_changed = self._sense_and_update_shared_map()

        old_target = self.target_state.position.copy()
        if self.target_continues_after_detection or not self.task_found:
            self.target_state = advance_target_state(
                self.target_state,
                self.dt,
                self.space_size.detach().cpu().numpy(),
                self.ground_truth_obstacles,
                clearance=self.target_obstacle_clearance,
                max_reflections=self.target_max_reflections_per_step,
            )
        self.target_distance_travelled += float(
            np.linalg.norm(self.target_state.position - old_target)
        )
        self._task_target.copy_(self._vec(self.target_state.position))
        self.map_module.predict_belief_motion()
        self._planner_step_update()
        self._maybe_detect_swept(agent_start, target_start)

        if self.executor_delivered_target_state is not None:
            physical_age = max(
                0,
                self.step_count - self.executor_delivered_target_state.sample_step,
            )
            self.handoff_payload_age_steps = int(physical_age)
            predicted_now = self._predict_delivered_target(physical_age)
            error = float(
                np.linalg.norm(predicted_now.position - self.target_state.position)
            )
            self._target_prediction_error_sum += error
            self._target_prediction_error_count += 1
            self._refresh_intercept(self.step_count)

        self._update_capture(agent_start, target_start)
        if not (self.task_found and self.executor_target_assigned):
            self._update_executor_hold_events(
                self._compute_nav_distances(),
                torch.norm(self._agent_vel, dim=1),
            )
        rewards = self._calculate_mission_rewards(previous_nav_distances)
        self._update_search_path_events()
        self._update_nav_targets(
            force=bool(map_changed and self.replan_on_map_change)
        )
        self.obstacle_collision_count += new_collisions
        observations = self._get_obs()
        done = self.mission_complete or self.step_count >= self.max_steps
        self._prev_nav_distances = self._compute_nav_distances()
        self._prev_acc.copy_(self._agent_acc)
        self._prev_coverage_ratio = self._current_coverage_ratio_internal()
        self._prev_search_task_min_dist = self._current_search_task_min_dist()
        self._sync_pse_diagnostics()
        return (
            self._obs_to_public(observations),
            self._rewards_to_public(rewards),
            [done] * 4,
        )


def _mission_init_signature(
    self,
    *,
    target_motion_mode="static",
    target_continues_after_detection=True,
    target_state_schema="moving_target_state_v1",
    target_capture_radius=0.80,
    target_capture_hold_steps=5,
    target_max_reflections_per_step=4,
    target_belief_transition_mode="static",
    target_belief_diffusion_rate=0.0,
    handoff_payload_schema="moving_target_position_velocity_timestamp_v1",
    executor_intercept_mode="constant_velocity_reflect_fixed_point_v1",
    executor_intercept_iterations=4,
    travel_cost_mode="grid_geodesic_v1",
    navigation_path_mode="grid_astar_subgoals_v1",
    planner_obstacle_clearance=0.40,
    target_obstacle_clearance=0.20,
    path_subgoal_radius=0.75,
    path_replan_interval=10,
    failure_penalty_steps=100,
    scenario_profile="S00_STATIC_CLEAR",
    base_candidate="ch3_v3_full_reference",
    **kwargs,
):
    """Signature-only declaration for current mission runtime introspection."""


UAVEnv.__init__.__signature__ = inspect.signature(_mission_init_signature)
del _mission_init_signature
