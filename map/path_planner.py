"""Deterministic obstacle-aware grid planning for Chapter 3."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from map.map_module import ProbabilisticTaskMapPlanner
from target_motion import segment_aabb_first_hit


class ObstacleAwareTaskMapPlanner(ProbabilisticTaskMapPlanner):
    def __init__(
        self, *args, planner_obstacle_clearance=0.40,
        target_belief_transition_mode="static",
        target_belief_diffusion_rate=0.0, **kwargs,
    ):
        self.planner_obstacle_clearance = float(planner_obstacle_clearance)
        self.target_belief_transition_mode = str(target_belief_transition_mode)
        self.target_belief_diffusion_rate = float(target_belief_diffusion_rate)
        self.grid_revision = 0
        self._geodesic_cache = {}
        self.obstacle_layout_hash = hashlib.sha256(b"none").hexdigest()
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.grid_revision += 1
        self._geodesic_cache = {}
        payload = [
            {"center": list(map(float, o["center"])), "size": list(map(float, o["size"]))}
            for o in self.obstacles
        ]
        self.obstacle_layout_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._xyz_centers_np = self.xyz_centers.detach().cpu().numpy()
        self._flat_xyz_centers_np = self.flat_xyz_centers.detach().cpu().numpy()
        self._valid_flats_np = np.flatnonzero(
            self.valid_mask.detach().cpu().numpy().reshape(-1)
        )
        self._space_size_np = self.space_size.detach().cpu().numpy()
        self._obstacle_boxes_np = [
            (
                np.asarray(obstacle["center"], dtype=np.float64)
                - np.asarray(obstacle["size"], dtype=np.float64) / 2.0,
                np.asarray(obstacle["center"], dtype=np.float64)
                + np.asarray(obstacle["size"], dtype=np.float64) / 2.0,
            )
            for obstacle in self.obstacles
        ]
        self.waypoint_unreachable_event_count = 0
        self.last_waypoint_failure_reason = None
        self.last_all_search_candidates_unreachable = False
        return result

    def _build_valid_mask(self):
        if not self.obstacles:
            return self._all_valid_mask.clone()
        points = self.flat_xyz_centers
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        for obstacle in self.obstacles:
            center = self._as_points(obstacle["center"])
            size = self._as_points(obstacle["size"]) + 2.0 * self.planner_obstacle_clearance
            lower, upper = center - size / 2.0, center + size / 2.0
            inside |= ((points >= lower) & (points <= upper)).all(dim=1)
        return (~inside).reshape(self.grid_size)

    def nearest_valid_grid_cell(self, point):
        candidates = self._connector_candidates(point)
        return None if not candidates else candidates[0][1]

    def endpoint_status(self, point, role="searcher"):
        """Describe whether a physical point is a legal planner endpoint.

        This query does not change claims, pheromone, belief, occupancy,
        reservations, waypoints, paths, or planner revision.  Normal
        connector-cache population is allowed.
        """

        current = self._as_points(point).reshape(3)
        point_valid = bool(self._point_is_valid(current))
        connectors = (
            self._connector_candidates(current, role=role)
            if point_valid
            else []
        )
        if not point_valid:
            failure_reason = "point_invalid"
        elif not connectors:
            failure_reason = "no_connector"
        else:
            failure_reason = None
        return {
            "point_valid": point_valid,
            "connector_count": int(len(connectors)),
            "reachable": bool(point_valid and connectors),
            "failure_reason": failure_reason,
        }

    def is_reachable_endpoint(self, point, role="searcher"):
        return bool(
            self.endpoint_status(point, role=role)["reachable"]
        )

    def _flatten(self, cell):
        _, ny, nz = self.grid_size
        return int(cell[0]) * ny * nz + int(cell[1]) * nz + int(cell[2])

    def _unflatten(self, flat):
        _, ny, nz = self.grid_size
        x, rest = divmod(int(flat), ny * nz)
        y, z = divmod(rest, nz)
        return (x, y, z)

    @staticmethod
    def _neighbor_offsets():
        return tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        )

    def _edge_time(self, left, right, role):
        delta = self.xyz_centers[right] - self.xyz_centers[left]
        horizontal = float(torch.norm(delta[:2]).item())
        vertical = abs(float(delta[2].item()))
        if str(role).lower().startswith("exec"):
            v_xy, v_z = 1.15, 0.60
        else:
            v_xy, v_z = 1.00, 0.55
        return horizontal / v_xy + 0.6 * vertical / v_z

    def _continuous_edge_time(self, left, right, role):
        left = self._as_points(left).reshape(3)
        right = self._as_points(right).reshape(3)
        delta = right - left
        horizontal = float(torch.norm(delta[:2]).item())
        vertical = abs(float(delta[2].item()))
        if str(role).lower().startswith("exec"):
            v_xy, v_z = 1.15, 0.60
        else:
            v_xy, v_z = 1.00, 0.55
        return horizontal / v_xy + 0.6 * vertical / v_z

    @staticmethod
    def _continuous_edge_time_np(left, right, role):
        delta = np.asarray(right, dtype=np.float64) - np.asarray(
            left, dtype=np.float64
        )
        horizontal = float(np.linalg.norm(delta[:2]))
        vertical = abs(float(delta[2]))
        if str(role).lower().startswith("exec"):
            v_xy, v_z = 1.15, 0.60
        else:
            v_xy, v_z = 1.00, 0.55
        return horizontal / v_xy + 0.6 * vertical / v_z

    def _point_is_valid(self, point, clearance=None):
        point = self._as_points(point).reshape(3)
        if not bool(torch.isfinite(point).all()):
            return False
        if bool(torch.any(point < 0.0)) or bool(torch.any(point > self.space_size)):
            return False
        clearance = (
            self.planner_obstacle_clearance
            if clearance is None else float(clearance)
        )
        for obstacle in self.obstacles:
            center = self._as_points(obstacle["center"]).reshape(3)
            size = self._as_points(obstacle["size"]).reshape(3)
            half = size / 2.0 + clearance
            if bool(torch.all(point >= center - half) and torch.all(point <= center + half)):
                return False
        return True

    def segment_is_free(self, start, end, clearance=None):
        """Return whether one continuous connector is bounded and collision-free."""

        clearance = (
            self.planner_obstacle_clearance
            if clearance is None else float(clearance)
        )
        start_np = self._as_points(start).reshape(3).detach().cpu().numpy()
        end_np = self._as_points(end).reshape(3).detach().cpu().numpy()
        return self._segment_is_free_np(start_np, end_np, clearance)

    def _segment_is_free_np(self, start, end, clearance):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if (
            not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or np.any(start < 0.0)
            or np.any(end < 0.0)
            or np.any(start > self._space_size_np)
            or np.any(end > self._space_size_np)
        ):
            return False
        for lower, upper in self._obstacle_boxes_np:
            expanded_lower = lower - clearance
            expanded_upper = upper + clearance
            if (
                np.all(start >= expanded_lower)
                and np.all(start <= expanded_upper)
            ) or (
                np.all(end >= expanded_lower)
                and np.all(end <= expanded_upper)
            ):
                return False
            hit = segment_aabb_first_hit(
                start, end, expanded_lower, expanded_upper
            )
            if hit is not None and -self.eps <= hit[0] <= 1.0 + self.eps:
                return False
        return True

    def _connector_candidates(
        self, point, role="searcher", *, per_component=True
    ):
        point = self._as_points(point).reshape(3)
        if not self._point_is_valid(point):
            return []
        point_np = point.detach().cpu().numpy().astype(np.float64, copy=False)
        cache_key = (
            "endpoint_connectors",
            self.obstacle_layout_hash,
            float(self.planner_obstacle_clearance),
            str(role),
            int(self.grid_revision),
            tuple(float(value) for value in point_np),
            bool(per_component),
        )
        cached = self._geodesic_cache.get(cache_key)
        if cached is not None:
            return [(float(cost), tuple(cell)) for cost, cell in cached]
        deltas = self._flat_xyz_centers_np - point_np
        labels = self._component_labels()
        components = set(labels.values())
        if len(components) == 1:
            squared = np.einsum("ij,ij->i", deltas, deltas)
            nearest_flat = int(np.argmin(squared))
            nearest_cell = self._unflatten(nearest_flat)
            if (
                bool(self.valid_mask[nearest_cell].item())
                and float(squared[nearest_flat]) <= self.eps ** 2
            ):
                candidates = [(0.0, nearest_cell)]
                self._geodesic_cache[cache_key] = tuple(candidates)
                return list(candidates)
        horizontal = np.linalg.norm(deltas[:, :2], axis=1)
        vertical = np.abs(deltas[:, 2])
        if str(role).lower().startswith("exec"):
            connector_costs = horizontal / 1.15 + 0.6 * vertical / 0.60
        else:
            connector_costs = horizontal / 1.00 + 0.6 * vertical / 0.55
        valid_costs = connector_costs[self._valid_flats_np]
        order = np.lexsort((self._valid_flats_np, valid_costs))
        component_count = len(components)
        best_by_component = {}
        all_candidates = []
        for index in order:
            flat = int(self._valid_flats_np[int(index)])
            cell = self._unflatten(flat)
            center_np = self._flat_xyz_centers_np[flat]
            if not self._segment_is_free_np(
                point_np, center_np, self.planner_obstacle_clearance
            ):
                continue
            cost = float(connector_costs[flat])
            item = (float(cost), cell)
            if not per_component:
                all_candidates.append(item)
                continue
            component = labels.get(cell)
            if component is None:
                continue
            current = best_by_component.get(component)
            candidate_key = (float(cost), flat)
            if current is None or candidate_key < current[0]:
                best_by_component[component] = (candidate_key, item)
                if len(best_by_component) == component_count:
                    break
        if per_component:
            candidates = [
                value[1] for _, value in sorted(
                    best_by_component.items(),
                    key=lambda pair: (
                        pair[1][1][0],
                        pair[0],
                        self._flatten(pair[1][1][1]),
                    ),
                )
            ]
        else:
            candidates = sorted(
                all_candidates,
                key=lambda item: (item[0], self._flatten(item[1])),
            )
        self._geodesic_cache[cache_key] = tuple(candidates)
        return list(candidates)

    def _edge_is_valid(self, left, right):
        if not self.obstacles:
            return True
        cache_key = (
            "edge",
            int(self.grid_revision),
            min(self._flatten(left), self._flatten(right)),
            max(self._flatten(left), self._flatten(right)),
        )
        if cache_key not in self._geodesic_cache:
            self._geodesic_cache[cache_key] = self._segment_is_free_np(
                self._xyz_centers_np[left],
                self._xyz_centers_np[right],
                self.planner_obstacle_clearance,
            )
        return self._geodesic_cache[cache_key]

    def _heuristic(self, cell, goal_cell, role):
        return self._continuous_edge_time(
            self.xyz_centers[cell], self.xyz_centers[goal_cell], role
        )

    def _core_cell_path(self, start_cell, goal_cell, role):
        cache_key = (
            "cell_core",
            self.obstacle_layout_hash,
            float(self.planner_obstacle_clearance),
            str(role),
            int(self.grid_revision),
            start_cell,
            goal_cell,
        )
        if cache_key in self._geodesic_cache:
            cached = self._geodesic_cache[cache_key]
            return {
                "reachable": cached["reachable"],
                "cells": list(cached["cells"]),
                "cost": float(cached["cost"]),
            }
        if start_cell == goal_cell:
            result = {"reachable": True, "cells": [start_cell], "cost": 0.0}
            self._geodesic_cache[cache_key] = result
            return dict(result, cells=list(result["cells"]))
        start_h = self._heuristic(start_cell, goal_cell, role)
        queue = [
            (start_h, 0.0, self._flatten(start_cell), start_cell)
        ]
        costs = {start_cell: 0.0}
        parents = {}
        nx, ny, nz = self.grid_size
        found = False
        while queue:
            _, cost, _, cell = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf) + 1e-12:
                continue
            if cell == goal_cell:
                found = True
                break
            for offset in self._neighbor_offsets():
                neighbor = tuple(cell[i] + offset[i] for i in range(3))
                if not (
                    0 <= neighbor[0] < nx
                    and 0 <= neighbor[1] < ny
                    and 0 <= neighbor[2] < nz
                ):
                    continue
                if not bool(self.valid_mask[neighbor].item()):
                    continue
                if not self._edge_is_valid(cell, neighbor):
                    continue
                new_cost = cost + self._edge_time(cell, neighbor, role)
                if new_cost + 1e-12 < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    parents[neighbor] = cell
                    heuristic = self._heuristic(neighbor, goal_cell, role)
                    heapq.heappush(
                        queue,
                        (
                            new_cost + heuristic,
                            new_cost,
                            self._flatten(neighbor),
                            neighbor,
                        ),
                    )
        if not found:
            result = {"reachable": False, "cells": [], "cost": math.inf}
        else:
            cells, current = [goal_cell], goal_cell
            while current != start_cell:
                current = parents[current]
                cells.append(current)
            cells.reverse()
            result = {
                "reachable": True,
                "cells": cells,
                "cost": float(costs[goal_cell]),
            }
        self._geodesic_cache[cache_key] = result
        return dict(result, cells=list(result["cells"]))

    def _component_labels(self):
        key = (
            "components",
            self.obstacle_layout_hash,
            float(self.planner_obstacle_clearance),
            int(self.grid_revision),
        )
        if key in self._geodesic_cache:
            return self._geodesic_cache[key]
        labels = {}
        nx, ny, nz = self.grid_size
        component = 0
        for flat in range(nx * ny * nz):
            start = self._unflatten(flat)
            if start in labels or not bool(self.valid_mask[start].item()):
                continue
            labels[start] = component
            queue = [start]
            cursor = 0
            while cursor < len(queue):
                cell = queue[cursor]
                cursor += 1
                for offset in self._neighbor_offsets():
                    neighbor = tuple(cell[i] + offset[i] for i in range(3))
                    if not (
                        0 <= neighbor[0] < nx
                        and 0 <= neighbor[1] < ny
                        and 0 <= neighbor[2] < nz
                    ):
                        continue
                    if neighbor in labels or not bool(self.valid_mask[neighbor].item()):
                        continue
                    if not self._edge_is_valid(cell, neighbor):
                        continue
                    labels[neighbor] = component
                    queue.append(neighbor)
            component += 1
        self._geodesic_cache[key] = labels
        return labels

    @staticmethod
    def _unreachable_result(reason):
        return {
            "reachable": False,
            "cells": [],
            "points": [],
            "cost": math.inf,
            "grid_cost": math.inf,
            "start_connector_cost": math.inf,
            "goal_connector_cost": math.inf,
            "resolved_start_cell": None,
            "resolved_goal_cell": None,
            "goal_projected": False,
            "resolved_component_id": None,
            "start_connector_valid": False,
            "goal_connector_valid": False,
            "exact_goal_reachable": False,
            "failure_reason": str(reason),
        }

    def grid_astar_path(self, start, goal, role="searcher"):
        start_t = self._as_points(start).reshape(3)
        goal_t = self._as_points(goal).reshape(3)
        if not self._point_is_valid(start_t):
            return self._unreachable_result("invalid_start")
        if not self._point_is_valid(goal_t):
            return self._unreachable_result("invalid_goal")
        if bool(torch.allclose(start_t, goal_t, atol=self.eps, rtol=0.0)):
            connectors = self._connector_candidates(start_t, role)
            if not connectors:
                return self._unreachable_result("no_start_connector")
            resolved_cell = connectors[0][1]
            resolved_component = self._component_labels().get(resolved_cell)
            return {
                "reachable": True,
                "cells": [],
                "points": [start_t.detach().cpu().tolist()],
                "cost": 0.0,
                "grid_cost": 0.0,
                "start_connector_cost": 0.0,
                "goal_connector_cost": 0.0,
                "resolved_start_cell": resolved_cell,
                "resolved_goal_cell": resolved_cell,
                "goal_projected": False,
                "resolved_component_id": int(resolved_component),
                "start_connector_valid": True,
                "goal_connector_valid": True,
                "exact_goal_reachable": True,
                "failure_reason": None,
            }
        starts = self._connector_candidates(start_t, role)
        goals = self._connector_candidates(goal_t, role)
        if not starts:
            return self._unreachable_result("no_start_connector")
        if not goals:
            return self._unreachable_result("no_goal_connector")
        labels = self._component_labels()
        start_by_component = {
            labels.get(cell): (cost, cell) for cost, cell in starts
        }
        goal_by_component = {
            labels.get(cell): (cost, cell) for cost, cell in goals
        }
        common = sorted(
            component
            for component in set(start_by_component) & set(goal_by_component)
            if component is not None
        )
        best = None
        # Endpoint connectors are deliberately not cached. Core cell paths are.
        for component in common:
            start_cost, start_cell = start_by_component[component]
            goal_cost, goal_cell = goal_by_component[component]
            core = self._core_cell_path(start_cell, goal_cell, role)
            if not core["reachable"]:
                continue
            total = float(start_cost + core["cost"] + goal_cost)
            route_key = (
                total,
                self._flatten(start_cell),
                self._flatten(goal_cell),
            )
            if best is None or route_key < best[0]:
                best = (
                    route_key, start_cost, goal_cost,
                    start_cell, goal_cell, core,
                )
        if best is None:
            return self._unreachable_result("disconnected_endpoint_components")
        _, start_cost, goal_cost, start_cell, goal_cell, core = best
        resolved_component = labels[start_cell]
        cells = list(core["cells"])
        point_tensors = [start_t.detach().clone()]
        for cell in cells:
            center = self.xyz_centers[cell]
            if float(torch.norm(center - point_tensors[-1]).item()) > self.eps:
                point_tensors.append(center.detach().clone())
        if float(torch.norm(goal_t - point_tensors[-1]).item()) > self.eps:
            point_tensors.append(goal_t.detach().clone())
        points = [point.detach().cpu().tolist() for point in point_tensors]
        if any(
            not self.segment_is_free(points[index], points[index + 1])
            for index in range(len(points) - 1)
        ):
            return self._unreachable_result("continuous_path_edge_blocked")
        return {
            "reachable": True,
            "cells": cells,
            "points": points,
            "cost": float(start_cost + core["cost"] + goal_cost),
            "grid_cost": float(core["cost"]),
            "start_connector_cost": float(start_cost),
            "goal_connector_cost": float(goal_cost),
            "resolved_start_cell": start_cell,
            "resolved_goal_cell": goal_cell,
            "goal_projected": False,
            "resolved_component_id": int(resolved_component),
            "start_connector_valid": True,
            "goal_connector_valid": True,
            "exact_goal_reachable": True,
            "failure_reason": None,
        }

    def sample_next_waypoint(
        self, agent_id, current_pos, reserved_positions=None, anchor=None
    ):
        """Select only reachable candidates and never revive an ``-inf`` score."""

        current = self._as_points(current_pos).reshape(3)
        if not self._point_is_valid(current) or not self._connector_candidates(current):
            raise RuntimeError("current_pos is not a legal reachable planner point")
        self._last_candidate_reachability_mask = None
        points, scores = self._candidate_points(
            int(agent_id),
            current,
            reserved_positions=reserved_positions,
            anchor=anchor,
        )
        points = self._as_points(points).reshape(-1, 3)
        scores = self._as_points(scores).reshape(-1)
        finite = torch.isfinite(scores)
        reachability = getattr(
            self, "_last_candidate_reachability_mask", None
        )
        if (
            reachability is not None
            and reachability.numel() == scores.numel()
        ):
            finite &= reachability.reshape(-1)
        else:
            finite &= torch.isfinite(
                self.estimate_travel_time(
                    current, points, role="searcher"
                )
            )
        if not bool(torch.any(finite)):
            self.last_waypoint_failure_reason = "all_candidates_unreachable"
            self.last_all_search_candidates_unreachable = True
            self.waypoint_unreachable_event_count += 1
            return current.clone()

        finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
        finite_scores = scores[finite_indices]
        self.last_waypoint_failure_reason = None
        self.last_all_search_candidates_unreachable = False
        if bool(torch.all(finite_scores <= self.eps)):
            finite_points = points[finite_indices]
            distances = torch.linalg.vector_norm(
                finite_points - current.unsqueeze(0), dim=1
            )
            local_index = int(torch.argmax(distances).item())
            chosen = finite_points[local_index]
        else:
            use_stochastic = (
                self.stochastic_topk > 1
                and self.stochastic_eps > 0.0
                and float(
                    torch.rand(
                        (), dtype=self.dtype, device=self.device
                    ).item()
                )
                < self.stochastic_eps
            )
            if use_stochastic:
                k = min(self.stochastic_topk, int(finite_scores.numel()))
                values, local_indices = torch.topk(finite_scores, k=k)
                values = torch.clamp(values, min=0.0)
                if float(values.sum().item()) <= self.eps:
                    local = int(torch.argmax(finite_scores).item())
                else:
                    probabilities = values / values.sum().clamp_min(self.eps)
                    pick = int(
                        torch.multinomial(probabilities, num_samples=1).item()
                    )
                    local = int(local_indices[pick].item())
            else:
                local = int(torch.argmax(finite_scores).item())
            chosen = points[int(finite_indices[local].item())]
        if not self._point_is_valid(chosen):
            raise RuntimeError("waypoint selection produced an illegal point")
        self.register_waypoint_claim(chosen)
        return chosen.clone()

    def grid_geodesic_cost(self, start, goal, role="searcher"):
        return self.grid_astar_path(start, goal, role=role)

    def path_to_subgoals(self, path_result, final_target=None):
        if not path_result.get("reachable"):
            return []
        points = [
            self._as_points(point).reshape(3).clone()
            for point in path_result["points"][1:]
        ]
        if final_target is not None and path_result.get("goal_connector_valid", False):
            final = self._as_points(final_target).reshape(3).clone()
            connector_start = (
                points[-1]
                if points
                else self._as_points(path_result["points"][0]).reshape(3)
            )
            if not self.segment_is_free(connector_start, final):
                return []
            if not points or float(torch.norm(points[-1] - final).item()) > self.eps:
                points.append(final)
        return points

    def _single_source_costs(self, start, role):
        start_cell = self.nearest_valid_grid_cell(start)
        if start_cell is None:
            return {}
        key = (
            self.obstacle_layout_hash, start_cell, str(role),
            int(self.grid_revision), "single_source",
        )
        if key in self._geodesic_cache:
            return self._geodesic_cache[key]
        nx, ny, nz = self.grid_size
        costs = {start_cell: 0.0}
        queue = [(0.0, self._flatten(start_cell), start_cell)]
        while queue:
            cost, _, cell = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf) + 1e-12:
                continue
            for offset in self._neighbor_offsets():
                neighbor = tuple(cell[i] + offset[i] for i in range(3))
                if not (
                    0 <= neighbor[0] < nx
                    and 0 <= neighbor[1] < ny
                    and 0 <= neighbor[2] < nz
                ):
                    continue
                if not bool(self.valid_mask[neighbor].item()):
                    continue
                if not self._edge_is_valid(cell, neighbor):
                    continue
                new_cost = cost + self._edge_time(cell, neighbor, role)
                if new_cost + 1e-12 < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    heapq.heappush(
                        queue, (new_cost, self._flatten(neighbor), neighbor)
                    )
        self._geodesic_cache[key] = costs
        return costs

    # CH3_M00_EQUIVALENT_OPTIMIZATION_V1
    def _planning_adjacency(self, role):
        """Build one immutable weighted adjacency snapshot per planning revision.

        The original Dijkstra implementation re-evaluated validity and edge
        weights for every source. Edge validity and weights are deterministic
        within one grid revision, so computing them once is exactly equivalent.
        """

        role_key = str(role)
        key = (
            "planning_adjacency_v1",
            self.obstacle_layout_hash,
            float(self.planner_obstacle_clearance),
            role_key,
            int(self.grid_revision),
        )
        cached = self._geodesic_cache.get(key)
        if cached is not None:
            return cached
        nx, ny, nz = self.grid_size
        offsets = self._neighbor_offsets()
        adjacency = {}
        for flat in self._valid_flats_np:
            cell = self._unflatten(int(flat))
            edges = []
            for offset in offsets:
                neighbor = tuple(cell[index] + offset[index] for index in range(3))
                if not (
                    0 <= neighbor[0] < nx
                    and 0 <= neighbor[1] < ny
                    and 0 <= neighbor[2] < nz
                ):
                    continue
                if not bool(self.valid_mask[neighbor].item()):
                    continue
                if not self._edge_is_valid(cell, neighbor):
                    continue
                edges.append((neighbor, float(self._edge_time(cell, neighbor, role))))
            adjacency[cell] = tuple(edges)
        self._geodesic_cache[key] = adjacency
        return adjacency

    def _single_source_cell_costs(self, start_cell, role):
        key = (
            "single_source_cell",
            self.obstacle_layout_hash,
            float(self.planner_obstacle_clearance),
            start_cell,
            str(role),
            int(self.grid_revision),
        )
        if key in self._geodesic_cache:
            return self._geodesic_cache[key]
        adjacency = self._planning_adjacency(role)
        if start_cell not in adjacency:
            return {}
        costs = {start_cell: 0.0}
        queue = [(0.0, self._flatten(start_cell), start_cell)]
        while queue:
            cost, _, cell = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf) + 1e-12:
                continue
            for neighbor, edge_cost in adjacency.get(cell, ()):
                new_cost = cost + edge_cost
                if new_cost + 1e-12 < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    heapq.heappush(
                        queue, (new_cost, self._flatten(neighbor), neighbor)
                    )
        self._geodesic_cache[key] = costs
        return costs

    def _endpoint_connectors_by_component(self, point, role, labels):
        result = {}
        for connector_cost, cell in self._connector_candidates(point, role):
            component = labels.get(cell)
            if component is None:
                continue
            result.setdefault(
                component, (float(connector_cost), tuple(cell))
            )
        return result

    def estimate_travel_time_matrix(self, starts, goals, role="searcher"):
        """Return exact pairwise endpoint travel times for many starts/goals.

        Dijkstra is run from the same start-side connector cells as the legacy
        scalar implementation. This preserves the original accumulation order
        while sharing one precomputed adjacency snapshot across all sources.
        Connector costs and component handling match ``estimate_travel_time``.
        """

        starts_tensor = self._as_points(starts).reshape(-1, 3)
        goals_tensor = self._as_points(goals).reshape(-1, 3)
        matrix = torch.full(
            (starts_tensor.shape[0], goals_tensor.shape[0]),
            math.inf,
            dtype=self.dtype,
            device=self.device,
        )
        if starts_tensor.numel() == 0 or goals_tensor.numel() == 0:
            return matrix
        labels = self._component_labels()
        start_connectors = [
            self._endpoint_connectors_by_component(point, role, labels)
            for point in starts_tensor
        ]
        goal_connectors = [
            self._endpoint_connectors_by_component(point, role, labels)
            for point in goals_tensor
        ]
        source_cells = {
            cell for mapping in start_connectors for _, cell in mapping.values()
        }
        source_costs = {
            cell: self._single_source_cell_costs(cell, role)
            for cell in sorted(source_cells, key=self._flatten)
        }
        values = []
        for start_mapping in start_connectors:
            row = []
            for goal_mapping in goal_connectors:
                best = math.inf
                common = sorted(set(start_mapping) & set(goal_mapping))
                for component in common:
                    start_cost, start_cell = start_mapping[component]
                    goal_cost, goal_cell = goal_mapping[component]
                    grid_cost = source_costs.get(start_cell, {}).get(
                        goal_cell, math.inf
                    )
                    best = min(best, start_cost + grid_cost + goal_cost)
                row.append(best)
            values.append(row)
        self.last_travel_time_matrix_source_count = int(len(source_cells))
        self.last_travel_time_matrix_start_count = int(starts_tensor.shape[0])
        self.last_travel_time_matrix_goal_count = int(goals_tensor.shape[0])
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    @staticmethod
    def _belief_support_signature(support, probabilities):
        support_cpu = support.detach().cpu().contiguous()
        probabilities_cpu = probabilities.detach().cpu().contiguous()
        return (
            tuple(int(value) for value in support_cpu.shape),
            support_cpu.numpy().tobytes(order="C"),
            tuple(int(value) for value in probabilities_cpu.shape),
            probabilities_cpu.numpy().tobytes(order="C"),
        )

    def _expected_response_cache_key(self, standby_point, support, probabilities):
        point = self._as_points(standby_point).reshape(3)
        return (
            "expected_response_cost_v1",
            int(self.grid_revision),
            self._belief_support_signature(support, probabilities),
            tuple(float(value) for value in point.detach().cpu().tolist()),
        )


    def estimate_travel_time(self, start, goals, role="searcher"):
        goals_tensor = self._as_points(goals)
        single = goals_tensor.ndim == 1
        labels = self._component_labels()
        starts = self._connector_candidates(start, role)
        start_by_component = {}
        for connector_cost, cell in starts:
            start_by_component.setdefault(
                labels.get(cell), (connector_cost, cell)
            )
        sources = {
            component: (
                connector_cost,
                self._single_source_cell_costs(cell, role),
            )
            for component, (connector_cost, cell) in start_by_component.items()
            if component is not None
        }
        values = []
        for goal in goals_tensor.reshape(-1, 3):
            best = math.inf
            for goal_cost, goal_cell in self._connector_candidates(goal, role):
                component = labels.get(goal_cell)
                if component not in sources:
                    continue
                start_cost, costs = sources[component]
                grid_cost = costs.get(goal_cell, math.inf)
                best = min(best, start_cost + grid_cost + goal_cost)
            values.append(best)
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        return tensor[0] if single else tensor

    @staticmethod
    def _normalize_reachable_cost(cost):
        finite = torch.isfinite(cost)
        normalized = torch.zeros_like(cost)
        if bool(torch.any(finite)):
            maximum = cost[finite].max().clamp_min(1e-9)
            normalized[finite] = cost[finite] / maximum
        return normalized, finite

    def score_search_candidates(self, agent_id, points, base_score, current_pos):
        """Score candidates without allowing unreachable infinities into scaling."""

        pts = self._as_points(points).reshape(-1, 3)
        base = self._as_points(base_score).reshape(-1)
        if pts.shape[0] != base.numel():
            raise ValueError("points/base_score size mismatch")
        base_norm = torch.clamp(base, min=0.0)
        base_norm = base_norm / base_norm.max().clamp_min(self.eps)
        score = self.pse_base_score_weight * base_norm
        if (
            pts.shape == self.flat_xyz_centers.shape
            and pts.data_ptr() == self.flat_xyz_centers.data_ptr()
        ):
            nearest_idx = torch.arange(
                pts.shape[0], dtype=torch.long, device=self.device
            )
        else:
            nearest_idx = torch.argmin(
                torch.cdist(pts, self.flat_xyz_centers), dim=1
            )
        if self.pse_use_gated_belief and self.belief_enabled:
            mixed_belief, gated_weight = self.gated_belief_distribution()
            belief_score = mixed_belief[nearest_idx]
        else:
            belief_score = self._normalized_flat_belief()[nearest_idx]
            gated_weight = self.pse_belief_weight
            self._reset_gated_belief_diagnostics()
        if self.belief_enabled:
            score = score + gated_weight * belief_score
        search_cost = self.estimate_travel_time(
            current_pos, pts, role="searcher"
        )
        search_norm, reachable = self._normalize_reachable_cost(search_cost)
        score = score - self.pse_search_cost_weight * search_norm
        if self.exec_cost_enabled and self.runtime_executor_pos is not None:
            exec_cost = self.estimate_travel_time(
                self.runtime_executor_pos, pts, role="executor"
            )
            exec_norm, exec_reachable = self._normalize_reachable_cost(exec_cost)
            reachable &= exec_reachable
            finite_exec = exec_cost[exec_reachable]
            self.last_exec_response_cost = (
                float(finite_exec.mean().item()) if finite_exec.numel() else math.inf
            )
            effective = float(
                getattr(
                    self,
                    "last_pse_exec_cost_weight_effective",
                    self.pse_exec_cost_weight,
                )
            )
            score = score - effective * exec_norm
        else:
            self.last_exec_response_cost = 0.0
        claims = self.claim_count.reshape(-1)[nearest_idx]
        if claims.numel():
            self.last_claim_overlap = float(
                torch.nan_to_num(claims.mean(), nan=0.0).item()
            )
        if hasattr(self, "flat_valid_mask"):
            reachable &= self.flat_valid_mask[nearest_idx]
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score[~reachable] = -torch.inf
        self._last_candidate_reachability_mask = reachable.detach().clone()
        self.last_all_search_candidates_unreachable = not bool(torch.any(reachable))
        finite_scores = score[reachable]
        self.last_search_score_mean = (
            float(finite_scores.mean().item()) if finite_scores.numel() else 0.0
        )
        return score

    def expected_response_cost(self, standby_point):
        support, probabilities = self.topk_belief_points(self.pse_standby_topk)
        if support.numel() == 0:
            raise RuntimeError("belief support is empty")
        cache_key = self._expected_response_cache_key(
            standby_point, support, probabilities
        )
        if cache_key in self._geodesic_cache:
            return float(self._geodesic_cache[cache_key])
        times = self.estimate_travel_time(standby_point, support, role="executor")
        unreachable = torch.isinf(times) & (probabilities > 0)
        if bool(torch.any(unreachable)):
            value = math.inf
        else:
            value = float(torch.sum(times * probabilities).item())
        self._geodesic_cache[cache_key] = float(value)
        return float(value)


    def plan_executor_standby(
        self, executor_pos, prev_standby=None,
        move_weight=None, hysteresis_weight=None,
    ):
        executor = self._as_points(executor_pos).reshape(3)
        fallback = (
            executor
            if prev_standby is None
            else self._as_points(prev_standby).reshape(3)
        )
        if not self.standby_enabled or self.flat_valid_points.numel() == 0:
            return fallback.clone()
        count = min(
            self.pse_standby_candidates,
            int(self.flat_valid_points.shape[0]),
        )
        indices = torch.floor(
            (torch.arange(count, dtype=self.dtype, device=self.device) + 0.5)
            * self.flat_valid_points.shape[0] / count
        ).long()
        candidates = self.flat_valid_points[indices]
        support, probabilities = self.topk_belief_points(self.pse_standby_topk)
        if support.numel() == 0:
            raise RuntimeError("belief support is empty")

        response_times = self.estimate_travel_time_matrix(
            candidates, support, role="executor"
        )
        responses = torch.sum(
            response_times * probabilities.reshape(1, -1), dim=1
        )
        unreachable = torch.any(
            torch.isinf(response_times)
            & (probabilities.reshape(1, -1) > 0),
            dim=1,
        )
        responses = responses.clone()
        responses[unreachable] = math.inf
        move_times = self.estimate_travel_time_matrix(
            executor.reshape(1, 3), candidates, role="executor"
        ).reshape(-1)

        # Populate the scalar response cache so the v2 acceptance check can
        # reuse an already evaluated candidate without another Dijkstra pass.
        for candidate, response in zip(candidates, responses):
            cache_key = self._expected_response_cache_key(
                candidate, support, probabilities
            )
            self._geodesic_cache[cache_key] = float(response.item())

        best = None
        move_weight = (
            self.pse_standby_move_weight
            if move_weight is None else float(move_weight)
        )
        hysteresis_weight = (
            self.pse_standby_hysteresis_weight
            if hysteresis_weight is None else float(hysteresis_weight)
        )
        for index, candidate in enumerate(candidates):
            response = float(responses[index].item())
            move = float(move_times[index].item())
            if not math.isfinite(response) or not math.isfinite(move):
                continue
            hysteresis = (
                0.0
                if prev_standby is None
                else float(torch.norm(candidate - fallback).item())
            )
            score = response + move_weight * move + hysteresis_weight * hysteresis
            item = (score, index, candidate)
            if best is None or item[:2] < best[:2]:
                best = item
        if best is None:
            self.last_standby_failure_reason = "all_candidates_unreachable"
            return fallback.clone()
        self.last_standby_failure_reason = None
        self.last_executor_standby = best[2].clone()
        self.last_exec_response_cost = float(best[0])
        return best[2].clone()


    def predict_belief_motion(self):
        if self.target_belief_transition_mode == "static":
            return self.belief_map
        if self.target_belief_transition_mode != "isotropic_diffusion_v1":
            raise ValueError(
                f"unknown belief transition={self.target_belief_transition_mode!r}"
            )
        alpha = min(1.0, max(0.0, self.target_belief_diffusion_rate))
        kernel = torch.zeros((3, 3, 3), dtype=self.dtype, device=self.device)
        kernel[1, 1, 1] = 1.0
        for offset in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            kernel[offset] = 1.0
        kernel /= kernel.sum()
        convolved = F.conv3d(
            self.belief_map[None, None], kernel[None, None], padding=1
        )[0, 0]
        belief = ((1.0 - alpha) * self.belief_map + alpha * convolved)
        belief = torch.clamp(belief, min=0.0) * self.valid_mask_float
        self.belief_map = belief / belief.sum().clamp_min(self.eps)
        self._sync_belief_diagnostics()
        return self.belief_map


# CH3_STAGE2_MERGED_UNKNOWN_PLANNER
_LOG2 = math.log(2.0)


class OnlineUnknownMapTaskPlanner(ObstacleAwareTaskMapPlanner):
    """Risk-aware online A* over a shared probabilistic occupancy map."""

    def __init__(
        self,
        *args,
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
        **kwargs,
    ):
        self.occupancy_free_logodds = float(occupancy_free_logodds)
        self.occupancy_occupied_logodds = float(occupancy_occupied_logodds)
        self.occupancy_logodds_clip = float(abs(occupancy_logodds_clip))
        self.occupancy_free_threshold = float(occupancy_free_threshold)
        self.occupancy_occupied_threshold = float(occupancy_occupied_threshold)
        if not 0.0 < self.occupancy_free_threshold < 0.5:
            raise ValueError("occupancy_free_threshold must be in (0, 0.5)")
        if not 0.5 < self.occupancy_occupied_threshold < 1.0:
            raise ValueError("occupancy_occupied_threshold must be in (0.5, 1)")
        if self.occupancy_free_threshold >= self.occupancy_occupied_threshold:
            raise ValueError("occupancy thresholds are not ordered")
        self.occupancy_unknown_cost_weight = float(
            max(0.0, occupancy_unknown_cost_weight)
        )
        self.occupancy_risk_cost_weight = float(
            max(0.0, occupancy_risk_cost_weight)
        )
        self.occupancy_replan_probability_delta = float(
            max(0.0, occupancy_replan_probability_delta)
        )
        self.target_negative_observation_strength = float(
            np.clip(target_negative_observation_strength, 0.0, 1.0)
        )
        self.target_negative_likelihood_floor = float(
            np.clip(target_negative_likelihood_floor, 1e-6, 1.0)
        )
        self.target_revisit_half_life_steps = float(
            max(target_revisit_half_life_steps, 1e-6)
        )
        self.target_recency_penalty_weight = float(
            max(0.0, target_recency_penalty_weight)
        )
        self.obstacle_information_gain_weight = float(
            max(0.0, obstacle_information_gain_weight)
        )
        self.reservation_decay = float(np.clip(reservation_decay, 0.0, 1.0))
        self.online_map_enabled = True
        self.map_revision = 0
        self.map_update_count = 0
        self.changed_cell_count_total = 0
        self.last_map_changed_cell_count = 0
        self.last_planning_changed_cell_count = 0
        self.planning_changed_cell_count_total = 0
        self.last_astar_expanded_nodes = 0
        self.last_path_unknown_fraction = 0.0
        self.last_path_risk = 0.0
        self._runtime_step_for_mapping = 0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Shared obstacle map
    # ------------------------------------------------------------------

    def reset(self, agent_positions=None, obstacles=None):
        """Reset with an empty planner map, regardless of ground truth.

        ``obstacles`` is intentionally ignored.  The environment may pass its
        physical obstacle list through inherited reset code, but this planner
        must learn obstacles only through local scan updates.
        """

        del obstacles
        result = super().reset(agent_positions, [])
        shape = tuple(int(value) for value in self.grid_size)
        self.occupancy_logodds = torch.zeros(
            shape, dtype=self.dtype, device=self.device
        )
        self.occupancy_probability = torch.full(
            shape, 0.5, dtype=self.dtype, device=self.device
        )
        self.occupancy_last_observed_step = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self.occupancy_observation_count = torch.zeros(
            shape, dtype=torch.int32, device=self.device
        )
        self.target_last_observed_step = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self.target_observation_confidence = torch.zeros(
            shape, dtype=self.dtype, device=self.device
        )
        self.map_revision = 0
        self.map_update_count = 0
        self.changed_cell_count_total = 0
        self.last_map_changed_cell_count = 0
        self.last_planning_changed_cell_count = 0
        self.planning_changed_cell_count_total = 0
        self.last_astar_expanded_nodes = 0
        self.last_path_unknown_fraction = 0.0
        self.last_path_risk = 0.0
        self._runtime_step_for_mapping = 0
        self._refresh_online_masks(force=True, increment_revision=False)
        self.reset_belief_map()
        return result

    def set_mapping_step(self, step: int) -> None:
        self._runtime_step_for_mapping = int(step)

    @property
    def sensor_directions(self) -> torch.Tensor:
        """Return deterministic 26-neighbour ray directions."""

        offsets = torch.as_tensor(
            self._neighbor_offsets(), dtype=self.dtype, device=self.device
        )
        return offsets / torch.linalg.vector_norm(
            offsets, dim=1, keepdim=True
        ).clamp_min(self.eps)

    def _refresh_online_masks(
        self, *, force=False, increment_revision=True
    ) -> None:
        old_valid = getattr(self, "valid_mask", None)
        self.occupancy_probability = torch.sigmoid(self.occupancy_logodds)
        self.known_free_mask = (
            self.occupancy_probability <= self.occupancy_free_threshold
        )
        self.known_occupied_mask = (
            self.occupancy_probability >= self.occupancy_occupied_threshold
        )
        self.unknown_mask = ~(self.known_free_mask | self.known_occupied_mask)
        self.valid_mask = ~self.known_occupied_mask
        self.flat_valid_mask = self.valid_mask.reshape(-1)
        self.valid_mask_float = self.valid_mask.to(self.dtype)
        self.flat_valid_mask_float = self.flat_valid_mask.to(self.dtype)
        self.flat_valid_points = self.flat_xyz_centers[self.flat_valid_mask]

        classification_changed = (
            old_valid is None
            or old_valid.shape != self.valid_mask.shape
            or not torch.equal(old_valid, self.valid_mask)
        )
        if force or classification_changed:
            self.pheromone.mul_(self.valid_mask_float)
        if hasattr(self, "belief_map"):
            self.belief_map = (
                torch.clamp(self.belief_map, min=0.0) * self.valid_mask_float
            )
            total = self.belief_map.sum()
            if float(total.item()) <= self.eps:
                fallback = self.valid_mask_float
                self.belief_map = fallback / fallback.sum().clamp_min(1.0)
            else:
                self.belief_map /= total

        if increment_revision:
            self.map_revision += 1
            self.grid_revision += 1
        self._geodesic_cache = {}
        signature = {
            "revision": int(self.map_revision),
            "occupied": torch.nonzero(
                self.known_occupied_mask, as_tuple=False
            ).detach().cpu().tolist(),
            "free": torch.nonzero(
                self.known_free_mask, as_tuple=False
            ).detach().cpu().tolist(),
        }
        self.obstacle_layout_hash = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._valid_flats_np = np.flatnonzero(
            self.valid_mask.detach().cpu().numpy().reshape(-1)
        )
        self._space_size_np = self.space_size.detach().cpu().numpy()
        # Exact AABBs are deliberately absent from an online unknown map.
        self._obstacle_boxes_np = []

    def mark_all_free(self, current_step=0) -> None:
        """Install a known-clear map for a diagnostic clear-space baseline."""

        self.occupancy_logodds.fill_(-self.occupancy_logodds_clip)
        self.occupancy_last_observed_step.fill_(int(current_step))
        self.occupancy_observation_count.add_(1)
        self.last_map_changed_cell_count = int(self.occupancy_logodds.numel())
        self.changed_cell_count_total += self.last_map_changed_cell_count
        self.map_update_count += 1
        self._refresh_online_masks(force=True)

    def _cell_from_point_np(self, point: np.ndarray) -> tuple[int, int, int]:
        point = np.asarray(point, dtype=np.float64)
        low_z, _ = self.z_range
        x = int(np.clip(np.floor(point[0] / max(self.cell_dx, self.eps)), 0, self.grid_size[0] - 1))
        y = int(np.clip(np.floor(point[1] / max(self.cell_dy, self.eps)), 0, self.grid_size[1] - 1))
        z_rel = (point[2] - low_z) / max(self.cell_dz, self.eps)
        z = int(np.clip(np.rint(z_rel), 0, self.grid_size[2] - 1))
        return x, y, z

    def integrate_obstacle_scan(
        self,
        origins,
        directions,
        distances,
        hit_mask,
        *,
        current_step: int,
    ) -> int:
        """Fuse deterministic ray scans into the shared log-odds map.

        Free cells are the cells traversed before a return.  The terminal cell
        of a hit ray receives occupied evidence.  Multiple agents update the
        same shared map, so a newly observed obstacle immediately becomes
        available to every planner role.
        """

        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        distances = np.asarray(distances, dtype=np.float64)
        hit_mask = np.asarray(hit_mask, dtype=bool)
        if distances.shape != (origins.shape[0], directions.shape[0]):
            raise ValueError("distances shape does not match origins/directions")
        if hit_mask.shape != distances.shape:
            raise ValueError("hit_mask shape does not match distances")

        free_counts: dict[tuple[int, int, int], int] = {}
        occupied_counts: dict[tuple[int, int, int], int] = {}
        sample_step = max(
            0.15,
            0.45 * min(
                self.cell_dx,
                self.cell_dy,
                self.cell_dz if self.cell_dz > self.eps else min(self.cell_dx, self.cell_dy),
            ),
        )

        for origin_index, origin in enumerate(origins):
            for ray_index, direction in enumerate(directions):
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-12:
                    continue
                unit = direction / direction_norm
                distance = max(0.0, float(distances[origin_index, ray_index]))
                hit = bool(hit_mask[origin_index, ray_index])
                free_limit = max(
                    0.0,
                    distance - (0.55 * sample_step if hit else 0.0),
                )
                if free_limit > 0.0:
                    samples = max(1, int(math.ceil(free_limit / sample_step)))
                    visited = set()
                    for value in np.linspace(0.0, free_limit, samples + 1):
                        point = origin + unit * float(value)
                        if np.any(point < 0.0) or np.any(point > self._space_size_np):
                            continue
                        visited.add(self._cell_from_point_np(point))
                    for cell in visited:
                        free_counts[cell] = free_counts.get(cell, 0) + 1
                if hit:
                    endpoint = origin + unit * distance
                    if np.all(endpoint >= 0.0) and np.all(endpoint <= self._space_size_np):
                        cell = self._cell_from_point_np(endpoint)
                        occupied_counts[cell] = occupied_counts.get(cell, 0) + 1

        before = self.occupancy_logodds.clone()
        before_probability = torch.sigmoid(before)
        before_free = before_probability <= self.occupancy_free_threshold
        before_occupied = (
            before_probability >= self.occupancy_occupied_threshold
        )
        for cell, count in free_counts.items():
            self.occupancy_logodds[cell] += (
                self.occupancy_free_logodds * min(int(count), 3)
            )
            self.occupancy_last_observed_step[cell] = int(current_step)
            self.occupancy_observation_count[cell] += int(count)
        for cell, count in occupied_counts.items():
            self.occupancy_logodds[cell] += (
                self.occupancy_occupied_logodds * min(int(count), 3)
            )
            self.occupancy_last_observed_step[cell] = int(current_step)
            self.occupancy_observation_count[cell] += int(count)

        self.occupancy_logodds.clamp_(
            -self.occupancy_logodds_clip, self.occupancy_logodds_clip
        )
        changed = torch.abs(self.occupancy_logodds - before) > 1e-9
        changed_count = int(torch.count_nonzero(changed).item())
        after_probability = torch.sigmoid(self.occupancy_logodds)
        after_free = after_probability <= self.occupancy_free_threshold
        after_occupied = (
            after_probability >= self.occupancy_occupied_threshold
        )
        classification_changed = (
            (before_free != after_free)
            | (before_occupied != after_occupied)
        )
        material_probability_change = (
            torch.abs(after_probability - before_probability)
            >= self.occupancy_replan_probability_delta
        )
        planning_changed = classification_changed | material_probability_change
        planning_changed_count = int(
            torch.count_nonzero(planning_changed).item()
        )
        self.last_map_changed_cell_count = changed_count
        self.last_planning_changed_cell_count = planning_changed_count
        if changed_count:
            self.map_update_count += 1
            self.changed_cell_count_total += changed_count
            self.planning_changed_cell_count_total += planning_changed_count
            self._refresh_online_masks(force=True)
        return planning_changed_count

    # ------------------------------------------------------------------
    # Online collision semantics and A*
    # ------------------------------------------------------------------

    def _point_is_valid(self, point, clearance=None):
        del clearance
        point = self._as_points(point).reshape(3)
        if not bool(torch.isfinite(point).all()):
            return False
        if bool(torch.any(point < 0.0)) or bool(torch.any(point > self.space_size)):
            return False
        cell = self._grid_index_from_point(point)
        return not bool(self.known_occupied_mask[cell].item())

    def _segment_is_free_np(self, start, end, clearance):
        del clearance
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if (
            not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or np.any(start < 0.0)
            or np.any(end < 0.0)
            or np.any(start > self._space_size_np)
            or np.any(end > self._space_size_np)
        ):
            return False
        length = float(np.linalg.norm(end - start))
        sample_step = max(
            0.15,
            0.40 * min(
                self.cell_dx,
                self.cell_dy,
                self.cell_dz if self.cell_dz > self.eps else min(self.cell_dx, self.cell_dy),
            ),
        )
        count = max(1, int(math.ceil(length / sample_step)))
        for ratio in np.linspace(0.0, 1.0, count + 1):
            point = start + ratio * (end - start)
            if bool(self.known_occupied_mask[self._cell_from_point_np(point)].item()):
                return False
        return True

    def _edge_is_valid(self, left, right):
        cache_key = (
            "online_edge",
            int(self.map_revision),
            min(self._flatten(left), self._flatten(right)),
            max(self._flatten(left), self._flatten(right)),
        )
        if cache_key not in self._geodesic_cache:
            self._geodesic_cache[cache_key] = self._segment_is_free_np(
                self._xyz_centers_np[left],
                self._xyz_centers_np[right],
                0.0,
            )
        return bool(self._geodesic_cache[cache_key])

    def _physical_edge_time(self, left, right, role):
        return ObstacleAwareTaskMapPlanner._edge_time(self, left, right, role)

    def _edge_time(self, left, right, role):
        physical = self._physical_edge_time(left, right, role)
        p_left = float(self.occupancy_probability[left].item())
        p_right = float(self.occupancy_probability[right].item())
        probability = 0.5 * (p_left + p_right)
        entropy = 0.0
        if 0.0 < probability < 1.0:
            entropy = -(
                probability * math.log(probability)
                + (1.0 - probability) * math.log(1.0 - probability)
            ) / _LOG2
        return (
            physical
            + self.occupancy_unknown_cost_weight * entropy
            + self.occupancy_risk_cost_weight * probability * physical
        )

    def _core_cell_path(self, start_cell, goal_cell, role):
        cache_key = (
            "online_cell_core",
            str(role),
            int(self.map_revision),
            start_cell,
            goal_cell,
        )
        cached = self._geodesic_cache.get(cache_key)
        if cached is not None:
            self.last_astar_expanded_nodes = int(cached["expanded_nodes"])
            return {
                "reachable": bool(cached["reachable"]),
                "cells": list(cached["cells"]),
                "cost": float(cached["cost"]),
                "expanded_nodes": int(cached["expanded_nodes"]),
            }
        if start_cell == goal_cell:
            result = {
                "reachable": True,
                "cells": [start_cell],
                "cost": 0.0,
                "expanded_nodes": 1,
            }
            self._geodesic_cache[cache_key] = result
            self.last_astar_expanded_nodes = 1
            return dict(result, cells=list(result["cells"]))

        queue = [
            (
                self._heuristic(start_cell, goal_cell, role),
                0.0,
                self._flatten(start_cell),
                start_cell,
            )
        ]
        costs = {start_cell: 0.0}
        parents = {}
        nx, ny, nz = self.grid_size
        expanded = 0
        found = False
        while queue:
            _, cost, _, cell = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf) + 1e-12:
                continue
            expanded += 1
            if cell == goal_cell:
                found = True
                break
            for offset in self._neighbor_offsets():
                neighbor = tuple(cell[index] + offset[index] for index in range(3))
                if not (
                    0 <= neighbor[0] < nx
                    and 0 <= neighbor[1] < ny
                    and 0 <= neighbor[2] < nz
                ):
                    continue
                if not bool(self.valid_mask[neighbor].item()):
                    continue
                if not self._edge_is_valid(cell, neighbor):
                    continue
                new_cost = cost + self._edge_time(cell, neighbor, role)
                if new_cost + 1e-12 < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    parents[neighbor] = cell
                    heapq.heappush(
                        queue,
                        (
                            new_cost + self._heuristic(neighbor, goal_cell, role),
                            new_cost,
                            self._flatten(neighbor),
                            neighbor,
                        ),
                    )

        if not found:
            result = {
                "reachable": False,
                "cells": [],
                "cost": math.inf,
                "expanded_nodes": expanded,
            }
        else:
            cells = [goal_cell]
            current = goal_cell
            while current != start_cell:
                current = parents[current]
                cells.append(current)
            cells.reverse()
            result = {
                "reachable": True,
                "cells": cells,
                "cost": float(costs[goal_cell]),
                "expanded_nodes": expanded,
            }
        self._geodesic_cache[cache_key] = result
        self.last_astar_expanded_nodes = expanded
        return dict(result, cells=list(result["cells"]))

    def grid_astar_path(self, start, goal, role="searcher"):
        result = super().grid_astar_path(start, goal, role=role)
        result["planning_cost"] = float(result["cost"])
        result["map_revision"] = int(self.map_revision)
        result["expanded_nodes"] = int(self.last_astar_expanded_nodes)
        if not result.get("reachable"):
            result.update(
                travel_time=math.inf,
                physical_grid_cost=math.inf,
                path_risk=math.inf,
                unknown_fraction=1.0,
            )
            return result

        cells = list(result.get("cells", []))
        physical_grid = 0.0
        for left, right in zip(cells, cells[1:]):
            physical_grid += self._physical_edge_time(left, right, role)
        travel_time = (
            float(result["start_connector_cost"])
            + physical_grid
            + float(result["goal_connector_cost"])
        )
        if cells:
            probabilities = torch.stack(
                [self.occupancy_probability[cell] for cell in cells]
            ).to(dtype=self.dtype, device=self.device)
            path_risk = float(probabilities.mean().item())
            unknown_fraction = float(
                torch.stack(
                    [self.unknown_mask[cell].to(self.dtype) for cell in cells]
                ).mean().item()
            )
        else:
            path_risk = 0.0
            unknown_fraction = 0.0
        self.last_path_risk = path_risk
        self.last_path_unknown_fraction = unknown_fraction
        result.update(
            travel_time=float(travel_time),
            physical_grid_cost=float(physical_grid),
            path_risk=path_risk,
            unknown_fraction=unknown_fraction,
        )
        return result

    def known_obstacle_aabbs(self) -> list[dict]:
        """Approximate discovered obstacles as occupied grid-cell AABBs."""

        occupied = torch.nonzero(self.known_occupied_mask, as_tuple=False)
        if occupied.numel() == 0:
            return []
        z_size = self.cell_dz if self.cell_dz > self.eps else 0.5
        size = [float(self.cell_dx), float(self.cell_dy), float(z_size)]
        return [
            {
                "center": self.xyz_centers[tuple(index.tolist())]
                .detach().cpu().numpy().astype(np.float64),
                "size": np.asarray(size, dtype=np.float64),
            }
            for index in occupied
        ]

    # ------------------------------------------------------------------
    # Dynamic moving-target belief
    # ------------------------------------------------------------------

    def reset_belief_map(self):
        if not hasattr(self, "occupancy_probability"):
            return super().reset_belief_map()
        free_likelihood = torch.clamp(
            1.0 - self.occupancy_probability, min=0.02
        ) * self.valid_mask_float
        total = free_likelihood.sum()
        if float(total.item()) <= self.eps:
            free_likelihood = self.valid_mask_float
            total = free_likelihood.sum()
        self.belief_map = free_likelihood / total.clamp_min(1.0)
        self._sync_belief_diagnostics()
        return self.belief_map

    def predict_belief_motion(self):
        mode = self.target_belief_transition_mode
        if mode == "static":
            return self.belief_map
        if mode not in {
            "isotropic_diffusion_v1",
            "occupancy_constrained_diffusion_v1",
        }:
            raise ValueError(f"unknown belief transition={mode!r}")

        alpha = float(np.clip(self.target_belief_diffusion_rate, 0.0, 1.0))
        kernel = torch.zeros((3, 3, 3), dtype=self.dtype, device=self.device)
        kernel[1, 1, 1] = 1.0
        for offset in (
            (0, 1, 1), (2, 1, 1), (1, 0, 1),
            (1, 2, 1), (1, 1, 0), (1, 1, 2),
        ):
            kernel[offset] = 1.0
        kernel /= kernel.sum()
        propagated = F.conv3d(
            self.belief_map[None, None],
            kernel[None, None],
            padding=1,
        )[0, 0]
        belief = (1.0 - alpha) * self.belief_map + alpha * propagated
        if mode == "occupancy_constrained_diffusion_v1":
            belief *= torch.clamp(
                1.0 - self.occupancy_probability, min=0.02
            )
        belief = torch.clamp(belief, min=0.0) * self.valid_mask_float
        self.belief_map = belief / belief.sum().clamp_min(self.eps)

        # Claims are temporary task reservations, not permanent exclusions.
        self.claim_count.mul_(self.reservation_decay)
        self._sync_belief_diagnostics()
        return self.belief_map

    def update_belief_negative(self, search_positions, sensor_ranges=None):
        if not self.belief_enabled:
            self._sync_belief_diagnostics()
            return self.belief_map

        positions = self._as_points(search_positions).reshape(-1, 3)
        if positions.numel() == 0:
            return self.belief_map
        if sensor_ranges is None:
            ranges = torch.full(
                (positions.shape[0],),
                1.2 * min(self.cell_dx, self.cell_dy),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            ranges = self._as_points(sensor_ranges).reshape(-1)
            if ranges.numel() == 1:
                ranges = ranges.repeat(positions.shape[0])
            elif ranges.numel() < positions.shape[0]:
                ranges = torch.cat(
                    [
                        ranges,
                        ranges[-1:].repeat(positions.shape[0] - ranges.numel()),
                    ]
                )

        flat_belief = self.belief_map.reshape(-1)
        flat_last = self.target_last_observed_step.reshape(-1)
        flat_confidence = self.target_observation_confidence.reshape(-1)
        for index, position in enumerate(positions):
            radius = float(max(ranges[index].item(), self.eps))
            distance = torch.linalg.vector_norm(
                self.flat_xyz_centers - position.unsqueeze(0), dim=1
            )
            visible = (distance <= radius) & self.flat_valid_mask
            if not torch.any(visible):
                continue
            radial = torch.exp(
                -0.5 * (distance[visible] / max(radius * 0.65, self.eps)) ** 2
            )
            confidence = torch.clamp(
                self.pse_belief_detect_prob * radial, 0.0, 1.0
            )
            likelihood = torch.clamp(
                1.0
                - self.target_negative_observation_strength * confidence,
                min=self.target_negative_likelihood_floor,
                max=1.0,
            )
            flat_belief[visible] *= likelihood
            visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
            flat_last[visible_indices] = int(self.runtime_step)
            flat_confidence[visible_indices] = torch.maximum(
                flat_confidence[visible_indices], confidence
            )

        self.belief_map = flat_belief.reshape(self.grid_size)
        return self.normalize_belief()

    def score_search_candidates(self, agent_id, points, base_score, current_pos):
        score = super().score_search_candidates(
            agent_id, points, base_score, current_pos
        )
        points_t = self._as_points(points).reshape(-1, 3)
        nearest_idx = torch.argmin(
            torch.cdist(points_t, self.flat_xyz_centers), dim=1
        )
        probability = self.occupancy_probability.reshape(-1)[nearest_idx]
        entropy = -(
            probability.clamp_min(self.eps)
            * torch.log(probability.clamp_min(self.eps))
            + (1.0 - probability).clamp_min(self.eps)
            * torch.log((1.0 - probability).clamp_min(self.eps))
        ) / _LOG2

        last = self.target_last_observed_step.reshape(-1)[nearest_idx]
        confidence = self.target_observation_confidence.reshape(-1)[nearest_idx]
        age = torch.clamp(
            torch.as_tensor(
                float(self.runtime_step),
                dtype=self.dtype,
                device=self.device,
            )
            - last.to(self.dtype),
            min=0.0,
        )
        never_seen = last < 0
        recency = confidence * torch.exp(
            -_LOG2 * age / self.target_revisit_half_life_steps
        )
        recency[never_seen] = 0.0

        finite = torch.isfinite(score)
        score[finite] += (
            self.obstacle_information_gain_weight * entropy[finite]
            - self.target_recency_penalty_weight * recency[finite]
        )
        finite_values = score[finite]
        self.last_search_score_mean = (
            float(finite_values.mean().item()) if finite_values.numel() else 0.0
        )
        return score

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def map_statistics(self) -> dict:
        probability = self.occupancy_probability
        entropy = -(
            probability.clamp_min(self.eps)
            * torch.log(probability.clamp_min(self.eps))
            + (1.0 - probability).clamp_min(self.eps)
            * torch.log((1.0 - probability).clamp_min(self.eps))
        ) / _LOG2
        total = float(probability.numel())
        observed = self.known_free_mask | self.known_occupied_mask
        return {
            "map_revision": int(self.map_revision),
            "map_update_count": int(self.map_update_count),
            "map_changed_cell_count_total": int(self.changed_cell_count_total),
            "map_planning_changed_cell_count_total": int(
                self.planning_changed_cell_count_total
            ),
            "last_planning_changed_cell_count": int(
                self.last_planning_changed_cell_count
            ),
            "map_known_fraction": float(observed.sum().item() / total),
            "map_unknown_fraction": float(self.unknown_mask.sum().item() / total),
            "map_known_free_fraction": float(self.known_free_mask.sum().item() / total),
            "map_known_occupied_fraction": float(
                self.known_occupied_mask.sum().item() / total
            ),
            "map_occupancy_entropy": float(entropy.mean().item()),
            "last_astar_expanded_nodes": int(self.last_astar_expanded_nodes),
            "last_path_unknown_fraction": float(self.last_path_unknown_fraction),
            "last_path_risk": float(self.last_path_risk),
        }
