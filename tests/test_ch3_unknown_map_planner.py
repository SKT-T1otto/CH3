import numpy as np
import torch

from map.path_planner import OnlineUnknownMapTaskPlanner


def _planner():
    planner = OnlineUnknownMapTaskPlanner(
        space_size=(20, 20, 8),
        grid_size=(10, 10, 8),
        z_range=(0.5, 7.5),
        device="cpu",
        target_belief_transition_mode=
            "occupancy_constrained_diffusion_v1",
        target_belief_diffusion_rate=0.12,
    )
    planner.reset(None, [])
    return planner


def test_obstacle_truth_is_not_installed_and_scan_updates_shared_map():
    planner = _planner()
    assert planner.obstacles == []
    assert torch.allclose(
        planner.occupancy_probability,
        torch.full_like(planner.occupancy_probability, 0.5),
    )

    origins = np.asarray([[2.0, 2.0, 2.0]])
    directions = np.asarray([[1.0, 0.0, 0.0]])
    distances = np.asarray([[4.0]])
    hits = np.asarray([[True]])
    changed = planner.integrate_obstacle_scan(
        origins, directions, distances, hits, current_step=1
    )
    assert changed > 0
    assert planner.map_revision > 0
    hit_cell = planner._grid_index_from_point(
        torch.tensor([6.0, 2.0, 2.0])
    )
    free_cell = planner._grid_index_from_point(
        torch.tensor([3.0, 2.0, 2.0])
    )
    assert planner.occupancy_probability[hit_cell] > 0.5
    assert planner.occupancy_probability[free_cell] < 0.5


def test_online_astar_allows_unknown_cells_but_blocks_discovered_occupied_cells():
    planner = _planner()
    result_unknown = planner.grid_astar_path(
        [2.0, 2.0, 2.0], [8.0, 2.0, 2.0]
    )
    assert result_unknown["reachable"]
    assert result_unknown["unknown_fraction"] > 0.0
    assert result_unknown["travel_time"] > 0.0

    origins = np.asarray([[2.0, 2.0, 2.0]])
    directions = np.asarray([[1.0, 0.0, 0.0]])
    distances = np.asarray([[4.0]])
    hits = np.asarray([[True]])
    for step in range(3):
        planner.integrate_obstacle_scan(
            origins, directions, distances, hits, current_step=step
        )
    cell = planner._grid_index_from_point(
        torch.tensor([6.0, 2.0, 2.0])
    )
    assert planner.known_occupied_mask[cell]
    assert not planner.segment_is_free(
        [2.0, 2.0, 2.0], [8.0, 2.0, 2.0]
    )


def test_negative_target_evidence_is_time_valid_not_permanent():
    planner = _planner()
    cell = planner._grid_index_from_point(
        torch.tensor([10.0, 10.0, 4.0])
    )
    before = float(planner.belief_map[cell])
    planner.set_runtime_context(
        executor_pos=torch.tensor([18.0, 18.0, 4.0]),
        executor_wait_point=torch.tensor([15.0, 15.0, 4.0]),
        step=1,
    )
    planner.update_belief_negative(
        torch.tensor([[10.0, 10.0, 4.0]]),
        sensor_ranges=torch.tensor([3.0]),
    )
    suppressed = float(planner.belief_map[cell])
    assert 0.0 < suppressed < before
    for _ in range(8):
        planner.predict_belief_motion()
    recovered = float(planner.belief_map[cell])
    assert recovered > suppressed
    assert torch.isclose(
        planner.belief_map.sum(), torch.tensor(1.0), atol=1e-5
    )
