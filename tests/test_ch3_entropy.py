import math
import torch
from map.map_module import ProbabilisticTaskMapPlanner
from train import _build_train_env, get_ch3_method_config


def _planner(obstacles=None):
    planner = ProbabilisticTaskMapPlanner(space_size=(20.0,20.0,8.0), grid_size=(6,5,4), z_range=(0.5,7.5), device="cpu")
    planner.reset(None, obstacles or [])
    return planner


def test_uniform_belief_has_unit_normalized_entropy_and_unit_mass():
    planner = _planner()
    assert torch.isclose(planner.belief_map.sum(), torch.tensor(1.0), atol=1e-6)
    assert math.isclose(float(planner.belief_entropy_normalized().item()), 1.0, abs_tol=1e-6)
    assert planner.valid_belief_cell_count() == 6*5*4


def test_single_cell_belief_has_zero_normalized_entropy():
    planner = _planner(); planner.belief_map.zero_()
    first_valid = int(torch.nonzero(planner.flat_valid_mask, as_tuple=False)[0].item())
    planner.belief_map.reshape(-1)[first_valid] = 1.0; planner.normalize_belief()
    assert float(planner.belief_entropy_normalized().item()) < 1e-7
    assert math.isclose(float(planner.belief_peak_probability().item()), 1.0, abs_tol=1e-7)


def test_entropy_uses_only_non_obstacle_cells():
    planner = _planner([{"center": torch.tensor([10.0,10.0,4.0]), "size": torch.tensor([8.0,8.0,4.0])}])
    expected = int(torch.count_nonzero(planner.valid_mask).item())
    assert expected < planner.belief_map.numel()
    assert planner.valid_belief_cell_count() == expected
    assert planner.last_valid_belief_cell_count == expected
    assert math.isclose(float(planner.belief_entropy_normalized().item()), 1.0, abs_tol=1e-6)


def test_no_belief_method_keeps_uniform_distribution_for_whole_episode_prefix():
    env, _ = _build_train_env(torch.device("cpu"), 8, get_ch3_method_config("ch3_pse_no_belief"))
    env.reset(); initial = env.map_module.belief_map.clone()
    zeros = torch.zeros((env.num_agents,3), dtype=env.dtype, device=env.device)
    for _ in range(5):
        env.step(zeros)
        assert torch.allclose(env.map_module.belief_map, initial, atol=0.0, rtol=0.0)
        assert math.isclose(float(env.map_module.belief_entropy_normalized().item()), 1.0, abs_tol=1e-6)
