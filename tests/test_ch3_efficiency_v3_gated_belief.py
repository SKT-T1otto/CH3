import math

import torch

from registry.ch3_efficiency_v3_registry import resolve_ch3_efficiency_v3_config
from train import build_ch3_runtime_from_resolved_config


def _planner(label="ch3_v3_gated_belief"):
    config = resolve_ch3_efficiency_v3_config(label)
    return build_ch3_runtime_from_resolved_config(label, config, seed=7, max_steps=4, replay_size=8).env.map_module


def test_gated_belief_boundary_math_and_probability_floor():
    planner = _planner()
    planner.belief_map.zero_()
    first = int(torch.nonzero(planner.flat_valid_mask_float, as_tuple=False)[0])
    planner.belief_map.reshape(-1)[first] = 1.0
    planner.normalize_belief()
    mixed, weight = planner.gated_belief_distribution(step=79, normalized_entropy=0.0)
    assert weight == 0.0
    mixed, weight = planner.gated_belief_distribution(step=160, normalized_entropy=0.90)
    assert weight == 0.0
    state = torch.random.get_rng_state().clone()
    mixed, weight = planner.gated_belief_distribution(step=160, normalized_entropy=0.65)
    assert torch.equal(state, torch.random.get_rng_state())
    assert math.isclose(weight, 0.25)
    assert math.isclose(planner.last_gated_belief_uniform_mix, 0.35)
    assert torch.all(mixed >= 0)
    assert math.isclose(float(mixed.sum()), 1.0, abs_tol=1e-6)
    valid = mixed[planner.flat_valid_mask]
    assert float(valid.min()) + 1e-8 >= 0.35 / planner.valid_belief_cell_count()


def test_no_belief_and_v2_do_not_enable_gated_branch():
    no_belief = resolve_ch3_efficiency_v3_config("ch3_v3_no_belief_reference")
    assert not no_belief.get("pse_use_gated_belief", False)
    from train import CH3_EFFICIENCY_V2, get_ch3_method_config
    assert "pse_use_gated_belief" not in get_ch3_method_config("ch3_pse_rmaddpg", CH3_EFFICIENCY_V2)


def test_execution_cost_reference_is_decoupled_only_when_requested():
    config = resolve_ch3_efficiency_v3_config("ch3_v3_full_reference")
    runtime = build_ch3_runtime_from_resolved_config("v3", config, seed=8, max_steps=4, replay_size=8)
    env = runtime.env
    env.reset()
    fixed = env.exec_cost_reference_position.clone()
    env._agent_pos[env.executor_idx] += torch.tensor([1.0, 0.0, 0.0])
    env._set_pse_planner_context()
    assert torch.allclose(env.exec_cost_reference_position, fixed)
    assert not torch.allclose(env.physical_executor_position, fixed)
    config["pse_exec_cost_reference_mode"] = "physical_position"
    old = build_ch3_runtime_from_resolved_config("old", config, seed=8, max_steps=4, replay_size=8).env
    old._agent_pos[old.executor_idx] += torch.tensor([1.0, 0.0, 0.0])
    old._set_pse_planner_context()
    assert torch.allclose(old.exec_cost_reference_position, old._agent_pos[old.executor_idx])
