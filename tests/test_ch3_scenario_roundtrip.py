import json
import torch
from tools.build_ch3_pilot_scenarios import build_pilot_scenarios
from train import _build_train_env, get_ch3_method_config


def test_scenario_roundtrip_is_exact_across_methods():
    manifest=build_pilot_scenarios(count=1,seed=123); restored=json.loads(json.dumps(manifest,sort_keys=True)); assert restored==manifest
    scenario=restored["scenarios"][0]
    for method in ("ch3_pheromone_prior","ch3_pse_rmaddpg"):
        env,_=_build_train_env(torch.device("cpu"),4,get_ch3_method_config(method)); env.reset(scenario=scenario)
        assert torch.equal(env._agent_pos,torch.tensor(scenario["initial_agent_positions"],dtype=env.dtype))
        assert torch.equal(env._task_target,torch.tensor(scenario["target_position"],dtype=env.dtype))
