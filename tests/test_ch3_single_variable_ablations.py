import pytest
from train import CH3_EFFICIENCY_V2, CH3_PILOT_V1, get_ch3_method_config


def _diff(left,right):
    return {k:(left.get(k),right.get(k)) for k in set(left)|set(right) if left.get(k)!=right.get(k)}

@pytest.mark.parametrize("method,allowed",(
    ("ch3_pse_no_belief",{"pse_use_belief"}),
    ("ch3_pse_no_exec_cost",{"pse_use_exec_cost"}),
    ("ch3_pse_no_standby",{"pse_use_standby"}),
    ("ch3_pse_no_residual",{"residual_scale_search","residual_scale_executor","run_type"}),
))
@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_pse_ablation_changes_only_allowed_keys(method,allowed,protocol):
    full = get_ch3_method_config("ch3_pse_rmaddpg", protocol=protocol)
    ablation = get_ch3_method_config(method, protocol=protocol)
    assert set(_diff(full, ablation)) == allowed


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_pheromone_and_pse_learning_baselines_share_training_and_environment(protocol):
    pheromone = get_ch3_method_config("ch3_pheromone_rmaddpg", protocol=protocol)
    pse = get_ch3_method_config("ch3_pse_rmaddpg", protocol=protocol)
    differences=_diff(pheromone, pse)
    assert set(differences)=={"use_pse_planner","pse_use_belief","pse_use_exec_cost","pse_use_standby"}


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_no_residual_is_pse_prior_not_pure_maddpg(protocol):
    cfg=get_ch3_method_config("ch3_pse_no_residual", protocol=protocol)
    assert cfg["use_pse_planner"] is True and cfg["use_residual_prior"] is True
    assert cfg["residual_scale_search"]==0.0 and cfg["residual_scale_executor"]==0.0 and cfg["run_type"]=="controller_only"
