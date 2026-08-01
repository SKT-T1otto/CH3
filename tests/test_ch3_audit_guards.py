from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tools.run_ch3 as acceptance
import tools.validate_ch3_config as validator
import train
from algorithms.maddpg import MADDPG
from env import UAVEnv
from evaluate_pse import _build_evaluation_runtime
from map.map_module import ProbabilisticTaskMapPlanner


def _scenario(*, phase_x=0.0, phase_y=0.0):
    return {
        "scenario_id": "audit_guard", "scenario_seed": 123,
        "initial_agent_positions": [[2.0,2.0,1.0],[8.0,2.0,2.0],[2.0,8.0,3.0],[16.0,16.0,4.0]],
        "target_position": [14.0,14.0,6.0], "flow_phase_x": phase_x,
        "flow_phase_y": phase_y, "initial_executor_wait_point": [10.0,10.0,4.0],
        "planner_seed": 456,
    }


def test_cli_resolves_pilot_and_formal_episode_defaults(monkeypatch):
    captured=[]
    def fake_train(method,**kwargs):
        captured.append((method,kwargs["episodes"],kwargs["pilot"],kwargs["output_dir"])); return {"captured":True},[],[]
    monkeypatch.setattr(train,"train_and_evaluate_method",fake_train)
    train.main(["--method","ch3_pse_rmaddpg","--pilot","--device","cpu"])
    train.main(["--method","ch3_pse_rmaddpg","--device","cpu"])
    train.main(["--method","ch3_pheromone_prior","--pilot","--device","cpu"])
    assert captured[0][1:3]==(200,True); assert captured[1][1:3]==(6000,False); assert captured[2][1:3]==(0,True)
    assert captured[0][3]==train.DEFAULT_PILOT_OUTPUT_DIR and captured[1][3]==train.DEFAULT_FORMAL_OUTPUT_DIR


def test_seed_isolated_checkpoint_has_identity_and_reloads(tmp_path):
    kwargs=dict(method="ch3_pheromone_rmaddpg",episodes=1,max_steps=1,device="cpu",output_dir=tmp_path,pilot=True,scenario_manifest=None)
    first,_,_=train.train_and_evaluate_method(seed=17,**kwargs); second,_,_=train.train_and_evaluate_method(seed=18,**kwargs)
    first_path=Path(first["checkpoint_path"]); second_path=Path(second["checkpoint_path"])
    assert first_path.parent.name=="seed_17" and second_path.parent.name=="seed_18" and first_path!=second_path
    loaded=MADDPG.init_from_save(first_path,device="cpu")
    assert loaded.checkpoint_metadata["method"]=="ch3_pheromone_rmaddpg" and loaded.checkpoint_metadata["seed"]==17
    same=_build_evaluation_runtime("ch3_pheromone_rmaddpg",model_path=first_path,seed=1,max_steps=1,device="cpu")
    assert same.maddpg.checkpoint_metadata["seed"]==17
    with pytest.raises(ValueError,match="checkpoint identity mismatch"):
        _build_evaluation_runtime("ch3_pse_rmaddpg",model_path=first_path,seed=1,max_steps=1,device="cpu")


def test_checkpoint_rejects_wrong_horizon(tmp_path):
    summary,_,_=train.train_and_evaluate_method("ch3_pheromone_rmaddpg",seed=5,episodes=1,max_steps=2,device="cpu",output_dir=tmp_path,pilot=True,scenario_manifest=None)
    with pytest.raises(ValueError,match="checkpoint identity mismatch"):
        _build_evaluation_runtime("ch3_pheromone_rmaddpg",model_path=Path(summary["checkpoint_path"]),seed=1,max_steps=3,device="cpu")


def test_flow_phases_are_applied_to_fixed_scenarios():
    first=UAVEnv(max_steps=2,return_numpy=False); second=UAVEnv(max_steps=2,return_numpy=False)
    first.reset(scenario=_scenario()); second.reset(scenario=_scenario(phase_x=1.1,phase_y=-0.7))
    zeros=torch.zeros((4,3)); first.step(zeros); second.step(zeros)
    assert first.current_scenario_seed==second.current_scenario_seed==123 and not torch.equal(first._agent_pos,second._agent_pos)


def test_standby_preserves_rng_and_propagates_internal_errors(monkeypatch):
    planner=ProbabilisticTaskMapPlanner(space_size=(20,20,8),grid_size=(10,10,8),z_range=(0.5,7.5))
    before=torch.random.get_rng_state().clone(); planner.plan_executor_standby(torch.tensor([4.0,5.0,2.0])); assert torch.equal(before,torch.random.get_rng_state())
    def fail(_): raise ValueError("injected")
    monkeypatch.setattr(planner,"topk_belief_points",fail)
    with pytest.raises(RuntimeError,match="standby planning failed"): planner.plan_executor_standby(torch.tensor([4.0,5.0,2.0]))


def test_validator_scans_checkout_below_data_named_parent(monkeypatch,tmp_path):
    root=tmp_path/"contains_data"/"project"; root.mkdir(parents=True); (root/"env.py").write_text("VALUE=1\n")
    generated=root/"data"/"generated.py"; generated.parent.mkdir(); generated.write_text("SKIP=True\n")
    monkeypatch.setattr(validator,"PROJECT_ROOT",root)
    assert [path for path,_ in validator._source_texts()]==[Path("env.py")]


def test_acceptance_preflight_rejects_missing_pytest_count(monkeypatch):
    def fake_run(name, command, *, timeout_seconds):
        return {
            "name": name,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "runtime_seconds": 0.0,
            "returncode": 0,
            "timed_out": False,
            "passed": True,
            "output": "zero without count",
        }

    monkeypatch.setattr(acceptance, "_run_preflight_stage", fake_run)
    monkeypatch.setattr(
        acceptance,
        "repository_source_fingerprint",
        lambda _root: "repository-fingerprint",
    )
    monkeypatch.setattr(
        acceptance,
        "base_algorithm_source_fingerprint",
        lambda _root: "base-fingerprint",
    )
    monkeypatch.setattr(
        acceptance,
        "mission_algorithm_source_fingerprint",
        lambda _root: "mission-fingerprint",
    )
    monkeypatch.setattr(
        acceptance,
        "unknown_map_algorithm_source_fingerprint",
        lambda _root: "unknown-fingerprint",
    )

    report = acceptance._run_acceptance_preflight()

    assert report["passed"] is False
    assert report["pytest_passed_count"] is None
    assert report["pytest_collected_count"] is None
    assert report["pytest_counts_verified"] is False
    assert report["source_unchanged"] is True
    assert report["source_before"] == report["source_after"]
