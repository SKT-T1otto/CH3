from pathlib import Path

import pytest

import tools.run_ch3_efficiency_v3_screen as runner
from tools.build_ch3_efficiency_v3_scenarios import build_v3_manifest


def test_v3_manifest_identity_and_pairing_contract():
    manifest = build_v3_manifest("validation")
    assert manifest["protocol"] == "ch3_efficiency_v3_screen"
    assert manifest["manifest_id"] == "ch3_efficiency_v3_screen_validation_scenarios_v1"
    assert manifest["scenario_count"] == 50
    assert manifest["scenarios"][0]["scenario_id"] == "efficiency_v3_screen_validation_0001"
    assert all(not row["use_obstacles"] and row["flow_phase_x"] == row["flow_phase_y"] == 0 for row in manifest["scenarios"])


def test_runner_rejects_output_outside_v3_root():
    with pytest.raises(ValueError, match="must be below"):
        runner._below_v3(Path(runner.PROJECT_ROOT) / "data" / "chapter3_efficiency_v2", "test")


def test_v3_validation_reuses_v2_geometry_and_fixed_flow_phases():
    from tools.build_ch3_efficiency_scenarios import build_efficiency_manifest

    v2 = build_efficiency_manifest(
        "validation", count=50, generator_seed=41001,
        use_obstacles=False, obstacle_layout_id="none",
    )
    v3 = build_v3_manifest("validation")
    for left, right in zip(v2["scenarios"], v3["scenarios"]):
        for key in (
            "scenario_seed", "planner_seed", "use_obstacles",
            "obstacle_layout_id", "initial_agent_positions",
            "target_position", "initial_executor_wait_point",
            "flow_phase_x", "flow_phase_y",
        ):
            assert left[key] == right[key]


def test_v3_budget_checkpoint_defaults_and_auto_device():
    assert runner.BUDGETS == {"smoke": 2, "pilot": 200}
    assert runner.CHECKPOINT_INTERVALS == {"smoke": 0, "pilot": 50}
    assert runner._device("cpu") == "cpu"
    assert runner._device("auto") in {"cpu", "cuda"}


def test_v3_runner_rejects_nonincreasing_resume_target(tmp_path, monkeypatch):
    import torch

    v3_root = tmp_path / "chapter3_efficiency_v3_screen"
    monkeypatch.setattr(runner, "V3_ROOT", v3_root)
    method = "ch3_v3_full_reference"
    runs = v3_root / "runs"
    method_dir = runs / method / "seed_1"
    method_dir.mkdir(parents=True)
    torch.save({"episode": 2}, method_dir / "resume_state.pt")
    manifest = tmp_path / "smoke.json"
    import json
    manifest.write_text(json.dumps(build_v3_manifest("smoke")), encoding="utf-8")

    with pytest.raises(ValueError, match="resume target episodes must exceed"):
        runner.run_candidate(
            method,
            seed=1,
            episodes=2,
            max_steps=2,
            device="cpu",
            output_dir=runs,
            manifest_path=manifest,
            resume=True,
            evaluation_limit=1,
            checkpoint_interval=0,
        )
