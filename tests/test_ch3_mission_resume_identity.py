import csv
import json
from types import SimpleNamespace

import pytest

import training
from ch3_config import build_mission_config
from training import train_and_evaluate


def _manifest(path):
    manifest = {
        "protocol": "ch3_mission_v1",
        "manifest_id": "resume_smoke_manifest",
        "scenario_profile": "S00_STATIC_CLEAR",
        "scenario_role": "smoke_train",
        "scenario_split": "smoke_train",
        "generator_seed": 73001,
        "scenario_count": 1,
        "scenarios": [{
            "scenario_id": "resume_static_0001",
            "scenario_seed": 301,
            "planner_seed": 302,
            "scenario_profile": "S00_STATIC_CLEAR",
            "initial_agent_positions": [
                [1, 1, 1], [4, 1, 1], [1, 4, 1], [4, 4, 1],
            ],
            "initial_executor_wait_point": [10, 10, 4],
            "target_position": [18, 18, 6],
            "target_initial_position": [18, 18, 6],
            "target_initial_velocity": [0, 0, 0],
            "target_motion_mode": "static",
            "obstacle_layout_id": "none",
            "obstacles": [],
        }],
    }
    path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _train(output, manifest, episodes, resume):
    return train_and_evaluate(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=303,
        episodes=episodes,
        max_steps=1,
        device="cpu",
        output_dir=output,
        scenario_manifest=manifest,
        resume=resume,
        evaluation_limit=0,
        checkpoint_interval=0,
        replay_size=8,
    )


def test_two_to_four_resume_is_contiguous_and_nonincrementing_targets_reject(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path / "manifest.json")
    output = tmp_path / "runs"
    _train(output, manifest, 2, False)
    _train(output, manifest, 4, True)
    run_dir = (
        output
        / "ch3_v3_full_reference"
        / "S00_STATIC_CLEAR"
        / "seed_303"
    )
    with (run_dir / "episode_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        episodes = [int(row["episode"]) for row in csv.DictReader(handle)]
    assert episodes == [1, 2, 3, 4]
    summary_before = (run_dir / "training_summary.json").read_bytes()
    checkpoint_before = (run_dir / "model_final.pt").read_bytes()
    config = build_mission_config(
        "ch3_v3_full_reference", "S00_STATIC_CLEAR"
    )
    config.update(max_steps=1, replay_size=8)
    monkeypatch.setattr(
        training,
        "build_runtime",
        lambda *args, **kwargs: SimpleNamespace(config=dict(config)),
    )
    monkeypatch.setattr(
        training,
        "_load_resume",
        lambda *args, **kwargs: {
            "episode": 4,
            "global_step": 4,
            "update_step": 0,
            "sigma": config["initial_sigma"],
            "repository_source_fingerprint": None,
        },
    )
    _train(output, manifest, 4, True)
    assert (run_dir / "training_summary.json").read_bytes() == summary_before
    assert (run_dir / "model_final.pt").read_bytes() == checkpoint_before
    with pytest.raises(ValueError, match="below completed"):
        _train(output, manifest, 3, True)
