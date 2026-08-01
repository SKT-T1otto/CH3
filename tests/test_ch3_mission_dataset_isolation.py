import hashlib
import json

import numpy as np
import pytest

import tools.build_ch3_scenarios as scenario_builder
from tools.build_ch3_scenarios import build_scenario_manifests
from training import validate_dataset_isolation


def _all_values(manifests, key):
    return {
        row[key]
        for manifest in manifests.values()
        for row in manifest["scenarios"]
        if row.get(key) is not None
    }


def test_train_validation_and_smoke_splits_are_independent_and_reproducible(
    monkeypatch,
):
    def fast_trajectory(position, velocity, mode, obstacles):
        identity = json.dumps(
            {
                "position": np.asarray(position).round(8).tolist(),
                "velocity": np.asarray(velocity).round(8).tolist(),
                "mode": mode,
                "obstacles": obstacles,
            },
            sort_keys=True,
        ).encode()
        return (
            hashlib.sha256(identity).hexdigest(),
            0.0 if mode == "static" else 11.0,
            np.asarray(position, dtype=np.float64).reshape(1, 3),
        )

    monkeypatch.setattr(scenario_builder, "_trajectory_record", fast_trajectory)
    monkeypatch.setattr(scenario_builder, "_BUILD_CACHE", {})
    train = build_scenario_manifests(1, 71001, "train")
    validation = build_scenario_manifests(1, 72001, "validation")
    smoke_train = build_scenario_manifests(1, 73001, "smoke_train")
    smoke_validation = build_scenario_manifests(
        1, 74001, "smoke_validation"
    )
    assert json.dumps(train, sort_keys=True) == json.dumps(
        build_scenario_manifests(1, 71001, "train"), sort_keys=True
    )
    for key in ("scenario_id", "scenario_seed", "pair_group_id"):
        assert not _all_values(train, key) & _all_values(validation, key)
        assert not _all_values(smoke_train, key) & _all_values(
            smoke_validation, key
        )
    for split, manifests in (
        ("train", train),
        ("validation", validation),
        ("smoke_train", smoke_train),
        ("smoke_validation", smoke_validation),
    ):
        assert all(
            manifest["scenario_role"] == split
            and manifest["scenario_split"] == split
            and f"_{split}_" in manifest["manifest_id"]
            for manifest in manifests.values()
        )
        pair_rows = [
            [
                row["pair_group_id"]
                for row in manifests[profile]["scenarios"]
            ]
            for profile in manifests
        ]
        assert all(row == pair_rows[0] for row in pair_rows[1:])


def _scenario(identifier, seed, pair, x):
    return {
        "scenario_id": identifier,
        "scenario_seed": seed,
        "planner_seed": seed + 100,
        "target_motion_seed": seed + 200,
        "pair_group_id": pair,
        "scenario_profile": "S00_STATIC_CLEAR",
        "scenario_role": "train",
        "scenario_split": "train",
        "initial_agent_positions": [
            [1 + x, 1, 1],
            [4 + x, 1, 1],
            [1 + x, 4, 1],
            [4 + x, 4, 1],
        ],
        "initial_executor_wait_point": [10, 10, 4],
        "target_position": [18 - x, 18, 6],
        "target_initial_position": [18 - x, 18, 6],
        "target_initial_velocity": [0, 0, 0],
        "target_motion_mode": "static",
        "obstacle_layout_id": "none",
        "obstacles": [],
    }


def _manifest(role, identifier, generator_seed, scenario):
    scenario = dict(scenario, scenario_role=role, scenario_split=role)
    return {
        "protocol": "ch3_mission_v1",
        "manifest_id": identifier,
        "manifest_sha256": f"sha-{identifier}",
        "scenario_profile": "S00_STATIC_CLEAR",
        "scenario_role": role,
        "scenario_split": role,
        "generator_seed": generator_seed,
        "scenario_count": 1,
        "scenarios": [scenario],
    }


def test_dataset_isolation_accepts_disjoint_identity_and_physical_content():
    training = _manifest(
        "train", "training", 71001,
        _scenario("train-1", 1, "train-pair", 0),
    )
    evaluation = _manifest(
        "validation", "evaluation", 72001,
        _scenario("evaluation-1", 2, "evaluation-pair", 0.5),
    )
    result = validate_dataset_isolation(
        training, training["scenarios"], evaluation, evaluation["scenarios"]
    )
    assert result == {
        "scenario_id_overlap_count": 0,
        "scenario_seed_overlap_count": 0,
        "pair_group_overlap_count": 0,
        "physical_content_overlap_count": 0,
    }


@pytest.mark.parametrize("overlap", ["scenario_id", "scenario_seed", "pair_group_id"])
def test_each_identity_overlap_is_rejected(overlap):
    train_scenario = _scenario("train-1", 1, "train-pair", 0)
    evaluation_scenario = _scenario("evaluation-1", 2, "eval-pair", 0.5)
    evaluation_scenario[overlap] = train_scenario[overlap]
    training = _manifest("train", "training", 71001, train_scenario)
    evaluation = _manifest(
        "validation", "evaluation", 72001, evaluation_scenario
    )
    with pytest.raises(
        RuntimeError, match="training/evaluation scenario leakage detected"
    ):
        validate_dataset_isolation(
            training, training["scenarios"], evaluation, evaluation["scenarios"]
        )


def test_same_physical_scenario_with_only_new_ids_and_seeds_is_rejected():
    train_scenario = _scenario("train-id", 1, "train-pair", 0)
    evaluation_scenario = dict(
        train_scenario,
        scenario_id="validation-id",
        scenario_seed=3,
        planner_seed=4,
        target_motion_seed=5,
        pair_group_id="validation-pair",
        scenario_role="validation",
        scenario_split="validation",
    )
    training = _manifest("train", "training", 71001, train_scenario)
    evaluation = _manifest(
        "validation", "evaluation", 72001, evaluation_scenario
    )
    with pytest.raises(
        RuntimeError, match="training/evaluation scenario leakage detected"
    ):
        validate_dataset_isolation(
            training, training["scenarios"], evaluation, evaluation["scenarios"]
        )


def test_same_manifest_hash_or_generator_seed_is_rejected():
    training = _manifest(
        "train", "training", 71001,
        _scenario("train-1", 1, "train-pair", 0),
    )
    evaluation = _manifest(
        "validation", "evaluation", 72001,
        _scenario("eval-1", 2, "eval-pair", 0.5),
    )
    evaluation["manifest_sha256"] = training["manifest_sha256"]
    with pytest.raises(RuntimeError):
        validate_dataset_isolation(
            training, training["scenarios"], evaluation, evaluation["scenarios"]
        )
    evaluation["manifest_sha256"] = "different"
    evaluation["generator_seed"] = training["generator_seed"]
    with pytest.raises(RuntimeError):
        validate_dataset_isolation(
            training, training["scenarios"], evaluation, evaluation["scenarios"]
        )
