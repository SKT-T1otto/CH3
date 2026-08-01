import pytest

import tools.build_ch3_scenarios as scenarios
from ch3_constants import (
    SCENARIO_PROFILES,
    SMOKE_TRAIN_GENERATOR_SEED,
    SMOKE_VALIDATION_GENERATOR_SEED,
    TRAIN_GENERATOR_SEED,
    UNKNOWN_MAP_PROFILES,
    UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED,
    UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED,
    UNKNOWN_TRAIN_GENERATOR_SEED,
    UNKNOWN_VALIDATION_GENERATOR_SEED,
    VALIDATION_GENERATOR_SEED,
)


@pytest.mark.parametrize(
    ("split", "mission_seed", "unknown_seed"),
    (
        ("train", TRAIN_GENERATOR_SEED, UNKNOWN_TRAIN_GENERATOR_SEED),
        (
            "validation",
            VALIDATION_GENERATOR_SEED,
            UNKNOWN_VALIDATION_GENERATOR_SEED,
        ),
        (
            "smoke_train",
            SMOKE_TRAIN_GENERATOR_SEED,
            UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED,
        ),
        (
            "smoke_validation",
            SMOKE_VALIDATION_GENERATOR_SEED,
            UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED,
        ),
    ),
)
def test_default_generator_seed_is_family_specific(
    monkeypatch, split, mission_seed, unknown_seed
):
    calls = []

    def mission_builder(count, seed, requested_split):
        calls.append(("mission", count, seed, requested_split))
        return {
            profile: {"scenario_profile": profile}
            for profile in SCENARIO_PROFILES
        }

    def unknown_builder(count, seed, requested_split):
        calls.append(("unknown", count, seed, requested_split))
        return {
            profile: {"scenario_profile": profile}
            for profile in UNKNOWN_MAP_PROFILES
        }

    monkeypatch.setattr(
        scenarios, "_build_mission_profile_manifests", mission_builder
    )
    monkeypatch.setattr(
        scenarios, "_build_unknown_profile_manifests", unknown_builder
    )

    manifests = scenarios.build_scenario_manifests(
        count=1,
        generator_seed=None,
        split=split,
        profiles="all",
    )

    assert set(manifests) == set(SCENARIO_PROFILES) | set(
        UNKNOWN_MAP_PROFILES
    )
    assert calls == [
        ("mission", 1, mission_seed, split),
        ("unknown", 1, unknown_seed, split),
    ]


def test_explicit_generator_seed_is_applied_to_both_families(monkeypatch):
    calls = []

    def mission_builder(count, seed, split):
        calls.append(("mission", count, seed, split))
        return {
            profile: {"scenario_profile": profile}
            for profile in SCENARIO_PROFILES
        }

    def unknown_builder(count, seed, split):
        calls.append(("unknown", count, seed, split))
        return {
            profile: {"scenario_profile": profile}
            for profile in UNKNOWN_MAP_PROFILES
        }

    monkeypatch.setattr(
        scenarios, "_build_mission_profile_manifests", mission_builder
    )
    monkeypatch.setattr(
        scenarios, "_build_unknown_profile_manifests", unknown_builder
    )

    scenarios.build_scenario_manifests(
        count=2,
        generator_seed=99123,
        split="train",
        profiles="all",
    )

    assert calls == [
        ("mission", 2, 99123, "train"),
        ("unknown", 2, 99123, "train"),
    ]
