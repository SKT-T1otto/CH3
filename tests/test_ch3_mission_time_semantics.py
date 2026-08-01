import numpy as np
import torch

from runtime import build_runtime
from tools.build_ch3_scenarios import build_scenario_manifests


def test_handoff_event_delay_one_but_physical_age_zero_at_delivery():
    scenario = build_scenario_manifests(1)["S10_MOVING_CLEAR"]["scenarios"][0]
    env = build_runtime(
        "ch3_v3_full_reference",
        "S10_MOVING_CLEAR",
        seed=81,
        max_steps=4,
        device="cpu",
        replay_size=8,
    ).env
    env.reset(scenario)
    env._publish_detection(0)
    sample = env.target_state.copy()
    assert env.step_count == sample.sample_step == 0
    env.step(torch.zeros((4, 3)))
    assert env.last_handoff_delay == 1.0
    assert env.handoff_event_delay_steps == 1
    assert env.handoff_delivery_phase == "pre_transition"
    assert env.handoff_physical_age_at_delivery_steps == 0
    assert env.target_prediction_error_at_delivery <= 1e-8
    assert np.array_equal(env.predicted_target_position_at_delivery, sample.position)
    assert env.handoff_payload_age_steps == 1
    assert env.ch3_handoff_count == 1


def test_handoff_target_state_metadata_is_deep_copied():
    scenario = build_scenario_manifests(1)["S00_STATIC_CLEAR"]["scenarios"][0]
    env = build_runtime(
        "ch3_v3_full_reference",
        "S00_STATIC_CLEAR",
        seed=82,
        max_steps=3,
        device="cpu",
        replay_size=8,
    ).env
    env.reset(scenario)
    env.target_state.metadata["nested"] = {"values": [1, 2]}
    env._publish_detection(0)
    env.target_state.metadata["nested"]["values"][0] = 99
    env.step(torch.zeros((4, 3)))
    delivered = env.executor_delivered_target_state
    assert delivered.metadata["nested"]["values"] == [1, 2]
    delivered.metadata["nested"]["values"][1] = 88
    assert env.fixed_reliable_handoff.state_dict()["target"].metadata[
        "nested"
    ]["values"] == [1, 2]
