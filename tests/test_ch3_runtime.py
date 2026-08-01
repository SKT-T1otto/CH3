from __future__ import annotations

from pathlib import Path

import pytest
import torch

from algorithms.maddpg import MADDPG
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES, CONTROLLER_ONLY_METHODS
from train import (
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    _checkpoint_metadata,
    build_ch3_runtime,
    build_ch3_runtime_from_resolved_config,
    get_ch3_method_config,
)


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_all_formal_methods_build_with_expected_interfaces(protocol):
    for index, method in enumerate(ACTIVE_CH3_FINAL_EXPERIMENT_MODES):
        runtime = build_ch3_runtime(
            method, seed=200 + index, max_steps=3, device="cpu",
            replay_size=8, protocol=protocol,
        )
        observations = runtime.env.reset()
        assert len(observations) == 4
        assert all(tuple(obs.shape) == (28,) for obs in observations)
        assert runtime.env.action_space["agent_0"].shape == (3,)
        if method in CONTROLLER_ONLY_METHODS:
            assert runtime.maddpg is None and runtime.replay_buffer is None
        else:
            assert runtime.maddpg is not None and runtime.replay_buffer is not None


def test_resolved_runtime_builder_matches_legacy_builder_for_v1_v2():
    for protocol in (CH3_PILOT_V1, CH3_EFFICIENCY_V2):
        config = get_ch3_method_config("ch3_pse_rmaddpg", protocol)
        direct = build_ch3_runtime_from_resolved_config(
            "ch3_pse_rmaddpg", config, seed=199, max_steps=3, replay_size=8
        )
        legacy = build_ch3_runtime(
            "ch3_pse_rmaddpg", seed=199, max_steps=3, replay_size=8, protocol=protocol
        )
        assert direct.config == legacy.config
        assert direct.env.obs_dim == 28
        assert direct.env.action_space["agent_0"].shape == (3,)


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_learning_runtime_produces_finite_action_and_transition(protocol):
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg", seed=301, max_steps=3, device="cpu",
        replay_size=8, protocol=protocol,
    )
    observations = runtime.env.reset()
    actions = runtime.maddpg.step(observations, explore=False)
    action_tensor = torch.stack([action.squeeze(0) for action in actions])
    next_observations, rewards, dones = runtime.env.step(action_tensor)
    assert tuple(action_tensor.shape) == (4, 3)
    assert torch.isfinite(action_tensor).all()
    assert all(torch.isfinite(item).all() for item in next_observations)
    assert torch.isfinite(torch.as_tensor(rewards)).all()
    assert len(dones) == 4


def test_real_actor_critic_replay_and_target_update(tmp_path: Path):
    runtime = build_ch3_runtime("ch3_pse_rmaddpg", seed=777, max_steps=20, device="cpu", replay_size=64)
    observations = runtime.env.reset()
    for _ in range(16):
        actions_list = runtime.maddpg.step(observations, explore=True)
        actions = torch.stack([item.squeeze(0) for item in actions_list])
        next_observations, rewards, dones = runtime.env.step(actions)
        runtime.replay_buffer.push(observations, actions, rewards, next_observations, dones, [False] * 4)
        observations = next_observations
        if all(dones):
            observations = runtime.env.reset()
    sample = runtime.replay_buffer.sample(8, norm_rews=False, device="cpu")
    runtime.maddpg.prep_training(device="cpu")
    errors = []
    for agent_i in range(4):
        critic_loss, actor_loss, td_error = runtime.maddpg.update(sample, agent_i)
        assert torch.isfinite(torch.tensor([critic_loss, actor_loss])).all()
        assert torch.isfinite(td_error).all()
        errors.append(td_error)
        for parameter in runtime.maddpg.agents[agent_i].policy.parameters():
            assert torch.isfinite(parameter).all()
    runtime.replay_buffer.update_priorities(sample[6], torch.stack(errors).mean(dim=0), sample[7])
    before = runtime.maddpg.niter
    runtime.maddpg.update_all_targets()
    assert runtime.maddpg.niter == before + 1

    metadata = _checkpoint_metadata(
        runtime, method="ch3_pse_rmaddpg", seed=777, requested_episodes=1,
        max_steps=20, pilot=True, checkpoint_episode=1, checkpoint_kind="final",
        global_step=16, update_step=1, manifest_id=None, manifest_sha256=None,
    )
    path = tmp_path / "model.pt"
    runtime.maddpg.save(path, metadata=metadata)
    loaded = MADDPG.init_from_save(path, device="cpu")
    assert loaded.checkpoint_metadata["max_steps"] == 20
    loaded.prep_rollouts(device="cpu")
    runtime.maddpg.prep_rollouts(device="cpu")
    reference_obs = runtime.env.reset()
    left = runtime.maddpg.step(reference_obs, explore=False)
    right = loaded.step(reference_obs, explore=False)
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_cuda_request_falls_back_to_cpu_when_unavailable():
    if torch.cuda.is_available():
        return
    runtime = build_ch3_runtime("ch3_pse_rmaddpg", seed=2, max_steps=2, device="cuda", replay_size=8)
    assert runtime.train_device.type == "cpu"


def test_v2_real_actor_update_applies_residual_regularization():
    runtime = build_ch3_runtime(
        "ch3_pse_rmaddpg",
        seed=778,
        max_steps=20,
        device="cpu",
        replay_size=64,
        protocol=CH3_EFFICIENCY_V2,
    )
    observations = runtime.env.reset()
    for _ in range(16):
        actions_list = runtime.maddpg.step(observations, explore=True)
        actions = torch.stack([item.squeeze(0) for item in actions_list])
        next_observations, rewards, dones = runtime.env.step(actions)
        runtime.replay_buffer.push(
            observations, actions, rewards, next_observations, dones, [False] * 4
        )
        observations = next_observations
        if all(dones):
            observations = runtime.env.reset()

    sample = runtime.replay_buffer.sample(8, norm_rews=False, device="cpu")
    runtime.maddpg.prep_training(device="cpu")
    critic_loss, actor_loss, td_error = runtime.maddpg.update(sample, 0)

    assert runtime.maddpg.residual_action_reg == 0.05
    assert runtime.maddpg.last_residual_regularization_term > 0.0
    assert torch.isfinite(torch.tensor([critic_loss, actor_loss])).all()
    assert torch.isfinite(td_error).all()
