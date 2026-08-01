import numpy as np
import torch

from runtime import build_runtime
from target_motion import TargetState
from tools.build_ch3_scenarios import build_scenario_manifests


def test_all_profiles_preserve_interfaces_and_run_bounded_steps():
    manifests = build_scenario_manifests(1)
    for profile, manifest in manifests.items():
        runtime = build_runtime(
            "ch3_v3_full_reference", profile,
            seed=33, max_steps=20, device="cpu", replay_size=32,
        )
        env = runtime.env
        observations = env.reset(manifest["scenarios"][0])
        assert len(observations) == 4
        assert all(tuple(item.shape) == (28,) for item in observations)
        assert env.action_space["agent_0"].shape == (3,)
        for _ in range(20):
            observations, rewards, _ = env.step(torch.zeros((4, 3)))
            assert all(torch.isfinite(item).all() for item in observations)
            assert torch.isfinite(torch.as_tensor(rewards)).all()
            assert np.all(np.isfinite(env.target_state.position))


def test_swept_detection_selects_distance_then_agent_index_and_does_not_leak():
    runtime = build_runtime(
        "ch3_v3_full_reference", "S10_MOVING_CLEAR",
        seed=34, max_steps=4, device="cpu", replay_size=8,
    )
    env = runtime.env
    observations = env.reset()
    assert not env._agent_task_known.any()
    assert all(torch.all(item[12:15] == 0) for item in observations)
    env._sensor_range[:3] = 1.0
    env._agent_pos[:3] = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 5, 1]], dtype=env.dtype)
    env.target_state = TargetState([1, 0, 1], [0, 0, 0], 1, "static")
    env._task_target.copy_(env._vec(env.target_state.position))
    starts = np.asarray([[0, 0, 1], [0, 0, 1], [0, 5, 1]], dtype=np.float64)
    env._maybe_detect_swept(starts, np.asarray([-1, 0, 1], dtype=np.float64))
    assert env.task_found and env.finder_idx == 0
