import math
import torch
from train import _build_train_env, get_ch3_method_config


def _build_env():
    return _build_train_env(torch.device("cpu"), 12, get_ch3_method_config("ch3_pse_rmaddpg"))[0]


def _force_detection_on_next_step(env):
    env._task_target.copy_(env._agent_pos[0]); env._agent_vel.zero_(); env._agent_acc.zero_(); env._prev_acc.zero_()


def test_fixed_handoff_is_reliable_exactly_one_step_and_once_per_episode():
    env = _build_env(); env.reset(); zeros = torch.zeros((env.num_agents,3), dtype=env.dtype, device=env.device)
    _force_detection_on_next_step(env); env.step(zeros)
    assert env.task_found and env.found_step == 1 and env.handoff_step == 1
    assert not bool(env._agent_task_known[env.executor_idx].item())
    assert env.executor_received_target_step is None and math.isnan(env.last_handoff_delay)
    env.step(zeros)
    assert env.executor_received_target_step == 2 and env.last_handoff_delay == 1.0 and env.ch3_handoff_count == 1
    assert torch.allclose(env._agent_task_est[env.executor_idx], env._task_target, atol=0.0, rtol=0.0)
    env.step(zeros); env.step(zeros)
    assert env.ch3_handoff_count == 1


def test_reset_clears_every_fixed_handoff_state():
    env = _build_env(); env.reset(); zeros = torch.zeros((env.num_agents,3), dtype=env.dtype, device=env.device)
    _force_detection_on_next_step(env); env.step(zeros); env.step(zeros); assert env.ch3_handoff_count == 1
    env.reset()
    assert not env.task_found and env.found_step is None and env.executor_received_target_step is None
    assert env.fixed_reliable_handoff.state_dict() is None and env.ch3_handoff_count == 0
    assert math.isnan(env.last_handoff_delay)
