import torch
from utils.ch3_buffer import CH3ReplayBuffer


def test_ch3_replay_buffer_returns_eight_items_without_ch5_metadata():
    buffer=CH3ReplayBuffer(max_steps=16,num_agents=4,obs_dims=(28,)*4,ac_dims=(3,)*4)
    obs=[torch.zeros(28) for _ in range(4)]; actions=torch.zeros((4,3))
    buffer.push(obs,actions,torch.arange(4.0),obs,[False]*4,[False]*4)
    sample=buffer.sample(1,norm_rews=False,device="cpu")
    assert len(sample)==8 and sample[0][0].shape==(1,28) and sample[1][0].shape==(1,3)
    forbidden=("tail","comm","scenario","graph","message","belief","quarantine")
    assert not any(token in attr.lower() for attr in vars(buffer) for token in forbidden)
