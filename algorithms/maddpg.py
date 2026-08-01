
import os
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from utils.misc import soft_update, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent


class MADDPG(object):
    def __init__(
        self,
        agent_init_params,
        alg_types,
        gamma=0.95,
        tau=5e-3,
        lr=0.01,
        lr_actor=None,
        lr_critic=None,
        hidden_dim=64,
        discrete_action=False,
        agent_role_names=None,
        agent_noise_sigmas=None,
        residual_action_reg=1e-2,
    ):
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.lr_actor = lr if lr_actor is None else lr_actor
        self.lr_critic = lr if lr_critic is None else lr_critic
        self.hidden_dim = hidden_dim
        self.discrete_action = discrete_action
        self.agent_role_names = agent_role_names or [f"agent_{i}" for i in range(self.nagents)]
        self.residual_action_reg = float(residual_action_reg)

        self.agents = [
            DDPGAgent(
                lr=lr,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                discrete_action=discrete_action,
                hidden_dim=hidden_dim,
                **params,
            )
            for params in agent_init_params
        ]

        if agent_noise_sigmas is not None:
            for ag, sigma in zip(self.agents, agent_noise_sigmas):
                ag.scale_noise(float(sigma), multiply=False)

        self.device = torch.device("cpu")
        self.niter = 0
        self.init_dict = None
        self.checkpoint_metadata = None
        self.last_residual_regularization_term = 0.0
        self._cached_sample_key = None
        self._cached_sample = None

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        device_str = str(device).lower()
        if device_str == "gpu":
            device_str = "cuda"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_str)

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=device, non_blocking=True)

    @staticmethod
    def _recursive_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: MADDPG._recursive_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MADDPG._recursive_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(MADDPG._recursive_to_cpu(v) for v in obj)
        return obj

    @staticmethod
    def _clone_agent_params_to_cpu(agent: DDPGAgent):
        return {
            "policy": MADDPG._recursive_to_cpu(agent.policy.state_dict()),
            "target_policy": MADDPG._recursive_to_cpu(agent.target_policy.state_dict()),
            "policy_opt": MADDPG._recursive_to_cpu(agent.policy_optimizer.state_dict()),
            "critic1": MADDPG._recursive_to_cpu(agent.critic1.state_dict()),
            "target_critic1": MADDPG._recursive_to_cpu(agent.target_critic1.state_dict()),
            "critic1_opt": MADDPG._recursive_to_cpu(agent.critic1_optimizer.state_dict()),
            "critic2": MADDPG._recursive_to_cpu(agent.critic2.state_dict()),
            "target_critic2": MADDPG._recursive_to_cpu(agent.target_critic2.state_dict()),
            "critic2_opt": MADDPG._recursive_to_cpu(agent.critic2_optimizer.state_dict()),
        }

    def _move_all_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        for a in self.agents:
            a.policy.to(device)
            a.target_policy.to(device)
            a.critic1.to(device)
            a.target_critic1.to(device)
            a.critic2.to(device)
            a.target_critic2.to(device)
            self._move_optimizer_state(a.policy_optimizer, device)
            self._move_optimizer_state(a.critic1_optimizer, device)
            self._move_optimizer_state(a.critic2_optimizer, device)
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()
        self.device = device
        self._clear_sample_cache()

    def _clear_sample_cache(self):
        self._cached_sample_key = None
        self._cached_sample = None

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def reset_noise(self):
        for agent in self.agents:
            agent.noise.reset()

    def _ensure_obs_tensor(self, obs: Any) -> torch.Tensor:
        if torch.is_tensor(obs):
            tensor = obs.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def step(self, observations, explore=False):
        with torch.no_grad():
            proc_obs = [self._ensure_obs_tensor(obs) for obs in observations]
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents, proc_obs)]

    def step_residual(self, observations, explore=False):
        return self.step(observations, explore=explore)

    def _unpack_sample(self, sample):
        if len(sample) != 8:
            raise ValueError(f"Chapter-3 replay samples must contain 8 items, got {len(sample)}")
        return sample

    def _to_device(self, tensors, dtype=torch.float32):
        out = []
        for x in tensors:
            if torch.is_tensor(x):
                out.append(x.to(device=self.device, dtype=dtype, non_blocking=True))
            else:
                out.append(torch.as_tensor(x, dtype=dtype, device=self.device))
        return out

    def _prepare_sample(self, sample):
        key = (id(sample), self.device)
        if self._cached_sample_key == key and self._cached_sample is not None:
            return self._cached_sample
        obs, acs, rews, next_obs, dones, weights, indices, success_flags = self._unpack_sample(sample)
        obs = self._to_device(obs)
        acs = self._to_device(acs)
        rews = self._to_device(rews)
        next_obs = self._to_device(next_obs)
        dones = self._to_device(dones)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1)
        with torch.no_grad():
            next_acs = self._build_target_actions(next_obs)
            target_vf_in = torch.cat((*next_obs, *next_acs), dim=1)
        joint_vf_in = torch.cat((*obs, *acs), dim=1)
        packed = {
            "obs": obs, "acs": acs, "rews": rews, "next_obs": next_obs,
            "dones": dones, "weights": weights, "indices": indices,
            "success_flags": success_flags, "next_acs": next_acs,
            "target_vf_in": target_vf_in, "joint_vf_in": joint_vf_in,
        }
        self._cached_sample_key = key
        self._cached_sample = packed
        return packed



    def _build_target_actions(self, next_obs):
        if self.discrete_action:
            return [onehot_from_logits(pi(no)) for pi, no in zip(self.target_policies, next_obs)]
        return [torch.clamp(pi(no), -1.0, 1.0) for pi, no in zip(self.target_policies, next_obs)]

    def _build_policy_action(self, agent_i, obs_i, explore_gumbel=True):
        ag = self.agents[agent_i]
        if self.discrete_action:
            logits = ag.policy(obs_i)
            act = gumbel_softmax(logits, hard=True) if explore_gumbel else onehot_from_logits(logits)
            return logits, act
        raw = torch.clamp(ag.policy(obs_i), -1.0, 1.0)
        return raw, raw

    def _compute_target_q(self, ag: DDPGAgent, target_vf_in, rew_i, done_i):
        with torch.no_grad():
            q1_next = ag.target_critic1(target_vf_in)
            q2_next = ag.target_critic2(target_vf_in)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rew_i.view(-1, 1) + self.gamma * min_q_next * (1 - done_i.view(-1, 1))
            return torch.clamp(target_q, -10.0, 10.0)

    def _critic_input(self, batch, agent_i):
        obs = batch["obs"]
        acs = batch["acs"]
        if self.alg_types[agent_i] == "MADDPG":
            return batch["joint_vf_in"] if batch["joint_vf_in"] is not None else torch.cat((*obs, *acs), dim=1)
        return torch.cat((obs[agent_i], acs[agent_i]), dim=1)

    def update_critic_only(self, sample, agent_i):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target_q = self._compute_target_q(ag, batch["target_vf_in"], batch["rews"][agent_i], batch["dones"][agent_i])
        vf_in = self._critic_input(batch, agent_i)

        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)

        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        td_loss = loss1 + loss2
        vf_loss = (td_loss * batch["weights"]).mean()

        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5)
        ag.critic1_optimizer.step()
        ag.critic2_optimizer.step()

        return float(vf_loss.item()), td_error

    def update(self, sample, agent_i, parallel=False, logger=None):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target_q = self._compute_target_q(ag, batch["target_vf_in"], batch["rews"][agent_i], batch["dones"][agent_i])
        vf_in = self._critic_input(batch, agent_i)

        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)

        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        td_loss = loss1 + loss2
        vf_loss = (td_loss * batch["weights"]).mean()

        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5)
        ag.critic1_optimizer.step()
        ag.critic2_optimizer.step()

        ag.policy_optimizer.zero_grad(set_to_none=True)
        curr_pol_out, curr_pol_act = self._build_policy_action(agent_i, batch["obs"][agent_i], explore_gumbel=True)

        all_pol_acs = []
        for idx, (pi, ob) in enumerate(zip(self.policies, batch["obs"])):
            if idx == agent_i:
                all_pol_acs.append(curr_pol_act)
            else:
                with torch.no_grad():
                    if self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob)))
                    else:
                        all_pol_acs.append(torch.clamp(pi(ob), -1.0, 1.0))

        vf_in_pol = torch.cat((*batch["obs"], *all_pol_acs), dim=1)
        q_pol = ag.critic1(vf_in_pol)
        pol_loss = -q_pol.mean()

        # In residual-prior control, the actor output is a correction term rather
        # than the full acceleration command. Penalizing large residuals keeps the
        # learned policy from fighting the waypoint prior unless the critic supports it.
        if self.residual_action_reg > 0.0:
            residual_regularization = (curr_pol_out ** 2).mean() * self.residual_action_reg
            self.last_residual_regularization_term = float(
                residual_regularization.detach().item()
            )
            pol_loss = pol_loss + residual_regularization
        else:
            self.last_residual_regularization_term = 0.0

        if self.discrete_action:
            probs = F.softmax(curr_pol_out, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            pol_loss = pol_loss - 2e-3 * entropy

        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.policy.parameters(), 0.2)
        ag.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars(
                f"agent{agent_i}/loss",
                {"critic": float(vf_loss.item()), "actor": float(pol_loss.item())},
                self.niter,
            )

        return float(vf_loss.item()), float(pol_loss.item()), td_error

    def update_all_targets(self, compute_diff=False):
        diffs = {"policy": [], "critic1": [], "critic2": []} if compute_diff else None
        for a in self.agents:
            diff_c1 = soft_update(a.target_critic1, a.critic1, self.tau, return_diff=compute_diff)
            diff_c2 = soft_update(a.target_critic2, a.critic2, self.tau, return_diff=compute_diff)
            diff_pol = soft_update(a.target_policy, a.policy, self.tau, return_diff=compute_diff)
            if compute_diff:
                diffs["critic1"].append(diff_c1)
                diffs["critic2"].append(diff_c2)
                diffs["policy"].append(diff_pol)

        self._clear_sample_cache()
        self.niter += 1
        return diffs

    def scale_noise(self, factor: float, multiply: bool = True):
        for agent in self.agents:
            agent.scale_noise(factor, multiply=multiply)

    def prep_training(self, device="cuda"):
        target_device = self._resolve_device(device)
        self._move_all_to(target_device)
        for a in self.agents:
            a.policy.train()
            a.critic1.train()
            a.critic2.train()
            a.target_policy.train()
            a.target_critic1.train()
            a.target_critic2.train()
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()

    def prep_rollouts(self, device=None):
        if device is not None:
            target_device = self._resolve_device(device)
            self._move_all_to(target_device)
        for a in self.agents:
            a.policy.eval()
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()

    def save(self, filename, metadata=None):
        save_dict = {
            "init_dict": self.init_dict,
            "agent_params": [self._clone_agent_params_to_cpu(a) for a in self.agents],
            "metadata": None if metadata is None else dict(metadata),
        }
        torch.save(save_dict, filename)

    def training_state_dict(self):
        return {
            "init_dict": self.init_dict,
            "agent_params": [self._clone_agent_params_to_cpu(a) for a in self.agents],
            "niter": int(self.niter),
        }

    def load_training_state_dict(self, state):
        if state.get("init_dict") != self.init_dict:
            raise ValueError("MADDPG initialization metadata mismatch during resume")
        params = state.get("agent_params", [])
        if len(params) != len(self.agents):
            raise ValueError("MADDPG agent count mismatch during resume")
        for agent, agent_params in zip(self.agents, params):
            agent.load_params(agent_params)
        self.niter = int(state.get("niter", 0))
        self._clear_sample_cache()

    @classmethod
    def init_from_env(
        cls,
        env,
        gamma=0.95,
        tau=0.01,
        lr=0.01,
        lr_actor=None,
        lr_critic=None,
        hidden_dim=64,
        residual_action_reg=1e-2,
    ):
        agent_init_params = []
        alg_types = ["MADDPG" for _ in range(env.num_agents)]

        obs_dims = [env.observation_space[f"agent_{i}"].shape[0] for i in range(env.num_agents)]
        ac_dims = [env.action_space[f"agent_{i}"].shape[0] for i in range(env.num_agents)]
        total_critic_in = sum(obs_dims) + sum(ac_dims)

        for i in range(env.num_agents):
            num_in_pol = obs_dims[i]
            num_out_pol = ac_dims[i]
            num_in_critic = total_critic_in if alg_types[i] == "MADDPG" else (obs_dims[i] + ac_dims[i])
            params = {
                "num_in_pol": num_in_pol,
                "num_out_pol": num_out_pol,
                "num_in_critic": num_in_critic,
            }
            agent_init_params.append(params)

        role_names = getattr(env, "role_names", [f"agent_{i}" for i in range(env.num_agents)])
        if hasattr(env, "agent_specs") and len(env.agent_specs) == env.num_agents:
            noise_sigmas = []
            for spec in env.agent_specs:
                name = spec.get("name", "")
                if name == "search_fast":
                    noise_sigmas.append(0.18)
                elif name == "search_balanced":
                    noise_sigmas.append(0.14)
                elif name == "search_precise":
                    noise_sigmas.append(0.10)
                elif name == "executor":
                    noise_sigmas.append(0.08)
                else:
                    noise_sigmas.append(0.12)
        else:
            noise_sigmas = None

        effective_lr_actor = lr if lr_actor is None else lr_actor
        effective_lr_critic = lr if lr_critic is None else lr_critic

        init_dict = {
            "gamma": gamma,
            "tau": tau,
            "lr": lr,
            "lr_actor": effective_lr_actor,
            "lr_critic": effective_lr_critic,
            "hidden_dim": hidden_dim,
            "alg_types": alg_types,
            "agent_init_params": agent_init_params,
            "discrete_action": False,
            "agent_role_names": role_names,
            "agent_noise_sigmas": noise_sigmas,
            "residual_action_reg": float(residual_action_reg),
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, device=None):
        load_device = cls._resolve_device(device)
        save_dict = torch.load(filename, map_location=load_device, weights_only=True)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        instance.checkpoint_metadata = save_dict.get("metadata")
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        instance.prep_training(device=load_device)
        return instance
