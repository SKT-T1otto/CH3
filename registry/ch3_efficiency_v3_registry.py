"""Registered single-seed candidates for Chapter-3 efficiency screening v3."""

from __future__ import annotations

from copy import deepcopy


CH3_EFFICIENCY_V3_SCREEN = "ch3_efficiency_v3_screen"

CH3_EFFICIENCY_V3_SCREEN_METHODS = (
    "ch3_v3_full_reference",
    "ch3_v3_no_belief_reference",
    "ch3_v3_no_belief_no_standby",
    "ch3_v3_gated_belief",
    "ch3_v3_no_belief_low_exec",
    "ch3_v3_no_belief_low_exec_no_standby",
)

_COMMON_OVERRIDES = {
    "protocol": CH3_EFFICIENCY_V3_SCREEN,
    "pse_exec_cost_reference_mode": "fixed_initial_wait_point",
    "failure_penalty_steps": 100,
}

CH3_EFFICIENCY_V3_SCREEN_REGISTRY = {
    "ch3_v3_full_reference": {
        "label": "ch3_v3_full_reference",
        "base_method": "ch3_pse_rmaddpg",
        "description": "v2 mechanism settings rerun under v3 decoupled execution-cost reference",
        "config_overrides": dict(_COMMON_OVERRIDES),
        "changed_mechanisms": ["exec_cost_reference"],
        "screening_role": "v3 full-mechanism reference",
    },
    "ch3_v3_no_belief_reference": {
        "label": "ch3_v3_no_belief_reference",
        "base_method": "ch3_pse_no_belief",
        "description": "v2 no-belief mechanism settings under the v3 execution-cost reference",
        "config_overrides": dict(_COMMON_OVERRIDES),
        "changed_mechanisms": ["exec_cost_reference"],
        "screening_role": "efficiency reference candidate",
    },
    "ch3_v3_no_belief_no_standby": {
        "label": "ch3_v3_no_belief_no_standby",
        "base_method": "ch3_pse_no_belief",
        "description": "no-belief reference with standby disabled",
        "config_overrides": dict(_COMMON_OVERRIDES, pse_use_standby=False),
        "changed_mechanisms": ["exec_cost_reference", "standby"],
        "screening_role": "standby necessity screen after belief removal",
    },
    "ch3_v3_gated_belief": {
        "label": "ch3_v3_gated_belief",
        "base_method": "ch3_pse_rmaddpg",
        "description": "full reference with deterministic time-and-entropy gated belief",
        "config_overrides": dict(
            _COMMON_OVERRIDES,
            pse_use_gated_belief=True,
            pse_belief_weight=0.0,
            pse_belief_weight_max=0.25,
            pse_belief_gate_start_step=80,
            pse_belief_gate_full_step=160,
            pse_belief_entropy_high=0.90,
            pse_belief_entropy_low=0.65,
            pse_belief_uniform_mix_high=0.75,
            pse_belief_uniform_mix_low=0.35,
        ),
        "changed_mechanisms": ["exec_cost_reference", "gated_belief"],
        "screening_role": "controlled belief reintroduction screen",
    },
    "ch3_v3_no_belief_low_exec": {
        "label": "ch3_v3_no_belief_low_exec",
        "base_method": "ch3_pse_no_belief",
        "description": "no-belief reference with lower execution-cost weight",
        "config_overrides": dict(_COMMON_OVERRIDES, pse_exec_cost_weight=0.08),
        "changed_mechanisms": ["exec_cost_reference", "execution_cost_weight"],
        "screening_role": "execution-cost strength screen",
    },
    "ch3_v3_no_belief_low_exec_no_standby": {
        "label": "ch3_v3_no_belief_low_exec_no_standby",
        "base_method": "ch3_pse_no_belief",
        "description": "no-belief reference with lower execution-cost weight and standby disabled",
        "config_overrides": dict(
            _COMMON_OVERRIDES,
            pse_exec_cost_weight=0.08,
            pse_use_standby=False,
        ),
        "changed_mechanisms": [
            "exec_cost_reference", "execution_cost_weight", "standby"
        ],
        "screening_role": "execution-cost and standby coupling screen",
    },
}


def get_ch3_efficiency_v3_candidate(label):
    if label not in CH3_EFFICIENCY_V3_SCREEN_REGISTRY:
        raise ValueError(
            f"unknown v3 screening candidate={label!r}; expected "
            f"{CH3_EFFICIENCY_V3_SCREEN_METHODS}"
        )
    return deepcopy(CH3_EFFICIENCY_V3_SCREEN_REGISTRY[label])


def resolve_ch3_efficiency_v3_config(label):
    """Resolve a candidate explicitly from the frozen efficiency-v2 base config."""

    from train import CH3_EFFICIENCY_V2, get_ch3_method_config

    entry = get_ch3_efficiency_v3_candidate(label)
    config = get_ch3_method_config(entry["base_method"], protocol=CH3_EFFICIENCY_V2)
    config.update(entry["config_overrides"])
    config["reward_profile"] = "task_efficiency_v2"
    return config


def config_diff(left, right):
    keys = sorted(set(left) | set(right))
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def validate_v3_candidate_registry():
    """Raise on any unregistered or stealth candidate difference."""

    if tuple(CH3_EFFICIENCY_V3_SCREEN_REGISTRY) != CH3_EFFICIENCY_V3_SCREEN_METHODS:
        raise RuntimeError("v3 registry order does not match candidate tuple")
    configs = {
        label: resolve_ch3_efficiency_v3_config(label)
        for label in CH3_EFFICIENCY_V3_SCREEN_METHODS
    }
    reference = configs["ch3_v3_no_belief_reference"]
    allowed = {
        "ch3_v3_no_belief_no_standby": {"pse_use_standby"},
        "ch3_v3_no_belief_low_exec": {"pse_exec_cost_weight"},
        "ch3_v3_no_belief_low_exec_no_standby": {
            "pse_exec_cost_weight", "pse_use_standby"
        },
    }
    for label, expected in allowed.items():
        actual = set(config_diff(reference, configs[label]))
        if actual != expected:
            raise RuntimeError(f"{label} diff mismatch: expected={expected}, actual={actual}")
    full = configs["ch3_v3_full_reference"]
    gated_allowed = {
        "pse_use_gated_belief", "pse_belief_weight", "pse_belief_weight_max",
        "pse_belief_gate_start_step", "pse_belief_gate_full_step",
        "pse_belief_entropy_high", "pse_belief_entropy_low",
        "pse_belief_uniform_mix_high", "pse_belief_uniform_mix_low",
    }
    actual = set(config_diff(full, configs["ch3_v3_gated_belief"]))
    if actual != gated_allowed:
        raise RuntimeError(f"gated-belief diff mismatch: {actual}")
    if any(config["run_type"] != "learning" for config in configs.values()):
        raise RuntimeError("all v3 screening candidates must be learning runs")
    return configs
