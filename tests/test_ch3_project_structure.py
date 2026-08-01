from __future__ import annotations

import inspect
from pathlib import Path

from env import UAVEnv
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES
from train import CH3_METHOD_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOVED_SYMBOLS = (
    "legacy_" + "ch5", "Dynamic" + "CommGraph", "Semantic" + "MessageEncoder",
    "VOI" + "Selector", "RoleAware" + "Topology", "Teammate" + "Predictor",
    "Belief" + "Fusion", "ChannelAware" + "Actor",
)
REMOVED_COMM_MODULES = (
    "comm_" + "graph", "semantic_" + "message", "voi_" + "selector",
    "role_" + "topology", "teammate_" + "predictor", "belief_" + "fusion",
    "normal_" + "comm", "weak_" + "comm", "severe_" + "comm",
    "robust_" + "full", "apply_" + "uploaded_update",
)


def _python_source():
    paths = []
    for path in PROJECT_ROOT.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if "__pycache__" in relative.parts:
            continue
        if relative.parts and relative.parts[0] == "data":
            continue
        paths.append(path)
    assert paths, "source scan unexpectedly found no Python files"
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_comm_package_contains_only_basic_communication():
    assert sorted(path.name for path in (PROJECT_ROOT / "comm").glob("*.py")) == ["__init__.py", "basic_communication.py"]


def test_no_removed_communication_modules_or_symbols():
    source = _python_source()
    for token in REMOVED_SYMBOLS + REMOVED_COMM_MODULES:
        assert token not in source


def test_env_constructor_is_ch3_only():
    parameters = set(inspect.signature(UAVEnv.__init__).parameters)
    forbidden = ("comm", "snr", "semantic", "voi", "topology", "reconnect", "prediction", "reliability", "quarantine", "channel", "tail", "disturbance", "noise", "actuator_lag")
    assert not {name for name in parameters if any(token in name.lower() for token in forbidden)}
    env = UAVEnv(return_numpy=False, max_steps=4)
    assert env.communication_mode == "ch3_fixed_reliable"
    assert env.communication_model_id == "fixed_reliable_one_step_v1"
    assert env.obs_dim == 28


def test_registry_contains_only_seven_chapter3_methods():
    assert tuple(CH3_METHOD_CONFIGS) == ACTIVE_CH3_FINAL_EXPERIMENT_MODES
    assert len(ACTIVE_CH3_FINAL_EXPERIMENT_MODES) == 7
