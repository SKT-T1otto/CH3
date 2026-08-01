from __future__ import annotations

import pytest

from train import (
    CH3_EFFICIENCY_V2,
    CH3_PILOT_V1,
    CONTROLLER_ONLY_METHODS,
    build_ch3_runtime,
    train_and_evaluate_method,
)


@pytest.mark.parametrize("protocol", (CH3_PILOT_V1, CH3_EFFICIENCY_V2))
def test_controller_only_methods_create_no_learning_runtime_or_checkpoint(
    tmp_path, protocol
):
    protocol_root = tmp_path / protocol
    for method in CONTROLLER_ONLY_METHODS:
        runtime = build_ch3_runtime(
            method, seed=1, max_steps=4, device="cpu", protocol=protocol
        )
        assert runtime.run_type == "controller_only"
        assert runtime.maddpg is None
        assert runtime.replay_buffer is None

        summary, training_rows, evaluation_rows = train_and_evaluate_method(
            method,
            seed=1,
            episodes=2,
            max_steps=4,
            device="cpu",
            output_dir=protocol_root,
            pilot=True,
            scenario_manifest=None,
            protocol=protocol,
        )
        assert summary["protocol"] == protocol
        assert summary["checkpoint_path"] == "N/A"
        assert summary["checkpoint_sha256"] is None
        assert summary["checkpoint_paths"] == []
        assert summary["checkpoint_metadata"] is None
        assert summary["training_time"] == 0.0
        assert summary["actor_runtime_ms"] == 0.0
        assert training_rows == []
        assert evaluation_rows == []
        assert not list((protocol_root / method).rglob("*.pt"))
