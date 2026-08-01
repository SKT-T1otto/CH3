from registry.ch3_efficiency_v3_registry import (
    CH3_EFFICIENCY_V3_SCREEN_METHODS,
    config_diff,
    resolve_ch3_efficiency_v3_config,
    validate_v3_candidate_registry,
)
from registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES
from train import CH3_EFFICIENCY_V2, CH3_PILOT_V1, _algorithm_config_hash, build_ch3_runtime


FROZEN_ALGORITHM_HASHES = {
    CH3_PILOT_V1: {
        "ch3_pheromone_prior": "587bdbf88759b78b292dd7f2d90a5738a26301b52159d0f48f881a7c46437073",
        "ch3_pheromone_rmaddpg": "3b7ec3f502a0c4a8a82f1e63efaaea8c9cd53cab3a073402f452e9ed851e4d06",
        "ch3_pse_rmaddpg": "40b14329564aa7abab86e06cc525f56872366ce8b0e28ca81686f3bc1653ecd0",
        "ch3_pse_no_belief": "1a0ac75bd8f0c344aea5d4100a9e6f53c58f8875aaf6cb854f1399ca8efe9aa4",
        "ch3_pse_no_exec_cost": "41a1757121fdc665242bd5032bac170a38c01f6af7cdbdd5f4a972ae2c2c2d33",
        "ch3_pse_no_standby": "aab541f37f23bec14c6df56ef73e03ee6f795c4c091f38ab98b3472e9e133665",
        "ch3_pse_no_residual": "d2f0dd9c90234f241086bfce09266e4d31ad24e5b73410cfc57f1473bd6c776d",
    },
    CH3_EFFICIENCY_V2: {
        "ch3_pheromone_prior": "d6aa94811866b06f259c8b393c6dc389255fa2da170e3a968006b8f5daf36242",
        "ch3_pheromone_rmaddpg": "f461176e3f29c4a980b2399243208843afaac26d7031554c9d36de0d685f5bf6",
        "ch3_pse_rmaddpg": "522725f5ade82681f634cfc332cc6e8130d92b6a6b14b43332f828b8a22840f1",
        "ch3_pse_no_belief": "6b7c9ee308c34f804aa2e570e42d64ace1d6a517e7415d7f25a867696f8d32c3",
        "ch3_pse_no_exec_cost": "5492483274f66616c0f0c247715889f5a8df535101a323493bd712cbb96c5094",
        "ch3_pse_no_standby": "5d2f76daa4602ba433b011782103d317dbfb38b6e86aca7effcbdbe17174a075",
        "ch3_pse_no_residual": "31165d0369137774dec9bf7969fd6ebc2197a2ed2fad2845c2489d0767d5cb59",
    },
}


def test_six_learning_candidates_are_isolated_from_formal_registry():
    configs = validate_v3_candidate_registry()
    assert len(configs) == 6
    assert not (set(configs) & set(ACTIVE_CH3_FINAL_EXPERIMENT_MODES))
    assert all(config["run_type"] == "learning" for config in configs.values())
    assert all(config["pse_exec_cost_reference_mode"] == "fixed_initial_wait_point" for config in configs.values())


def test_v1_v2_algorithm_config_hashes_match_pre_v3_snapshot():
    actual = {}
    for protocol in (CH3_PILOT_V1, CH3_EFFICIENCY_V2):
        actual[protocol] = {}
        for method in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
            runtime = build_ch3_runtime(
                method, seed=1, max_steps=4, device="cpu", replay_size=8,
                protocol=protocol,
            )
            actual[protocol][method] = _algorithm_config_hash(runtime.config)
    assert actual == FROZEN_ALGORITHM_HASHES


def test_registered_reconstructions_have_only_declared_relative_diffs():
    configs = {label: resolve_ch3_efficiency_v3_config(label) for label in CH3_EFFICIENCY_V3_SCREEN_METHODS}
    base = configs["ch3_v3_no_belief_reference"]
    assert set(config_diff(base, configs["ch3_v3_no_belief_no_standby"])) == {"pse_use_standby"}
    assert set(config_diff(base, configs["ch3_v3_no_belief_low_exec"])) == {"pse_exec_cost_weight"}
    assert set(config_diff(base, configs["ch3_v3_no_belief_low_exec_no_standby"])) == {"pse_use_standby", "pse_exec_cost_weight"}


def test_gated_candidate_preserves_reward_rl_and_residual_fields():
    full = resolve_ch3_efficiency_v3_config("ch3_v3_full_reference")
    gated = resolve_ch3_efficiency_v3_config("ch3_v3_gated_belief")
    protected = (
        "reward_profile", "reward_scale", "optimizer", "hidden_dim", "lr_actor",
        "lr_critic", "gamma", "tau", "batch_size", "replay_size", "initial_sigma",
        "residual_scale_search", "residual_scale_executor", "residual_action_reg",
    )
    assert all(full[key] == gated[key] for key in protected)
    assert gated["pse_belief_weight"] == 0.0
    assert gated["pse_belief_weight_max"] == 0.25
