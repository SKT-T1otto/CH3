"""Registry for the seven formal Chapter-3 experiment methods."""

ACTIVE_CH3_FINAL_EXPERIMENT_MODES = (
    "ch3_pheromone_prior",
    "ch3_pheromone_rmaddpg",
    "ch3_pse_rmaddpg",
    "ch3_pse_no_belief",
    "ch3_pse_no_exec_cost",
    "ch3_pse_no_standby",
    "ch3_pse_no_residual",
)

CONTROLLER_ONLY_METHODS = (
    "ch3_pheromone_prior",
    "ch3_pse_no_residual",
)

METHOD_DESCRIPTIONS = {
    "ch3_pheromone_prior": "Pheromone prior controller without a learned residual.",
    "ch3_pheromone_rmaddpg": "Pheromone prior with RMADDPG residual control.",
    "ch3_pse_rmaddpg": "Full Chapter-3 PSE-RMADDPG method.",
    "ch3_pse_no_belief": "PSE ablation without probabilistic belief scoring.",
    "ch3_pse_no_exec_cost": "PSE ablation without execution-cost scoring.",
    "ch3_pse_no_standby": "PSE ablation without dynamic executor standby.",
    "ch3_pse_no_residual": "Full PSE prior controller without learned residual.",
}


def assert_ch3_method(method):
    method = str(method)
    if method not in ACTIVE_CH3_FINAL_EXPERIMENT_MODES:
        raise ValueError(
            f"Unknown Chapter-3 method {method!r}; expected one of "
            f"{ACTIVE_CH3_FINAL_EXPERIMENT_MODES}"
        )
    return method
