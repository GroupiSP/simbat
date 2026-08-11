from __future__ import annotations

from pathlib import Path

import numpy as np

import simbat as sb


def main():
    """Shows a simple simulation with a custom discharge policy."""

    rng = np.random.default_rng(seed=42)

    policies = [
        sb.simulate.ConstantCurrentDischarge(current_value=-1.0),
        sb.simulate.ConstantCurrentDischarge(current_value=-2.0),
        sb.simulate.ConstantCurrentDischarge(current_value=-3.0),
    ]

    def policy_choice_distribution() -> int:
        return rng.choice(len(policies), p=[0.2, 0.4, 0.4])

    sim_config = sb.SimulationConfig(
        current_policies=policies,
        policy_choice_distribution=policy_choice_distribution,
        dt=50.0,
    )

    sim_results = sb.simulate_constant_capacity_simple(n_sim=10, config=sim_config)

    df = sim_results.to_dataframe()
    df.to_csv(Path().cwd() / "simulation_results.csv", index=False)


if __name__ == "__main__":
    main()
