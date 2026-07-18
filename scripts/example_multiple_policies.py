from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt

import simbat as sb


def main():
    """Shows a simulation with two possible discharge policies, picked with a choice distribution."""

    rng = np.random.default_rng(0)

    def discharge_policy_0(soc: Any, t: float) -> float:
        return -2

    def discharge_policy_1(soc: Any, t: float) -> float:
        if t < 4000:
            return -1
        else:
            return -2

    sim_config = sb.SimulationConfig(
        current_policies=[discharge_policy_0, discharge_policy_1],
        policy_choice_distribution=lambda: rng.choice([0, 1], p=[0.3, 0.7]),
        voc_model=sb.VOC_Bustos_Baeza(),
        ec_model=sb.ECMTheveninZeroOrder(),
        process_noise_distribution=lambda: rng.normal(0, 0.001),
        measurement_noise_distribution=lambda: rng.normal(0, 0.01),
        dt=100.0,
    )

    sim_results = sb.simulate_constant_capacity_simple(n_sim=100, config=sim_config)

    _, axs = plt.subplots(1, 3, figsize=(10, 6))

    sb.plot_soc_results(axs[0], [sim_results])
    sb.plot_voltage_results(axs[1], [sim_results])
    sb.plot_rul_bars(axs[2], [sim_results])

    plt.show()


if __name__ == "__main__":
    main()
