from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt

import lib_eod_simulation as les


def main():
    """Shows a simple simulation with a custom discharge policy."""

    def discharge_policy(soc: Any, t: float) -> float:
        if t < 4000:
            return -1
        else:
            return -2

    sim_config = les.SimulationConfig(
        current_policy=discharge_policy,
        voc_model=les.VOC_Bustos_Baeza(),
        ec_model=les.ECMTheveninZeroOrder(),
        process_noise_distribution=lambda: np.random.normal(0, 0.01),
        measurement_noise_distribution=lambda: np.random.normal(0, 0.01),
        dt=100.0,
    )

    sim_results = les.simulate_constant_capacity_simple(n_sim=100, config=sim_config)

    _, axs = plt.subplots(1, 3, figsize=(10, 6))

    les.plot_soc_results(axs[0], [sim_results])
    les.plot_voltage_results(axs[1], [sim_results])
    les.plot_rul_bars(axs[2], [sim_results])

    plt.show()


# # Simple simulation with custom discharge profile
# battery_config = {
#     "Q": 2.8 * 3600,
#     "R": 0.1,
#     "voc_params": {
#         "vL": 1.35531394,
#         "v0": 4.12017677,
#         "gamma": 0.13286143,
#         "alpha": 0.16945463,
#         "beta": 2.34538224,
#     },
# }

# battery = les.BatteryModel(battery_config)


# simulator_config = {
#     "N_simu": 1,
#     "v_cut": 2.5,
#     "SoC_0": 0.8,
#     "dt": 1.0,
#     "omega_std": 0.0,
#     "eta_std": 0.0,
#     "I": Policy(),
#     "battery": battery,
# }

# sim = les.SimulatorSimple(simulator_config)
# sim.simulate()

# les.plot_soc_results([sim])
# les.plot_voltage_results([sim])
# les.plot_rul_results([sim])


# # %%

# # Case study with different constant current discharge and gaussian process and sensor noise
# disch_policies = []
# for i in [-1, -1.5, -2]:
#     disch_policies.append(les.ConstantCurrentDischarge(i))

# configs = [
#     {
#         "N_simu": 500,
#         "v_cut": 2.5,
#         "SoC_0": 0.8,
#         "dt": 1.0,
#         "omega_std": 0.0,
#         "eta_std": 0.0,
#         "I": policy,
#         "battery": battery,
#     }
#     for policy in disch_policies
# ]

# sim = les.SimulatorComplete(configs)
# sim.simulate()
# les.plot_soc_results(sim.sim_results)
# les.plot_rul_results(sim.sim_results)

# sim.join_results()
# les.plot_rul_results([sim])

# # %%

# configs = [
#     {
#         "N_simu": 100,
#         "v_cut": 2.5,
#         "SoC_0": soc_0,
#         "dt": 1.0,
#         "omega_std": 0.0005,
#         "eta_std": 0.0005,
#         "I": les.ConstantCurrentDischarge(-1),
#         "battery": battery,
#     }
#     for soc_0 in np.random.normal(0.7, 0.1, 50)
# ]

# sim = les.SimulatorComplete(configs)
# sim.simulate()
# sim.join_results()
# # plot_soc_results(sim.sim_results)
# # plot_rul_results(sim.sim_results)


# configs = [
#     {
#         "N_simu": 100,
#         "v_cut": 2.5,
#         "SoC_0": soc_0,
#         "dt": 1.0,
#         "omega_std": 0.0005,
#         "eta_std": 0.0005,
#         "I": les.ConstantCurrentDischarge(-1),
#         "battery": battery,
#     }
#     for soc_0 in np.random.normal(0.7, 0.05, 50)
# ]
# sim2 = les.SimulatorComplete(configs)
# sim2.simulate()
# sim2.join_results()


# configs = [
#     {
#         "N_simu": 100,
#         "v_cut": 2.5,
#         "SoC_0": soc_0,
#         "dt": 1.0,
#         "omega_std": 0.0005,
#         "eta_std": 0.0005,
#         "I": les.ConstantCurrentDischarge(-1),
#         "battery": battery,
#     }
#     for soc_0 in np.random.normal(0.7, 0.01, 50)
# ]
# sim3 = les.SimulatorComplete(configs)
# sim3.simulate()
# sim3.join_results()

# # plot_soc_results(sim.sim_results)
# # plot_rul_results(sim2.sim_results)

# les.plot_rul_results([sim, sim2, sim3])


# %%
# TODO: "Nice to have some voltage histories with dynamic operational conditions. Current"
# TODO: Compute actual RUL from Simulations
# TODO: "Implement continuous run probability score (CRPS)"
# %%

if __name__ == "__main__":
    main()
