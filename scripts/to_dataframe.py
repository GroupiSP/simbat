from __future__ import annotations

from pathlib import Path

import lib_eod_simulation as les


def main():
    """Shows a simple simulation with a custom discharge policy."""

    sim_config = les.SimulationConfig(dt=200.0)

    sim_results = les.simulate_constant_capacity_simple(n_sim=100, config=sim_config)

    df = sim_results.to_dataframe()
    df.to_csv(Path().cwd() / "simulation_results.csv", index=False)


if __name__ == "__main__":
    main()
