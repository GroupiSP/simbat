import matplotlib.pyplot as plt
import numpy as np

from .simulate import SimulationResult


def plot_soc_results(ax: plt.Axes, results: list[SimulationResult]) -> None:
    color = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    for i, (r, c) in enumerate(zip(results, color)):
        ax.plot(r.times, r.soc_histories.mean(axis=1), color=c, label=f"Scenario {i}")
        ax.fill_between(
            r.times,
            np.percentile(r.soc_histories, 5, axis=1),
            np.percentile(r.soc_histories, 95, axis=1),
            alpha=0.2,
            color=c,
        )

    ax.legend()
    ax.grid()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SoC [/]")


def plot_voltage_results(ax: plt.Axes, results: list[SimulationResult]) -> None:
    color = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    for i, (r, c) in enumerate(zip(results, color)):
        ax.plot(
            r.times, r.voltage_histories.mean(axis=1), color=c, label=f"Scenario {i}"
        )
        ax.fill_between(
            r.times,
            np.percentile(r.voltage_histories, 5, axis=1),
            np.percentile(r.voltage_histories, 95, axis=1),
            alpha=0.2,
            color=c,
        )

    ax.legend()
    ax.grid()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")


def plot_rul_results(ax: plt.Axes, results: list[SimulationResult]) -> None:
    color = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    for i, (r, c) in enumerate(zip(results, color)):
        ax.fill_between(
            r.times, r.rul_probability, color=c, label=f"Scenario {i}", alpha=0.4
        )

    ax.legend()
    ax.grid()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("P(RUL)")


def plot_rul_bars(ax: plt.Axes, results: list[SimulationResult]) -> None:
    color = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    for i, (r, c) in enumerate(zip(results, color)):
        ax.bar(
            x=r.times,
            height=r.rul_probability,
            width=(r.times[-1] - r.times[0]) / 100,
            color=c,
            label=f"Scenario {i}",
            alpha=0.4,
        )

    ax.legend()
    ax.grid()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("P(RUL)")


def expected_RUL(results: SimulationResult):
    return np.sum(results.rul_probability * results.times)


def variance_RUL(results: SimulationResult):
    mu = expected_RUL(results)

    time_diff = (results.times - mu) * (results.times - mu)

    return np.sum(time_diff * results.rul_probability)
