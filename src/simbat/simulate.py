from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# =====================================================================
# CURRENT POLICIES
# =====================================================================


class DischargePolicyTemplate:
    def __call__(self, soc: float, t: float) -> float:
        """Returns the current values at time `t` for the given SoC values."""


class ConstantCurrentDischarge:
    def __init__(self, current_value: float) -> None:
        self.current_value = current_value

    def __call__(self, soc: float, t: float) -> float:
        return self.current_value


# =====================================================================
# OPEN-CIRCUIT VOLTAGE MODELS
# =====================================================================


class VOCModelTemplate(Protocol):
    def __call__(self, soc: ArrayLike) -> ArrayLike: ...


@dataclass(frozen=True)
class VOC_Bustos_Baeza:
    """https://doi.org/10.1016/j.engappai.2024.109925"""

    vL: float = 1.35531394
    v0: float = 4.12017677
    gamma: float = 0.13286143
    alpha: float = 0.16945463
    beta: float = 2.34538224

    def __call__(self, soc: float) -> float:
        a = (self.v0 - self.vL) * np.exp(self.gamma * (soc - 1))
        b = self.alpha * self.vL * (soc - 1)
        c = (
            (1 - self.alpha)
            * self.vL
            * (np.exp(-1 * self.beta) - np.exp(-1 * self.beta * np.sqrt(soc)))
        )

        return self.vL + a + b + c


# =====================================================================
# EQUIVALENT CIRCUIT MODELS
# =====================================================================


class EquivalentCircuitModelTemplate(Protocol):
    def v_terminals(self, voc: ArrayLike, current: ArrayLike) -> ArrayLike: ...


class ECMTheveninZeroOrder:
    def __init__(self, R: float = 0.1):
        self.R = R

    def v_terminals(self, voc: float, current: float) -> float:
        return voc + self.R * current


# =====================================================================
# SIMULATOR
# =====================================================================

type Distribution = Callable[[], float]


@dataclass(frozen=True)
class SimulationConfig:
    # TODO: add data validation (Pydantic?)
    current_policy: DischargePolicyTemplate = ConstantCurrentDischarge(-2.8 * 0.75)
    voc_model: VOCModelTemplate = VOC_Bustos_Baeza()
    ec_model: EquivalentCircuitModelTemplate = ECMTheveninZeroOrder()
    process_noise_distribution: Distribution = lambda: np.random.normal(0, 1e-3)
    measurement_noise_distribution: Distribution = lambda: np.random.normal(0, 1e-2)
    dt: float = 1.0
    nominal_capacity: float = 10080.0  # in Coulombs, i.e., 2.8Ah at 1.2V
    soc_0: float = 1.0
    t_0: float = 0.0
    v_cutoff: float = 2.5  # in Volts


@dataclass
class SimulationResult:
    times: np.ndarray
    soc_histories: np.ndarray
    voltage_histories: np.ndarray
    rul_probability: np.ndarray
    times_eod: np.ndarray

    def __len__(self) -> int:
        return len(self.times)

    @property
    def n_sim(self) -> int:
        return self.soc_histories.shape[1]

    def to_dataframe(self) -> pd.DataFrame:
        to_df = {
            "time": self.times,
            "rul_probability": self.rul_probability,
        }
        for i in range(self.n_sim):
            to_df[f"soc_sim_{i}"] = self.soc_histories[:, i]
            to_df[f"voltage_sim_{i}"] = self.voltage_histories[:, i]
            to_df[f"eod_reached_sim_{i}"] = self.times_eod[i] <= self.times

        return pd.DataFrame(to_df)


class _BatteryParticle:
    """Utility class to track the particle state during Monte Carlo simulatons."""

    id: int

    soc: float
    alive: bool

    process_noise_model: Distribution
    measurement_noise_model: Distribution
    ec_model: EquivalentCircuitModelTemplate

    def __init__(
        self,
        id: int,
        soc_init: float,
        ec_model: EquivalentCircuitModelTemplate,
    ):
        self.id = id
        self.ec_model = ec_model

        self.soc = soc_init

        self.alive = True

    def update_state(
        self,
        I_t: ArrayLike,
        Q_t: ArrayLike,
        dt: float,
        perturbation: float = 0.0,
    ) -> None:
        """Returns SoC_{t+1}."""
        self.soc = np.clip(
            self.soc + I_t * dt / Q_t + perturbation,
            0,
            1,
        )

    def measure_voltage(
        self,
        voc_t: float,
        I_t: float,
        perturbation: float = 0.0,
    ) -> float:
        """Returns the voltage at the terminals at time t."""
        return self.ec_model.v_terminals(voc_t, I_t) + perturbation


def simulate_constant_capacity_simple(
    n_sim: int, config: SimulationConfig
) -> SimulationResult:
    """Simulates battery discharge of n_sim particles, following a constant capacity model."""
    if n_sim <= 0:
        raise ValueError("n_sim must be greater than 0.")

    # Initialization
    particles = [
        _BatteryParticle(
            id=i,
            soc_init=config.soc_0,
            ec_model=config.ec_model,
        )
        for i in range(n_sim)
    ]

    times_eod = np.empty(shape=(n_sim,))
    times = []  # Final shape: (n_time_steps,)
    soc_histories = []  # Final shape: (n_time_steps, n_sim)
    voltage_histories = []  # Final shape: (n_time_steps, n_sim)
    rul_probability = []  # Final shape: (n_time_steps,)

    def append_to_histories(
        t: float, particles: list[_BatteryParticle], new_deaths: int
    ) -> None:
        """Update the histories at time t, based on the particles SoC and alive status."""
        times.append(t)
        soc_histories.append(np.array([p.soc for p in particles]))
        voltage_histories.append(
            np.array(
                [
                    p.measure_voltage(
                        voc_t=config.voc_model(p.soc),
                        I_t=config.current_policy(p.soc, t),
                    )
                    for p in particles
                ]
            )
        )
        rul_probability.append(new_deaths / n_sim)

    t = config.t_0
    append_to_histories(
        t, particles, new_deaths=0
    )  # assumes all particles alive at t_0

    keep = True

    # Evolve the particles until all of them are dead.
    while keep:
        t += config.dt
        new_deaths = 0
        for p in particles:
            if p.alive:
                # update states
                I_t = config.current_policy(p.soc, t)
                state_perturbation = config.process_noise_distribution()
                p.update_state(
                    I_t,
                    config.nominal_capacity,
                    config.dt,
                    perturbation=state_perturbation,
                )

                # update alive-ness and count new deaths (p_RUL)
                voc_t = config.voc_model(p.soc)
                measurement_perturbation = config.measurement_noise_distribution()
                v_t = p.measure_voltage(
                    voc_t, I_t, perturbation=measurement_perturbation
                )

                if v_t <= config.v_cutoff:
                    p.alive = False
                    times_eod[p.id] = t
                    new_deaths += 1

        append_to_histories(t, particles, new_deaths=new_deaths)

        keep = any(p.alive for p in particles)

    # Compose the result and return it
    return SimulationResult(
        times=np.array(times),
        soc_histories=np.stack(soc_histories),
        voltage_histories=np.stack(voltage_histories),
        rul_probability=np.array(rul_probability),
        times_eod=times_eod,
    )


def join_simulation_results(results: list[SimulationResult]) -> SimulationResult:
    """Joins the results of multiple simulations into a single result, by averaging the probabilities and concatenating the SoC and voltage histories."""

    longest_time_sequence, max_time_sequence_length = max(
        enumerate(len(r) for r in results), key=lambda x: x[1]
    )

    # Array padding and add extra dimension.
    padded_p = [
        # pad at the end
        np.pad(r.rul_probability, (0, max_time_sequence_length - len(r)), "constant")
        for r in results
    ]

    # Compute mean probablity RUL distribution across simulations.
    joined_probabilities = np.sum(padded_p, 0) / len(results)

    # Extend the SoC histories with the last value
    padded_soc = [
        np.pad(
            r.soc_histories, ((0, max_time_sequence_length - len(r)), (0, 0)), "edge"
        )
        for r in results
    ]
    # Concatenate the SoC histories across simulations, resulting in a shape of (max(n_time_steps), n_simulations * n_simulators).
    joined_soc_histories = np.concatenate(padded_soc, 1)

    # Extend the voltage histories with the last value
    padded_v = [
        np.pad(
            r.voltage_histories,
            ((0, max_time_sequence_length - len(r)), (0, 0)),
            "edge",
        )
        for r in results
    ]
    joined_voltage_histories = np.concatenate(padded_v, 1)

    return SimulationResult(
        times=results[longest_time_sequence].times,
        soc_histories=joined_soc_histories,
        voltage_histories=joined_voltage_histories,
        rul_probability=joined_probabilities,
        times_eod=np.concatenate([r.times_eod for r in results]),
    )
