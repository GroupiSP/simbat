from __future__ import annotations

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

import simbat as sb


@pytest.fixture(scope="session")
def constant_current_discharge() -> sb.simulate.DischargePolicyTemplate:
    return sb.simulate.ConstantCurrentDischarge(current_value=-1.0)


@pytest.fixture(scope="session")
def voc_bustos_baeza() -> sb.simulate.VOCModelTemplate:
    return sb.simulate.VOC_Bustos_Baeza()


@pytest.fixture(scope="session")
def ecm_thevenin_zero_order() -> sb.simulate.EquivalentCircuitModelTemplate:
    return sb.simulate.ECMTheveninZeroOrder(R=0.1)


@pytest.fixture
def mock_simulation_results() -> sb.SimulationResult:
    time = np.array([0, 1, 2, 3, 4])
    soc = np.random.rand(5, 10)  # 5 time steps, 10 simulations
    voltage = np.random.rand(5, 10) * 4 + 2.5  # random voltages between 2.5V and 6.5V
    p_rul = np.random.rand(5)  # random probabilities for each time step
    p_rul /= p_rul.sum()  # normalize to sum to 1
    time_eod = (
        np.random.rand(10) * 10000
    )  # random EoD times between 0 and 10,000 seconds

    return sb.SimulationResult(
        times=time,
        soc_histories=soc,
        voltage_histories=voltage,
        rul_probability=p_rul,
        times_eod=time_eod,
    )


@pytest.fixture(scope="session")
def mock_simulation_config() -> sb.SimulationConfig:
    return sb.SimulationConfig(
        current_policies=[sb.simulate.ConstantCurrentDischarge(current_value=-1.0)],
        policy_choice_distribution=lambda: 0,
        voc_model=sb.simulate.VOC_Bustos_Baeza(),
        ec_model=sb.simulate.ECMTheveninZeroOrder(R=0.1),
        process_noise_distribution=lambda: np.random.normal(0, 0.0005),
        measurement_noise_distribution=lambda: np.random.normal(0, 0.0005),
        dt=100.0,
        nominal_capacity=10000.0,
        soc_0=1.0,
        t_0=0.0,
        v_cutoff=2.5,
    )


@hp.given(soc=st.floats(0, 1))
def test_constant_discharge_values_different_soc(constant_current_discharge, soc):
    t = 1.0
    assert (
        constant_current_discharge(soc, t) == constant_current_discharge.current_value
    )


@hp.given(t=st.floats(0, 10))
def test_constant_discharge_values_different_time(constant_current_discharge, t):
    soc = 1.0
    assert (
        constant_current_discharge(soc, t) == constant_current_discharge.current_value
    )


@hp.given(soc=st.floats(0, 1))
def test_voc_bustos_baeza_min(voc_bustos_baeza, soc):
    assert voc_bustos_baeza(soc) >= voc_bustos_baeza.vL


@hp.given(soc=st.floats(0, 1))
def test_voc_bustos_baeza_max(voc_bustos_baeza, soc):
    assert voc_bustos_baeza(soc) <= voc_bustos_baeza.v0


def test_voc_bustos_baeza_monotonic(voc_bustos_baeza):
    soc_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    voc_values = [voc_bustos_baeza(soc) for soc in soc_values]

    assert all(
        (earlier <= later for earlier, later in zip(voc_values[:-1], voc_values[1:]))
    )


def test_ecm_thevenin_zero_order(ecm_thevenin_zero_order):
    vocs = (3.0, 3.5, 4.0)
    current = -1.0

    true_v_terminals = (2.9, 3.4, 3.9)

    assert np.allclose(
        [ecm_thevenin_zero_order.v_terminals(voc, current) for voc in vocs],
        true_v_terminals,
    )


def test_simulation_result_length(mock_simulation_results):
    assert len(mock_simulation_results) == 5


def test_simulation_result_particles_have_same_history_length(mock_simulation_results):
    assert (
        mock_simulation_results.soc_histories.shape[0]
        == mock_simulation_results.voltage_histories.shape[0]
        == 5
    )


def test_simulation_result_to_dataframe_length(mock_simulation_results):
    df = mock_simulation_results.to_dataframe()
    assert len(df) == len(mock_simulation_results.times)


def test_expected_columns_of_dataframe(mock_simulation_results):
    df = mock_simulation_results.to_dataframe()
    expected_columns = (
        {"time", "rul_probability"}.union(
            {f"soc_sim_{i}" for i in range(mock_simulation_results.n_sim)}
        )
        .union({f"voltage_sim_{i}" for i in range(mock_simulation_results.n_sim)})
        .union({f"eod_reached_sim_{i}" for i in range(mock_simulation_results.n_sim)})
    )
    assert set(df.columns) == expected_columns


@hp.given(current=st.floats(-10, 0), capacity=st.floats(1, 10000))
def test_battery_particle_update_state_monotonic_when_negative_current(
    current, capacity
):
    particle = sb.simulate._BatteryParticle(
        id=0,
        soc_init=1.0,
        ec_model=ecm_thevenin_zero_order,
    )
    socs = [particle.soc]

    for _ in range(10):
        particle.update_state(current, capacity, dt=1.0, perturbation=0.0)
        socs.append(particle.soc)

    assert all(earlier >= later for earlier, later in zip(socs[:-1], socs[1:]))


@hp.given(current=st.floats(-10, 10), voc=st.floats(3, 5))
def test_battery_particle_measure_voltage_matches_ecm_when_no_noise(
    ecm_thevenin_zero_order, current, voc
):
    particle = sb.simulate._BatteryParticle(
        id=0,
        soc_init=1.0,
        ec_model=ecm_thevenin_zero_order,
    )

    expected_voltage = ecm_thevenin_zero_order.v_terminals(voc, current)

    measured_voltage = particle.measure_voltage(voc, current, perturbation=0.0)

    assert np.isclose(measured_voltage, expected_voltage)


@hp.given(n_sim=st.integers(-100, 0))
def test_simulate_constant_capacity_simple_invalid_n_sim(mock_simulation_config, n_sim):
    with pytest.raises(ValueError):
        sb.simulate_constant_capacity_simple(n_sim=n_sim, config=mock_simulation_config)


def test_simulate_constant_capacity_simple_output_shape(mock_simulation_config):
    result = sb.simulate_constant_capacity_simple(
        n_sim=5, config=mock_simulation_config
    )
    assert (
        result.times.shape[0]
        == result.soc_histories.shape[0]
        == result.voltage_histories.shape[0]
        == result.rul_probability.shape[0]
    )
    assert result.soc_histories.shape[1] == result.voltage_histories.shape[1] == 5


def test_simulate_constant_capacity_simple_unit_probability(mock_simulation_config):
    result = sb.simulate_constant_capacity_simple(
        n_sim=5, config=mock_simulation_config
    )
    assert np.isclose(result.rul_probability.sum(), 1.0)


def test_simulate_constant_capacity_simple_soc_zero():
    config = sb.SimulationConfig(
        current_policies=[sb.simulate.ConstantCurrentDischarge(current_value=-1.0)],
        policy_choice_distribution=lambda: 0,
        voc_model=sb.simulate.VOC_Bustos_Baeza(),
        ec_model=sb.simulate.ECMTheveninZeroOrder(R=0.1),
        process_noise_distribution=lambda: 0.0,
        measurement_noise_distribution=lambda: 0.0,
        dt=100.0,
        nominal_capacity=10000.0,
        soc_0=0.0,
        t_0=0.0,
        v_cutoff=2.5,
    )
    result = sb.simulate_constant_capacity_simple(n_sim=5, config=config)
    assert np.all(result.soc_histories == 0.0)
    # even though the particles are dead at t_0, the algorithms assumes they are alive.
    assert len(result) == 2


@pytest.mark.xfail(
    reason="Fails because t_0 is purely a reference since the current is constant in time."
)
def test_simulate_constant_capacity_simple_t_zero_gt_eod():
    config = sb.SimulationConfig(
        current_policies=[sb.simulate.ConstantCurrentDischarge(current_value=-1.0)],
        policy_choice_distribution=lambda: 0,
        voc_model=sb.simulate.VOC_Bustos_Baeza(),
        ec_model=sb.simulate.ECMTheveninZeroOrder(R=0.1),
        process_noise_distribution=lambda: 0.0,
        measurement_noise_distribution=lambda: 0.0,
        dt=100.0,
        nominal_capacity=10000.0,
        soc_0=1.0,
        t_0=20_000.0,
        v_cutoff=2.5,
    )
    result = sb.simulate_constant_capacity_simple(n_sim=5, config=config)
    assert np.all(result.soc_histories == 0.0)
    assert len(result) == 1


def test_simulate_constant_capacity_simple_dt_gt_eod():
    config = sb.SimulationConfig(
        current_policies=[sb.simulate.ConstantCurrentDischarge(current_value=-1.0)],
        policy_choice_distribution=lambda: 0,
        voc_model=sb.simulate.VOC_Bustos_Baeza(),
        ec_model=sb.simulate.ECMTheveninZeroOrder(R=0.1),
        process_noise_distribution=lambda: 0.0,
        measurement_noise_distribution=lambda: 0.0,
        dt=10_000.0,
        nominal_capacity=10000.0,
        soc_0=1.0,
        t_0=0.0,
        v_cutoff=2.5,
    )
    result = sb.simulate_constant_capacity_simple(n_sim=5, config=config)
    assert len(result) == 2


def test_simulate_constant_capacity_simple_last_time_eq_t_eod(mock_simulation_config):
    result = sb.simulate_constant_capacity_simple(
        n_sim=5, config=mock_simulation_config
    )
    assert np.isclose(result.times[-1], result.times_eod.max())
