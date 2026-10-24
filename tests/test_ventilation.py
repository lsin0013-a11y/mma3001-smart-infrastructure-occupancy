"""Tests for smart_infra.ventilation."""

import numpy as np
import pytest

from smart_infra.ventilation import assess, required_airflow, simulate_co2, steady_state_co2


def test_required_airflow_scales_with_people():
    assert list(required_airflow([0, 1, 2.5])) == [0.0, 10.0, 25.0]
    with pytest.raises(ValueError):
        required_airflow([-1])


def test_steady_state_at_10_litres_per_person_is_940_ppm():
    assert steady_state_co2() == pytest.approx(940.0)


def test_exact_solution_matches_analytical_curve():
    # 4 people at 40 L/s in 81 m3 for 2 h, from outdoor level:
    # C(t) = C_ss + (420 - C_ss) exp(-Q t / V).
    n, q, v, dt = 4, 40.0, 81.0, 60.0
    steps = 120
    co2 = simulate_co2(np.full(steps, n), np.full(steps, q), dt, volume=v)
    t = dt * np.arange(1, steps + 1)
    c_ss = 420 + 0.0052 * n / q * 1e6
    expected = c_ss + (420 - c_ss) * np.exp(-(q / 1000) * t / v)
    np.testing.assert_allclose(co2, expected, rtol=1e-12)


def test_exact_solution_does_not_depend_on_step_size():
    people, airflow = np.full(60, 3.0), np.full(60, 30.0)
    coarse = simulate_co2(people, airflow, dt=60.0)
    fine = simulate_co2(np.repeat(people, 6), np.repeat(airflow, 6), dt=10.0)
    assert coarse[-1] == pytest.approx(fine[-1], rel=1e-12)


def test_euler_error_shrinks_linearly_with_step():
    # Explicit Euler is first-order: halving dt roughly halves the error.
    def final_error(dt):
        steps = int(3600 / dt)
        args = (np.full(steps, 5.0), np.full(steps, 50.0), dt)
        return abs(simulate_co2(*args, method="euler")[-1] - simulate_co2(*args)[-1])

    ratio = final_error(120.0) / final_error(60.0)
    assert ratio == pytest.approx(2.0, rel=0.15)


def test_euler_is_unstable_beyond_its_step_limit():
    # Stability needs Q dt / V < 2. 81 m3 at 300 L/s gives a limit of 540 s.
    people, airflow = np.full(30, 2.0), np.full(30, 300.0)
    stable = simulate_co2(people, airflow, dt=300.0, method="euler")
    unstable = simulate_co2(people, airflow, dt=900.0, method="euler")
    assert np.all(np.isfinite(stable)) and stable.max() < 2000
    assert np.abs(np.diff(unstable[-5:])).min() > 1000  # growing oscillation


def test_no_people_and_no_air_keeps_co2_constant():
    co2 = simulate_co2(np.zeros(10), np.zeros(10), dt=60.0, initial=800.0)
    assert np.all(co2 == 800.0)


def test_simulate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulate_co2(np.ones(3), np.ones(2), dt=60.0)
    with pytest.raises(ValueError):
        simulate_co2(np.ones(3), np.ones(3), dt=60.0, method="rk4")


def test_assess_counts_person_hours_over_limit():
    people = np.array([0, 2, 2, 0])
    airflow = np.array([0, 20, 20, 0])
    co2 = np.array([500, 900, 1100, 1200])
    result = assess(people, airflow, co2, dt=1800.0)
    assert result["person_hours_over_limit"] == pytest.approx(1.0)  # 2 people x 0.5 h
    assert result["share_occupied_time_over_limit"] == pytest.approx(0.5)
    assert result["air_m3"] == pytest.approx(72.0)  # 40 L/s x 1800 s
    assert result["peak_ppm"] == 1100
