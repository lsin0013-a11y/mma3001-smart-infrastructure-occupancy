"""Ventilation requirement and indoor CO2 for a mechanically ventilated room.

The required outdoor-air supply is the people present times a per-person rate
(AS 1668.2 gives 10 L/s per person for office-type use, the assumption here).

To judge a ventilation strategy, the room's CO2 concentration is simulated
with a well-mixed single-zone mass balance:

    V dC/dt = G N(t) + Q(t) (C_out - C)

with room volume V (m3), people N, CO2 generation G per person (m3/s),
outdoor-air supply Q (m3/s) and outdoor concentration C_out. Within a time
step N and Q are constant, so the equation is linear with constant
coefficients and has the exact solution

    C(t + dt) = C_ss + (C(t) - C_ss) exp(-Q dt / V),   C_ss = C_out + G N / Q

(or C(t) + G N dt / V when Q = 0). :func:`simulate_co2` uses this by default.
The explicit Euler method is also provided for comparison: it is only stable
while Q dt / V < 2, and only accurate when Q dt / V is much smaller than 1.
"""

from __future__ import annotations

import numpy as np

RATE_PER_PERSON = 10.0
"""Outdoor air per person, L/s (AS 1668.2, office-type use; an assumption)."""

CO2_GENERATION = 0.0052
"""CO2 generated per person, L/s: an adult doing office work (about 1.2 met;
Persily and de Jonge, 2017)."""

OUTDOOR_CO2 = 420.0
"""Outdoor CO2 concentration, ppm."""

CO2_LIMIT = 1000.0
"""Indoor CO2 level used as the air-quality threshold, ppm."""

ROOM_AREA = 300.0
"""Floor area of G.20, m2 (the student's estimate: about 20 m by 15 m)."""

DESIGN_QUANTILE = 0.999
"""Quantile of the occupied-time headcount that the design occupancy covers."""

ROOM_HEIGHT = 2.7
"""Ceiling height of G.20, m (assumed)."""

INFILTRATION_ACH = 0.3
"""Background outdoor air leaking in through the building envelope, air
changes per hour (assumed; the room has few operable windows). It flows even
when the ventilation is off."""


def infiltration_airflow(ach: float = INFILTRATION_ACH, volume: float = ROOM_AREA * ROOM_HEIGHT) -> float:
    """Convert air changes per hour to an airflow in L/s (volume x ACH / 3.6)."""
    return ach * volume / 3.6


def required_airflow(people: np.ndarray, rate_per_person: float = RATE_PER_PERSON) -> np.ndarray:
    """Outdoor air needed for the people present.

    Parameters
    ----------
    people : array_like
        Number of people (may be fractional, e.g. an interval's mean headcount).
    rate_per_person : float, default 10.0
        Outdoor air per person, L/s.

    Returns
    -------
    numpy.ndarray
        Required outdoor-air supply, L/s.

    Raises
    ------
    ValueError
        If any value of ``people`` is negative.
    """
    people = np.asarray(people, dtype=float)
    if (people < 0).any():
        raise ValueError("people must be non-negative")
    return rate_per_person * people


def design_occupancy(headcount: np.ndarray, quantile: float = DESIGN_QUANTILE) -> int:
    """Number of people a fixed ventilation design is sized for.

    The ``quantile`` of the headcount over occupied time, rounded up to whole
    people. Sizing for the very highest reading would let one brief crowd (or
    sensor glitch) set the airflow for every hour of the year; the 99.9th
    percentile covers all but the rarest peaks.

    Parameters
    ----------
    headcount : array_like
        Headcount per equal time step (e.g. 1-min means). Zeros (empty room)
        are ignored.
    quantile : float, default 0.999
        Share of occupied time the design covers, between 0 and 1.

    Returns
    -------
    int
        Design occupancy, people.

    Raises
    ------
    ValueError
        If ``quantile`` is outside (0, 1], any headcount is negative, or the
        room is never occupied.
    """
    headcount = np.asarray(headcount, dtype=float)
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if (headcount < 0).any():
        raise ValueError("headcount must be non-negative")
    occupied = headcount[headcount > 0]
    if occupied.size == 0:
        raise ValueError("the room is never occupied")
    return int(np.ceil(np.quantile(occupied, quantile)))


def fallback_airflow(usual_headcount: np.ndarray, design_people: float, threshold: float,
                     rate_per_person: float = RATE_PER_PERSON) -> np.ndarray:
    """Outdoor air to supply while the headcount sensor is known to be dead.

    Without a live reading the controller cannot see how many people are in
    the room, so the fallback is sized to be safe rather than efficient: full
    design air whenever the room is usually in use (its average headcount for
    that time of the week exceeds ``threshold``), and none otherwise.

    Parameters
    ----------
    usual_headcount : array_like
        Average headcount for each step's time of the week, from past data.
    design_people : float
        Design occupancy, people (see :func:`design_occupancy`).
    threshold : float
        Average headcount above which the room counts as usually in use.
    rate_per_person : float, default 10.0
        Outdoor air per person, L/s.

    Returns
    -------
    numpy.ndarray
        Outdoor-air supply, L/s: ``rate_per_person * design_people`` or 0.

    Raises
    ------
    ValueError
        If ``usual_headcount``, ``design_people`` or ``threshold`` is negative.
    """
    usual_headcount = np.asarray(usual_headcount, dtype=float)
    if (usual_headcount < 0).any() or design_people < 0 or threshold < 0:
        raise ValueError("usual_headcount, design_people and threshold must be non-negative")
    return np.where(usual_headcount > threshold, rate_per_person * design_people, 0.0)


def steady_state_co2(rate_per_person: float = RATE_PER_PERSON, generation: float = CO2_GENERATION,
                     outdoor: float = OUTDOOR_CO2) -> float:
    """CO2 level a room settles at when supplied exactly ``rate_per_person``.

    With Q = rate x N, the steady state C_out + G N / Q does not depend on N.
    At 10 L/s per person it is 420 + 0.0052 / 10 x 1e6 = 940 ppm, just under
    the 1000 ppm limit, which is why that rate is a common design value.
    """
    return outdoor + generation / rate_per_person * 1e6


def simulate_co2(
    people: np.ndarray,
    airflow: np.ndarray,
    dt: float,
    volume: float = ROOM_AREA * ROOM_HEIGHT,
    generation: float = CO2_GENERATION,
    outdoor: float = OUTDOOR_CO2,
    initial: float | None = None,
    method: str = "exact",
) -> np.ndarray:
    """Simulate indoor CO2 for piecewise-constant occupancy and airflow.

    Parameters
    ----------
    people : array_like
        People present in each time step.
    airflow : array_like
        Outdoor-air supply in each time step, L/s (same length as ``people``).
    dt : float
        Time step, seconds.
    volume : float, default 810.0
        Room volume, m3.
    generation : float, default 0.0052
        CO2 generated per person, L/s.
    outdoor : float, default 420.0
        Outdoor CO2, ppm.
    initial : float, optional
        CO2 at the start, ppm. Defaults to ``outdoor``.
    method : {"exact", "euler"}, default "exact"
        ``"exact"`` uses the analytical solution within each step;
        ``"euler"`` uses the explicit Euler method.

    Returns
    -------
    numpy.ndarray
        CO2 at the end of each step, ppm (same length as ``people``).

    Raises
    ------
    ValueError
        If the inputs differ in length, contain negative values, ``dt`` or
        ``volume`` is not positive, or ``method`` is unknown.
    """
    people = np.asarray(people, dtype=float)
    q = np.asarray(airflow, dtype=float) / 1000.0           # m3/s
    if people.shape != q.shape:
        raise ValueError("people and airflow must have the same length")
    if (people < 0).any() or (q < 0).any():
        raise ValueError("people and airflow must be non-negative")
    if dt <= 0 or volume <= 0:
        raise ValueError("dt and volume must be positive")
    if method not in ("exact", "euler"):
        raise ValueError(f"Unknown method '{method}'")

    source = generation / 1000.0 * people * 1e6               # ppm * m3/s added by occupants
    c = outdoor if initial is None else float(initial)
    out = np.empty(len(people))
    if method == "euler":
        for i in range(len(people)):
            c += dt * (source[i] + q[i] * (outdoor - c)) / volume
            out[i] = c
        return out

    # Exact solution of the linear ODE over each step; decay factor precomputed.
    decay = np.exp(-q * dt / volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        steady = np.where(q > 0, outdoor + source / q, np.inf)
    for i in range(len(people)):
        c = steady[i] + (c - steady[i]) * decay[i] if q[i] > 0 else c + source[i] * dt / volume
        out[i] = c
    return out


def assess(people: np.ndarray, airflow: np.ndarray, co2: np.ndarray, dt: float,
           limit: float = CO2_LIMIT, mask: np.ndarray | None = None) -> dict[str, float]:
    """Energy and air-quality measures of a ventilation strategy.

    Parameters
    ----------
    people, airflow, co2 : array_like
        People, outdoor-air supply (L/s) and simulated CO2 (ppm) per step.
    dt : float
        Time step, seconds.
    limit : float, default 1000.0
        CO2 threshold, ppm.
    mask : array_like of bool, optional
        Steps to include (e.g. excluding sensor outages). All if ``None``.

    Returns
    -------
    dict
        - ``air_m3``: total outdoor air supplied, m3 (proxy for fan and
          conditioning energy);
        - ``required_m3``: outdoor air needed at the per-person rate, m3;
        - ``person_hours_over_limit``: person-hours spent above ``limit``;
        - ``share_occupied_time_over_limit``: share of occupied time above it;
        - ``peak_ppm``: highest CO2 while occupied;
        - ``mean_occupied_ppm``: mean CO2 while occupied.
    """
    people, airflow, co2 = (np.asarray(a, dtype=float) for a in (people, airflow, co2))
    m = np.ones(len(people), bool) if mask is None else np.asarray(mask, bool)
    people, airflow, co2 = people[m], airflow[m], co2[m]
    occupied = people > 0
    over = co2 > limit
    return {
        "air_m3": float(airflow.sum() * dt / 1000.0),
        "required_m3": float(required_airflow(people).sum() * dt / 1000.0),
        "person_hours_over_limit": float((people * over).sum() * dt / 3600.0),
        "share_occupied_time_over_limit": float((over & occupied).sum() / max(occupied.sum(), 1)),
        "peak_ppm": float(co2[occupied].max()) if occupied.any() else float("nan"),
        "mean_occupied_ppm": float(co2[occupied].mean()) if occupied.any() else float("nan"),
    }
