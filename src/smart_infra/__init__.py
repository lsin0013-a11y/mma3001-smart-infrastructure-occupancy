"""smart_infra: occupancy-based ventilation for a room in Monash's Smart Infrastructure dataset.

The package turns the raw people-counter event stream for room G.20 into a
clean occupancy series, forecasts its headcount, and evaluates ventilation
strategies by simulating the room's CO2.

Modules, in pipeline order:

- :mod:`smart_infra.occupancy`: clean the events and integrate them into
  fixed intervals (held-state occupancy, exact time integration);
- :mod:`smart_infra.academic_calendar`: semester periods and public holidays;
- :mod:`smart_infra.forecasting`: leak-free forecasting inputs, baselines,
  models, tuning and metrics;
- :mod:`smart_infra.ventilation`: outdoor-air demand and the CO2 mass balance.

See the project README for the problem definition, inputs, outputs and how to
reproduce the results.
"""

__version__ = "0.2.0"
