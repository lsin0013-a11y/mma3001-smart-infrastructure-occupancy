"""smart_infra: utilities for the Monash Smart Infrastructure occupancy and environmental dataset.

This package provides reusable building blocks for loading, cleaning and
analysing time-stamped building sensor data (occupancy counts and
environmental measurements such as CO2, PM2.5/PM10, temperature and
humidity) collected from Monash Smart Infrastructure.

The project uses them to forecast whether room G.20 will be occupied in each
time interval (10 min, 30 min or 1 h) from calendar information, so that its
HVAC can be scheduled on demand
rather than on a fixed timetable. See the project README for the full problem
definition, inputs, outputs and current status.
"""

__version__ = "0.1.0"
