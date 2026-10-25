# Decision log

The key decisions in this project, in the order they were made, with the evidence behind each. Every number comes from a script in `scripts/` and is reproduced in the results files linked here.

## 1. Problem framing

| Date | Decision | Reason |
|---|---|---|
| 2026-10-24 | Start with the simplest problem the data supports: is room G.20 occupied in each hour? | The environmental (CO2, temperature) data is unusable (see 2), but the people-counter data for G.20 is complete from March 2024. |
| 2026-10-24 | Reframe from heating/cooling schedules to **ventilation**. | Outdoor-air demand is proportional to headcount (AS 1668.2), and ventilation can follow occupancy within minutes, whereas heating and cooling take 30 to 75 minutes. Ventilation matters more for the people in the room than reaching a target temperature. |
| 2026-10-24 | Room parameters: about 30 m² (5 m × 6 m, student's estimate), 2.7 m ceiling (81 m³), 10 L/s per person. | G.20 was once the Monash Motorsport workshop and is now used mainly as an office/lab space. The area and height are assumptions; their effect is tested in the sensitivity analysis. |
| 2026-10-25 | Research question: can live headcount sensors, combined with short-term forecasting, provide demand-controlled ventilation without relying on CO2 sensors? | The CO2 sensors in the building are frozen, short-lived or implausible (see 2). |

## 2. Data

- **The environmental data was excluded.** No environmental sensor in the extract is in G.20; its listed CO2 sensors (ES1G01 to ES1G03) are missing. Of the five sensors that are present, three are frozen, one is live for only four months, and one shows CO2 falling during the day. See [`data_audit_findings.md`](data_audit_findings.md).
- **Every CO2 value in this project is modelled, not measured.** CO2 is modelled from the measured headcount with a mass balance. Figures and tables label it as modelled.
- **The people counters record presence, not bookings.** Reports arrive about every 2 s while someone is present and stop when the room is empty.
- **Suspect data is flagged, not deleted**, so each check can be tested in the sensitivity analysis:
  - duplicates;
  - headcounts above a capacity of 20 (up to 72 people were reported in 30 m²);
  - sensor outages (silences over 24 h that cover weekday working time);
  - detections between 01:00 and 05:00.

## 3. Occupancy series and resolution

- **Occupancy is treated as a held state.** Each report holds until the next one, for at most 5 minutes. The first method, which labelled an interval occupied only when a report fell inside it, did not converge as the resolution was refined: it fell to 13.4% at 10 s against a true 29.4%. The convergence study requested by the student exposed this. See [`resolution_convergence.csv`](resolution_convergence.csv).
- **Person-time per interval is integrated exactly**, using a cumulative sum and binary search. The mean headcount (0.853 people) is identical at every resolution, which serves as a correctness check.
- **Control interval: 5 minutes.** The room's air responds over about 20 to 30 minutes. Live-sensor control gives 20 person-hours above 1000 ppm at 5-minute control, 90 at 10 minutes, 421 at 30 minutes and 755 at 1 hour.

## 4. Forecast horizon: 10 minutes (changed from 30)

- 10 minutes ahead covers two delays: the live reading's delay (the current 5-minute interval has not finished) and the time the ventilation system takes to respond.
- Longer horizons are less accurate. The forecast MAE is 0.37 people at 10 minutes, 0.57 at 30 minutes and 0.68 at 1 hour.
- With a longer horizon the hybrid also uses more air (14.8% to 17.1% of the fixed schedule) for about the same breaches. See [`horizon_comparison.md`](horizon_comparison.md).

## 5. Final design: live headcount + 15% safety margin (the hybrid was rejected)

The hybrid strategy (the larger of the live reading and the forecast) at first looked like the forecast was anticipating arrivals: it halved the live sensor's breaches. Testing it against strategies without a forecast showed otherwise. Over the test period, at 5-minute control (see [`ventilation_results.md`](ventilation_results.md)):

| Strategy | Air vs fixed | Person-hours > 1000 ppm (modelled) |
|---|---|---|
| Live sensor | 13.1% | 19.9 |
| Larger of last two readings (no forecast) | 14.8% | 10.1 |
| Hybrid (live sensor + 10-min forecast) | 14.8% | 9.8 |
| **Live sensor + 15% margin** | **15.1%** | **1.6** |

- **The hybrid's gain comes from supplying more air, not from the forecast.** Ten minutes ahead, persistence ("same as now") is the most accurate forecast: MAE 0.349 people, against 0.366 for gradient boosting.
- **The margin was chosen on 2025, before the test period.** It is the smallest margin, in 5% steps, that cuts person-hours above the limit by at least 90%.
- **The forecasting work stays in the report as a tested, negative result.** The models were compared against baselines on a withheld test period.

## 6. Assumptions to keep in view

- **The fixed schedule is an assumed comparison**, not G.20's actual HVAC schedule. It is sized for 20 people and runs Monday to Friday, 07:00 to 19:00, then switches off. The modelled CO2 peaks after 19:00 depend on this assumption and on the 0.3 air-changes-per-hour infiltration rate.
- **Air settings are assumptions:**
  - CO2 generation of 0.0052 L/s per person;
  - outdoor CO2 of 420 ppm;
  - a 1000 ppm limit.

  See [`sensitivity.md`](sensitivity.md) for which of these matter.
- **The test period is short.** It runs from 1 January to 22 April 2026: the summer break and the first seven weeks of semester 1.

## 7. Future work

- Obtain G.20's CO2 data (ES1G01 to ES1G03) to validate the CO2 model.
- Test forecasting where it could matter:
  - slower ventilation systems;
  - pre-purging before scheduled classes;
  - a calendar-only forecast as a fallback during sensor outages.
- Repeat on G.25 and later semesters.
