# MMA3001 Project — Monash Smart Infrastructure

Individual project for **MMA3001: Numerical Methods and Machine Learning** (Monash University), using **Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data**.

## Status

🚧 **Scaffold stage.** The repository structure, tooling, and generic data-handling utilities are in place. The specific engineering question (see brief below) and the corresponding numerical/ML method are being defined and will be implemented on top of this scaffold as the project progresses.

## Engineering problem

The dataset provides time-stamped building occupancy readings and environmental sensor measurements (CO2, PM1/PM2.5/PM10, temperature, humidity, pressure) collected across rooms/zones at Monash Smart Infrastructure between May and December 2024. The general problem, per the project brief, is:

> Develop a computational method that uses time-stamped building sensor data to understand, estimate or predict occupancy and environmental behaviour.

Candidate directions being considered include: predicting room occupancy from environmental readings, reconstructing missing sensor measurements, detecting unusual environmental conditions, and characterising daily/weekly operating patterns. The final scope will be recorded here once selected.

## Repository structure

```
.
├── src/smart_infra/       # installable Python package (source code)
│   ├── data_loader.py     # CSV loading + validation
│   ├── preprocessing.py   # gap detection, missing-value handling
│   └── analysis.py        # summary statistics, occupancy/environment merging
├── tests/                  # pytest test suite
├── reports/
│   └── test_report.html   # generated pytest HTML test report
├── docs/                   # generated HTML API documentation (pdoc)
├── data/                   # not tracked in git — see data/README.md
├── notebooks/               # exploratory analysis notebooks
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # install smart_infra as an editable package
```

Place the raw dataset locally per `data/README.md` (the raw files are large and are not committed to this repository).

## Running the tests

```bash
pytest --html=reports/test_report.html --self-contained-html
```

The current suite (16 tests) covers data loading/validation, missing-data handling and gap detection, and the summary/merge analysis utilities, using small synthetic fixtures rather than the full raw dataset. The generated report is at `reports/test_report.html`.

## Building the documentation

HTML API documentation is generated from the docstrings with [`pdoc`](https://pdoc.dev/):

```bash
pdoc smart_infra --output-dir docs -d numpy
```

Open `docs/smart_infra.html` in a browser to view it.

## Computational approach

To be documented once the specific method is selected. This section will explain how the method works, why it was chosen, its assumptions and parameters, how it was implemented, and its known limitations, per the project brief.

## Validation

To be documented once the specific method is selected. Validation will follow the project brief's requirements: a justified method (e.g. withheld test data, cross-validation, comparison against a baseline), clear metrics/acceptance criteria, and an honest account of what the validation does and does not establish.

## Alternative solutions / baseline comparison

To be documented once at least one baseline/alternative approach has been implemented and compared against the final approach.

## AI use and reflection

AI assistance (Claude) has been used so far for: scaffolding the repository structure, drafting the initial data-loading/preprocessing/analysis utility functions and their docstrings, and configuring the testing and documentation tooling (pytest, pdoc). All generated code was reviewed and run locally (`pytest`, `pdoc`) before being committed. The choice of engineering question, dataset interpretation, and all analytical/methodological decisions remain the student's responsibility and will be detailed further in the final written report as the project develops.

## License

Source code is released under the [MIT License](LICENSE). The underlying dataset is supplied by Monash University for MMA3001 coursework and is governed by the unit's own data usage terms.
