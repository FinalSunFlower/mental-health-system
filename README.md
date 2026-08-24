# RCEP Companion Implementation

This directory contains the paper-specific implementation for:

**When External Knowledge Should Abstain: Risk-Controlled Evidence Projection
for Causal Orientation**

It is the reproducibility companion for the public preprint by Luchang Jiang.
The package contains only the RCEP contracts, source-admission experiments,
projection audits, data-source download helpers, locked manuscript artifacts,
figure-generation code, and regression tests used by the paper. The parent
repository's unrelated application, API, frontend, and legacy experiments are
not part of this package.

## Layout

- `backend/app/cuspnet/`: risk gates, evidence semantics, and support-preserving
  acyclic projection.
- `backend/app/data/`: repository-relative input and source loaders.
- `backend/app/experiments/`: manuscript experiments 7--24 and shared utilities.
- `backend/run_core_validation.py`: regenerates the main locked validation
  artifact when the documented inputs are available.
- `backend/tests/`: core contract and regression tests.
- `backend/results/`: locked result artifacts used by the paper.
- `paper/make_figures.py`: deterministic vector-figure generator.
- `paper/RESULTS_LOCK.json`: authoritative artifact hashes.

## Reproduce

Run from the repository root:

```powershell
python -m pip install -r backend/requirements-lock.txt
python -B -m pytest -p no:cacheprovider backend/tests -q
python backend/run_core_validation.py
python paper/make_figures.py
```

Third-party benchmark files are not redistributed. Download helpers in
`backend/` document their acquisition routes; tests that require restricted or
unavailable inputs skip with an explicit reason. The locked artifacts remain
available for verifying the manuscript-facing values without those downloads.

## Paper Link

The companion preprint and its Zenodo record are the authoritative descriptions
of the method and claims. This implementation is maintained on the
the `rcep-preprint` branch of the repository:

https://github.com/FinalSunFlower/mental-health-system/tree/rcep-preprint

The code is released under the repository's MIT license. The manuscript and
figures use the separate CC BY 4.0 release described by the preprint.
