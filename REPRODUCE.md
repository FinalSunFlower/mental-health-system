# Reproduction Guide

## Environment

The locked manuscript artifact was generated on CPU with Python 3.12.10. Install
the exact recorded package versions:

```powershell
python -m pip install -r backend/requirements-lock.txt
```

## Locked-Artifact Verification

The following command executes contract, solver, synthetic, and locked-artifact
tests without replacing the supplied result artifact:

```powershell
python -B -m pytest -p no:cacheprovider backend/tests -q
```

On an archive without the optional raw benchmark downloads, data-dependent
tests are expected to be skipped with an explicit reason. They are not treated
as passing data tests.

## Figures and Paper

```powershell
python paper/make_figures.py
```

The generated PDF/SVG figures are deterministic functions of
`paper/RESULTS_LOCK.json` and the hash-pinned artifact in `backend/results/`.
The editable manuscript source and submission PDF are distributed through the
associated Zenodo release rather than this code-only branch.

## Full Experiment Regeneration

After acquiring the public inputs listed in `DATA_AVAILABILITY.md`, run:

```powershell
python backend/run_core_validation.py
python backend/run_exp24.py
python paper/make_figures.py
```

The runners write timestamped artifacts under `backend/results/`. They do not
overwrite the supplied locked artifact automatically. The manuscript reports
the included lock files, not a newly generated output.

## Expected Checks

- All available tests pass.
- Missing-data tests are skipped only when their documented public input is not
  installed.
- Figure regeneration preserves the locked-result hash check.
- The compiled PDF reports the source-risk, structural-permission, and
  scope-boundary claims stated in the manuscript.
