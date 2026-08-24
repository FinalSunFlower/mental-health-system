# Data Availability and Reuse

This release does not redistribute raw benchmark files. It includes code,
locked result artifacts, source hashes, extraction metadata, and download
helpers sufficient to identify the manuscript inputs. This separation avoids
re-licensing or rehosting third-party datasets.

| Input | Availability | Acquisition route |
| --- | --- | --- |
| Sachs observations | Public benchmark | Place the public benchmark at `data/raw/sachs_data.csv`. |
| JOBS II | Public source subject to its terms | `python backend/download_jobs_ii.py` |
| Project STAR | Public source subject to its terms | `python backend/download_star.py` |
| Tuebingen Cause-Effect Pairs | Public download subject to provider terms | `python backend/download_tuebingen.py` |
| CausalBench inputs | Public source/workbook subject to provider terms | Included extraction metadata and `backend/download_causalbench_labels.py` |
| Frozen OmniPath snapshot | Included derived snapshot | `backend/resources/omnipath_sachs_prior.json` |

The exact input hashes, versions, query metadata, and expected locations are
recorded in the locked result artifact and in the manuscript's reproducibility
appendix. Users are responsible for complying with the licenses and terms of
the original data providers.

No direct identifiers or newly collected human-subject data are included in
this release.
