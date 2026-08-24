"""Verify the frozen public RPE1 source audit and its fail-closed claims."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "RPE1_RESULTS_LOCK.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    artifact = ROOT / lock["artifact"]
    failures: list[str] = []
    if not artifact.exists():
        failures.append(f"missing artifact: {artifact}")
    elif _sha256(artifact) != lock["sha256"]:
        failures.append("artifact hash does not match RPE1_RESULTS_LOCK.json")
    if failures:
        print("RPE1 public-source audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    for relative, expected in payload["meta"]["input_manifest"].items():
        path = ROOT / relative
        if not path.exists() or _sha256(path).lower() != expected["sha256"].lower():
            failures.append(f"input hash mismatch: {relative}")
    for name in ("rpe1_hepg2_chip", "rpe1_omnipath"):
        result = payload["experiments"][name]
        calibration = set(result["split"]["calibration_sources"])
        held_out = set(result["split"]["held_out_sources"])
        if calibration & held_out:
            failures.append(f"source leakage in {name}")
        if result["gate_certificate"]["status"] != "abstain_no_certified_threshold":
            failures.append(f"unexpected certification in {name}")
        summary = result["summary"]
        if summary["held_out_accepted_support"]["coverage"] != 0.0:
            failures.append(f"uncertified source accepted proposals in {name}")
        direction = summary["directional_positive_labels"]
        if direction["rcep_delta_vs_data"] != 0.0 or direction["rcep_harm_fraction_vs_data"] != 0.0:
            failures.append(f"fail-closed direction mismatch in {name}")
    contextual = payload["experiments"]["rpe1_context_conditioned"]
    for name, result in contextual.items():
        if result["gate_certificate"]["status"] != "abstain_no_certified_threshold":
            failures.append(f"unexpected contextual certification in {name}")
        if result["held_out"]["coverage"] != 0.0:
            failures.append(f"uncertified contextual source accepted proposals in {name}")
    controlled = payload["experiments"]["contextual_controlled_validation"]["summary"]
    if controlled["static_source"]["certificate_rate"] != 0.0:
        failures.append("static source unexpectedly certified in contextual control")
    if controlled["contextual"]["certificate_rate"] < 0.95:
        failures.append("contextual gate did not reliably certify in positive control")
    if controlled["contextual"]["mean_deployment_coverage_when_certified"] < 0.50:
        failures.append("contextual gate coverage too low in positive control")
    if controlled["contextual"]["target_violation_fraction_when_certified"] != 0.0:
        failures.append("contextual gate exceeded target error in positive control")
    cross_cell = payload["experiments"]["cross_cell_context_representation"]
    primary = cross_cell["primary"]
    if primary["gate_certificate"]["status"] != "certified":
        failures.append("cross-cell context representation did not certify")
    if primary["test"]["accepted_precision"] < 0.95:
        failures.append("cross-cell context held-out precision is below 0.95")
    if primary["test"]["coverage"] < 0.40:
        failures.append("cross-cell context held-out coverage is below 0.40")
    sensitivity = cross_cell["split_sensitivity"]
    if sensitivity["target_violation_fraction_when_certified"] != 0.0:
        failures.append("cross-cell context sensitivity exceeded target error")
    if failures:
        print("RPE1 public-source audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"RPE1 public-source audit passed: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
