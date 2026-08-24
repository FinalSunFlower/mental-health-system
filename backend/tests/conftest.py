"""Pytest behavior for public-data-dependent supplement regressions."""
from pathlib import Path

import pytest


RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_TESTS = {
    "test_project_star_protocol_is_predeclared_and_cross_study",
    "test_real_source_holdout_keeps_calibration_labels_out_of_deployment",
    "test_tuebingen_loader_and_group_split_are_deterministic",
    "test_causalbench_tf_holdout_preserves_the_explicit_abstention_boundary",
    "test_causalbench_independent_source_variants_do_not_hide_low_precision",
    "test_public_rpe1_sources_fail_closed_when_precision_is_not_certified[run_exp19]",
    "test_public_rpe1_sources_fail_closed_when_precision_is_not_certified[run_exp20]",
    "test_context_conditioning_does_not_turn_rpe1_unknowns_into_positive_claims",
    "test_cross_cell_context_representation_is_precise_and_risk_controlled",
    "test_cross_cell_pairwise_transfer_refuses_nontransferable_proposals",
}


def pytest_collection_modifyitems(config, items):
    required = {
        RAW_ROOT / "sachs_data.csv",
        RAW_ROOT / "psychology" / "star.csv",
        RAW_ROOT / "tuebingen" / "pairs" / "pairmeta.txt",
        RAW_ROOT / "causalbench" / "summary_stats.xlsx",
    }
    if all(path.exists() for path in required):
        return
    skip = pytest.mark.skip(
        reason=(
            "public benchmark inputs are absent; acquire the files listed in "
            "REPRODUCIBILITY.md to run this data-dependent regression"
        )
    )
    for item in items:
        if item.name in DATA_TESTS:
            item.add_marker(skip)
