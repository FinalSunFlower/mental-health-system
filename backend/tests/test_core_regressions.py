import itertools
import builtins
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cuspnet.projection import CausalDiscoveryLayer, TheoryConstraint, TheoryConstraintEngine
from app.cuspnet.contracts import (
    ContextualRiskControlledEvidenceGate,
    RiskControlConfig,
    RiskControlledEvidenceGate,
    StratifiedRiskControlledEvidenceGate,
)
from app.cuspnet.evidence_semantics import EvidenceSemantics, route_evidence
from app.data import sachs
from app.experiments.exp7_risk_controlled_projection import (
    _fit_empirical_gate,
    run_exp7,
)
from app.experiments.exp9_psychology_design_validation import (
    CALIBRATION_OUTCOMES,
    DEPLOYMENT_OUTCOMES,
    _fit_design_gate,
    _fit_reversed_design_gate,
)
from app.experiments.exp10_solver_scaling import run_exp10
from app.experiments.exp11_finite_sample_gate_audit import run_exp11
from app.experiments.exp12_cross_study_design_transfer import (
    STAR_DESIGN_RELATIONS,
    STAR_VARIABLES,
    _load_star,
    run_exp12,
)
from app.experiments.exp13_source_holdout_and_baselines import run_exp13
from app.experiments.exp14_hyperparameter_robustness import run_exp14
from app.experiments.exp15_tuebingen_source_holdout import (
    _integral_igci_score,
    _load_pairs,
    _fixed_group_split,
)
from app.experiments.exp16_causalbench_intervention_holdout import run_exp16
from app.experiments.exp17_causalbench_source_variants import run_exp17
from app.experiments.exp18_stratified_shift_audit import run_exp18
from app.experiments.exp19_rpe1_intervention_holdout import run_exp19
from app.experiments.exp20_rpe1_omnipath_holdout import run_exp20
from app.experiments.exp21_context_conditioned_rpe1 import run_exp21
from app.experiments.exp22_contextual_gate_validation import run_exp22
from app.experiments.exp23_cross_cell_context_representation import run_exp23
from app.experiments.exp24_cross_cell_pairwise_transfer import run_exp24
from download_omnipath_prior import _normalized_values


def test_empty_theory_list_means_no_constraints():
    engine = TheoryConstraintEngine(theories=[])
    assert engine.get_applicable_constraints(["insomnia", "fatigue"]) == []


def test_omnipath_reference_strings_remain_whole_provenance_records():
    assert _normalized_values("SPIKE:11259588;SIGNOR:11259588") == [
        "SIGNOR:11259588",
        "SPIKE:11259588",
    ]
    assert _normalized_values(["HPRD:8885868", "SPIKE:11259588;HPRD:8885868"]) == [
        "HPRD:8885868",
        "SPIKE:11259588",
    ]


def test_external_constraints_fail_closed_without_required_risk_certificate():
    engine = TheoryConstraintEngine(
        theories=[],
        constraints=[TheoryConstraint("cause", "effect", confidence=1.0)],
        require_risk_certificate=True,
    )
    assert engine.get_applicable_constraints(["cause", "effect"]) == []


def test_open_world_authority_is_diagnostic_only_not_a_fake_negative_label():
    semantics = EvidenceSemantics.open_world_curated(
        "documented literature TF-target relation"
    )
    route = route_evidence(semantics)
    assert route.route == "diagnostic_only"
    assert semantics.certificate_status() == "diagnostic_only_open_world"
    with pytest.raises(ValueError, match="open-world"):
        semantics.validate_for_certificate([True, False])


def test_complete_design_direction_labels_can_use_the_risk_route():
    semantics = EvidenceSemantics.documented_design(
        "randomized assignment precedes post-assignment measurement"
    )
    route = route_evidence(semantics)
    assert route.route == "risk_controlled"
    labels = semantics.validate_for_certificate([True, False], [True, True])
    assert labels.tolist() == [True, False]


def test_contextual_gate_controls_score_family_selection_and_abstains_safely():
    labels = np.asarray([1] * 80 + [0] * 20)
    source = np.asarray([0.55] * 80 + [0.10] * 20)
    context = np.asarray([0.90] * 80 + [0.05] * 20)
    gate = ContextualRiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.20,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.2, 0.4, 0.6, 0.8),
            min_accepted_calibration=20,
        )
    ).fit(source, context, labels)
    assert gate.certificate_["status"] == "certified"
    assert gate.score_family_ in {"source", "product", "geomean", "min"}
    assert gate.accepts_context(0.55, 0.90)
    assert not gate.accepts_context(0.10, 0.05)


def test_stratified_gate_requires_every_declared_source_environment_to_pass():
    confidence = np.concatenate([np.linspace(0.0, 1.0, 400), np.linspace(0.0, 1.0, 400)])
    correct = np.concatenate([
        np.ones(400, dtype=bool),
        np.linspace(0.0, 1.0, 400) >= 0.55,
    ])
    strata = np.asarray(["clean"] * 400 + ["noisy"] * 400)
    config = RiskControlConfig(
        target_error=0.20,
        failure_probability=0.05,
        candidate_thresholds=(0.0, 0.4, 0.6, 0.8),
        min_accepted_calibration=40,
    )
    gate = StratifiedRiskControlledEvidenceGate(config).fit(
        confidence, correct, strata
    )
    assert gate.certificate_["status"] == "certified"
    assert gate.threshold_ >= 0.6
    assert gate.certificate_["n_strata"] == 2
    assert all(row["feasible"] for row in gate.certificate_["selected"]["strata"])


def test_theory_constraint_does_not_inject_an_unsupported_edge():
    engine = TheoryConstraintEngine(
        theories=[],
        constraints=[TheoryConstraint("cause", "effect", confidence=1.0)],
    )
    layer = CausalDiscoveryLayer(
        constraint_engine=engine,
        score_threshold=0.9,
    )
    partial_corr = np.eye(2)
    adjacency = layer._theory_constrained_orientation(
        partial_corr,
        topo_order=[0, 1],
        names=["cause", "effect"],
        score_adj=np.zeros((2, 2)),
        score_skeleton=np.zeros((2, 2)),
    )
    assert not np.any(adjacency)


def test_theory_constraint_orients_a_supported_edge_without_changing_skeleton():
    engine = TheoryConstraintEngine(
        theories=[],
        constraints=[TheoryConstraint("effect", "cause", confidence=1.0)],
    )
    layer = CausalDiscoveryLayer(
        constraint_engine=engine,
        score_threshold=0.9,
        orientation_weights={
            "partial_correlation": 1.0,
            "score_direction": 0.0,
            "theory_prior": 1.0,
            "topological_compatibility": 0.0,
        },
    )
    partial_corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    skeleton = np.array([[0.0, 1.0], [1.0, 0.0]])
    adjacency = layer._theory_constrained_orientation(
        partial_corr,
        topo_order=[0, 1],
        names=["cause", "effect"],
        score_adj=np.zeros((2, 2)),
        score_skeleton=skeleton,
    )
    assert adjacency[1, 0] != 0
    assert adjacency[0, 1] == 0


def test_exact_orientation_projection_matches_brute_force():
    layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(theories=[]),
        orientation_solver="exact_order",
    )
    skeleton = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
        dtype=float,
    )
    scores = np.array(
        [[0, 1.2, 0.2, 0], [0.1, 0, 0.8, 0.3],
         [1.0, 0.4, 0, 1.1], [0, 1.3, 0.2, 0]],
        dtype=float,
    )
    order, objective = layer._exact_maximum_evidence_order(skeleton, scores)
    brute_force = max(
        layer._order_objective(list(candidate), skeleton, scores)
        for candidate in itertools.permutations(range(4))
    )
    assert objective == pytest.approx(brute_force)
    assert layer._order_objective(order, skeleton, scores) == pytest.approx(brute_force)


def test_refined_order_never_reduces_score_sort_objective():
    rng = np.random.RandomState(19)
    skeleton = (rng.rand(20, 20) < 0.2).astype(float)
    skeleton = np.triu(skeleton, 1)
    skeleton += skeleton.T
    scores = rng.rand(20, 20)
    np.fill_diagonal(scores, 0)
    initial = CausalDiscoveryLayer._score_sort_order(skeleton, scores)
    initial_objective = CausalDiscoveryLayer._order_objective(
        initial, skeleton, scores
    )
    refined, objective, _ = CausalDiscoveryLayer._refined_score_sort_order(
        skeleton, scores
    )
    assert sorted(refined) == list(range(20))
    assert objective >= initial_objective - 1e-10
    assert objective == pytest.approx(
        CausalDiscoveryLayer._order_objective(refined, skeleton, scores)
    )


def test_order_upper_bound_is_valid_and_fallback_reports_instancewise_certificate():
    rng = np.random.RandomState(71)
    p = 11
    skeleton = (rng.rand(p, p) < 0.35).astype(float)
    skeleton = np.triu(skeleton, 1)
    skeleton += skeleton.T
    scores = rng.rand(p, p)
    np.fill_diagonal(scores, 0.0)
    upper = CausalDiscoveryLayer._order_objective_upper_bound(skeleton, scores)
    for _ in range(100):
        order = rng.permutation(p).tolist()
        assert CausalDiscoveryLayer._order_objective(order, skeleton, scores) <= upper + 1e-12
    layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(theories=[]),
        orientation_solver="exact_order",
        max_exact_variables=4,
    )
    partial = np.full((p, p), 0.2)
    np.fill_diagonal(partial, 1.0)
    _, metadata = layer._project_acyclic_orientation(skeleton, scores, partial)
    assert metadata["objective_upper_bound"] >= metadata["objective"] - 1e-10
    assert 0.0 <= metadata["instancewise_approximation_lower_bound"] <= 1.0
    assert metadata["objective_gap_upper_bound"] >= -1e-10


def test_exact_solver_falls_back_to_refined_order_above_limit():
    rng = np.random.RandomState(23)
    p = 8
    skeleton = np.ones((p, p)) - np.eye(p)
    scores = rng.rand(p, p)
    layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(theories=[]),
        orientation_solver="exact_order",
        max_exact_variables=4,
    )
    partial = np.full((p, p), 0.2)
    np.fill_diagonal(partial, 1.0)
    _, metadata = layer._project_acyclic_orientation(skeleton, scores, partial)
    assert metadata["solver"] == "deterministic_relocation_search"
    assert metadata["refinement_passes"] >= 0


def test_solver_audit_records_a_conservative_instancewise_bound():
    result = run_exp10(variable_counts=(6,), n_repeats=3, edge_probability=0.20)
    for row in result["replicates"]:
        refined = row["refined_order"]
        exact = row["exact_order"]
        assert refined["independent_pair_upper_bound"] >= refined["objective"] - 1e-10
        assert 0.0 <= refined["instancewise_approximation_lower_bound"] <= 1.0
        assert refined["objective_gap_upper_bound"] >= 0.0
        assert exact["independent_pair_bound_relative_slack"] >= -1e-10


def test_data_evidence_fallback_never_builds_support_from_external_proposals(
    monkeypatch,
):
    engine = TheoryConstraintEngine(
        theories=[],
        constraints=[TheoryConstraint("A", "B", confidence=1.0)],
        min_confidence=0.0,
    )
    layer = CausalDiscoveryLayer(constraint_engine=engine)
    real_import = builtins.__import__

    def reject_causal_learn(name, *args, **kwargs):
        if name.startswith("causallearn"):
            raise ImportError("causal-learn deliberately unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_causal_learn)
    order, adjacency, skeleton = layer._score_fallback_pc_ges(
        np.ones((8, 2)), ["A", "B"]
    )

    assert order == [0, 1]
    assert not np.any(adjacency)
    assert not np.any(skeleton)


def test_precision_fallback_is_data_only_when_estimation_fails(monkeypatch):
    engine = TheoryConstraintEngine(
        theories=[],
        constraints=[TheoryConstraint("A", "B", confidence=1.0)],
        min_confidence=0.0,
    )
    layer = CausalDiscoveryLayer(constraint_engine=engine)

    def fail_estimation(*args, **kwargs):
        raise RuntimeError("deliberate estimation failure")

    monkeypatch.setattr(layer, "_ebic_glasso_python", fail_estimation)
    _, partial_corr = layer._ebic_glasso(
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]]), ["A", "B"]
    )
    data_only_layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(theories=[])
    )
    monkeypatch.setattr(data_only_layer, "_ebic_glasso_python", fail_estimation)
    _, data_only_partial_corr = data_only_layer._ebic_glasso(
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]]), ["A", "B"]
    )

    assert np.isfinite(partial_corr).all()
    assert np.allclose(partial_corr, data_only_partial_corr)


def test_risk_control_gate_certifies_only_low_error_confidence_region():
    confidence = np.concatenate([np.full(80, 0.95), np.full(80, 0.35)])
    correct = np.concatenate([np.ones(80, dtype=bool), np.zeros(80, dtype=bool)])
    gate = RiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.15,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.5, 0.9),
            min_accepted_calibration=30,
        )
    ).fit(confidence, correct)
    assert gate.certificate_["status"] == "certified"
    assert gate.threshold_ == pytest.approx(0.5)
    assert gate.accepts(0.95)
    assert not gate.accepts(0.35)


def test_empirical_gate_is_explicitly_uncertified():
    gate = _fit_empirical_gate(
        source_accuracy=0.85,
        n_calibration=100,
        target_error=0.20,
        seed=31,
        confidence_regime="informative",
    )
    assert gate.certificate_["status"] in {
        "empirically_selected", "abstain_no_empirical_threshold"
    }
    assert "No finite-sample" in gate.certificate_["warning"]


def test_psychology_design_calibration_targets_are_held_out_from_deployment():
    assert set(CALIBRATION_OUTCOMES).isdisjoint(DEPLOYMENT_OUTCOMES)
    gate = _fit_design_gate()
    assert gate.certificate_["status"] == "certified"
    assert gate.certificate_["selected"]["simultaneous_error_upper"] <= 0.20


def test_reversed_psychology_design_source_is_not_certified():
    gate = _fit_reversed_design_gate()
    assert gate.certificate_["status"] == "abstain_no_certified_threshold"
    assert not gate.accepts(0.90)


def test_project_star_protocol_is_predeclared_and_cross_study():
    frame = _load_star()
    assert list(frame.columns) == list(STAR_VARIABLES)
    assert len(frame) == 2621
    assert len(STAR_DESIGN_RELATIONS) == 20
    result = run_exp12(n_bootstrap=3, seed=41)
    assert result["protocol"]["independent_studies"] is True
    assert result["protocol"]["independent_source_calibration_labels"] is True
    assert result["protocol"]["protocol_status"] == "predeclared_before_participant_bootstrap"
    assert result["protocol"]["n_calibration_relations"] == 24
    assert len(result["protocol"]["calibration_relations"]) == 24
    assert all(row[3] is True for row in result["protocol"]["calibration_relations"])
    assert result["calibration_certificate"]["status"] == "certified"
    assert result["summary"]["maximum_support_violations"] == 0
    assert result["summary"]["all_outputs_dag"] is True
    assert (
        result["summary"]["risk_controlled_design_consistent_coverage"]
        > result["summary"]["data_only_design_consistent_coverage"]
    )


def test_finite_sample_gate_audit_keeps_protocols_separate():
    result = run_exp11(
        calibration_sizes=(60,),
        source_accuracies=(0.55,),
        confidence_regimes=("informative",),
        n_repeats=8,
        n_deployment=200,
        seed=37,
    )
    row = result["settings"][0]
    for method in ("risk_controlled", "empirical_threshold"):
        assert 0.0 <= row[method]["mean_acceptance_rate"] <= 1.0
        assert 0.0 <= row[method]["deployment_nonempty_rate"] <= 1.0


def test_risk_controlled_graph_projection_abstains_when_confidence_is_uninformative():
    result = run_exp7(
        n_repeats=4,
        n_samples=100,
        n_variables=5,
        direction_accuracies=(0.55,),
        spurious_ratios=(0.5,),
        sem_families=("linear_gaussian",),
        confidence_regimes=("uninformative",),
        n_calibration=300,
        seed=73,
    )
    condition = result["conditions"][0]
    assert condition["risk_certificate"]["status"] == "abstain_no_certified_threshold"
    controlled = condition["summary"]["risk_controlled_global_contract"]
    assert controlled["support_violations_mean"] == 0.0
    assert controlled["paired_delta_f1_vs_data"]["estimate"] == pytest.approx(0.0)


def test_sachs_loader_reorders_bnlearn_columns(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    row = {name: idx for idx, name in enumerate(sachs.SachsLoader.BNLEARN_SACHS_COLUMNS)}
    import pandas as pd
    pd.DataFrame([row]).to_csv(raw_dir / "sachs_data.csv", index=False)

    X, adj, names = sachs.SachsLoader(raw_dir / "sachs_data.csv").load()

    assert names == sachs.SachsLoader.SACHS_VAR_NAMES
    assert X.shape == (1, 11)
    assert X[0, names.index("PKC")] == row["PKC"]
    assert X[0, names.index("Raf")] == row["Raf"]
    assert adj.shape == (11, 11)


def test_real_source_holdout_keeps_calibration_labels_out_of_deployment():
    result = run_exp13(n_bootstrap=8, seed=89)
    protocol = result["protocol"]
    assert protocol["n_calibration_proposals"] + protocol["n_held_out_proposals"] == 12
    assert protocol["source_split"] == "fixed source-proposal split before data bootstrap"
    assert result["summary"]["rcep_source_holdout"]["support_violations_max"] == 0
    if result["gate_certificate"]["status"] != "certified":
        assert result["summary"]["rcep_source_holdout"]["mean_delta_f1_vs_data"] == pytest.approx(0.0)


def test_hyperparameter_audit_covers_both_backbones_and_invariants():
    result = run_exp14(
        theory_weights=(0.85,),
        support_thresholds=(0.08,),
        glasso_alphas=(0.08,),
        topology_weights=(0.15,),
        backbones=("ges_auto", "pc"),
        n_repeats=4,
        seed=97,
    )
    assert len(result["settings"]) == 2
    assert result["aggregate"]["all_support_violations_zero"]
    assert result["aggregate"]["all_outputs_dag"]


def test_tuebingen_loader_and_group_split_are_deterministic():
    rows_a = _load_pairs(max_rows=80, seed=101)
    rows_b = _load_pairs(max_rows=80, seed=101)
    assert len(rows_a) == 102
    assert [row["pair_id"] for row in rows_a] == [row["pair_id"] for row in rows_b]
    calibration, test = _fixed_group_split(rows_a, seed=102, calibration_fraction=0.67)
    assert set(calibration).isdisjoint(test)
    assert {rows_a[i]["group"] for i in calibration}.isdisjoint(
        {rows_a[i]["group"] for i in test}
    )
    x = np.linspace(-1.0, 1.0, 100)
    score = _integral_igci_score(x, 2.0 * x + 0.01 * np.sin(x))
    assert np.isfinite(score)
    assert score == pytest.approx(
        -_integral_igci_score(2.0 * x + 0.01 * np.sin(x), x)
    )


def test_causalbench_tf_holdout_preserves_the_explicit_abstention_boundary():
    result = run_exp16(n_bootstrap=50, seed=20260827)
    protocol = result["protocol"]
    calibration = set(result["split"]["calibration_sources"])
    held_out = set(result["split"]["held_out_sources"])
    summary = result["summary"]
    assert protocol["n_calibration_sources"] == 23
    assert protocol["n_held_out_sources"] == 12
    assert calibration.isdisjoint(held_out)
    assert protocol["n_held_out_positive_labels"] >= 10
    assert result["gate_certificate"]["status"] == "abstain_no_certified_threshold"
    assert summary["held_out_support"]["support_precision"] < 0.10
    assert summary["held_out_support"]["support_precision_clopper_pearson_95"][0] < summary["held_out_support"]["support_precision"]
    assert summary["held_out_tf_cluster_bootstrap"]["n_tf_groups"] == 12
    assert summary["held_out_accepted_support"]["coverage"] == 0.0
    directional = summary["directional_positive_labels"]
    assert directional["rcep_delta_vs_data"] == pytest.approx(0.0)
    assert directional["rcep_harm_fraction_vs_data"] == pytest.approx(0.0)


def test_causalbench_independent_source_variants_do_not_hide_low_precision():
    result = run_exp17(n_bootstrap=20, seed=20260827)
    variants = result["variants"]
    assert set(variants) == {
        "k562_chip_only",
        "k562_hepg2_replicated",
        "k562_dorothea_abc_consensus",
        "k562_hepg2_dorothea_consensus",
    }
    for variant in variants.values():
        assert variant["gate_certificate"]["status"] == "abstain_no_certified_threshold"
        assert variant["held_out_accepted_support"]["coverage"] == 0.0
    assert variants["k562_hepg2_replicated"]["held_out_support"]["support_precision"] < 0.10
    assert variants["k562_dorothea_abc_consensus"]["n_held_out_positive_labels"] == 0


def test_stratified_certificate_controls_declared_mixture_shift():
    result = run_exp18(
        repetitions=30,
        calibration_size=5_000,
        deployment_size=3_000,
        seed=20260829,
    )
    pooled = result["summary"]["pooled_certificate"]
    empirical_stratified = result["summary"]["empirical_stratified"]
    stratified = result["summary"]["stratified_certificate"]
    assert pooled["target_violation_fraction"] >= 0.90
    assert empirical_stratified["target_violation_fraction"] > 0.0
    assert stratified["target_violation_fraction"] == 0.0
    assert stratified["mean_accepted_coverage"] > 0.10


@pytest.mark.parametrize("runner", [run_exp19, run_exp20])
def test_public_rpe1_sources_fail_closed_when_precision_is_not_certified(runner):
    result = runner()
    protocol = result["protocol"]
    summary = result["summary"]
    assert protocol["n_calibration_sources"] >= 10
    assert protocol["n_held_out_sources"] >= 5
    assert set(result["split"]["calibration_sources"]).isdisjoint(
        result["split"]["held_out_sources"]
    )
    assert result["gate_certificate"]["status"] == "abstain_no_certified_threshold"
    assert summary["held_out_accepted_support"]["coverage"] == 0.0
    assert summary["directional_positive_labels"]["rcep_delta_vs_data"] == pytest.approx(0.0)
    assert summary["directional_positive_labels"]["rcep_harm_fraction_vs_data"] == pytest.approx(0.0)


def test_context_conditioning_does_not_turn_rpe1_unknowns_into_positive_claims():
    result = run_exp21()
    for source in result.values():
        assert source["protocol"]["context_score_uses_intervention_labels"] is False
        assert source["gate_certificate"]["status"] == "abstain_no_certified_threshold"
        assert source["held_out"]["coverage"] == 0.0


def test_contextual_gate_uses_task_signal_when_static_confidence_is_uninformative():
    result = run_exp22(
        repetitions=20,
        n_calibration=1_000,
        n_deployment=1_500,
        seed=20260901,
    )
    static = result["summary"]["static_source"]
    contextual = result["summary"]["contextual"]
    assert static["certificate_rate"] <= 0.10
    assert contextual["certificate_rate"] >= 0.90
    assert contextual["mean_deployment_coverage_when_certified"] > 0.25
    assert contextual["mean_deployment_error_when_certified"] < 0.18


def test_cross_cell_context_representation_is_precise_and_risk_controlled():
    result = run_exp23(sensitivity_repetitions=2)
    primary = result["primary"]
    assert result["protocol"]["n_shared_genes"] >= 2_000
    assert primary["gate_certificate"]["status"] == "certified"
    assert primary["test"]["accepted_precision"] > 0.95
    assert primary["test"]["coverage"] > 0.40
    assert primary["test"]["target_violation"] is False
    assert result["split_sensitivity"]["target_violation_fraction_when_certified"] == 0.0


def test_cross_cell_pairwise_transfer_refuses_nontransferable_proposals():
    result = run_exp24()
    summary = result["summary"]
    assert result["split"]["source_sets_disjoint"]
    assert summary["n_pairs"] == 1_500
    assert summary["n_cross_cell_replicated_detectable_pairs"] == 5
    assert result["gate_certificate"]["status"] == "abstain_no_certified_threshold"
    assert summary["accepted"]["coverage"] == 0.0
    assert summary["fail_closed_equals_data_only"]
    assert result["source_group_sensitivity"]["all_folds_abstain"]
