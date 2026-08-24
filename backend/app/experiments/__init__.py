"""Submission-facing RCEP experiment entry points."""
from .exp7_risk_controlled_projection import (
    run_exp7,
    run_exp7_sensitivity,
    run_gate_calibration_sensitivity,
)
from .exp8_external_prior_transfer import run_exp8
from .exp9_psychology_design_validation import run_exp9
from .exp10_solver_scaling import run_exp10
from .exp11_finite_sample_gate_audit import run_exp11
from .exp12_cross_study_design_transfer import run_exp12
from .exp13_source_holdout_and_baselines import run_exp13
from .exp14_hyperparameter_robustness import run_exp14
from .exp15_tuebingen_source_holdout import run_exp15
from .exp16_causalbench_intervention_holdout import run_exp16
from .exp17_causalbench_source_variants import run_exp17
from .exp18_stratified_shift_audit import run_exp18
from .exp19_rpe1_intervention_holdout import run_exp19
from .exp20_rpe1_omnipath_holdout import run_exp20
from .exp21_context_conditioned_rpe1 import run_exp21
from .exp22_contextual_gate_validation import run_exp22
from .exp23_cross_cell_context_representation import run_exp23
from .exp24_cross_cell_pairwise_transfer import run_exp24

__all__ = [
    "run_exp7",
    "run_exp7_sensitivity",
    "run_gate_calibration_sensitivity",
    "run_exp8",
    "run_exp9",
    "run_exp10",
    "run_exp11",
    "run_exp12",
    "run_exp13",
    "run_exp14",
    "run_exp15",
    "run_exp16",
    "run_exp17",
    "run_exp18",
    "run_exp19",
    "run_exp20",
    "run_exp21",
    "run_exp22",
    "run_exp23",
    "run_exp24",
]
