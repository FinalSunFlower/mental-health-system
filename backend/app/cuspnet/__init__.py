"""
CUSPNet module initialization.
Exports the three-layer architecture: causal discovery, CUSP dynamics, and LLM appraisal,
along with the main CuspNetEngine orchestrator and utility functions.
"""
from .layer1_causal import CausalDiscoveryLayer, TheoryConstraintEngine
from .layer2_dynamics import CuspDynamicsLayer, NormProvider, POPULATION_NORMS
from .layer3_llm import LazarusAppraisalChain
from .cuspnet_engine import CuspNetEngine
from .utils import compute_resilience_reserve, compute_potential
from .statistics import (
    bootstrap_ci,
    bootstrap_ci_metric,
    paired_significance_test,
    bonferroni_correction,
    fdr_correction,
    mcnemar_test,
    compute_full_statistical_report,
)
