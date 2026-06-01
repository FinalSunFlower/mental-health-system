"""
Baseline algorithms module initialization.
Exports causal discovery baselines (PC, GES, NOTEARS) and ML classifiers (RF, XGBoost)
for comparative evaluation against CUSPNet.
"""
from .pc_algorithm import run_pc
from .ges_algorithm import run_ges
from .notears_baseline import run_notears
from .ml_baselines import train_rf, train_xgb
