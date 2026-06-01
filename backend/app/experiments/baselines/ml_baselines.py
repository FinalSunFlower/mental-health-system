"""
Machine learning baseline classifiers.
Provides Random Forest and XGBoost implementations with standard evaluation metrics
(AUC-ROC, accuracy, classification report) for comparison with CUSPNet.
"""
import numpy as np
from typing import Dict

from .pc_algorithm import run_pc
from .ges_algorithm import run_ges
from .notears_baseline import run_notears


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob[:, 1]) if y_prob.shape[1] > 1 else 0.0
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    return {
        "model": clf,
        "auc_roc": auc,
        "accuracy": acc,
        "classification_report": report,
        "feature_importances": clf.feature_importances_.tolist(),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob[:, 1]) if y_prob.shape[1] > 1 else 0.0
    except Exception:
        auc = 0.0
        y_prob = None
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    return {
        "model": clf,
        "auc_roc": auc,
        "accuracy": acc,
        "classification_report": report,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
