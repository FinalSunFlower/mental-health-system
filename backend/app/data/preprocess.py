import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


def preprocess_questionnaire(X: np.ndarray, handle_missing: str = "mean") -> Tuple[np.ndarray, Dict]:
    X = X.copy()
    if handle_missing == "mean":
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    meta = {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}
    return X_scaled, meta


def extract_cusp_params_from_student(student_data: dict) -> dict:
    pss10_raw = sum(student_data.get("pss10", [0] * 10))
    cdrisc_raw = sum(student_data.get("cdrisc", [0] * 10))
    mspss_raw = sum(student_data.get("mspss", [0] * 12))
    return {
        "pss10_norm": pss10_raw / 50.0,
        "cdrisc_norm": cdrisc_raw / 40.0,
        "mspss_norm": mspss_raw / 84.0,
    }
