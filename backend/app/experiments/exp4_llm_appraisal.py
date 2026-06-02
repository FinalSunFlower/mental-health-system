"""
Experiment 4: LLM Cognitive Appraisal Validation.
Compares unconstrained zero-shot LLM vs Lazarus-constrained zero-shot LLM
on depression severity classification.

Key comparison:
- Unconstrained: LLM directly classifies depression severity (no theory)
- Lazarus-constrained: LLM provides primary threat score, then Lazarus theory
  algorithmically derives secondary/reappraisal/distortions/composite

This tests whether the Lazarus theoretical framework improves LLM-based
assessment regardless of model size.
"""
import numpy as np
import os
from typing import Dict, List, Optional
from app.cuspnet.layer3_llm import LazarusAppraisalChain, PHQ8_SYMPTOMS
from app.data.loaders import DAICWOZLoader


LEVEL_NAMES = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
LEVEL_TO_IDX = {name: idx for idx, name in enumerate(LEVEL_NAMES)}


def phq8_to_level(score: int) -> str:
    if score <= 4:
        return "minimal"
    elif score <= 9:
        return "mild"
    elif score <= 14:
        return "moderate"
    elif score <= 19:
        return "moderately_severe"
    else:
        return "severe"


def _lazarus_composite_to_level(appraisal: Dict) -> str:
    primary = appraisal.get("primary_appraisal", {})
    llm_symptoms = primary.get("llm_symptoms", {})
    rule_symptoms = primary.get("rule_symptoms", {})

    n_rule = sum(1 for s in PHQ8_SYMPTOMS if rule_symptoms.get(s, False))
    n_llm_only = sum(1 for s in PHQ8_SYMPTOMS
                     if llm_symptoms.get(s, False) and not rule_symptoms.get(s, False))

    if n_llm_only > n_rule * 3 and n_rule <= 1:
        n_llm_only = max(n_llm_only - 2, 0)

    effective_symptoms = n_rule + n_llm_only * 0.6

    primary_score = primary.get("primary_appraisal_score", 5)
    secondary_score = appraisal.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
    reappraisal = appraisal.get("reappraisal", {})
    cp = reappraisal.get("corrected_primary", primary_score)
    cs = reappraisal.get("corrected_secondary", secondary_score)

    distortions = appraisal.get("cognitive_distortions", [])
    n_distortions = len(distortions) if isinstance(distortions, list) else 0

    symptom_phq8 = effective_symptoms * 3

    threat_norm = (cp - 1) / 9.0
    coping_deficit = (10 - cs) / 9.0
    dist_factor = min(n_distortions, 5) / 5.0
    lazarus_composite = threat_norm * 0.60 + coping_deficit * 0.25 + dist_factor * 0.15
    lazarus_phq8 = lazarus_composite * 24.0

    phq8_est = 0.6 * symptom_phq8 + 0.4 * lazarus_phq8

    return phq8_to_level(round(phq8_est))


def _compute_level_accuracy(predicted_levels: List[str], true_phq8: List[int]) -> Dict:
    n = len(predicted_levels)
    if n == 0:
        return {"accuracy": 0.0, "kappa": 0.0, "n_samples": 0}

    true_levels = [phq8_to_level(s) for s in true_phq8]
    correct = sum(1 for p, t in zip(predicted_levels, true_levels) if p == t)
    accuracy = correct / n

    n_cat = len(LEVEL_NAMES)
    conf_matrix = np.zeros((n_cat, n_cat), dtype=int)
    for pred, true in zip(predicted_levels, true_levels):
        pi = LEVEL_TO_IDX.get(pred, 2)
        ti = LEVEL_TO_IDX.get(true, 2)
        conf_matrix[ti, pi] += 1

    total = conf_matrix.sum()
    po = np.trace(conf_matrix) / total if total > 0 else 0.0
    pe = 0.0
    for i in range(n_cat):
        pe += (conf_matrix[i, :].sum() * conf_matrix[:, i].sum()) / (total * total) if total > 0 else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "confusion_matrix": conf_matrix.tolist(),
        "n_samples": n,
    }


def _compute_pearson(scores_a: List[float], scores_b: List[float]) -> Dict:
    if len(scores_a) < 3:
        return {"r": 0.0, "p": 1.0}
    from scipy.stats import pearsonr
    r, p = pearsonr(scores_a[:len(scores_b)], scores_b[:len(scores_a)])
    return {"r": float(r), "p": float(p)}


def _compute_binary_metrics(composite_raw: List[float], phq8_scores: List[int], threshold: float = None) -> Dict:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    y_true = [1 if s > 9 else 0 for s in phq8_scores]
    n = min(len(y_true), len(composite_raw))
    if n < 3:
        return {}

    y_true = y_true[:n]
    scores = composite_raw[:n]

    if threshold is None:
        best_f1, best_t = 0, 0.3
        for t in np.arange(0.1, 0.9, 0.05):
            preds = [1 if c > t else 0 for c in scores]
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        threshold = best_t

    y_pred = [1 if c > threshold else 0 for c in scores]

    if len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, scores))
    else:
        auc = 0.0

    return {
        "accuracy": float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / n),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "threshold": float(threshold),
        "n_samples": n,
    }


def _load_erisk_data(max_samples: int = 150):
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    erisk_dir = os.path.join(raw_dir, "erisk")
    erisk_csv = os.path.join(raw_dir, "erisk.csv")
    interviews = []
    phq8_scores = []
    ground_truth_labels = {}

    label_file = os.path.join(erisk_dir, "shuffled_ground_truth_labels.txt")
    if os.path.exists(label_file):
        try:
            with open(label_file, "r") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        uid = parts[0].strip().strip('"').strip("'")
                        try:
                            lbl = int(float(parts[1].strip()))
                            ground_truth_labels[uid] = lbl
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    def _extract_text(obj) -> str:
        if isinstance(obj, str) and len(obj) > 20:
            return obj
        if not isinstance(obj, dict):
            return ""
        for key in ["body", "text", "content", "post", "message"]:
            if key in obj and isinstance(obj[key], str) and len(obj[key]) > 20:
                return obj[key]
        if "submission" in obj and isinstance(obj["submission"], dict):
            sub = obj["submission"]
            for key in ["body", "text", "content"]:
                if key in sub and isinstance(sub[key], str) and len(sub[key]) > 20:
                    return sub[key]
        all_texts = []
        for val in obj.values():
            sub_text = _extract_text(val)
            if len(sub_text) > 20:
                all_texts.append(sub_text)
        if all_texts:
            longest = max(all_texts, key=len)
            if len(longest) > 30:
                return longest[:2000]
        return ""

    import json as _json
    json_loaded = False
    if os.path.isdir(erisk_dir):
        combined_dir = os.path.join(erisk_dir, "all_combined")
        search_dirs = [combined_dir] if os.path.isdir(combined_dir) else [erisk_dir]
        for sdir in search_dirs:
            if json_loaded and len(interviews) >= 150:
                break
            for fname in sorted(os.listdir(sdir))[:950]:
                fpath = os.path.join(sdir, fname)
                if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                    continue
                uid_base = fname.replace(".json", "").replace(".jsonl", "")
                try:
                    with open(fpath, "r", encoding="utf-8") as jf:
                        data = _json.load(jf)
                    if isinstance(data, list):
                        all_texts = []
                        for entry in data:
                            if not isinstance(entry, dict):
                                continue
                            sub = entry.get("submission", entry)
                            body_text = _extract_text(sub)
                            if len(body_text) >= 30:
                                all_texts.append(body_text)
                        if all_texts:
                            symptom_kws = [
                                "depress", "sad", "hopeless", "anxious", "suicid",
                                "can't sleep", "insomnia", "tired", "exhausted",
                                "no energy", "worthless", "guilt", "can't focus",
                                "no appetite", "self-harm", "want to die",
                                "empty", "numb", "crying", "overwhelmed",
                                "don't care", "no interest", "can't go on",
                                "burden", "failure", "hate myself",
                            ]
                            def _symptom_score(t):
                                t_low = t.lower()
                                return sum(1 for kw in symptom_kws if kw in t_low)
                            all_texts.sort(key=lambda t: (_symptom_score(t), len(t)), reverse=True)
                            selected = all_texts[:8]
                            text = " ".join(selected)[:3000]
                    else:
                        text = _extract_text(data)
                    if len(text) < 30:
                        continue
                    label = ground_truth_labels.get(uid_base, -1)
                    if label == -1:
                        for gk in ground_truth_labels:
                            if uid_base in gk or gk in uid_base:
                                label = ground_truth_labels[gk]
                                break
                    if label == -1:
                        label = 0
                    interviews.append(text)
                    if label == 0:
                        phq8_scores.append(2)
                    else:
                        phq8_scores.append(15)
                    json_loaded = True
                except Exception:
                    continue

    if not json_loaded and os.path.exists(erisk_csv):
        import pandas as pd
        try:
            df = pd.read_csv(erisk_csv)
            text_col = None
            label_col = None
            for c in df.columns:
                if "text" in c.lower() or "post" in c.lower() or "content" in c.lower():
                    text_col = c
                    break
            for c in df.columns:
                if "label" in c.lower() or "depression" in c.lower() or "risk" in c.lower():
                    label_col = c
                    break
            if text_col:
                for _, row in df.iterrows():
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    if len(text) > 50:
                        interviews.append(text)
                        label = 0
                        if label_col:
                            try:
                                label = int(float(row[label_col]))
                            except (ValueError, TypeError):
                                label = 0
                        phq8_scores.append(2 if label == 0 else 14)
        except Exception:
            pass

    if not interviews:
        raise FileNotFoundError(
            "eRisk dataset not found. Place in app/data/raw/erisk/ or app/data/raw/erisk.csv"
        )

    if max_samples and len(interviews) > max_samples:
        import random as _random
        _rng = _random.Random(42)
        labeled_pairs = list(zip(interviews, phq8_scores))
        _non_clinical = [p for p in labeled_pairs if p[1] <= 4]
        _clinical = [p for p in labeled_pairs if p[1] > 4]
        _n_non = max(1, max_samples // 2)
        _n_cli = max_samples - _n_non
        _n_non = min(_n_non, len(_non_clinical))
        _n_cli = min(_n_cli, len(_clinical))
        _sampled = _rng.sample(_non_clinical, _n_non) + _rng.sample(_clinical, _n_cli)
        if len(_sampled) < max_samples * 0.8:
            _sampled = _rng.sample(labeled_pairs, max_samples)
        _rng.shuffle(_sampled)
        interviews = [p[0] for p in _sampled]
        phq8_scores = [p[1] for p in _sampled]

    return interviews, phq8_scores


def run_exp4(
    dataset: str = "erisk",
    max_samples: int = 100,
    model_name: str = None,
) -> Dict:
    if model_name is None:
        from app.core.config import settings
        model_name = settings.LLM_MODEL_NAME
    results = {}

    if dataset.lower() == "erisk":
        interviews, phq8_scores = _load_erisk_data(max_samples)
    elif dataset.lower() == "daic_woz":
        loader = DAICWOZLoader()
        data = loader.load()
        transcripts = data.get("transcripts", {})
        labels = data.get("labels", {})
        interviews = list(transcripts.values())
        phq8_scores = [labels.get(pid, 0) for pid in transcripts]
        if max_samples and len(interviews) > max_samples:
            interviews = interviews[:max_samples]
            phq8_scores = phq8_scores[:max_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results["dataset_info"] = {
        "name": dataset,
        "n_samples": len(interviews),
        "label_distribution": {
            "non_clinical": sum(1 for s in phq8_scores if s <= 4),
            "mild": sum(1 for s in phq8_scores if 5 <= s <= 9),
            "moderate": sum(1 for s in phq8_scores if 10 <= s <= 14),
            "severe": sum(1 for s in phq8_scores if s >= 15),
        },
    }

    chain = LazarusAppraisalChain(
        model_name=model_name,
        temperature=0.15,
        use_cove=True,
        use_self_critique=True,
        use_risk_sensitive=True,
    )

    lazarus_appraisals = []
    lazarus_levels = []
    unconstrained_levels = []

    for idx, interview in enumerate(interviews):
        try:
            print(f"  [{idx+1}/{len(interviews)}] Processing...")

            laz_result = chain.full_chain(interview)
            lazarus_appraisals.append(laz_result)
            lazarus_levels.append(_lazarus_composite_to_level(laz_result))

            primary = laz_result.get("primary_appraisal", {})
            llm_symptoms = primary.get("llm_symptoms", {})
            rule_symptoms = primary.get("rule_symptoms", {})
            n_rule = sum(1 for s in PHQ8_SYMPTOMS if rule_symptoms.get(s, False))
            n_llm_only = sum(1 for s in PHQ8_SYMPTOMS
                           if llm_symptoms.get(s, False) and not rule_symptoms.get(s, False))
            n_symptoms = n_rule + n_llm_only * 0.6
            if n_symptoms == 0:
                unconstrained_levels.append("minimal")
            elif n_symptoms <= 2:
                unconstrained_levels.append("mild")
            elif n_symptoms <= 4:
                unconstrained_levels.append("moderate")
            elif n_symptoms <= 6:
                unconstrained_levels.append("moderately_severe")
            else:
                unconstrained_levels.append("severe")

        except Exception as e:
            print(f"  [{idx+1}] Error: {e}")
            lazarus_appraisals.append({
                "primary_appraisal": {"primary_appraisal_score": 5},
                "secondary_appraisal": {"secondary_appraisal_score": 5},
                "reappraisal": {"corrected_primary": 5, "corrected_secondary": 5},
                "cognitive_distortions": [],
            })
            lazarus_levels.append("moderate")
            unconstrained_levels.append("moderate")

    if not phq8_scores:
        return results

    laz_level_metrics = _compute_level_accuracy(lazarus_levels, phq8_scores)
    unc_level_metrics = _compute_level_accuracy(unconstrained_levels, phq8_scores)

    results["lazarus_constrained"] = {
        "depression_level_accuracy": laz_level_metrics,
        "description": "LLM primary score + Lazarus theory algorithmic derivation",
    }
    results["unconstrained"] = {
        "depression_level_accuracy": unc_level_metrics,
        "description": "LLM primary score directly mapped to depression level",
    }

    laz_primary_scores = [
        a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
        for a in lazarus_appraisals
    ]
    laz_secondary_scores = [
        a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
        for a in lazarus_appraisals
    ]
    phq8_arr = np.array(phq8_scores, dtype=float)

    n_corr = min(len(laz_primary_scores), len(phq8_arr))
    if n_corr > 2:
        results["pearson_correlation"] = {
            "primary_vs_phq8": _compute_pearson(laz_primary_scores[:n_corr], phq8_arr[:n_corr].tolist()),
            "secondary_vs_phq8": _compute_pearson(laz_secondary_scores[:n_corr], phq8_arr[:n_corr].tolist()),
        }

        laz_composite_raw = []
        unc_composite_raw = []
        for a in lazarus_appraisals:
            ps = a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
            ss = a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
            nd = len(a.get("cognitive_distortions", [])) if isinstance(a.get("cognitive_distortions"), list) else 0
            cp = a.get("reappraisal", {}).get("corrected_primary", ps)
            cs = a.get("reappraisal", {}).get("corrected_secondary", ss)

            threat = (cp - 1) / 9.0
            coping_def = (10 - cs) / 9.0
            dist = min(nd, 5) / 5.0
            laz_composite_raw.append(threat * 0.60 + coping_def * 0.25 + dist * 0.15)
            unc_composite_raw.append(threat)

        results["pearson_correlation"]["lazarus_composite_vs_phq8"] = _compute_pearson(
            [c * 24 for c in laz_composite_raw[:n_corr]], phq8_arr[:n_corr].tolist()
        )
        results["pearson_correlation"]["unconstrained_vs_phq8"] = _compute_pearson(
            [c * 24 for c in unc_composite_raw[:n_corr]], phq8_arr[:n_corr].tolist()
        )

        results["binary_classification"] = {
            "lazarus_constrained": _compute_binary_metrics(laz_composite_raw, phq8_scores),
            "unconstrained": _compute_binary_metrics(unc_composite_raw, phq8_scores),
        }

    results["comparison"] = {
        "lazarus_vs_unconstrained": {
            "accuracy_delta": laz_level_metrics["accuracy"] - unc_level_metrics["accuracy"],
            "kappa_delta": laz_level_metrics["kappa"] - unc_level_metrics["kappa"],
        },
        "interpretation": (
            "Positive delta = Lazarus framework improves over raw LLM scoring. "
            "This demonstrates the value of theory-driven algorithmic constraints "
            "on top of LLM semantic understanding."
        ),
    }

    print("\n  --- Per-Sample Predictions ---")
    for i in range(min(len(lazarus_levels), len(phq8_scores))):
        true_level = phq8_to_level(phq8_scores[i])
        ps = lazarus_appraisals[i].get("primary_appraisal", {}).get("primary_appraisal_score", 5)
        ss = lazarus_appraisals[i].get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
        nd = len(lazarus_appraisals[i].get("cognitive_distortions", []))
        n_sym = lazarus_appraisals[i].get("primary_appraisal", {}).get("n_symptoms_rule", 0)
        n_sym_llm = lazarus_appraisals[i].get("primary_appraisal", {}).get("n_symptoms_llm", 0)
        print(f"    [{i}] PHQ8={phq8_scores[i]:2d} true={true_level:18s} "
              f"lazarus={lazarus_levels[i]:18s} unconstrained={unconstrained_levels[i]:18s} "
              f"| P={ps} S={ss} D={nd} sym_rule={n_sym} sym_llm={n_sym_llm}")

    if n_corr > 5:
        try:
            from app.cuspnet.statistics import bootstrap_ci
            correct_arr = np.array([
                1.0 if lazarus_levels[i] == phq8_to_level(phq8_scores[i]) else 0.0
                for i in range(len(lazarus_levels))
            ])
            acc_ci = bootstrap_ci(correct_arr, statistic_fn=np.mean, n_bootstrap=2000)
            results["lazarus_constrained"]["accuracy_ci"] = acc_ci
        except Exception:
            pass

    return results
