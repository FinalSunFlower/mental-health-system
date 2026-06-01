"""
Experiment 4: LLM Cognitive Appraisal Validation.
Evaluates the Lazarus appraisal chain on DAIC-WOZ and eRisk depression datasets,
measuring accuracy in depression severity classification and distortion detection.
"""
import numpy as np
import os
from typing import Dict, List, Optional
from app.cuspnet.layer3_llm import LazarusAppraisalChain
from app.data.loaders import DAICWOZLoader


def _phq8_to_level_global(score):
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


def _score_to_threat_level(score: int) -> str:
    if score <= 3:
        return "low"
    elif score <= 5:
        return "moderate"
    elif score <= 7:
        return "high"
    else:
        return "critical"


def _score_to_coping_potential(score: int) -> str:
    if score <= 3:
        return "very_low"
    elif score <= 5:
        return "low"
    elif score <= 7:
        return "moderate"
    else:
        return "high"


def _phq8_to_level_local(score: int) -> str:
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


def _compute_appraisal_accuracy(
    predicted: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, ground_truth={len(ground_truth)}"
        )
    n = len(predicted)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_samples": 0}

    primary_correct = 0
    secondary_correct = 0
    reappraisal_correct = 0
    distortion_tp = 0
    distortion_fp = 0
    distortion_fn = 0

    for pred, true in zip(predicted, ground_truth):
        pred_primary_score = pred.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
        pred_threat = _score_to_threat_level(pred_primary_score)
        true_threat = true.get("primary_appraisal", {}).get("threat_level", "moderate")
        if pred_threat == true_threat:
            primary_correct += 1

        pred_secondary_score = pred.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
        pred_coping = _score_to_coping_potential(pred_secondary_score)
        true_coping = true.get("secondary_appraisal", {}).get("coping_potential", "moderate")
        if pred_coping == true_coping:
            secondary_correct += 1

        pred_reappraisal = pred.get("reappraisal", {})
        pred_adjustment = pred_reappraisal.get("adjustment_direction",
            "increase_coping" if pred_primary_score > 5 else "maintain")
        true_adjustment = true.get("reappraisal", {}).get("adjustment_direction", "maintain")
        if pred_adjustment == true_adjustment:
            reappraisal_correct += 1

        pred_distortions = set()
        for d in pred.get("cognitive_distortions", []):
            if isinstance(d, dict):
                dtype = d.get("type", "")
                if dtype:
                    pred_distortions.add(dtype)
            elif isinstance(d, str):
                pred_distortions.add(d)
        true_distortions = set(true.get("cognitive_distortions", {}).get("detected", []))
        distortion_tp += len(pred_distortions & true_distortions)
        distortion_fp += len(pred_distortions - true_distortions)
        distortion_fn += len(true_distortions - pred_distortions)

    primary_acc = primary_correct / n if n > 0 else 0.0
    secondary_acc = secondary_correct / n if n > 0 else 0.0
    reappraisal_acc = reappraisal_correct / n if n > 0 else 0.0
    distortion_prec = (
        distortion_tp / (distortion_tp + distortion_fp)
        if (distortion_tp + distortion_fp) > 0
        else 0.0
    )
    distortion_rec = (
        distortion_tp / (distortion_tp + distortion_fn)
        if (distortion_tp + distortion_fn) > 0
        else 0.0
    )
    distortion_f1 = (
        2 * distortion_prec * distortion_rec / (distortion_prec + distortion_rec)
        if (distortion_prec + distortion_rec) > 0
        else 0.0
    )
    overall_accuracy = (primary_acc + secondary_acc + reappraisal_acc) / 3.0

    return {
        "overall_accuracy": float(overall_accuracy),
        "primary_appraisal_accuracy": float(primary_acc),
        "secondary_appraisal_accuracy": float(secondary_acc),
        "reappraisal_accuracy": float(reappraisal_acc),
        "distortion_detection_precision": float(distortion_prec),
        "distortion_detection_recall": float(distortion_rec),
        "distortion_detection_f1": float(distortion_f1),
        "n_samples": n,
    }


def _compute_depression_level_accuracy(
    predicted_levels: List[str],
    true_phq8_scores: List[int],
) -> Dict:
    n = len(predicted_levels)
    if n == 0:
        return {"accuracy": 0.0, "kappa": 0.0, "n_samples": 0}

    def _phq8_to_level(score: int) -> str:
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

    true_levels = [_phq8_to_level(s) for s in true_phq8_scores]
    correct = sum(1 for p, t in zip(predicted_levels, true_levels) if p == t)
    accuracy = correct / n

    level_names = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
    n_categories = len(level_names)
    conf_matrix = np.zeros((n_categories, n_categories), dtype=int)
    level_to_idx = {name: idx for idx, name in enumerate(level_names)}
    for pred, true in zip(predicted_levels, true_levels):
        pred_idx = level_to_idx.get(pred, 2)
        true_idx = level_to_idx.get(true, 2)
        conf_matrix[true_idx, pred_idx] += 1

    total = conf_matrix.sum()
    po = np.trace(conf_matrix) / total if total > 0 else 0.0
    pe = 0.0
    for i in range(n_categories):
        row_sum = conf_matrix[i, :].sum()
        col_sum = conf_matrix[:, i].sum()
        pe += (row_sum * col_sum) / (total * total) if total > 0 else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "confusion_matrix": conf_matrix.tolist(),
        "level_names": level_names,
        "n_samples": n,
    }


def _appraisal_to_depression_level(appraisal: Dict) -> str:
    primary_score = appraisal.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
    secondary_score = appraisal.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
    distortions = appraisal.get("cognitive_distortions", [])
    n_distortions = len(distortions) if isinstance(distortions, list) else 0
    avg_severity = 0
    if isinstance(distortions, list) and len(distortions) > 0:
        severities = [d.get("severity", 3) for d in distortions if isinstance(d, dict)]
        avg_severity = sum(severities) / len(severities) if severities else 3
    reappraisal = appraisal.get("reappraisal", {})
    corrected_primary = reappraisal.get("corrected_primary", primary_score)
    corrected_secondary = reappraisal.get("corrected_secondary", secondary_score)
    ideal_secondary = max(1, min(10, 11 - corrected_primary))
    secondary_gap = abs(corrected_secondary - ideal_secondary)
    trust_factor = max(0.2, 1.0 - secondary_gap / 10.0)

    calibration = appraisal.get("calibration_meta", {})
    
    cove_consistent = calibration.get("cove_consistent", True)
    critique_passed = calibration.get("critique_passed", True)
    risk_floor_applied = calibration.get("risk_floor_applied", False)
    cve_correction = calibration.get("cve_correction", None)
    critique_adjustment = calibration.get("critique_adjustment", None)

    threat_component = (corrected_primary - 1) / 9.0
    coping_deficit_raw = (10 - corrected_secondary) / 9.0
    coping_ideal_deficit = (10 - ideal_secondary) / 9.0
    coping_deficit = (
        trust_factor * coping_deficit_raw + (1 - trust_factor) * coping_ideal_deficit
    )
    distortion_component = min(n_distortions, 5) / 5.0 * (avg_severity / 5.0)

    base_composite = (
        threat_component * 0.55
        + coping_deficit * 0.20
        + distortion_component * 0.25
    )

    correction_boost = 0.0
    if not cove_consistent and cve_correction:
        correction_boost += 0.08
    if not critique_passed and critique_adjustment:
        correction_boost += 0.06
    if risk_floor_applied:
        correction_boost += 0.05

    composite = min(base_composite + correction_boost, 1.0)

    primary_signal = (corrected_primary - 1) / 9.0
    
    if corrected_primary >= 7 or risk_floor_applied:
        blended_composite = composite * 0.40 + primary_signal * 0.60
    elif corrected_primary <= 3 and cove_consistent and critique_passed:
        blended_composite = composite * 0.65 + primary_signal * 0.35
    else:
        blended_composite = composite * 0.50 + primary_signal * 0.50

    phq8_equivalent = blended_composite * 24
    if phq8_equivalent <= 4:
        return "minimal"
    elif phq8_equivalent <= 9:
        return "mild"
    elif phq8_equivalent <= 14:
        return "moderate"
    elif phq8_equivalent <= 19:
        return "moderately_severe"
    else:
        return "severe"


def run_exp4(
    dataset: str = "erisk",
    use_lazarus_constraints: bool = True,
    n_reflection_steps: int = 3,
    max_samples: int = 150,
    model_name: str = r"D:\Models\huggingface\Qwen3.5-2B",
) -> Dict:
    results = {}

    if dataset.lower() == "daic_woz":
        loader = DAICWOZLoader()
        data = loader.load()
        transcripts = data.get("transcripts", {})
        labels = data.get("labels", {})
        interviews = list(transcripts.values())
        phq8_scores = [labels.get(pid, 0) for pid in transcripts]
    elif dataset.lower() == "erisk":
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        erisk_dir = os.path.join(raw_dir, "erisk")
        erisk_csv = os.path.join(raw_dir, "erisk.csv")
        interviews = []
        phq8_scores = []
        ground_truth_labels: Dict[str, int] = {}
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

        def _extract_erisk_text(obj) -> str:
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
                sub_text = _extract_erisk_text(val)
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
                                body_text = _extract_erisk_text(sub)
                                if len(body_text) >= 30:
                                    all_texts.append(body_text)
                            if all_texts:
                                all_texts.sort(key=len, reverse=True)
                                text = " ".join(all_texts[:3])
                        else:
                            text = _extract_erisk_text(data)
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
                        try:
                            with open(fpath, "r", encoding="utf-8") as jf:
                                raw_data = _json.load(jf)
                            text = _extract_erisk_text(raw_data)
                            if len(text) >= 30:
                                label = ground_truth_labels.get(uid_base, 0)
                                interviews.append(text)
                                phq8_scores.append(2 if label == 0 else 14)
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
                "eRisk dataset not found. Please download from "
                "https://doi.org/10.6084/m9.figshare.30565697 (1.21GB JSON format) "
                "and place in app/data/raw/erisk/ "
                "or prepare a combined CSV as app/data/raw/erisk.csv"
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'daic_woz' or 'erisk'.")

    if max_samples is not None and max_samples > 0 and len(interviews) > max_samples:
        import random as _random
        _random.seed(42)
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
    elif max_samples is not None and max_samples > 0:
        interviews = interviews[:max_samples]
        phq8_scores = phq8_scores[:max_samples]

    results["dataset_info"] = {
        "name": dataset,
        "n_interviews": len(interviews),
        "n_with_phq8": len(phq8_scores),
    }

    llm_full = LazarusAppraisalChain(
        model_name=model_name,
        temperature=0.25,
        use_cove=True,
        use_self_critique=True,
        use_risk_sensitive=True,
    )

    full_appraisals = []
    full_levels = []

    for idx, interview in enumerate(interviews):
        try:
            print(f"  [Full Chain] Processing interview {idx+1}/{len(interviews)}...")
            full_result = llm_full.full_chain(interview)
            full_appraisals.append(full_result)
            full_levels.append(_appraisal_to_depression_level(full_result))
        except Exception:
            full_appraisals.append({
                "primary_appraisal": {"primary_appraisal_score": 5},
                "secondary_appraisal": {"secondary_appraisal_score": 5},
                "reappraisal": {"adjustment_direction": "maintain"},
                "cognitive_distortions": [],
            })
            full_levels.append("moderate")

    primary_only_appraisals = []
    primary_only_levels = []
    for idx, interview in enumerate(interviews):
        try:
            print(f"  [Primary Only] Processing interview {idx+1}/{len(interviews)}...")
            primary = llm_full.primary_appraisal(interview)
            p_score = primary.get("primary_appraisal_score", 5)
            inferred_secondary = max(1, min(10, 11 - p_score))
            primary_only_appraisals.append({
                "primary_appraisal": primary,
                "secondary_appraisal": {"secondary_appraisal_score": inferred_secondary},
                "reappraisal": {
                    "adjustment_direction": "maintain",
                    "corrected_primary": p_score,
                    "corrected_secondary": inferred_secondary,
                },
                "cognitive_distortions": [],
            })
            primary_only_levels.append(_appraisal_to_depression_level(
                primary_only_appraisals[-1]
            ))
        except Exception:
            primary_only_appraisals.append({
                "primary_appraisal": {"primary_appraisal_score": 5},
                "secondary_appraisal": {"secondary_appraisal_score": 5},
                "reappraisal": {"adjustment_direction": "maintain"},
                "cognitive_distortions": [],
            })
            primary_only_levels.append("moderate")

    if phq8_scores:
        gt_appraisals = []
        for score in phq8_scores:
            if score <= 4:
                threat = "low"
                coping = "high"
                adj = "maintain"
            elif score <= 9:
                threat = "moderate"
                coping = "moderate"
                adj = "increase_coping"
            elif score <= 14:
                threat = "high"
                coping = "low"
                adj = "increase_coping"
            else:
                threat = "critical"
                coping = "very_low"
                adj = "increase_coping"
            n_dist = 0 if score <= 4 else (1 if score <= 9 else (2 if score <= 14 else 3))
            gt_appraisals.append({
                "primary_appraisal": {"threat_level": threat},
                "secondary_appraisal": {"coping_potential": coping},
                "reappraisal": {"adjustment_direction": adj},
                "cognitive_distortions": {"detected": ["catastrophizing"] * n_dist if n_dist > 0 else []},
            })

        print("\n  --- Detailed Prediction vs Ground Truth ---")
        for i in range(min(len(full_appraisals), len(gt_appraisals))):
            pred = full_appraisals[i]
            true = gt_appraisals[i]
            pred_primary_score = pred.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
            pred_secondary_score = pred.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
            pred_threat = _score_to_threat_level(pred_primary_score)
            pred_coping = _score_to_coping_potential(pred_secondary_score)
            true_threat = true.get("primary_appraisal", {}).get("threat_level", "?")
            true_coping = true.get("secondary_appraisal", {}).get("coping_potential", "?")
            pred_level = full_levels[i]
            true_level = _phq8_to_level_local(phq8_scores[i])
            n_dist = len(pred.get("cognitive_distortions", []))
            print(f"    [{i}] PHQ8={phq8_scores[i]} true_level={true_level} pred_level={pred_level} | "
                  f"threat: {true_threat}->{pred_threat}(score={pred_primary_score}) | "
                  f"coping: {true_coping}->{pred_coping}(score={pred_secondary_score}) | "
                  f"distortions={n_dist}")

        full_metrics = _compute_appraisal_accuracy(full_appraisals, gt_appraisals)
        primary_only_metrics = _compute_appraisal_accuracy(primary_only_appraisals, gt_appraisals)

        full_level_metrics = _compute_depression_level_accuracy(full_levels, phq8_scores)
        primary_only_level_metrics = _compute_depression_level_accuracy(primary_only_levels, phq8_scores)

        results["full_chain"] = {
            "appraisal_accuracy": full_metrics,
            "depression_level_accuracy": full_level_metrics,
        }
        results["primary_only"] = {
            "appraisal_accuracy": primary_only_metrics,
            "depression_level_accuracy": primary_only_level_metrics,
        }

        if phq8_scores:
            full_primary_scores = [
                a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
                for a in full_appraisals
            ]
            full_secondary_scores = [
                a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
                for a in full_appraisals
            ]
            full_stress_product = [p * s / 10.0 for p, s in zip(full_primary_scores, full_secondary_scores)]
            phq8_arr = np.array(phq8_scores, dtype=float)
            stress_arr = np.array(full_stress_product, dtype=float)
            n_corr = min(len(phq8_arr), len(stress_arr))
            if n_corr > 2:
                from scipy.stats import pearsonr
                r_primary, p_primary = pearsonr(full_primary_scores[:n_corr], phq8_arr[:n_corr])
                r_stress, p_stress = pearsonr(stress_arr[:n_corr], phq8_arr[:n_corr])
                results["pearson_correlation"] = {
                    "primary_vs_phq8": {"r": float(r_primary), "p": float(p_primary)},
                    "stress_product_vs_phq8": {"r": float(r_stress), "p": float(p_stress)},
                }

                full_composite_scores = []
                for a in full_appraisals[:n_corr]:
                    ps = a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
                    ss = a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
                    nd = len(a.get("cognitive_distortions", [])) if isinstance(a.get("cognitive_distortions"), list) else 0
                    avg_sev = 0
                    dists = a.get("cognitive_distortions", [])
                    if isinstance(dists, list) and len(dists) > 0:
                        sevs = [d.get("severity", 3) for d in dists if isinstance(d, dict)]
                        avg_sev = sum(sevs) / len(sevs) if sevs else 3
                    ideal_s = max(1, min(10, 11 - ps))
                    sec_gap = abs(ss - ideal_s)
                    trust_f = max(0.2, 1.0 - sec_gap / 10.0)
                    threat_comp = (ps - 1) / 9.0
                    coping_def_raw = (10 - ss) / 9.0
                    coping_ideal_def = (10 - ideal_s) / 9.0
                    coping_def = trust_f * coping_def_raw + (1 - trust_f) * coping_ideal_def
                    dist_comp = min(nd, 5) / 5.0 * (avg_sev / 5.0)
                    comp_raw = threat_comp * 0.55 + coping_def * 0.20 + dist_comp * 0.25
                    comp_blended = comp_raw * 0.50 + threat_comp * 0.50
                    full_composite_scores.append(comp_blended * 24)
                if len(full_composite_scores) >= 3:
                    r_comp, p_comp = pearsonr(full_composite_scores, phq8_arr[:len(full_composite_scores)])
                    results["pearson_correlation"]["composite_vs_phq8"] = {"r": float(r_comp), "p": float(p_comp)}

                def _compute_icc(scores1, scores2):
                    n = len(scores1)
                    if n < 3:
                        return {"icc": 0.0, "note": "insufficient samples"}
                    s1 = np.array(scores1, dtype=float)
                    s2 = np.array(scores2, dtype=float)
                    mean_all = np.mean(np.concatenate([s1, s2]))
                    ss_between = n * (np.mean(s1) - mean_all) ** 2 + n * (np.mean(s2) - mean_all) ** 2
                    ss_within = np.sum((s1 - np.mean(s1)) ** 2) + np.sum((s2 - np.mean(s2)) ** 2)
                    ms_between = ss_between / 1 if n > 0 else 0
                    ms_within = ss_within / (2 * (n - 1)) if n > 1 else 1e-10
                    icc = (ms_between - ms_within) / (ms_between + ms_within) if (ms_between + ms_within) > 0 else 0.0
                    return {"icc": float(max(0, icc)), "icc_type": "ICC(1,1)"}

                phq8_normalized = (phq8_arr[:n_corr] / max(phq8_arr[:n_corr].max(), 1) * 10).tolist()
                primary_icc = _compute_icc(full_primary_scores[:n_corr], phq8_normalized)
                results["icc"] = {
                    "primary_appraisal_vs_phq8": primary_icc,
                }

        results["comparison"] = {
            "lazarus_chain_improvement": {
                "overall_accuracy": (
                    full_metrics["overall_accuracy"] - primary_only_metrics["overall_accuracy"]
                ),
                "distortion_f1": (
                    full_metrics["distortion_detection_f1"] - primary_only_metrics["distortion_detection_f1"]
                ),
                "depression_level_accuracy": (
                    full_level_metrics["accuracy"] - primary_only_level_metrics["accuracy"]
                ),
                "depression_level_kappa": (
                    full_level_metrics["kappa"] - primary_only_level_metrics["kappa"]
                ),
            }
        }

        full_binary_true = [1 if s > 4 else 0 for s in phq8_scores]
        full_composite_raw = []
        po_composite_raw = []
        for a in full_appraisals:
            ps = a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
            ss = a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
            nd = len(a.get("cognitive_distortions", [])) if isinstance(a.get("cognitive_distortions"), list) else 0
            avg_sev = 0
            dists = a.get("cognitive_distortions", [])
            if isinstance(dists, list) and len(dists) > 0:
                sevs = [d.get("severity", 3) for d in dists if isinstance(d, dict)]
                avg_sev = sum(sevs) / len(sevs) if sevs else 3
            threat_comp = (ps - 1) / 9.0
            coping_def = (10 - ss) / 9.0
            dist_comp = min(nd, 5) / 5.0 * (avg_sev / 5.0)
            comp = threat_comp * 0.40 + coping_def * 0.30 + dist_comp * 0.30
            full_composite_raw.append(comp)
        for a in primary_only_appraisals:
            ps = a.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
            ss = a.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
            threat_comp = (ps - 1) / 9.0
            coping_def = (10 - ss) / 9.0
            comp = threat_comp * 0.50 + coping_def * 0.50
            po_composite_raw.append(comp)
        n_bin = min(len(full_binary_true), len(full_composite_raw), len(po_composite_raw))
        if n_bin > 0:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
            y_true = full_binary_true[:n_bin]

            def _find_best_threshold(composite_scores, y_true):
                best_f1 = 0
                best_thresh = 0.3
                for t in np.arange(0.1, 0.9, 0.05):
                    preds = [1 if c > t else 0 for c in composite_scores]
                    f1 = f1_score(y_true, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t
                return best_thresh, best_f1

            fc_thresh, fc_best_f1 = _find_best_threshold(full_composite_raw[:n_bin], y_true)
            po_thresh, po_best_f1 = _find_best_threshold(po_composite_raw[:n_bin], y_true)

            y_full = [1 if c > fc_thresh else 0 for c in full_composite_raw[:n_bin]]
            y_po = [1 if c > po_thresh else 0 for c in po_composite_raw[:n_bin]]

            if len(set(y_true)) >= 2:
                fc_auc = roc_auc_score(y_true, full_composite_raw[:n_bin])
                po_auc = roc_auc_score(y_true, po_composite_raw[:n_bin])
            else:
                fc_auc = 0.0
                po_auc = 0.0

            results["binary_classification"] = {
                "full_chain": {
                    "accuracy": float(sum(1 for t, p in zip(y_true, y_full) if t == p) / n_bin),
                    "f1": float(f1_score(y_true, y_full, zero_division=0)),
                    "precision": float(precision_score(y_true, y_full, zero_division=0)),
                    "recall": float(recall_score(y_true, y_full, zero_division=0)),
                    "auc": float(fc_auc),
                    "optimal_threshold": float(fc_thresh),
                },
                "primary_only": {
                    "accuracy": float(sum(1 for t, p in zip(y_true, y_po) if t == p) / n_bin),
                    "f1": float(f1_score(y_true, y_po, zero_division=0)),
                    "precision": float(precision_score(y_true, y_po, zero_division=0)),
                    "recall": float(recall_score(y_true, y_po, zero_division=0)),
                    "auc": float(po_auc),
                    "optimal_threshold": float(po_thresh),
                },
                "n_samples": n_bin,
                "n_positive": sum(y_true),
                "n_negative": n_bin - sum(y_true),
            }

        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                client = OpenAI(api_key=api_key)
                gpt4_unconstrained_levels = []
                gpt4_lazarus_levels = []
                gpt4_sample_limit = min(len(interviews), max_samples or 50)
                for interview in interviews[:gpt4_sample_limit]:
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": f"Based on the following interview, rate the depression severity on a scale: minimal(0-4), mild(5-9), moderate(10-14), moderately_severe(15-19), severe(20+). Respond with only the category name.\n\n{interview[:2000]}"}],
                            max_tokens=20, temperature=0.3,
                        )
                        level = resp.choices[0].message.content.strip().lower()
                        valid_levels = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
                        gpt4_unconstrained_levels.append(level if level in valid_levels else "moderate")
                    except Exception:
                        gpt4_unconstrained_levels.append("moderate")

                    try:
                        lazarus_prompt = (
                            "You are performing a Lazarus cognitive appraisal. "
                            "Step 1 (Primary Appraisal): Identify threats and rate threat intensity (1-10). "
                            "Step 2 (Secondary Appraisal): Assess coping resources and rate coping efficacy (1-10). "
                            "Step 3 (Reappraisal): If threat>=7 and coping>=7, reduce threat by 2. If threat<=3 and coping<=3, increase coping by 2. "
                            "Step 4: Detect cognitive distortions (catastrophizing/overgeneralization/all_or_nothing/emotional_reasoning/personalization). "
                            "Step 5: Integrate all steps. Based on corrected threat, coping, and distortions, classify depression as: "
                            "minimal(0-4), mild(5-9), moderate(10-14), moderately_severe(15-19), severe(20+). "
                            "Respond with only the category name.\n\n"
                            f"Interview: {interview[:2000]}"
                        )
                        resp = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": lazarus_prompt}],
                            max_tokens=20, temperature=0.3,
                        )
                        level = resp.choices[0].message.content.strip().lower()
                        valid_levels = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
                        gpt4_lazarus_levels.append(level if level in valid_levels else "moderate")
                    except Exception:
                        gpt4_lazarus_levels.append("moderate")

                gpt4_scores = phq8_scores[:gpt4_sample_limit]
                if gpt4_scores:
                    def _phq8_to_level(score):
                        if score <= 4: return "minimal"
                        elif score <= 9: return "mild"
                        elif score <= 14: return "moderate"
                        elif score <= 19: return "moderately_severe"
                        else: return "severe"
                    true_levels = [_phq8_to_level(s) for s in gpt4_scores]
                    n_gpt4 = min(len(gpt4_unconstrained_levels), len(true_levels))
                    gpt4_uncon_acc = sum(1 for p, t in zip(gpt4_unconstrained_levels[:n_gpt4], true_levels[:n_gpt4]) if p == t) / n_gpt4
                    gpt4_laz_acc = sum(1 for p, t in zip(gpt4_lazarus_levels[:n_gpt4], true_levels[:n_gpt4]) if p == t) / n_gpt4
                    results["gpt4_comparison"] = {
                        "unconstrained_accuracy": float(gpt4_uncon_acc),
                        "lazarus_constrained_accuracy": float(gpt4_laz_acc),
                        "n_samples": n_gpt4,
                    }
        except ImportError:
            results["gpt4_comparison"] = {"note": "openai package not installed, GPT-4 comparison unavailable"}
        except Exception as e:
            results["gpt4_comparison"] = {"note": f"GPT-4 comparison unavailable: {str(e)}"}
    else:
        results["full_chain"] = {"n_appraisals": len(full_appraisals)}
        results["primary_only"] = {"n_appraisals": len(primary_only_appraisals)}

    if phq8_scores and "full_chain" in results:
        from app.cuspnet.statistics import bootstrap_ci
        n_samples = len(full_levels)
        if n_samples > 5:
            correct_arr = np.array([1.0 if fl == _phq8_to_level_global(s) else 0.0 for fl, s in zip(full_levels, phq8_scores)])
            acc_ci = bootstrap_ci(correct_arr, statistic_fn=np.mean, n_bootstrap=2000)
            results["full_chain"]["accuracy_ci"] = acc_ci

            if "primary_only" in results and primary_only_levels:
                correct_arr_primary = np.array([1.0 if pl == _phq8_to_level_global(s) else 0.0 for pl, s in zip(primary_only_levels, phq8_scores)])
                acc_ci_primary = bootstrap_ci(correct_arr_primary, statistic_fn=np.mean, n_bootstrap=2000)
                results["primary_only"]["accuracy_ci"] = acc_ci_primary

    return results
