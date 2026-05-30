import numpy as np
import os
from typing import Dict, List, Optional
from app.cuspnet.layer3_llm import LazarusAppraisalChain
from app.data.loaders import DAICWOZLoader


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
    composite = (primary_score * 0.4 + (10 - secondary_score) * 0.3 + min(n_distortions, 5) * 0.6)
    if composite <= 3:
        return "minimal"
    elif composite <= 5:
        return "mild"
    elif composite <= 7:
        return "moderate"
    elif composite <= 9:
        return "moderately_severe"
    else:
        return "severe"


def run_exp4(
    dataset: str = "daic_woz",
    use_lazarus_constraints: bool = True,
    n_reflection_steps: int = 3,
    max_samples: Optional[int] = None,
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
        if os.path.isdir(erisk_dir):
            import pandas as pd
            for fname in sorted(os.listdir(erisk_dir)):
                fpath = os.path.join(erisk_dir, fname)
                if fname.endswith(".csv"):
                    try:
                        df = pd.read_csv(fpath)
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
                                    phq8_scores.append(label * 5)
                    except Exception:
                        continue
        elif os.path.exists(erisk_csv):
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
                            phq8_scores.append(label * 5)
            except Exception:
                pass
        if not interviews:
            raise FileNotFoundError(
                "eRisk dataset not found. Please download from https://early.irlab.org/ "
                "and place data in app/data/raw/erisk/ or as erisk.csv"
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'daic_woz' or 'erisk'.")

    if max_samples is not None and max_samples > 0:
        interviews = interviews[:max_samples]
        phq8_scores = phq8_scores[:max_samples]

    results["dataset_info"] = {
        "name": dataset,
        "n_interviews": len(interviews),
        "n_with_phq8": len(phq8_scores),
    }

    llm_full = LazarusAppraisalChain(model_name=model_name)

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
            primary_only_appraisals.append({
                "primary_appraisal": primary,
                "secondary_appraisal": {"secondary_appraisal_score": 5},
                "reappraisal": {"adjustment_direction": "maintain"},
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
            elif score <= 9:
                threat = "moderate"
                coping = "moderate"
            elif score <= 14:
                threat = "high"
                coping = "low"
            else:
                threat = "critical"
                coping = "very_low"
            gt_appraisals.append({
                "primary_appraisal": {"threat_level": threat},
                "secondary_appraisal": {"coping_potential": coping},
                "reappraisal": {"adjustment_direction": "maintain" if threat == "low" else "increase_coping"},
                "cognitive_distortions": {"detected": []},
            })

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

        try:
            from openai import OpenAI
            import os
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

    return results
