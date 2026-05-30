from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
import numpy as np
from datetime import datetime
from typing import Optional

from app.core.database import engine, get_db
from app.core.models import Base, Student, CuspNetRecord
from app.core.schemas import (
    CuspNetAssessmentRequest,
    CuspNetAssessmentResponse,
    CausalNetworkResult,
    DynamicsResult,
    LLMAppraisalResult,
    RiskLevel,
)
from app.cuspnet import CuspNetEngine
from app.cuspnet.utils import (
    compute_resilience_reserve,
    compute_potential,
    compute_critical_distance,
    find_fixed_points,
    classify_fixed_points,
)
from app.visualization.potential_vis import plot_potential_surface
from app.visualization.causal_graph_vis import plot_causal_dag
from app.visualization.network_vis import plot_centrality_heatmap
from app.core.config import settings

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CuspNet Mental Health System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_instance = CuspNetEngine(
    config={
        "ebic_gamma": settings.EBIC_GAMMA,
        "score_threshold": settings.SCORE_THRESHOLD,
        "theta_bifurcation": settings.CUSP_THETA_BIFURCATION,
        "lambda_a": settings.ALLOCATION_LAMBDA_A,
        "lambda_b": settings.ALLOCATION_LAMBDA_B,
        "lambda_c": settings.ALLOCATION_LAMBDA_C,
        "drift_eta": settings.STATE_DRIFT_ETA,
        "llm_model_name": settings.LLM_MODEL_NAME,
        "llm_device": settings.LLM_DEVICE,
        "llm_load_in_4bit": settings.LLM_LOAD_IN_4BIT,
        "llm_max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
        "llm_temperature": settings.LLM_TEMPERATURE,
    }
)

_last_assessment: Optional[dict] = None


def _extract_student_scores(student: Student):
    phq9 = student.phq9_scores or [0] * 9
    gad7 = student.gad7_scores or [0] * 7
    pss10 = float(np.sum(student.pss10_scores)) if student.pss10_scores else 25.0
    cdrisc = float(np.sum(student.cdrisc_scores)) if student.cdrisc_scores else 20.0
    mspss = float(np.sum(student.mspss_scores)) if student.mspss_scores else 42.0
    return phq9, gad7, pss10, cdrisc, mspss


def _build_single_sample_result(variable_names, a, b, c, resilience, cd, warning, risk_score, risk_level, l3_result=None):
    from app.cuspnet.layer2_dynamics import CuspDynamicsLayer
    x_range = np.linspace(-2, 2, 200)
    V = compute_potential(x_range, a, b, c)
    roots = find_fixed_points(a, b, c)
    classified = classify_fixed_points(roots, b, c)
    stable = [p for p in classified if p["stability"] == "stable"]
    unstable = [p for p in classified if p["stability"] == "unstable"]

    attractor_states = {
        "fixed_points": classified,
        "is_bistable": len(stable) >= 2,
        "num_stable": len(stable),
        "num_unstable": len(unstable),
    }

    potential_data = {
        "a": a, "b": b, "c": c,
        "x_range": x_range.tolist(),
        "V": V.tolist(),
    }
    if attractor_states["is_bistable"]:
        if stable:
            potential_data["stable_fixed_points"] = [
                {"x": fp["value"], "V": float(compute_potential(np.array([fp["value"]]), a, b, c)[0])}
                for fp in stable
            ]
        if unstable:
            potential_data["unstable_fixed_points"] = [
                {"x": fp["value"], "V": float(compute_potential(np.array([fp["value"]]), a, b, c)[0])}
                for fp in unstable
            ]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "causal_network": {
            "precision_matrix": np.eye(len(variable_names)).tolist(),
            "partial_correlation": np.zeros((len(variable_names), len(variable_names))).tolist(),
            "causal_adjacency": np.zeros((len(variable_names), len(variable_names))).tolist(),
            "topological_order": list(range(len(variable_names))),
            "centrality_ranking": [0.0] * len(variable_names),
            "bridge_centrality": [0.0] * len(variable_names),
            "bridge_symptoms": [],
            "positive_feedback_loops": [],
            "variable_names": variable_names,
        },
        "dynamics": {
            "global_a": a,
            "global_b": b,
            "global_c": c,
            "local_params": [{"name": variable_names[i] if i < len(variable_names) else f"V{i}", "a": a, "b": b, "c": c} for i in range(len(variable_names))],
            "attractor_states": attractor_states,
            "resilience_reserve": resilience,
            "critical_distance": cd,
            "tipping_point_warning": warning,
            "potential_function": potential_data,
            "drift_prediction": None,
            "simulation": None,
        },
        "llm_appraisal": l3_result,
        "model_version": "CuspNet-1.0",
    }


def _save_record(db: Session, student_id: int, result: dict):
    record = CuspNetRecord(
        student_id=student_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        global_a=result["dynamics"]["global_a"],
        global_b=result["dynamics"]["global_b"],
        global_c=result["dynamics"]["global_c"],
        resilience_reserve=result["dynamics"]["resilience_reserve"],
        critical_distance=result["dynamics"]["critical_distance"],
        tipping_point_warning=result["dynamics"]["tipping_point_warning"],
        causal_adjacency=result["causal_network"]["causal_adjacency"],
        centrality_ranking=result["causal_network"]["centrality_ranking"],
        positive_feedback_loops=result["causal_network"]["positive_feedback_loops"],
        attractor_states=result["dynamics"]["attractor_states"],
        llm_primary=result.get("llm_appraisal", {}).get("primary_appraisal") if result.get("llm_appraisal") else None,
        llm_secondary=result.get("llm_appraisal", {}).get("secondary_appraisal") if result.get("llm_appraisal") else None,
        llm_distortions=result.get("llm_appraisal", {}).get("cognitive_distortions") if result.get("llm_appraisal") else None,
        llm_intervention=result.get("llm_appraisal", {}).get("intervention") if result.get("llm_appraisal") else None,
        llm_explanation=result.get("llm_appraisal", {}).get("explanation") if result.get("llm_appraisal") else None,
        model_version="CuspNet-1.0",
        assessed_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()


def _build_response(result: dict):
    cn = result["causal_network"]
    dy = result["dynamics"]
    llm = result.get("llm_appraisal")

    dynamics_result = DynamicsResult(
        global_a=dy["global_a"],
        global_b=dy["global_b"],
        global_c=dy["global_c"],
        local_params=dy["local_params"],
        attractor_states=dy["attractor_states"],
        resilience_reserve=dy["resilience_reserve"],
        critical_distance=dy["critical_distance"],
        tipping_point_warning=dy["tipping_point_warning"],
        potential_function=dy["potential_function"],
        drift_prediction=dy.get("drift_prediction"),
        simulation=dy.get("simulation"),
    )

    return CuspNetAssessmentResponse(
        risk_score=result["risk_score"],
        risk_level=RiskLevel(result["risk_level"]),
        causal_network=CausalNetworkResult(
            precision_matrix=cn["precision_matrix"],
            partial_correlation=cn["partial_correlation"],
            causal_adjacency=cn["causal_adjacency"],
            topological_order=cn["topological_order"],
            centrality_ranking=cn["centrality_ranking"],
            bridge_centrality=cn.get("bridge_centrality", []),
            bridge_symptoms=cn["bridge_symptoms"],
            positive_feedback_loops=cn["positive_feedback_loops"],
            variable_names=cn.get("variable_names", []),
        ),
        dynamics=dynamics_result,
        llm_appraisal=LLMAppraisalResult(**llm) if llm else None,
        model_version="CuspNet-1.0",
        assessed_at=datetime.utcnow(),
    )


@app.post("/api/assess", response_model=CuspNetAssessmentResponse)
async def assess_student(request: CuspNetAssessmentRequest, db: Session = Depends(get_db)):
    global _last_assessment
    student = db.query(Student).filter(Student.id == request.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    phq9, gad7, pss10, cdrisc, mspss = _extract_student_scores(student)
    questionnaire_data = np.array([phq9 + gad7], dtype=float)
    variable_names = [f"PHQ9_{i}" for i in range(9)] + [f"GAD7_{i}" for i in range(7)]

    if questionnaire_data.shape[0] < 2:
        pss_norm = pss10 / 50.0
        cdrisc_norm = cdrisc / 40.0
        mspss_norm = mspss / 84.0
        a = pss_norm - cdrisc_norm
        b = cdrisc_norm * mspss_norm - 0.5
        c = mspss_norm * 0.5
        resilience = compute_resilience_reserve(a, b, c)
        cd = compute_critical_distance(b)
        warning = cd < 0.15
        risk_score = min(1.0, (0.4 if resilience < 0.5 else 0.2) + (0.3 if warning else 0))
        risk_level = "critical" if warning else ("high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low")

        l3_result = None
        if request.text_input:
            causal_info = {"central_symptoms": [], "top_loops": []}
            dynamics_info = {"resilience_reserve": resilience, "critical_distance": cd, "tipping_point_warning": warning}
            try:
                l3_result = await run_in_threadpool(
                    engine_instance.layer3.full_chain, request.text_input, causal_info, dynamics_info
                )
            except Exception:
                pass

        result = _build_single_sample_result(variable_names, a, b, c, resilience, cd, warning, risk_score, risk_level, l3_result)
    else:
        if request.text_input:
            result = await run_in_threadpool(
                engine_instance.assess,
                questionnaire_data=questionnaire_data,
                variable_names=variable_names,
                pss10=pss10,
                cdrisc=cdrisc,
                mspss=mspss,
                text_input=request.text_input,
            )
        else:
            result = engine_instance.assess(
                questionnaire_data=questionnaire_data,
                variable_names=variable_names,
                pss10=pss10,
                cdrisc=cdrisc,
                mspss=mspss,
            )

    _save_record(db, student.id, result)
    _last_assessment = result
    return _build_response(result)


@app.get("/api/students/{student_id}/history")
def get_student_history(student_id: int, db: Session = Depends(get_db)):
    records = (
        db.query(CuspNetRecord)
        .filter(CuspNetRecord.student_id == student_id)
        .order_by(CuspNetRecord.assessed_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "risk_score": r.risk_score,
            "risk_level": r.risk_level,
            "resilience_reserve": r.resilience_reserve,
            "critical_distance": r.critical_distance,
            "tipping_point_warning": r.tipping_point_warning,
            "assessed_at": r.assessed_at.isoformat() if r.assessed_at else None,
        }
        for r in records
    ]


@app.get("/api/visualization/potential")
def get_potential_data(a: float = -0.2, b: float = 0.3, c: float = 1.0):
    x_range = np.linspace(-2, 2, 200)
    V = compute_potential(x_range, a, b, c)
    roots = find_fixed_points(a, b, c)
    classified = classify_fixed_points(roots, b, c)
    fixed_points = [
        {"x": p["value"], "V": float(compute_potential(np.array([p["value"]]), a, b, c)[0]), "stability": p["stability"]}
        for p in classified
    ]
    result = {"x": x_range.tolist(), "V": V.tolist(), "fixed_points": fixed_points}

    if _last_assessment and _last_assessment.get("dynamics"):
        dy = _last_assessment["dynamics"]
        result["current_state"] = {
            "a": dy["global_a"],
            "b": dy["global_b"],
            "c": dy["global_c"],
            "resilience_reserve": dy["resilience_reserve"],
            "tipping_point_warning": dy["tipping_point_warning"],
            "risk_level": _last_assessment["risk_level"],
        }
        if dy.get("drift_prediction"):
            result["current_state"]["drift_prediction"] = dy["drift_prediction"]

    return result


@app.get("/api/visualization/causal-graph")
def get_causal_graph_data():
    if _last_assessment is None or _last_assessment.get("causal_network") is None:
        return {"nodes": [], "edges": [], "message": "Run /api/assess first to generate causal graph"}

    cn = _last_assessment["causal_network"]
    adj = cn["causal_adjacency"]
    var_names = cn.get("variable_names", [])
    centrality = cn["centrality_ranking"]
    bridge_symptoms = cn.get("bridge_symptoms", [])

    if isinstance(adj, np.ndarray):
        adj = adj.tolist()

    dag_data = plot_causal_dag(adj, var_names, centrality, bridge_symptoms)
    heatmap_data = plot_centrality_heatmap(adj, centrality, var_names)

    return {
        "dag": dag_data,
        "heatmap": heatmap_data,
        "bridge_symptoms": bridge_symptoms,
        "positive_feedback_loops": cn.get("positive_feedback_loops", []),
    }


@app.get("/api/visualization/simulation")
def get_simulation_data():
    if _last_assessment is None or _last_assessment.get("dynamics") is None:
        return {"message": "Run /api/assess first to generate simulation data"}

    dy = _last_assessment["dynamics"]
    sim = dy.get("simulation")
    if sim is None:
        return {"message": "No simulation data available for this assessment"}

    return {
        "time_steps": list(range(len(sim))),
        "trajectories": sim,
        "variable_names": _last_assessment["causal_network"].get("variable_names", []),
        "global_params": {
            "a": dy["global_a"],
            "b": dy["global_b"],
            "c": dy["global_c"],
        },
        "attractor_states": dy["attractor_states"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": "CuspNet-1.0"}
