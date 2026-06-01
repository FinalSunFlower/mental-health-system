"""
Pydantic schemas for API request/response validation.
Defines data models for risk classification, causal network results,
CUSP dynamics analysis, LLM appraisal outputs, and complete assessment responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CuspNetAssessmentRequest(BaseModel):
    student_id: int
    text_input: Optional[str] = None
    behavior_data: Optional[Dict[str, Any]] = None


class CausalNetworkResult(BaseModel):
    precision_matrix: List[List[float]]
    partial_correlation: List[List[float]]
    causal_adjacency: List[List[float]]
    topological_order: List[int]
    centrality_ranking: List[float]
    bridge_centrality: List[float] = []
    bridge_symptoms: List[str]
    positive_feedback_loops: List[Dict[str, Any]]
    variable_names: List[str] = []


class DynamicsResult(BaseModel):
    global_a: float
    global_b: float
    global_c: float
    local_params: List[Dict[str, Any]]
    attractor_states: Dict[str, Any]
    resilience_reserve: float
    critical_distance: float
    tipping_point_warning: bool
    potential_function: Dict[str, Any]
    drift_prediction: Optional[Dict[str, Any]] = None
    simulation: Optional[List[List[float]]] = None


class LLMAppraisalResult(BaseModel):
    primary_appraisal: Dict[str, Any]
    secondary_appraisal: Dict[str, Any]
    reappraisal: Dict[str, Any]
    cognitive_distortions: List[Dict[str, Any]]
    cusp_proxies: Dict[str, float]
    explanation: str
    intervention: str


class CuspNetAssessmentResponse(BaseModel):
    risk_score: float
    risk_level: RiskLevel
    causal_network: CausalNetworkResult
    dynamics: DynamicsResult
    llm_appraisal: Optional[LLMAppraisalResult] = None
    model_version: str = "CuspNet-1.0"
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class StudentBrief(BaseModel):
    id: int
    student_id: str
    name: str
    risk_level: Optional[RiskLevel] = None

    class Config:
        from_attributes = True
