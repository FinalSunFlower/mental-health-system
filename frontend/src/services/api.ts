import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export interface PotentialData {
  x: number[]
  V: number[]
  fixed_points: { x: number; V: number; stability: string }[]
  current_state?: {
    a: number
    b: number
    c: number
    resilience_reserve: number
    tipping_point_warning: boolean
    risk_level: string
  }
}

export interface CausalGraphData {
  nodes: any[]
  edges: any[]
  message?: string
}

export interface AssessmentRequest {
  student_id: number
  text_input?: string
}

export interface AssessmentResponse {
  risk_score: number
  risk_level: string
  causal_network: any
  dynamics: any
  llm_appraisal: any
  model_version: string
  assessed_at: string
}

export const fetchPotentialData = (a: number, b: number, c: number) =>
  api.get<PotentialData>('/visualization/potential', { params: { a, b, c } })

export const fetchCausalGraph = () =>
  api.get<CausalGraphData>('/visualization/causal-graph')

export const submitAssessment = (data: AssessmentRequest) =>
  api.post<AssessmentResponse>('/assess', data)

export const fetchStudentHistory = (studentId: number) =>
  api.get(`/students/${studentId}/history`)

export const fetchSimulationData = () =>
  api.get('/visualization/simulation')

export const healthCheck = () =>
  api.get('/health')

export default api
