import { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Tag, Slider, Progress, Space, Spin, Alert } from 'antd'
import ReactECharts from 'echarts-for-react'
import {
  ApartmentOutlined,
  LineChartOutlined,
  MessageOutlined,
  ArrowRightOutlined,
  ArrowDownOutlined,
  ApiOutlined,
  DisconnectOutlined,
} from '@ant-design/icons'
import { fetchCausalGraph, fetchPotentialData } from '../services/api'

const { Title, Paragraph, Text } = Typography

interface CausalNode {
  id: string
  label: string
  x?: number
  y?: number
}

interface CausalEdge {
  source: string
  target: string
  weight: number
}

interface FixedPoint {
  x: number
  V: number
  stability: string
}

function solveCubic(a: number, b: number, c: number): number[] {
  const roots: number[] = []
  const p = -b / c
  const q = -a / c
  const disc = -4 * p * p * 3 - 27 * q * q
  if (disc > 0) {
    for (let k = 0; k < 3; k++) {
      const theta = Math.acos(Math.max(-1, Math.min(1, (3 * q) / (2 * p) * Math.sqrt(-3 / p))))
      const x = 2 * Math.sqrt(-p / 3) * Math.cos((theta + 2 * Math.PI * k) / 3)
      if (Math.abs(x) < 10) roots.push(x)
    }
  } else {
    const sqrtTerm = Math.sqrt((q / 2) ** 2 + (p / 3) ** 3)
    const A = -q / 2 + sqrtTerm
    const B = -q / 2 - sqrtTerm
    const x = Math.cbrt(A) + Math.cbrt(B)
    if (Math.abs(x) < 10) roots.push(x)
  }
  return roots.sort((x, y) => x - y)
}

function computePotential(xArr: number[], a: number, b: number, c: number): number[] {
  return xArr.map((x) => -a * x - (b / 2) * x * x + (c / 4) * x * x * x * x)
}

function computeResilience(a: number, b: number, c: number): number {
  const roots = solveCubic(a, b, c)
  if (roots.length < 3) return Infinity
  const V = (x: number) => -a * x - (b / 2) * x * x + (c / 4) * x * x * x * x
  return Math.min(V(roots[1]) - V(roots[0]), V(roots[1]) - V(roots[2]))
}

const DEFAULT_NODES: CausalNode[] = [
  { id: 'insomnia', label: '失眠', x: 50, y: 25 },
  { id: 'fatigue', label: '疲劳', x: 140, y: 40 },
  { id: 'anhedonia', label: '快感缺失', x: 40, y: 100 },
  { id: 'depressed_mood', label: '情绪低落', x: 135, y: 105 },
  { id: 'stress', label: '压力', x: 15, y: 62 },
  { id: 'anxiety', label: '焦虑', x: 90, y: 65 },
  { id: 'concentration', label: '注意力', x: 190, y: 55 },
]

const DEFAULT_EDGES: CausalEdge[] = [
  { source: 'stress', target: 'anxiety', weight: 0.8 },
  { source: 'anxiety', target: 'insomnia', weight: 0.75 },
  { source: 'insomnia', target: 'fatigue', weight: 0.85 },
  { source: 'fatigue', target: 'concentration', weight: 0.8 },
  { source: 'anhedonia', target: 'depressed_mood', weight: 0.9 },
  { source: 'depressed_mood', target: 'insomnia', weight: 0.55 },
  { source: 'stress', target: 'insomnia', weight: 0.5 },
  { source: 'fatigue', target: 'depressed_mood', weight: 0.6 },
  { source: 'anxiety', target: 'depressed_mood', weight: 0.55 },
]

function CausalMiniGraph({ nodes, edges, isLoading }: { nodes: CausalNode[]; edges: CausalEdge[]; isLoading: boolean }) {
  const displayNodes = nodes.length > 0 ? nodes : DEFAULT_NODES
  const displayEdges = edges.length > 0 ? edges : DEFAULT_EDGES

  const nodePositions: Record<string, { x: number; y: number }> = {}
  displayNodes.forEach((n, i) => {
    nodePositions[n.id] = { x: n.x ?? (30 + (i % 4) * 50), y: n.y ?? (20 + Math.floor(i / 4) * 40) }
  })

  const option = {
    animation: false,
    grid: { top: 4, right: 4, bottom: 4, left: 4 },
    xAxis: { type: 'value', min: 0, max: 210, show: false },
    yAxis: { type: 'value', min: 0, max: 135, show: false },
    series: [{
      type: 'graph',
      layout: 'none',
      data: displayNodes.map((n) => ({
        name: n.label,
        value: n.label,
        x: nodePositions[n.id]?.x ?? 50,
        y: nodePositions[n.id]?.y ?? 50,
        symbolSize: n.label.length > 3 ? 24 : 20,
        itemStyle: {
          color: ['情绪低落', '快感缺失', '失眠', '疲劳'].includes(n.label) ? '#8B5CF6' : '#1E1E38',
          borderColor: '#8B5CF6',
          borderWidth: ['情绪低落', '快感缺失', '失眠', '疲劳'].includes(n.label) ? 2 : 1,
          shadowColor: 'rgba(139, 92, 246, 0.25)',
          shadowBlur: ['情绪低落', '快感缺失', '失眠', '疲劳'].includes(n.label) ? 6 : 0,
        },
        label: { show: true, fontSize: 9, color: '#E0E0F0', fontFamily: '"Chakra Petch", sans-serif' },
      })),
      links: displayEdges.map((e) => ({
        source: e.source,
        target: e.target,
        lineStyle: {
          color: e.weight > 0.7 ? 'rgba(139, 92, 246, 0.6)' : 'rgba(42, 42, 74, 0.4)',
          width: Math.max(e.weight * 2.2, 1),
          curveness: 0.15,
        },
        symbol: ['none', 'arrow'] as const,
        symbolSize: 4,
      })),
    }],
  }

  return (
    <div>
      {isLoading && <Spin size="small" style={{ display: 'block', marginBottom: 4 }} />}
      <ReactECharts option={option} style={{ width: '100%', height: 300 }} />
    </div>
  )
}

function CuspMiniChart({ fixedPoints, a, b, c, isLoading }: { fixedPoints: FixedPoint[]; a: number; b: number; c: number; isLoading: boolean }) {
  const [pss10, setPss10] = useState(25)
  const [cdrisc, setCdrisc] = useState(28)

  const computedA = a !== 0 ? a : ((pss10 - 19.6) / 6.4 - (cdrisc - 30.5) / 6.3) * 0.8
  const computedB = b !== 0 ? b : ((cdrisc - 30.5) / 6.3) * ((55 - 60.2) / 14.8) * 1.5 - 0.5
  const computedC = c !== 0 ? c : Math.max(((55 - 60.2) / 14.8) * 0.5 + 0.5, 0.1)

  const resilience = computeResilience(computedA, computedB, computedC)
  const roots = solveCubic(computedA, computedB, computedC)
  const isBistable = roots.length >= 3
  const resilienceNorm = resilience === Infinity ? 1.0 : Math.min(Math.max(resilience / 2, 0), 1)
  const riskLevel = resilienceNorm > 0.6 ? 'low' : resilienceNorm > 0.3 ? 'moderate' : 'high'

  const xArr = Array.from({ length: 80 }, (_, i) => -2 + (4 * i) / 79)
  const yArr = computePotential(xArr, computedA, computedB, computedC)
  const yMin = Math.min(...yArr)
  const yMax = Math.max(...yArr)
  const yRange = yMax - yMin || 1

  const displayFixedPoints = fixedPoints.length > 0 ? fixedPoints : roots.slice(0, 3).map((r, i) => ({
    x: r,
    V: -computedA * r - (computedB / 2) * r * r + (computedC / 4) * r * r * r * r,
    stability: i === 1 ? 'unstable' : 'stable',
  }))

  const markPoints = displayFixedPoints.map((fp, i) => ({
    coord: [fp.x, ((fp.V - yMin) / yRange) * 100],
    name: fp.stability === 'unstable' ? '鞍点' : i === 0 ? '健康态' : '抑郁态',
    itemStyle: { color: fp.stability === 'unstable' ? '#FF6B00' : i === 0 ? '#39FF14' : '#FF0080' },
    symbolSize: 8,
  }))

  const option = {
    animation: true,
    grid: { top: 14, right: 12, bottom: 24, left: 36 },
    xAxis: { type: 'value', min: -2, max: 2, axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.25)' } }, axisLabel: { fontSize: 8, color: '#6B6B8A' } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.25)' } }, axisLabel: { fontSize: 8, color: '#6B6B8A' } },
    series: [{
      type: 'line',
      data: xArr.map((x, i) => [x, ((yArr[i] - yMin) / yRange) * 100]),
      smooth: true,
      lineStyle: { color: '#8B5CF6', width: 2 },
      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(139, 92, 246, 0.2)' }, { offset: 1, color: 'rgba(139, 92, 246, 0.02)' }] } },
      markPoint: { data: markPoints, label: { show: true, formatter: '{b}', fontSize: 8, color: '#E0E0F0' } },
    }],
  }

  return (
    <div>
      {isLoading && <Spin size="small" style={{ display: 'block', marginBottom: 4 }} />}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
        <Space size={8}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ color: '#00F0FF', fontSize: 9, fontFamily: 'var(--font-mono)' }}>压力</Text>
            <Slider min={0} max={50} value={pss10} onChange={setPss10} style={{ width: 70 }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ color: '#39FF14', fontSize: 9, fontFamily: 'var(--font-mono)' }}>韧性</Text>
            <Slider min={0} max={40} value={cdrisc} onChange={setCdrisc} style={{ width: 70 }} />
          </div>
        </Space>
        <Tag color={riskLevel === 'low' ? 'green' : riskLevel === 'moderate' ? 'orange' : 'red'} style={{ fontSize: 9, padding: '0 6px', marginLeft: 'auto' }}>
          {riskLevel === 'low' ? '低风险' : riskLevel === 'moderate' ? '中风险' : '高风险'}
        </Tag>
      </div>
      <ReactECharts option={option} style={{ width: '100%', height: 240 }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <Space size={10}>
          <span><Text style={{ color: 'var(--ash)', fontSize: 9 }}>a=</Text><Text strong style={{ color: '#8B5CF6', fontSize: 10 }}>{computedA.toFixed(2)}</Text></span>
          <span><Text style={{ color: 'var(--ash)', fontSize: 9 }}>b=</Text><Text strong style={{ color: '#00F0FF', fontSize: 10 }}>{computedB.toFixed(2)}</Text></span>
          <Tag color={isBistable ? 'purple' : 'blue'} style={{ fontSize: 8, margin: 0 }}>{isBistable ? '双稳态' : '单稳态'}</Tag>
        </Space>
        <Text style={{ color: 'var(--ash)', fontSize: 9 }}>韧性: <Text strong style={{ color: resilienceNorm > 0.5 ? '#39FF14' : '#FF6B00' }}>{Math.round(resilienceNorm * 100)}%</Text></Text>
      </div>
    </div>
  )
}

function AppraisalFlow() {
  const STEPS = [
    { icon: '\u{1F50D}', title: 'Primary', sub: '初级评估', color: '#FF0080' },
    { icon: '\u{1F6E1}\uFE0F', title: 'Secondary', sub: '次级评估', color: '#00F0FF' },
    { icon: '\u{1F504}', title: 'Reappraisal', sub: '重评估', color: '#8B5CF6' },
    { icon: '\u26A0\uFE0F', title: 'Distortion', sub: '扭曲检测', color: '#FFD700' },
  ]

  const TECHS = [
    { name: 'CoVe 验证', tag: 'Meta 2023', pct: 94 },
    { name: 'Self-Critique', tag: 'Constitutional AI', pct: 88 },
    { name: 'Risk Floor', tag: 'Safety Guardrail', pct: 96 },
    { name: '逆关系约束', tag: 'Lazarus Theory', pct: 82 },
  ]

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        {STEPS.map((step, idx) => (
          <div key={step.title} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: idx < STEPS.length - 1 ? 4 : 0 }}>
            <span style={{ fontSize: 13 }}>{step.icon}</span>
            <div style={{
              flex: 1, background: 'var(--graphite)', borderRadius: 4, padding: '3px 8px',
              borderLeft: `2px solid ${step.color}`,
            }}>
              <Text strong style={{ color: step.color, fontSize: 10, fontFamily: 'var(--font-display)' }}>{step.title}</Text>
              <Text style={{ color: 'var(--ash)', fontSize: 9, marginLeft: 6 }}>{step.sub}</Text>
            </div>
            {idx < STEPS.length - 1 && <ArrowDownOutlined style={{ color: step.color, fontSize: 10, opacity: 0.4, flexShrink: 0 }} />}
          </div>
        ))}
      </div>

      <div style={{ borderTop: '1px solid rgba(255, 0, 128, 0.15)', paddingTop: 7 }}>
        <Row gutter={[6, 4]}>
          {TECHS.map((tech) => (
            <Col span={12} key={tech.name}>
              <div style={{ background: 'rgba(139, 92, 246, 0.04)', borderRadius: 4, padding: '4px 6px', border: '1px solid rgba(139, 92, 246, 0.1)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text strong style={{ color: 'var(--ghost)', fontSize: 9.5, fontFamily: 'var(--font-display)' }}>{tech.name}</Text>
                  <Progress percent={tech.pct} size={60}
                    strokeColor={tech.pct >= 90 ? '#39FF14' : tech.pct >= 80 ? '#00F0FF' : '#FFD700'}
                    format={(p) => <span style={{ fontSize: 8, fontWeight: 700 }}>{p}%</span>}
                  />
                </div>
                <Text style={{ color: 'var(--ash)', fontSize: 8, display: 'block' }}>{tech.tag}</Text>
              </div>
            </Col>
          ))}
        </Row>
      </div>
    </div>
  )
}

export default function Architecture() {
  const [zoom, setZoom] = useState(1)
  const [apiStatus, setApiStatus] = useState<'loading' | 'api' | 'local'>('loading')
  const [causalNodes, setCausalNodes] = useState<CausalNode[]>([])
  const [causalEdges, setCausalEdges] = useState<CausalEdge[]>([])
  const [fixedPoints, setFixedPoints] = useState<FixedPoint[]>([])
  const [cuspParams, setCuspParams] = useState({ a: 0, b: 0, c: 0 })

  useEffect(() => {
    const update = () => {
      const baseW = 1200
      const baseH = 600
      const scaleX = (window.innerWidth - 48) / baseW
      const scaleY = (window.innerHeight - 144) / baseH
      setZoom(Math.max(0.85, Math.min(scaleX, scaleY, 1.8)))
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  useEffect(() => {
    const fetchData = async () => {
      setApiStatus('loading')
      try {
        const [causalRes, potentialRes] = await Promise.all([
          fetchCausalGraph(),
          fetchPotentialData(0.5, -1.5, 0),
        ])

        const causalData = causalRes.data
        const potentialData = potentialRes.data

        if (causalData && causalData.nodes && causalData.nodes.length > 0) {
          setCausalNodes(causalData.nodes)
          setCausalEdges(causalData.edges || [])
        }

        if (potentialData && potentialData.fixed_points) {
          setFixedPoints(potentialData.fixed_points)
          if (potentialData.current_state) {
            setCuspParams({
              a: potentialData.current_state.a,
              b: potentialData.current_state.b,
              c: potentialData.current_state.c,
            })
          }
        }

        setApiStatus('api')
      } catch (error) {
        console.log('API unavailable, using local data')
        setApiStatus('local')
      }
    }
    fetchData()
  }, [])

  const LAYERS = [
    {
      num: '01',
      icon: <ApartmentOutlined />,
      title: '因果发现',
      subtitle: 'Layer 1 · Causal Discovery',
      accent: '#00F0FF',
      bg: 'rgba(0, 240, 255, 0.03)',
      border: 'rgba(0, 240, 255, 0.18)',
      features: ['EBIC Glasso 稀疏精度最优', '理论约束注入 DSM-5/Borsboom', '桥接症状 中心性+介数双指标', '正反馈环 SCC算法检测'],
      output: '因果邻接矩阵 → 核心症状集 → 反馈环列表',
      chart: <CausalMiniGraph nodes={causalNodes} edges={causalEdges} isLoading={apiStatus === 'loading'} />,
    },
    {
      num: '02',
      icon: <LineChartOutlined />,
      title: 'Cusp动力学',
      subtitle: 'Layer 2 · Cusp Dynamics',
      accent: '#8B5CF6',
      bg: 'rgba(139, 92, 246, 0.03)',
      border: 'rgba(139, 92, 246, 0.18)',
      features: ['势函数 V(x)=-ax-bx²/2+cx⁴/4', '分岔分析 双稳态/单稳态判定', '韧性量化 ΔV势垒高度度量', '早期预警 临界慢化+EWS信号'],
      output: 'Cusp参数(a,b,c) → 韧性储备ΔV → Tipping预警',
      chart: <CuspMiniChart fixedPoints={fixedPoints} a={cuspParams.a} b={cuspParams.b} c={cuspParams.c} isLoading={apiStatus === 'loading'} />,
    },
    {
      num: '03',
      icon: <MessageOutlined />,
      title: 'LLM认知评估',
      subtitle: 'Layer 3 · LLM Appraisal',
      accent: '#FF0080',
      bg: 'rgba(255, 0, 128, 0.03)',
      border: 'rgba(255, 0, 128, 0.18)',
      features: ['Lazarus链式 Primary→Secondary→Reappraisal', 'CoVe验证 生成-验证-修正闭环', 'Self-Critique 自我批判+规则约束', 'Risk-Sensitive 关键词兜底+动态加权'],
      output: '威胁评分+应对评分+认知扭曲 → 抑郁等级',
      chart: <AppraisalFlow />,
    },
  ]

  return (
    <div style={{
      height: 'calc(100vh - 144px)',
      overflow: 'hidden',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'flex-start',
    }}>
      <div style={{ zoom, width: 1400 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ flexShrink: 0, display: 'flex', alignItems: 'baseline', gap: 16 }}>
            <Title level={3} style={{ color: 'var(--ghost)', margin: 0, fontFamily: 'var(--font-display)', fontSize: 19 }}>
              核心架构
            </Title>
            <Paragraph style={{ color: 'var(--ash)', margin: 0, fontSize: 12, fontFamily: 'var(--font-body)' }}>
              三层融合 — 从症状网络到语义理解的跨粒度评估流水线
            </Paragraph>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
              {apiStatus === 'loading' && <Spin size="small" />}
              {apiStatus === 'api' && (
                <Tag icon={<ApiOutlined />} color="success" style={{ fontSize: 10 }}>
                  后端API
                </Tag>
              )}
              {apiStatus === 'local' && (
                <Tag icon={<DisconnectOutlined />} color="warning" style={{ fontSize: 10 }}>
                  本地数据
                </Tag>
              )}
            </div>
          </div>

          {apiStatus === 'api' && (
            <Alert title="已连接后端API，显示真实因果网络和势函数数据" type="success" showIcon style={{ marginBottom: 4, background: 'rgba(57, 255, 20, 0.08)', border: '1px solid rgba(57, 255, 20, 0.3)' }} />
          )}

          <Row gutter={[12, 12]}>
            {LAYERS.map((layer) => (
              <Col xs={24} md={8} key={layer.title}>
                <Card
                  style={{
                    height: '100%',
                    background: layer.bg,
                    border: `1px solid ${layer.border}`,
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                  styles={{ body: { padding: 14, flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' } }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexShrink: 0 }}>
                    <div style={{
                      width: 30, height: 30, borderRadius: 8,
                      background: `${layer.accent}12`, border: `1px solid ${layer.accent}25`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: layer.accent, fontSize: 14,
                    }}>
                      {layer.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <Text style={{ color: `${layer.accent}80`, fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{layer.num}</Text>
                        <Text strong style={{ color: 'var(--ghost)', fontSize: 14, fontFamily: 'var(--font-display)' }}>{layer.title}</Text>
                      </div>
                      <Text style={{ color: 'var(--ash)', fontSize: 9, fontFamily: 'var(--font-mono)' }}>{layer.subtitle}</Text>
                    </div>
                  </div>

                  <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', marginBottom: 8 }}>
                    {layer.chart}
                  </div>

                  <div style={{ flexShrink: 0 }}>
                    <div style={{ marginBottom: 6 }}>
                      {layer.features.map((f) => (
                        <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3 }}>
                          <div style={{ width: 3, height: 3, borderRadius: '50%', background: layer.accent, flexShrink: 0 }} />
                          <Text style={{ color: 'var(--silver)', fontSize: 9.5, lineHeight: 1.35 }}>{f}</Text>
                        </div>
                      ))}
                    </div>

                    <div style={{
                      background: `${layer.accent}06`, borderRadius: 4, padding: '5px 8px',
                      border: `1px solid ${layer.accent}12`,
                    }}>
                      <Text style={{ color: `${layer.accent}AA`, fontSize: 8, fontFamily: 'var(--font-mono)', display: 'block' }}>OUTPUT</Text>
                      <Text style={{ color: 'var(--silver)', fontSize: 9.5, lineHeight: 1.4 }}>{layer.output}</Text>
                    </div>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>

          <Card
            style={{
              background: 'linear-gradient(90deg, rgba(0, 240, 255, 0.05), rgba(139, 92, 246, 0.06), rgba(255, 0, 128, 0.05))',
              border: '1px solid rgba(139, 92, 246, 0.18)',
            }}
            styles={{ body: { padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 16 } }}
          >
            {[
              { label: 'LAYER 1', desc: '因果网络结构 → 桥接症状集合', color: '#00F0FF' },
              { label: 'LAYER 2', desc: 'Cusp参数(a,b,c) → 韧性ΔV', color: '#8B5CF6' },
              { label: 'LAYER 3', desc: '抑郁等级 + 干预建议', color: '#FF0080' },
            ].map((item, idx) => (
              <div key={item.label} style={{ flex: 1, textAlign: 'center', display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ flex: 1 }}>
                  <Text style={{ color: item.color, fontSize: 10, fontWeight: 700, fontFamily: 'var(--font-mono)', display: 'block' }}>{item.label}</Text>
                  <Text style={{ color: 'var(--ghost)', fontSize: 11, fontFamily: 'var(--font-display)' }}>{item.desc}</Text>
                </div>
                {idx < 2 && <ArrowRightOutlined style={{ color: item.color, fontSize: 13, flexShrink: 0 }} />}
              </div>
            ))}
          </Card>
        </div>
      </div>
    </div>
  )
}
