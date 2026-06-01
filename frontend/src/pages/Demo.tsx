import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, Row, Col, Typography, Slider, Button, Tag, Alert, Spin } from 'antd'
import { PlayCircleOutlined, ReloadOutlined, ExperimentOutlined, ApiOutlined, DisconnectOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { fetchPotentialData } from '../services/api'

const { Title, Paragraph, Text } = Typography

interface PotentialPoint {
  x: number
  y: number
}

interface BifurcationPoint {
  a: number
  x_eq: number
  stable: boolean
}

interface FixedPoint {
  x: number
  V: number
  stability: string
}

export default function Demo() {
  const [a, setA] = useState(0.5)
  const [b, setB] = useState(-1.5)
  const [c, setC] = useState(0)
  const [potentialData, setPotentialData] = useState<PotentialPoint[]>([])
  const [bifurcationData, setBifurcationData] = useState<BifurcationPoint[]>([])
  const [fixedPoints, setFixedPoints] = useState<FixedPoint[]>([])
  const [apiStatus, setApiStatus] = useState<'loading' | 'api' | 'local'>('loading')
  const [animating, setAnimating] = useState(false)
  const animRef = useRef<number | null>(null)
  const [zoom, setZoom] = useState(1)

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

  const cuspPotential = (x: number, a: number, b: number, c: number) =>
    a * x ** 4 / 4 + b * x ** 2 / 2 + c * x

  const generatePotential = useCallback((aVal: number, bVal: number, cVal: number) => {
    const pts: PotentialPoint[] = []
    for (let x = -3; x <= 3; x += 0.02) {
      pts.push({ x: parseFloat(x.toFixed(2)), y: cuspPotential(x, aVal, bVal, cVal) })
    }
    return pts
  }, [])

  const generateBifurcation = useCallback((aVal: number, cVal: number) => {
    const pts: BifurcationPoint[] = []
    for (let bVal = -3; bVal <= 3; bVal += 0.05) {
      const roots = findEquilibria(aVal, bVal, cVal)
      roots.forEach((x_eq) => {
        const stable = isStable(aVal, bVal, x_eq)
        pts.push({ a: parseFloat(bVal.toFixed(2)), x_eq: parseFloat(x_eq.toFixed(4)), stable })
      })
    }
    return pts
  }, [])

  const findEquilibria = (a: number, b: number, c: number): number[] => {
    const roots: number[] = []
    for (let x = -3; x <= 3; x += 0.01) {
      const f = a * x ** 3 + b * x + c
      const fx = x + 0.01
      const ff = a * fx ** 3 + b * fx + c
      if (f * ff < 0) {
        let r = (x + fx) / 2
        for (let i = 0; i < 50; i++) {
          const fr = a * r ** 3 + b * r + c
          const dfr = 3 * a * r ** 2 + b
          if (Math.abs(dfr) < 1e-12) break
          r = r - fr / dfr
        }
        const isDup = roots.some((rr) => Math.abs(rr - r) < 0.01)
        if (!isDup && Math.abs(a * r ** 3 + b * r + c) < 0.01) {
          roots.push(r)
        }
      }
    }
    return roots
  }

  const isStable = (a: number, b: number, x: number): boolean => {
    return 3 * a * x ** 2 + b < 0
  }

  const loadDataFromAPI = useCallback(async () => {
    setApiStatus('loading')
    try {
      const response = await fetchPotentialData(a, b, c)
      const data = response.data
      
      if (data && data.x && data.x.length > 0 && data.V && data.V.length > 0) {
        const pts = data.x.map((x: number, i: number) => ({
          x,
          y: data.V[i],
        }))
        setPotentialData(pts)
        setFixedPoints(data.fixed_points || [])
        setApiStatus('api')
        return true
      }
    } catch (error) {
      console.log('API unavailable, using local calculation')
    }
    
    setPotentialData(generatePotential(a, b, c))
    const localFp = findEquilibria(a, b, c).map(x => ({
      x,
      V: cuspPotential(x, a, b, c),
      stability: isStable(a, b, x) ? 'stable' : 'unstable'
    }))
    setFixedPoints(localFp)
    setApiStatus('local')
    return false
  }, [a, b, c, generatePotential])

  useEffect(() => {
    loadDataFromAPI()
    setBifurcationData(generateBifurcation(a, c))
  }, [a, b, c, loadDataFromAPI, generateBifurcation])

  const handleAnimate = () => {
    if (animating) {
      if (animRef.current) cancelAnimationFrame(animRef.current)
      setAnimating(false)
      return
    }
    setAnimating(true)
    let bVal = -3
    const step = () => {
      bVal += 0.03
      if (bVal > 3) {
        setAnimating(false)
        return
      }
      setB(bVal)
      animRef.current = requestAnimationFrame(step)
    }
    animRef.current = requestAnimationFrame(step)
  }

  const equilibria = findEquilibria(a, b, c)
  const stableEq = equilibria.filter((x) => isStable(a, b, x))
  const unstableEq = equilibria.filter((x) => !isStable(a, b, x))
  const isBistable = stableEq.length >= 2

  const potentialOption = {
    title: { text: '势函数 V(x)', textStyle: { color: '#E0E0F0', fontFamily: '"Chakra Petch", sans-serif', fontSize: 14, fontWeight: 700 } },
    tooltip: { trigger: 'axis', backgroundColor: '#151528', borderColor: '#2A2A4A', textStyle: { color: '#E0E0F0' } },
    grid: { top: 40, right: 20, bottom: 30, left: 50 },
    xAxis: { type: 'value', name: 'x', min: -3, max: 3, axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.3)' } } },
    yAxis: { type: 'value', name: 'V(x)', axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.3)' } } },
    series: [
      {
        type: 'line',
        data: potentialData.map((p) => [p.x, p.y]),
        smooth: true,
        lineStyle: { color: '#8B5CF6', width: 3 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(139, 92, 246, 0.3)' },
              { offset: 1, color: 'rgba(139, 92, 246, 0.02)' },
            ],
          },
        },
        markPoint: {
          data: [
            ...fixedPoints.filter(fp => fp.stability === 'stable').map((fp) => ({
              coord: [fp.x, fp.V],
              name: '稳定',
              symbolSize: 14,
              itemStyle: { color: '#39FF14' },
            })),
            ...fixedPoints.filter(fp => fp.stability === 'unstable').map((fp) => ({
              coord: [fp.x, fp.V],
              name: '不稳定',
              symbolSize: 12,
              itemStyle: { color: '#FF0080' },
              symbol: 'diamond',
            })),
          ],
          label: { show: true, formatter: '{b}', fontSize: 10, color: '#E0E0F0' },
        },
      },
    ],
  }

  const stablePts = bifurcationData.filter((p) => p.stable)
  const unstablePts = bifurcationData.filter((p) => !p.stable)

  const bifurcationOption = {
    title: { text: '分岔图', textStyle: { color: '#E0E0F0', fontFamily: '"Chakra Petch", sans-serif', fontSize: 14, fontWeight: 700 } },
    tooltip: { trigger: 'axis', backgroundColor: '#151528', borderColor: '#2A2A4A', textStyle: { color: '#E0E0F0' } },
    grid: { top: 40, right: 20, bottom: 30, left: 50 },
    xAxis: { type: 'value', name: 'b (分岔参数)', min: -3, max: 3, axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.3)' } } },
    yAxis: { type: 'value', name: 'x* (平衡态)', min: -3, max: 3, axisLine: { lineStyle: { color: '#2A2A4A' } }, splitLine: { lineStyle: { color: 'rgba(42, 42, 74, 0.3)' } } },
    series: [
      {
        type: 'scatter',
        name: '稳定平衡态',
        data: stablePts.map((p) => [p.a, p.x_eq]),
        symbolSize: 3,
        itemStyle: { color: '#39FF14' },
      },
      {
        type: 'scatter',
        name: '不稳定平衡态',
        data: unstablePts.map((p) => [p.a, p.x_eq]),
        symbolSize: 3,
        itemStyle: { color: '#FF0080' },
      },
      {
        type: 'line',
        markLine: {
          data: [{ xAxis: b, label: { formatter: `b=${b.toFixed(2)}`, color: '#FFD700' }, lineStyle: { color: '#FFD700', type: 'dashed' } }],
          symbol: 'none',
        },
        data: [],
      },
    ],
  }

  return (
    <div style={{
      height: 'calc(100vh - 144px)',
      overflow: 'hidden',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'flex-start',
    }}>
      <div style={{ zoom, width: 1400 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginBottom: 8 }}>
              <Title level={3} style={{ color: 'var(--ghost)', margin: 0, fontFamily: 'var(--font-display)', fontSize: 20 }}>
                交互式演示
              </Title>
              <Paragraph style={{ color: 'var(--silver)', margin: 0, fontSize: 13, fontFamily: 'var(--font-body)' }}>
                V(x) = ax⁴/4 + bx²/2 + cx — 拖动滑块观察势函数和分岔图的实时变化
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
                    本地计算
                  </Tag>
                )}
              </div>
            </div>

            {apiStatus === 'local' && (
              <Alert title="后端未连接，使用本地计算" type="info" showIcon style={{ marginBottom: 10, background: 'rgba(0, 240, 255, 0.08)', border: '1px solid rgba(0, 240, 255, 0.3)' }} />
            )}
            {apiStatus === 'api' && (
              <Alert title="已连接后端API，使用真实数据" type="success" showIcon style={{ marginBottom: 10, background: 'rgba(57, 255, 20, 0.08)', border: '1px solid rgba(57, 255, 20, 0.3)' }} />
            )}

            <Card styles={{ body: { padding: '12px 16px' } }}>
              <Row gutter={[24, 0]} align="middle">
                <Col flex="auto">
                  <Row gutter={[20, 0]}>
                    <Col>
                      <Text style={{ color: 'var(--ash)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>a (稳定性)</Text>
                      <Slider min={0.1} max={3} step={0.1} value={a} onChange={setA} style={{ width: 120 }} />
                      <Text code style={{ color: 'var(--neon-purple)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>{a.toFixed(1)}</Text>
                    </Col>
                    <Col>
                      <Text style={{ color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>b (分岔参数)</Text>
                      <Slider min={-3} max={3} step={0.05} value={b} onChange={setB} style={{ width: 140 }} />
                      <Text code style={{ color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>{b.toFixed(2)}</Text>
                    </Col>
                    <Col>
                      <Text style={{ color: 'var(--neon-pink)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>c (不对称)</Text>
                      <Slider min={-3} max={3} step={0.05} value={c} onChange={setC} style={{ width: 140 }} />
                      <Text code style={{ color: 'var(--neon-pink)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>{c.toFixed(2)}</Text>
                    </Col>
                  </Row>
                </Col>
                <Col flex="none">
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleAnimate} danger={animating}>
                      {animating ? '停止' : '动画'}
                    </Button>
                    <Button icon={<ReloadOutlined />} onClick={() => { setA(0.5); setB(-1.5); setC(0) }}>
                      重置
                    </Button>
                  </div>
                </Col>
                <Col flex="none">
                  <Card styles={{ body: { padding: '4px 12px', background: 'var(--graphite)' } }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <span><Text style={{ color: 'var(--ash)', fontSize: 10 }}>平衡态</Text><Text strong style={{ color: 'var(--neon-purple)', fontSize: 13 }}>{equilibria.length}</Text></span>
                      <Tag color={isBistable ? 'green' : 'default'} style={{ fontSize: 9, margin: 0 }}>{isBistable ? '双稳态' : '单稳态'}</Tag>
                    </div>
                  </Card>
                </Col>
              </Row>
            </Card>
          </div>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={14}>
              <Card
                styles={{ body: { padding: 8 } }}
                title={<span style={{ fontFamily: 'var(--font-display)', fontSize: 13 }}>势函数 V(x)</span>}
              >
                <ReactECharts option={potentialOption} style={{ height: 340, minHeight: 260 }} />
              </Card>
            </Col>
            <Col xs={24} lg={6}>
              <Card
                styles={{ body: { padding: 8 } }}
                title={<span style={{ fontFamily: 'var(--font-display)', fontSize: 13 }}>分岔图</span>}
              >
                <ReactECharts option={bifurcationOption} style={{ height: 340, minHeight: 260 }} />
              </Card>
            </Col>
            <Col xs={24} lg={4}>
              <Card
                styles={{ body: { padding: '10px 8px' } }}
                title="系统状态"
              >
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
                  {[
                    { label: '稳定平衡态', value: stableEq.length, color: '#39FF14' },
                    { label: '不稳定平衡态', value: unstableEq.length, color: '#FF0080' },
                  ].map((item) => (
                    <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={{ color: 'var(--ash)', fontSize: 11 }}>{item.label}</Text>
                      <Text strong style={{ color: item.color, fontSize: 14, fontFamily: 'var(--font-display)' }}>{item.value}</Text>
                    </div>
                  ))}
                  {fixedPoints.length > 0 && (
                    <div style={{ borderTop: '1px solid var(--smoke)', paddingTop: 6, marginTop: 4 }}>
                      <Text style={{ color: 'var(--ash)', fontSize: 10, display: 'block', marginBottom: 4 }}>不动点 (后端)</Text>
                      {fixedPoints.slice(0, 3).map((fp, i) => (
                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9 }}>
                          <Text style={{ color: fp.stability === 'stable' ? '#39FF14' : '#FF0080' }}>x={fp.x.toFixed(3)}</Text>
                          <Text style={{ color: 'var(--ash)' }}>{fp.stability}</Text>
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{ borderTop: '1px solid var(--smoke)', paddingTop: 6 }}>
                    <Text style={{ color: 'var(--ash)', fontSize: 11, display: 'block', marginBottom: 6 }}>快速预设</Text>
                    {[
                      { label: '双稳态', a: 1, b: -2, c: 0, tag: 'green' as const },
                      { label: '单稳态', a: 1, b: 1, c: 0, tag: 'blue' as const },
                      { label: '临界分岔', a: 1, b: 0, c: 0, tag: 'orange' as const },
                      { label: '偏移双稳', a: 1, b: -2, c: 0.5, tag: 'purple' as const },
                    ].map((p) => (
                      <Button
                        key={p.label}
                        block
                        onClick={() => { setA(p.a); setB(p.b); setC(p.c) }}
                        style={{ marginBottom: 4, textAlign: 'left', fontFamily: 'var(--font-mono)', fontSize: 10, height: 28 }}
                      >
                        <Tag color={p.tag} style={{ marginRight: 6, fontSize: 9 }}>{p.label}</Tag>
                        <Text style={{ fontSize: 9 }}>a={p.a} b={p.b} c={p.c}</Text>
                      </Button>
                    ))}
                  </div>
                </div>
              </Card>
            </Col>
          </Row>

          <div style={{ flexShrink: 0 }}>
            <Alert
              type="info"
              showIcon
              icon={<ExperimentOutlined style={{ color: 'var(--neon-purple)' }} />}
              style={{ background: 'rgba(139, 92, 246, 0.06)', border: '1px solid rgba(139, 92, 246, 0.15)' }}
              title={
                <Text style={{ color: 'var(--silver)', fontFamily: 'var(--font-body)', fontSize: 12 }}>
                  当 b &lt; 0 时系统可能出现双稳态（两个稳定平衡态），微小的扰动可能导致系统在两个状态间发生突变跳跃——这就是尖点突变的核心特征。点击"动画"观察 b 从 -3 到 +3 变化时平衡态的演化。
                </Text>
              }
            />
          </div>
        </div>
      </div>
    </div>
  )
}
