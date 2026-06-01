import { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Tag, Space, Progress } from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  ApartmentOutlined,
  LineChartOutlined,
  MessageOutlined,
  ThunderboltOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons'

const { Paragraph, Text } = Typography

const LAYERS = [
  {
    key: 'layer1',
    icon: <ApartmentOutlined style={{ fontSize: 22, color: '#00F0FF' }} />,
    title: 'Layer 1',
    subtitle: '因果发现',
    desc: 'EBIC Glasso + 理论约束引擎，从症状数据发现因果网络，识别核心症状与正反馈环',
    tags: ['Borsboom网络', 'DSM-5约束', '桥接检测'],
    accent: '#00F0FF',
    metrics: { label: '特征提取', value: 94 },
  },
  {
    key: 'layer2',
    icon: <LineChartOutlined style={{ fontSize: 22, color: '#8B5CF6' }} />,
    title: 'Layer 2',
    subtitle: 'Cusp动力学',
    desc: '尖点突变模型刻画抑郁非线性突变，势函数量化韧性储备，预测tipping point',
    tags: ['势函数V(x)', '分岔分析', '早期预警'],
    accent: '#8B5CF6',
    metrics: { label: '动力学建模', value: 87 },
  },
  {
    key: 'layer3',
    icon: <MessageOutlined style={{ fontSize: 22, color: '#FF0080' }} />,
    title: 'Layer 3',
    subtitle: 'LLM认知评估',
    desc: 'Lazarus认知评估链 + CoVe验证 + Self-Critique，结构化心理评估输出Cusp参数',
    tags: ['CoVe验证', 'Risk-Sensitive', '逆关系约束'],
    accent: '#FF0080',
    metrics: { label: '语义理解', value: 85 },
  },
]

const PIPELINE = [
  { from: '症状问卷', to: '因果网络', color: '#00F0FF' },
  { from: '因果结构', to: 'Cusp参数', color: '#8B5CF6' },
  { from: '文本输入', to: '抑郁等级', color: '#FF0080' },
]

export default function Home() {
  const navigate = useNavigate()
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

  return (
    <div style={{
      height: 'calc(100vh - 144px)',
      overflow: 'hidden',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'flex-start',
    }}>
      <div style={{ zoom, width: 1400 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div
            style={{
              flexShrink: 0,
              borderRadius: 'var(--radius)',
              padding: '14px 24px',
              background: 'linear-gradient(135deg, var(--carbon) 0%, #120a2e 50%, var(--graphite) 100%)',
              border: '1px solid rgba(139, 92, 246, 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Paragraph style={{ color: 'var(--silver)', fontSize: 13.5, maxWidth: 680, lineHeight: 1.65, fontFamily: 'var(--font-body)', margin: 0 }}>
              融合<span style={{ fontWeight: 700, color: '#00F0FF' }}> 尖点突变理论</span>、
              <span style={{ fontWeight: 700, color: '#8B5CF6' }}> Lazarus认知评估</span>与
              <span style={{ fontWeight: 700, color: '#FF0080' }}> 因果发现</span>的三层架构，
              实现从症状网络到动力学突变再到语义理解的跨粒度心理健康评估
            </Paragraph>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 }}>
              {PIPELINE.map((p, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Tag style={{
                    background: `${p.color}15`, color: p.color, border: `1px solid ${p.color}30`,
                    borderRadius: 10, fontSize: 10, padding: '1px 7px', margin: 0, fontFamily: 'var(--font-mono)',
                  }}>{p.from}</Tag>
                  <ArrowRightOutlined style={{ color: p.color, fontSize: 9 }} />
                  <Tag style={{
                    background: `${p.color}08`, color: p.color, border: `1px solid ${p.color}20`,
                    borderRadius: 10, fontSize: 10, padding: '1px 7px', margin: 0, fontFamily: 'var(--font-mono)',
                  }}>{p.to}</Tag>
                </div>
              ))}
            </div>
          </div>

          <Row gutter={[16, 16]}>
            {LAYERS.map((layer) => (
              <Col xs={24} md={8} key={layer.key}>
                <Card
                  onClick={() => navigate('/architecture')}
                  style={{
                    height: '100%',
                    cursor: 'pointer',
                    background: layer.accent === '#00F0FF' ? 'rgba(0, 240, 255, 0.04)' : layer.accent === '#8B5CF6' ? 'rgba(139, 92, 246, 0.04)' : 'rgba(255, 0, 128, 0.04)',
                    border: `1px solid ${layer.accent}20`,
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                  styles={{ body: { padding: 18, flex: 1, display: 'flex', flexDirection: 'column' } }}
                  hoverable
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 36, height: 36, borderRadius: 10,
                        background: `${layer.accent}12`, border: `1px solid ${layer.accent}25`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        {layer.icon}
                      </div>
                      <div>
                        <Text strong style={{ color: layer.accent, fontSize: 13, fontFamily: 'var(--font-display)', display: 'block' }}>{layer.title}</Text>
                        <Text style={{ color: 'var(--ghost)', fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-display)' }}>{layer.subtitle}</Text>
                      </div>
                    </div>
                    <Progress type="circle" percent={layer.metrics.value} size={40}
                      strokeColor={layer.accent}
                      format={(p) => <Text style={{ fontSize: 11, fontWeight: 800, color: layer.accent }}>{p}</Text>}
                    />
                  </div>

                  <Paragraph style={{ color: 'var(--silver)', lineHeight: 1.7, fontSize: 12.5, fontFamily: 'var(--font-body)', flex: 1, marginBottom: 12 }}>
                    {layer.desc}
                  </Paragraph>

                  <Space size={[5, 5]} wrap style={{ marginBottom: 10 }}>
                    {layer.tags.map((tag) => (
                      <Tag key={tag} style={{
                        background: `${layer.accent}10`, color: layer.accent,
                        border: `1px solid ${layer.accent}25`, borderRadius: 12,
                        fontFamily: 'var(--font-mono)', fontSize: 10, padding: '1px 8px',
                      }}>{tag}</Tag>
                    ))}
                  </Space>

                  <div style={{
                    borderTop: `1px solid ${layer.accent}15`, paddingTop: 10,
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  }}>
                    <Text style={{ color: 'var(--ash)', fontSize: 11, fontFamily: 'var(--font-body)' }}>{layer.metrics.label}: <Text strong style={{ color: layer.accent }}>{layer.metrics.value}%</Text></Text>
                    <ArrowRightOutlined style={{ color: layer.accent, fontSize: 12 }} />
                  </div>
                </Card>
              </Col>
            ))}
          </Row>

          <div style={{
            flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'linear-gradient(90deg, rgba(139, 92, 246, 0.06), rgba(0, 240, 255, 0.06), rgba(255, 0, 128, 0.06))',
            borderRadius: 'var(--radius)', padding: '12px 20px',
            border: '1px solid rgba(139, 92, 246, 0.15)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
              <div style={{ textAlign: 'center' }}>
                <Text style={{ color: 'var(--neon-cyan)', fontSize: 20, fontWeight: 800, fontFamily: 'var(--font-display)', display: 'block' }}>3</Text>
                <Text style={{ color: 'var(--ash)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>融合层级</Text>
              </div>
              <div style={{ width: 1, height: 30, background: 'rgba(139, 92, 246, 0.2)' }} />
              <div style={{ textAlign: 'center' }}>
                <Text style={{ color: 'var(--neon-purple)', fontSize: 20, fontWeight: 800, fontFamily: 'var(--font-display)', display: 'block' }}>CoVe+SC+RS</Text>
                <Text style={{ color: 'var(--ash)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>前沿技术栈</Text>
              </div>
              <div style={{ width: 1, height: 30, background: 'rgba(139, 92, 246, 0.2)' }} />
              <div style={{ textAlign: 'center' }}>
                <Text style={{ color: 'var(--neon-pink)', fontSize: 20, fontWeight: 800, fontFamily: 'var(--font-display)', display: 'block' }}>Qwen3.5-2B</Text>
                <Text style={{ color: 'var(--ash)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>基座模型</Text>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Text style={{ color: 'var(--ghost)', fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-display)', cursor: 'pointer' }}
                onClick={() => navigate('/architecture')}>
                查看完整架构 →
              </Text>
              <ThunderboltOutlined onClick={() => navigate('/demo')}
                style={{ color: 'var(--neon-yellow)', fontSize: 17, cursor: 'pointer' }} />
              <Text onClick={() => navigate('/demo')}
                style={{ color: 'var(--neon-yellow)', fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-display)', cursor: 'pointer' }}>
                Demo
              </Text>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
