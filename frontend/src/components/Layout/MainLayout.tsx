import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  HomeOutlined,
  ApartmentOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Header, Content, Footer } = Layout

const menuItems = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  { key: '/architecture', icon: <ApartmentOutlined />, label: '核心架构' },
  { key: '/demo', icon: <ThunderboltOutlined />, label: '交互Demo' },
]

export default function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Layout style={{ minHeight: '100vh', background: 'var(--void)' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          background: 'linear-gradient(135deg, var(--carbon) 0%, var(--graphite) 100%)',
          padding: '0 32px',
          borderBottom: '1px solid rgba(139, 92, 246, 0.3)',
          boxShadow: '0 0 20px rgba(139, 92, 246, 0.1), 0 2px 8px rgba(0, 0, 0, 0.5)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          backdropFilter: 'blur(12px)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginRight: 40,
            cursor: 'pointer',
          }}
          onClick={() => navigate('/')}
        >
          <span style={{ fontSize: 28, lineHeight: 1, filter: 'drop-shadow(0 0 6px rgba(255, 215, 0, 0.5))' }}>🌻</span>
          <span
            className="glitch-text"
            style={{
              color: 'var(--neon-purple)',
              fontSize: 20,
              fontWeight: 800,
              fontFamily: 'var(--font-display)',
              letterSpacing: 2,
            }}
          >
            CuspNet
          </span>
        </div>
        <Menu
          mode="horizontal"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{
            flex: 1,
            background: 'transparent',
            borderBottom: 'none',
            fontSize: 14,
            fontWeight: 600,
            fontFamily: 'var(--font-display)',
          }}
          theme="dark"
        />
      </Header>

      <Content
        style={{
          padding: '16px 24px',
          width: '100%',
          position: 'relative',
          zIndex: 1,
          minHeight: 'calc(100vh - 112px)',
        }}
      >
        <Outlet />
      </Content>

      <Footer
        style={{
          textAlign: 'center',
          background: 'var(--carbon)',
          color: 'var(--ash)',
          fontFamily: 'var(--font-display)',
          borderTop: '1px solid rgba(139, 92, 246, 0.2)',
          fontSize: 13,
          letterSpacing: 0.5,
        }}
      >
        CuspNet — 基于尖点突变理论与Lazarus认知评估的心理健康智能评估系统
      </Footer>
    </Layout>
  )
}
