import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider, theme } from 'antd'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#8B5CF6',
          colorBgContainer: '#0D0D1A',
          colorBgLayout: '#06060C',
          colorBgElevated: '#151528',
          borderRadius: 12,
          fontFamily: "'Inter', 'Chakra Petch', sans-serif",
          colorText: '#E0E0F0',
          colorTextSecondary: '#9CA3B8',
          colorBorder: '#2A2A4A',
          colorSplit: '#2A2A4A',
        },
        components: {
          Menu: {
            itemBg: 'transparent',
            itemSelectedBg: 'rgba(139, 92, 246, 0.15)',
            itemSelectedColor: '#8B5CF6',
            itemHoverBg: 'rgba(139, 92, 246, 0.08)',
            itemHoverColor: '#8B5CF6',
            darkItemBg: 'transparent',
            darkItemSelectedBg: 'rgba(139, 92, 246, 0.15)',
          },
          Card: {
            borderRadiusLG: 12,
          },
        },
      }}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ConfigProvider>
  </React.StrictMode>,
)
