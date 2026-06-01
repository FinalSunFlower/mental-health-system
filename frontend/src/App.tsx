import { Routes, Route } from 'react-router-dom'
import MainLayout from './components/Layout/MainLayout'
import Home from './pages/Home'
import Architecture from './pages/Architecture'
import DemoPage from './pages/Demo'

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Home />} />
        <Route path="architecture" element={<Architecture />} />
        <Route path="demo" element={<DemoPage />} />
      </Route>
    </Routes>
  )
}

export default App
