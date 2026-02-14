import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Classify from './pages/Classify';
import Dashboard from './pages/Dashboard';
import Explorer from './pages/Explorer';
import WhatIf from './pages/WhatIf';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/classify" element={<Classify />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/what-if" element={<WhatIf />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
