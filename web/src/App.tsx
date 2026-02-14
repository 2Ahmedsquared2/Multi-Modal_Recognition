import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import { ModelProvider } from './contexts/ModelContext';

// Lazy-load heavy pages (Plotly charts, canvas editor, large scatter plots)
const Classify = lazy(() => import('./pages/Classify'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Explorer = lazy(() => import('./pages/Explorer'));
const WhatIf = lazy(() => import('./pages/WhatIf'));

function PageLoader() {
  return (
    <div className="flex items-center gap-3 justify-center py-32">
      <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-sm text-slate-500">Loading…</span>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ModelProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/classify" element={<Suspense fallback={<PageLoader />}><Classify /></Suspense>} />
            <Route path="/dashboard" element={<Suspense fallback={<PageLoader />}><Dashboard /></Suspense>} />
            <Route path="/explorer" element={<Suspense fallback={<PageLoader />}><Explorer /></Suspense>} />
            <Route path="/what-if" element={<Suspense fallback={<PageLoader />}><WhatIf /></Suspense>} />
          </Route>
        </Routes>
      </ModelProvider>
    </BrowserRouter>
  );
}
