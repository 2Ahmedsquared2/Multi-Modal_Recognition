import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import ErrorBoundary from './ErrorBoundary';
import { useModel } from '../contexts/ModelContext';

export default function Layout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { engine, engineLabel, modality, dataset } = useModel();

  /* Key includes modality + dataset so switching triggers a fade transition */
  const contentKey = `${location.pathname}-${engine}-${modality}-${dataset}`;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main content area — offset by sidebar width on desktop */}
      <main className="lg:ml-60 min-h-screen flex flex-col">
        {/* Mobile top bar */}
        <div className="sticky top-0 z-20 flex items-center gap-3 px-4 py-3 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b border-slate-200 dark:border-slate-800 lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors duration-150 active:scale-95"
            aria-label="Open navigation"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-sm font-semibold text-slate-900 dark:text-white tracking-tight">
            MPRE
          </span>
          <span className={`ml-auto text-[10px] font-medium px-2 py-0.5 rounded-full
            ${engine === 'pytorch'
              ? 'bg-orange-100 text-orange-700 dark:bg-orange-500/10 dark:text-orange-400'
              : 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400'
            }`}
          >
            {engineLabel}
          </span>
        </div>

        <div className="flex-1">
          <ErrorBoundary key={contentKey}>
            <div className="animate-fade-in" key={contentKey}>
              <Outlet />
            </div>
          </ErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="border-t border-slate-200 dark:border-slate-800 px-8 py-6">
          <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 dark:text-slate-500">
            <p>
              {engine === 'pytorch'
                ? 'PyTorch engine — industry-standard ML framework.'
                : 'Built from scratch with NumPy — no TensorFlow, no PyTorch.'
              }
            </p>
            <p className="text-slate-400 dark:text-slate-600">
              Multi-Modal Pattern Recognition Engine
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
