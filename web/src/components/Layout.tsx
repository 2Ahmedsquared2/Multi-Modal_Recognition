import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import ErrorBoundary from './ErrorBoundary';
import { useModel } from '../contexts/ModelContext';

export default function Layout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const { engine, engineLabel, modality, dataset } = useModel();

  const contentKey = `${location.pathname}-${engine}-${modality}-${dataset}`;

  return (
    <div className="min-h-screen bg-warm-50 dark:bg-warm-900">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-warm-900/30 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
      />

      <main
        className={`min-h-screen flex flex-col transition-[margin] duration-300 ease-out
          ${collapsed ? 'lg:ml-16' : 'lg:ml-60'}`}
      >
        <div className="sticky top-0 z-20 flex items-center gap-3 px-4 py-3 bg-warm-50/80 dark:bg-warm-900/80 backdrop-blur border-b border-warm-300 dark:border-warm-700 lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-lg text-warm-600 dark:text-warm-400 hover:bg-warm-200 dark:hover:bg-warm-800 transition-colors duration-150 active:scale-95"
            aria-label="Open navigation"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="font-display text-sm font-semibold text-warm-900 dark:text-warm-100 tracking-tight">
            MPRE
          </span>
          <span className={`ml-auto text-[10px] font-medium px-2 py-0.5 rounded-full
            ${engine === 'pytorch'
              ? 'bg-warm-200 text-warm-600 dark:bg-warm-700 dark:text-warm-400'
              : 'bg-accent-subtle text-accent dark:bg-accent/10 dark:text-accent-light'
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

        <footer className="border-t border-warm-300 dark:border-warm-700 px-8 py-6">
          <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-warm-500 dark:text-warm-500">
            <p>
              {engine === 'pytorch'
                ? 'PyTorch engine — industry-standard ML framework.'
                : 'Hand-built with NumPy — no frameworks, no shortcuts.'
              }
            </p>
            <p className="text-warm-400 dark:text-warm-600">
              Multi-Modal Pattern Recognition Engine
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
