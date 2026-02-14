import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import ErrorBoundary from './ErrorBoundary';

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <Sidebar />
      {/* Main content area — offset by sidebar width */}
      <main className="ml-60 min-h-screen">
        <ErrorBoundary key={location.pathname}>
          <div className="animate-fade-in">
            <Outlet />
          </div>
        </ErrorBoundary>
      </main>
    </div>
  );
}
