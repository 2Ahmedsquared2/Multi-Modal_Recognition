import { NavLink, useLocation } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';

/* ── Inline SVG Icons (no icon library needed) ── */

function IconHome({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21V13.6a.6.6 0 0 1 .6-.6h4.8a.6.6 0 0 1 .6.6V21M12 3l9 7.5V21H3V10.5L12 3z" />
    </svg>
  );
}

function IconWaveform({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v18M6 7v10M18 7v10M3 10v4M21 10v4M9 5v14M15 5v14" />
    </svg>
  );
}

function IconChart({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="M7 16l4-6 4 3 5-7" />
    </svg>
  );
}

function IconScatter({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="7" cy="8" r="1.5" />
      <circle cx="12" cy="14" r="1.5" />
      <circle cx="17" cy="6" r="1.5" />
      <circle cx="9" cy="17" r="1.5" />
      <circle cx="16" cy="16" r="1.5" />
      <circle cx="5" cy="13" r="1.5" />
      <circle cx="19" cy="11" r="1.5" />
    </svg>
  );
}

function IconSliders({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 21V14M4 10V3M12 21V12M12 8V3M20 21V16M20 12V3M1 14h6M9 8h6M17 16h6" />
    </svg>
  );
}

function IconSun({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
}

function IconMoon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

/* ── Nav config ── */

const navItems = [
  { path: '/', label: 'Home', icon: IconHome, description: 'Overview' },
  { path: '/classify', label: 'Classify', icon: IconWaveform, description: 'Audio recognition' },
  { path: '/dashboard', label: 'Dashboard', icon: IconChart, description: 'Training metrics' },
  { path: '/explorer', label: 'Explorer', icon: IconScatter, description: 'Feature space' },
  { path: '/what-if', label: 'What-If', icon: IconSliders, description: 'Experiment' },
];

/* ── Sidebar Component ── */

export default function Sidebar() {
  const { theme, toggle } = useTheme();
  const location = useLocation();

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-60 flex flex-col
      bg-white dark:bg-slate-900
      border-r border-slate-200 dark:border-slate-800
      z-40"
    >
      {/* Brand */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-3">
          {/* Waveform logo mark */}
          <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 8h1M5 4v8M8 2v12M11 4v8M14 8h1" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-white leading-none">
              APRE
            </h1>
            <p className="text-[10px] text-slate-500 dark:text-slate-500 mt-0.5 leading-none">
              Acoustic Recognition
            </p>
          </div>
        </div>
      </div>

      {/* Thin separator */}
      <div className="mx-4 h-px bg-slate-200 dark:bg-slate-800" />

      {/* Navigation */}
      <nav className="flex-1 px-3 pt-4 space-y-0.5">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                transition-colors duration-150
                ${isActive
                  ? 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                }`}
            >
              <item.icon className={`w-[18px] h-[18px] flex-shrink-0 transition-colors duration-150
                ${isActive
                  ? 'text-indigo-500 dark:text-indigo-400'
                    : 'text-slate-500 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-300'
                }`}
              />
              <div className="flex flex-col min-w-0">
                <span className={`font-medium leading-none ${isActive ? '' : ''}`}>
                  {item.label}
                </span>
                <span className={`text-[10px] mt-1 leading-none
                  ${isActive
                    ? 'text-indigo-500/70 dark:text-indigo-400/50'
                    : 'text-slate-500 dark:text-slate-600'
                  }`}
                >
                  {item.description}
                </span>
              </div>

              {/* Active indicator dot */}
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-500 dark:bg-indigo-400" />
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-4 space-y-2">
        {/* Thin separator */}
        <div className="mx-1 h-px bg-slate-200 dark:bg-slate-800" />

        {/* Theme toggle */}
        <button
          onClick={toggle}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
            text-slate-500 dark:text-slate-400
            hover:text-slate-700 dark:hover:text-slate-200
            hover:bg-slate-50 dark:hover:bg-slate-800/50
            transition-colors duration-150"
        >
          {theme === 'dark' ? (
            <IconSun className="w-[18px] h-[18px] text-amber-500" />
          ) : (
            <IconMoon className="w-[18px] h-[18px] text-slate-500" />
          )}
          <span className="font-medium">
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </span>
        </button>

        {/* Backend status indicator */}
        <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-slate-500 dark:text-slate-500">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <span>Backend connected</span>
        </div>
      </div>
    </aside>
  );
}
