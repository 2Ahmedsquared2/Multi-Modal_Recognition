import { NavLink, useLocation } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import { useModel } from '../contexts/ModelContext';
import type { ModelEngine, Modality } from '../types';

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

function IconAudio({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </svg>
  );
}

function IconImage({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </svg>
  );
}

function IconCheck({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

/* ── Nav config ── */

const navItems = [
  { path: '/', label: 'Home', icon: IconHome, description: 'Overview' },
  { path: '/classify', label: 'Classify', icon: IconWaveform, description: 'Pattern recognition' },
  { path: '/dashboard', label: 'Dashboard', icon: IconChart, description: 'Training metrics' },
  { path: '/explorer', label: 'Explorer', icon: IconScatter, description: 'Feature space' },
  { path: '/what-if', label: 'What-If', icon: IconSliders, description: 'Experiment' },
];

/* ── Engine toggle options ── */

const ENGINE_OPTIONS: { value: ModelEngine; label: string; sub: string }[] = [
  { value: 'custom', label: 'From-Scratch', sub: 'NumPy' },
  { value: 'pytorch', label: 'PyTorch', sub: 'Framework' },
];

/* ── Modality toggle options ── */

const MODALITY_OPTIONS: { value: Modality; label: string; icon: typeof IconAudio }[] = [
  { value: 'audio', label: 'Audio', icon: IconAudio },
  { value: 'image', label: 'Image', icon: IconImage },
];

/* ── Sidebar Component ── */

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const {
    engine, setEngine,
    modality, setModality,
    dataset, setDataset,
    datasetsForModality,
    datasetsLoading,
  } = useModel();

  return (
    <aside
      className={`fixed left-0 top-0 bottom-0 w-60 flex flex-col
        bg-white dark:bg-slate-900
        border-r border-slate-200 dark:border-slate-800
        z-40
        transition-transform duration-200 ease-out
        ${open ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0`}
    >
      {/* Brand */}
      <div className="px-5 pt-6 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Multi-modal logo mark — waveform + grid */}
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              {/* Waveform (audio) */}
              <path d="M2 8h1M5 5v6M8 3v10M11 5v6M14 8h1" stroke="white" strokeWidth="1.2" strokeLinecap="round" opacity="0.9" />
              {/* Grid dots (image) */}
              <circle cx="3" cy="13" r="0.8" fill="white" opacity="0.5" />
              <circle cx="6" cy="13" r="0.8" fill="white" opacity="0.5" />
              <circle cx="3" cy="3" r="0.8" fill="white" opacity="0.5" />
              <circle cx="13" cy="3" r="0.8" fill="white" opacity="0.5" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-white leading-none">
              MPRE
            </h1>
            <p className="text-[10px] text-slate-500 dark:text-slate-500 mt-0.5 leading-none">
              Multi-Modal Recognition
            </p>
          </div>
        </div>

        {/* Close button (mobile only) */}
        <button
          onClick={onClose}
          className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          aria-label="Close navigation"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Thin separator */}
      <div className="mx-4 h-px bg-slate-200 dark:bg-slate-800" />

      {/* ── Modality toggle ── */}
      <div className="px-3 pt-4 pb-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 px-2 mb-2">
          Modality
        </p>
        <div className="relative flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
          {/* Sliding background indicator */}
          <div
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-md bg-white dark:bg-slate-700 shadow-sm
              transition-transform duration-200 ease-out"
            style={{ transform: modality === 'image' ? 'translateX(calc(100% + 8px))' : 'translateX(0)' }}
          />
          {MODALITY_OPTIONS.map((opt) => {
            const active = modality === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setModality(opt.value)}
                className="relative flex-1 flex items-center justify-center gap-1.5 py-2 px-1 rounded-md text-center
                  transition-all duration-200 active:scale-[0.97] z-10"
              >
                <opt.icon className={`w-3.5 h-3.5 transition-colors duration-200
                  ${active
                    ? 'text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-500 dark:text-slate-400'
                  }`}
                />
                <span className={`text-xs font-semibold leading-none transition-colors duration-200
                  ${active
                    ? 'text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-500 dark:text-slate-400'
                  }`}
                >
                  {opt.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Dataset selector ── */}
      <div className="px-3 pt-3 pb-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 px-2 mb-2">
          Dataset
        </p>
        <div className="space-y-0.5">
          {datasetsLoading ? (
            <div className="px-3 py-2 text-xs text-slate-400 dark:text-slate-600">Loading…</div>
          ) : datasetsForModality.length === 0 ? (
            <div className="px-3 py-2 text-xs text-slate-400 dark:text-slate-600">No datasets</div>
          ) : (
            datasetsForModality.map((ds) => {
              const active = dataset === ds.key;
              return (
                <button
                  key={ds.key}
                  onClick={() => setDataset(ds.key)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left
                    transition-all duration-150 active:scale-[0.98]
                    ${active
                      ? 'bg-indigo-50 dark:bg-indigo-500/10'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}
                >
                  {/* Selection indicator */}
                  <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-colors duration-150
                    ${active
                      ? 'border-indigo-500 bg-indigo-500'
                      : 'border-slate-300 dark:border-slate-600'
                    }`}
                  >
                    {active && <IconCheck className="w-2.5 h-2.5 text-white" />}
                  </div>

                  <div className="flex flex-col min-w-0">
                    <span className={`text-xs font-medium leading-none truncate transition-colors duration-150
                      ${active
                        ? 'text-indigo-600 dark:text-indigo-400'
                        : 'text-slate-700 dark:text-slate-300'
                      }`}
                    >
                      {ds.name}
                    </span>
                    <span className="text-[9px] mt-1 leading-none text-slate-400 dark:text-slate-600 truncate">
                      {ds.num_classes} classes
                    </span>
                  </div>

                  {/* Ready badge */}
                  {ds.ready ? (
                    <span className="ml-auto flex-shrink-0 w-1.5 h-1.5 rounded-full bg-emerald-500" title="Model trained" />
                  ) : (
                    <span className="ml-auto flex-shrink-0 w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-600" title="Not trained yet" />
                  )}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* ── Engine toggle ── */}
      <div className="px-3 pt-3 pb-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-500 px-2 mb-2">
          Engine
        </p>
        <div className="relative flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
          {/* Sliding background indicator */}
          <div 
            className={`absolute top-1 bottom-1 w-[calc(50%-4px)] rounded-md bg-white dark:bg-slate-700 shadow-sm
              transition-transform duration-200 ease-out`}
            style={{ transform: engine === 'pytorch' ? 'translateX(calc(100% + 8px))' : 'translateX(0)' }}
          />
          {ENGINE_OPTIONS.map((opt) => {
            const active = engine === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setEngine(opt.value)}
                className={`relative flex-1 flex flex-col items-center py-2.5 px-1 rounded-md text-center
                  transition-all duration-200 active:scale-[0.97] z-10`}
              >
                <span className={`text-xs font-semibold leading-none transition-colors duration-200
                  ${active
                    ? 'text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-500 dark:text-slate-400'
                  }`}
                >
                  {opt.label}
                </span>
                <span className={`text-[9px] mt-1 leading-none transition-colors duration-200
                  ${active
                    ? 'text-indigo-500/70 dark:text-indigo-400/60'
                    : 'text-slate-400 dark:text-slate-600'
                  }`}
                >
                  {opt.sub}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 pt-3 space-y-0.5">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                transition-all duration-150 active:scale-[0.98]
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
            transition-all duration-150 active:scale-[0.98]"
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
