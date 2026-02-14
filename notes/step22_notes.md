# Step 22: React + TypeScript Frontend Setup

**Estimated Time:** 30 minutes
**Status:** ✅ Complete

---

## Goal
Scaffold a modern React + TypeScript frontend with Tailwind CSS, routing, and a clean layout shell ready for features.

## Tasks
- [x] Create `web/` directory at project root
- [x] Scaffold with Vite: `npm create vite@latest web -- --template react-ts`
- [x] Install Tailwind CSS v3 + configure (dark mode via `class` strategy)
- [x] Install React Router for page navigation
- [x] Create layout shell: fixed sidebar nav + main content area
- [x] Create pages: Home, Classify, Dashboard, Explorer, What-If
- [x] Set up typed API client (fetch wrapper with Vite proxy to FastAPI)
- [x] Configure dark/light mode toggle (persisted to localStorage)
- [x] Add React Error Boundary to catch page-level crashes
- [x] Verify dev server runs and connects to backend

## Actual Directory Structure
```
web/
├── public/
├── src/
│   ├── components/
│   │   ├── Layout.tsx          # Shell: sidebar + <Outlet /> with ErrorBoundary
│   │   ├── Sidebar.tsx         # Fixed sidebar: nav, theme toggle, status
│   │   └── ErrorBoundary.tsx   # Catches render errors per-page
│   ├── pages/
│   │   ├── Home.tsx            # Pipeline viz, stats, feature cards, class list
│   │   ├── Classify.tsx        # Drag-and-drop upload + results panel
│   │   ├── Dashboard.tsx       # Confusion matrix, per-class metrics, chart slots
│   │   ├── Explorer.tsx        # t-SNE scatter with class legend + hover filter
│   │   └── WhatIf.tsx          # Spectrogram editor with modification presets
│   ├── api/
│   │   └── client.ts           # Typed fetch wrapper for all FastAPI endpoints
│   ├── hooks/
│   │   └── useTheme.ts         # Dark/light toggle with localStorage persistence
│   ├── types.ts                # Shared TypeScript interfaces (matches backend schemas)
│   ├── App.tsx                 # BrowserRouter + route definitions
│   ├── main.tsx                # Entry point
│   └── index.css               # Tailwind directives + custom .card/.badge utilities
├── tailwind.config.js          # darkMode: 'class', custom fonts + animations
├── postcss.config.js
├── vite.config.ts              # API proxy: /api → localhost:8000
├── index.html                  # Flash-free theme init script
└── package.json
```

## Design Decisions

### Theme
- **Dark mode** as default, light mode via toggle in sidebar
- Theme persisted to `localStorage`, flash-free via inline `<script>` in `index.html`
- All theming uses Tailwind `dark:` classes — no JS dark mode detection

### Colors (Tailwind only, no hex codes)
- **Light mode**: `slate-50` bg, `white` cards, `slate-300` borders, `slate-600`–`slate-700` text
- **Dark mode**: `slate-950` bg, `slate-900` cards, `slate-800` borders, `slate-100`–`slate-400` text
- **Accent**: `indigo-500` (active nav, buttons, active states)
- **Status**: `emerald` (success), `amber` (warning), `rose` (error)

### Icons
- Inline SVG components — no icon library dependency
- Stroke-based, 24×24 viewBox, 1.5px stroke weight

### API Client
- Uses Vite dev proxy (`/api` → `localhost:8000`) — no CORS issues in dev
- Typed `request<T>()` wrapper with error handling
- Supports all backend endpoints: health, classify, spectrogram, confusion matrix, class metrics, training history, t-SNE, what-if

### Error Handling
- `ErrorBoundary` wraps each page via `<Outlet />` in Layout
- Keyed by `location.pathname` so errors reset on navigation
- Dashboard wrapped in try/catch for data transformations

## Page Details

### Home (`/`)
- Badges: "Research Project" + "Model Loaded" (from `/api/health`)
- Processing pipeline: 4 stages with math notation (`signal[t]`, `f(Wx+b)`, `argmax(p)`)
- Stats row: Architecture, Classes, Implementation, Parameters
- Feature cards linking to each section
- Sound class tags from API

### Classify (`/classify`)
- Drag-and-drop file upload (WAV, MP3, OGG, FLAC)
- Microphone record button (placeholder for Step 25)
- Prediction display with confidence percentage
- Full probability breakdown with progress bars

### Dashboard (`/dashboard`)
- Fetches confusion matrix + per-class metrics from API
- Computes accuracy from confusion matrix diagonal
- Best/Hardest class summary cards
- Confusion matrix table with intensity-based highlighting
- Per-class F1/Precision/Recall bars
- Chart placeholders for Step 27

### Explorer (`/explorer`)
- t-SNE scatter plot rendered as positioned divs
- Class legend with hover-to-highlight filtering
- Graceful fallback when t-SNE data not yet generated

### What-If (`/what-if`)
- Original vs Modified spectrogram side-by-side
- 6 modification presets (Low-Pass, High-Pass, Noise, Shift, Scale, Mask)
- Intensity slider
- Prediction comparison area

## How to Run
```bash
cd web
npm install
npm run dev
# → http://localhost:5173
```

## Verification
- [x] `npm run dev` starts without errors
- [x] All 5 pages route correctly (`/`, `/classify`, `/dashboard`, `/explorer`, `/what-if`)
- [x] Dark/light mode toggle works and persists
- [x] API client reaches FastAPI backend (health check on Home page)
- [x] Layout: fixed sidebar + scrollable content area
- [x] TypeScript compiles with zero errors (`npx tsc --noEmit`)
- [x] Production build passes (`npx vite build` — 80KB gzipped)
- [x] Error Boundary catches page crashes without killing sidebar

## Issues Encountered
1. **Tailwind v3 `@apply` cross-reference** — `.card-hover` using `@apply card` failed; fixed by inlining the card styles
2. **Type mismatch with backend** — original types used `predicted_class`, `probabilities`, `labels`, `coordinates`; backend returns `prediction`, `all_confidences`, `class_names`, `points` — fixed all pages to match actual API schemas
3. **Dashboard crash** — initially crashed the entire React tree; added Error Boundary + wrapped data transformations in try/catch
4. **Light mode colors too washed out** — bumped light-mode text from `slate-400`→`slate-500`, body from `slate-500`→`slate-600`, borders from `slate-200`→`slate-300`

## Next
Step 23: Audio Upload & Classification
