# Step 22: React + TypeScript Frontend Setup

**Estimated Time:** 30 minutes
**Status:** ⬜ Not Started

---

## Goal
Scaffold a modern React + TypeScript frontend with Tailwind CSS, routing, and a clean layout shell ready for features.

## Tasks
- [ ] Create `web/` directory at project root
- [ ] Scaffold with Vite: `npm create vite@latest web -- --template react-ts`
- [ ] Install Tailwind CSS + configure
- [ ] Install React Router for page navigation
- [ ] Create layout shell: sidebar/nav + main content area
- [ ] Create placeholder pages: Home, Classify, Dashboard, Explorer, What-If
- [ ] Set up API client utility (fetch wrapper pointing to FastAPI backend)
- [ ] Configure dark mode (Tailwind `dark:` classes)
- [ ] Verify dev server runs and connects to backend

## Implementation Plan

### Directory Structure
```
web/
├── public/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Layout.tsx     # Main layout with nav
│   │   ├── Navbar.tsx     # Top navigation bar
│   │   └── LoadingSpinner.tsx
│   ├── pages/             # Route pages
│   │   ├── Home.tsx       # Landing / overview
│   │   ├── Classify.tsx   # Upload + mic + results
│   │   ├── Dashboard.tsx  # Training curves + confusion matrix
│   │   ├── Explorer.tsx   # t-SNE + feature explorer
│   │   └── WhatIf.tsx     # Spectrogram editor
│   ├── api/               # API client
│   │   └── client.ts      # Fetch wrapper for FastAPI
│   ├── types/             # TypeScript interfaces
│   │   └── index.ts       # Shared types
│   ├── App.tsx            # Router setup
│   ├── main.tsx           # Entry point
│   └── index.css          # Tailwind imports
├── tailwind.config.js
├── tsconfig.json
├── vite.config.ts
└── package.json
```

### Design Direction
- **Dark theme** primary (matches the existing plot aesthetics)
- **Accent color**: Blue/purple gradient (signals "audio / signal processing")
- **Clean, minimal layout**: sidebar nav on the left, content on the right
- **Smooth transitions** between pages
- **Responsive**: works on desktop, degrades gracefully on mobile

### Page Routing
```
/              → Home (project overview, quick stats, architecture diagram)
/classify      → Classify (upload + mic + results)
/dashboard     → Dashboard (interactive training curves + confusion matrix)
/explorer      → Explorer (t-SNE + feature browser)
/what-if       → What-If (spectrogram editor)
```

### API Client Pattern
```typescript
// src/api/client.ts
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  health: () => fetch(`${API_BASE}/api/health`).then(r => r.json()),
  
  modelInfo: () => fetch(`${API_BASE}/api/model/info`).then(r => r.json()),
  
  classify: (file: File) => {
    const form = new FormData();
    form.append('audio', file);
    return fetch(`${API_BASE}/api/classify`, { method: 'POST', body: form }).then(r => r.json());
  },
  
  // ... more endpoints
};
```

### Key Dependencies
```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "react-router-dom": "^6",
    "plotly.js": "^2",
    "react-plotly.js": "^2",
    "recharts": "^2"
  },
  "devDependencies": {
    "typescript": "^5",
    "tailwindcss": "^3",
    "autoprefixer": "^10",
    "postcss": "^8",
    "@types/react": "^18"
  }
}
```

## How to Run (Once Built)
```bash
cd web
npm install
npm run dev
# → http://localhost:5173
```

## Verification
- [ ] `npm run dev` starts without errors
- [ ] All 5 pages route correctly
- [ ] Tailwind dark mode works
- [ ] API client can reach FastAPI backend (health check)
- [ ] Layout looks clean with sidebar navigation
- [ ] TypeScript compiles with zero errors

## Next
Step 23: Audio Upload & Classification
