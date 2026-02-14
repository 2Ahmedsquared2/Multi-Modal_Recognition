# Step 30: Polish & Deployment

**Estimated Time:** 60 minutes
**Status:** ⬜ Not Started

---

## Goal
Final polish pass on the entire web app — responsive design, loading states, error handling, transitions — then deploy to the web so it's accessible via a public URL.

## Tasks

### Polish
- [ ] Loading states: skeleton screens or spinners for every data fetch
- [ ] Error handling: graceful messages for API failures, mic issues, unsupported browsers
- [ ] Empty states: helpful messages when no data/results yet
- [ ] Transitions: smooth page transitions (fade in/out between routes)
- [ ] Responsive: ensure all pages work at common breakpoints (1440px, 1024px, 768px)
- [ ] Keyboard navigation: all interactive elements focusable and usable
- [ ] Favicon + page title + meta tags
- [ ] "About" section or footer: brief project description, link to GitHub, your name

### Performance
- [ ] Lazy load heavy pages (Dashboard, Explorer, What-If) with `React.lazy()`
- [ ] Optimize Plotly bundle (only import needed chart types)
- [ ] Image optimization: compress any static assets
- [ ] API response caching: cache model info, training history, t-SNE data (they never change)

### Deployment — Backend (Railway or Render)
- [ ] Create `Procfile` or `railway.json` for deployment config
- [ ] Set environment variables (PORT, CORS origins)
- [ ] Ensure model file (`models/model.npz`) is included in deployment
- [ ] Ensure prepared data (`data/prepared/prepared_data.npz`) is included
- [ ] Health check endpoint works
- [ ] Test all API endpoints on deployed URL

### Deployment — Frontend (Vercel)
- [ ] Build production bundle: `npm run build`
- [ ] Configure Vercel project (link to GitHub repo)
- [ ] Set `VITE_API_URL` environment variable to deployed backend URL
- [ ] Configure SPA fallback (all routes → index.html)
- [ ] Test all pages on deployed URL
- [ ] Custom domain (optional, nice for portfolio)

### Final Checklist
- [ ] Full flow works: visit URL → classify audio → see results → explore dashboard → use What-If
- [ ] Mic recording works on deployed site (requires HTTPS — both Vercel and Railway provide this)
- [ ] No console errors in production
- [ ] Loads in <3 seconds on first visit
- [ ] Share the URL and it just works

## Implementation Plan

### Deployment Architecture
```
GitHub Repo
    ├── Push to main triggers:
    │
    ├── Vercel (Frontend)
    │   ├── Builds web/ directory
    │   ├── Serves static React app
    │   ├── URL: your-app.vercel.app
    │   └── VITE_API_URL → Railway backend
    │
    └── Railway (Backend)
        ├── Builds from project root
        ├── Runs: uvicorn api.server:app --host 0.0.0.0 --port $PORT
        ├── URL: your-app.up.railway.app
        └── Includes model weights + prepared data
```

### Backend Deployment Files

**Procfile:**
```
web: uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}
```

**railway.json:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}",
    "healthcheckPath": "/api/health"
  }
}
```

### Frontend Deployment

**vercel.json:**
```json
{
  "buildCommand": "cd web && npm run build",
  "outputDirectory": "web/dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### Error Boundary Pattern
```typescript
// Wrap each page in an error boundary
<ErrorBoundary fallback={<ErrorPage message="Something went wrong" />}>
  <Dashboard />
</ErrorBoundary>
```

### Loading States
Every data-dependent component should have 3 states:
```typescript
type DataState<T> = 
  | { status: 'loading' }
  | { status: 'success', data: T }
  | { status: 'error', error: string };
```

### Responsive Breakpoints
| Screen | Layout Adjustment |
|--------|-------------------|
| Desktop (>1024px) | Full layout, side-by-side panels |
| Tablet (768-1024px) | Stack panels vertically, keep sidebar |
| Mobile (<768px) | Hide sidebar → hamburger menu, full-width panels |

### SEO / Meta Tags
```html
<title>Acoustic Pattern Recognition Engine</title>
<meta name="description" content="From-scratch neural network for instrument classification. Upload audio or use your microphone to identify musical instruments in real time." />
<meta property="og:title" content="Acoustic Pattern Recognition Engine" />
<meta property="og:description" content="Real-time instrument classification powered by a neural network built from scratch with NumPy." />
<meta property="og:image" content="preview-image.png" />
```

### Footer / About Section
```
Built from scratch with NumPy — no TensorFlow, no PyTorch.
94.5% accuracy across 10 instrument families.
[GitHub] [Architecture Docs] [Math Derivations]
© 2025 [Your Name] — Built for USC IYA Portfolio
```

## Verification
- [ ] Visit deployed frontend URL → page loads in <3 seconds
- [ ] Upload audio → classification works end-to-end
- [ ] Microphone recording works (HTTPS required)
- [ ] Dashboard charts are interactive
- [ ] t-SNE explorer loads and is interactive
- [ ] What-If tool works with live re-prediction
- [ ] No console errors
- [ ] Responsive at all breakpoints
- [ ] Share URL with someone → they can use it immediately
- [ ] GitHub README links to the live demo

## Post-Deployment
- [ ] Add live demo URL to README.md
- [ ] Add live demo URL to hub.md
- [ ] Screenshot the app for portfolio/resume
- [ ] Record a 30-second demo GIF for GitHub README

## Why Deployment Matters
- A deployed URL is 100× more portfolio impact than "clone and run locally"
- Recruiters and admissions won't install Python — they will click a link
- HTTPS is required for microphone access
- Shows you can ship to production, not just develop locally

## Done!
Phase 2 complete. You now have a from-scratch neural network with a polished interactive web interface, deployed to the web.
