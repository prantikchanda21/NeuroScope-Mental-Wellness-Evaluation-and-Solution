<div align="center">

# 🧠 NeuroScope

**An adaptive, AI-assisted nervous-system / wellbeing check-in app**
React + TypeScript frontend · Express API · Gemini/Groq AI with offline fallback · 140-language support

</div>

---

## 1. What this project actually does

NeuroScope asks a person a short, **adaptive sequence of questions** about how they're
feeling, then produces a personal, research-grounded report:

- What's likely happening in their **nervous system** right now
- An **empathetic reflection** and a **cognitive reframe**
- One **immediate, concrete coping action**
- A **biopsychosocial breakdown** (body / mind / social) with a "detailed insight" dropdown per area
- Safety handling: if answers suggest crisis-level distress, it interrupts the flow with crisis
  resources instead of continuing the quiz

It works **with or without an AI key**. If `GEMINI_API_KEY` / `GROQ_API_KEY` are set, every
section is freshly written by an LLM, grounded in a local research corpus (papers + clinical
books). Without keys, the app composes the same sections locally from that same corpus, so it
never goes offline or breaks — it's just less varied in phrasing.

> ⚠️ This is a self-reflection / screening tool, not a diagnostic or medical product.

---

## 2. Quick start

```bash
npm install
cp .env.example .env        # then set GEMINI_API_KEY (and optionally GROQ_API_KEY)
npm run dev                 # http://localhost:3000
```

No keys? The app still runs — see [§5 AI vs offline fallback](#5-ai-vs-offline-fallback).

---

## 3. High-level architecture

```
  ┌─────────────────────────── BROWSER (React + TypeScript) ───────────────────────────┐
  │                                                                                      │
  │   App.tsx  ───────────►  QuestionCard                                              │
  │  (question flow             │                                                       │
  │   & state)                  ▼                                                       │
  │      │              AssessmentResultsView (report dashboard)                        │
  │      │                                                                              │
  │      └── wraps everything in ──► i18n layer (auto-translate DOM)                    │
  │                                                                                      │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                       │  fetch('/api/...')
                                       ▼
  ┌────────────────────────── SERVER  (server-app.ts, Express) ────────────────────────┐
  │                                                                                      │
  │   /api/* routes  ───────►  Research retrieval (researchCorpus.ts)                   │
  │                                     │                                                │
  │                    ┌────────────────┴────────────────┐                              │
  │                    ▼                                  ▼                             │
  │              Gemini API                          Groq API                           │
  │                    │  fails / no key                  │  fails / no key              │
  │                    └────────────────┬────────────────┘                              │
  │                                     ▼                                                │
  │                         Local / offline composer  ──► response back to /api/* route  │
  │                                                                                      │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

`server-app.ts` holds **every route and all AI logic** in one Express app, so the exact same
code can run in three different hosting shapes:

```
                         ┌───────────────────────────────────────┐
                         │            server-app.ts               │
                         │   (all /api routes, all AI logic)      │
                         └───────────────────┬─────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
   │      server.ts       │        │    api/index.ts      │        │  Netlify function    │
   │  long-running Node    │        │  Vercel serverless   │        │  (netlify.toml)      │
   │  process (local dev /│        │      function         │        │                       │
   │      Cloud Run)      │        │                       │        │                       │
   └─────────────────────┘        └─────────────────────┘        └─────────────────────┘
```

This is why `vercel.json` just rewrites `/api/*` to one function that re-exports the same
Express app — no logic is duplicated per platform.

---

## 4. The check-in flow (what happens per question)

```
  User            App.tsx           riskEngine        /api/* (server)      researchCorpus
   │                 │                    │                    │                   │
   │── answers a ───►│                    │                    │                   │
   │   question      │── assessAnswerRisk │                    │                   │
   │                 │───────────────────►│                    │                   │
   │                 │                    │                    │                   │
   │                 │      ┌── crisis-level risk detected ─────────────────────┐  │
   │                 │      │  riskEngine raises immediate support               │  │
   │◄── ImmediateCrisisCard (helplines) ───┤                                     │  │
   │                 │      └─────────────────────────────────────────────────────┘  │
   │                 │                    │                    │                   │
   │                 │      ┌── normal path ─────────────────────────────────────┐  │
   │                 │      │ adaptiveEngine picks next question                 │  │
   │                 │      │── POST /api/feeling-solution ────►│                │  │
   │                 │      │                                   │── retrieve ───►│  │
   │                 │      │                                   │◄── passages ───│  │
   │                 │      │                       ask Gemini → Groq → else     │  │
   │                 │      │                       compose locally              │  │
   │                 │      │◄── solution + nervous-system + reframe sections ───│  │
   │◄── DynamicSolutionCard ┤                                   │                │  │
   │                 │      └─────────────────────────────────────────────────────┘  │
   │                 │      (repeats until assessment length is reached)             │
   │                 │                    │                    │                   │
   │                 │── POST /api/assess ────────────────────►│                   │
   │                 │◄── AssessmentResult (biopsychosocial breakdown) ────────────│
   │◄── AssessmentResultsView (report + charts) ┤               │                   │
```

Key building blocks in `src/`:

| Piece | File | Role |
|---|---|---|
| Question selection | `utils/adaptiveEngine.ts` | Picks the next question based on prior answers (not a fixed order) |
| Safety | `utils/riskEngine.ts`, `ImmediateCrisisCard.tsx` | Detects crisis-level answers, interrupts the flow |
| Per-answer AI sections | `utils/dynamicFeelingSolutions.ts` + server `POST /api/feeling-solution` | Empathetic reflection, nervous-system explainer, reframe, solution |
| Final report | `AssessmentResultsView.tsx` + server `POST /api/assess` | Full biopsychosocial breakdown, charts, history |
| Per-area deep dive | `DimensionInsightDropdown.tsx` + server `POST /api/dimension-insight` | Gemini + Groq both answer independently, then a synthesis pass reconciles them |
| Research grounding | `data/researchCorpus.ts`, `utils/researchKnowledge.ts` | Papers/books the AI (or the offline composer) must cite from |

---

## 5. AI vs. offline fallback

Every AI-backed endpoint follows the same graceful-degradation ladder:

```
  Request arrives
        │
        ▼
  Gemini key set and call succeeds? ── yes ──► Use Gemini's answer ──┐
        │                                                              │
        no                                                             │
        ▼                                                              │
  Groq key set and call succeeds?   ── yes ──► Use Groq's answer ────┤
        │                                                              │
        no                                                             ▼
        ▼                                                    Grounded response
  Compose locally from researchCorpus.ts ─────────────────────► returned to caller
```

Safety-critical content (crisis helplines, severe-tone copy, coping steps) is **always
deterministic** — it never depends on which branch above was taken.

---

## 6. Multilingual layer (140 languages)

The app is built and rendered **only in English**. Translation is a separate, non-invasive
layer that runs *after* render:

```
  Rendered DOM text
        │
        ▼
  autoTranslate.ts walks text nodes + attributes  ◄── MutationObserver watches
        │                                              for newly rendered content
        ▼
  POST /api/translate (batched)
        │
        ▼
  Cached in localStorage? ── yes ──► Swap text in place
        │
        no
        ▼
  Call Gemini, then Groq, then echo ──► Swap text in place
```

- One stable English source of truth means switching Kannada → Hindi → English is instant and
  reversible — nothing is ever translated from a translation.
- Canvas/SVG/WebGL visuals and anything the user typed are never touched.
- Add `data-no-translate` to exempt an element.

---

## 7. Project layout

```
src/
├─ App.tsx                  # top-level flow/state machine
├─ components/               # UI (QuestionCard, results dashboard, modals, visualizers)
├─ data/                     # question pool + research corpus
├─ i18n/                     # language registry, auto-translate engine
├─ utils/                    # adaptive engine, risk engine, AI clients, retrieval
└─ types.ts                  # shared TypeScript types

server-app.ts                # all Express routes + AI/fallback logic (shared everywhere)
server.ts                    # local/Cloud Run entry point (adds Vite + static serving)
api/index.ts                 # Vercel serverless entry (re-exports server-app.ts)
netlify.toml                 # Netlify function entry (same server-app.ts)
```

---

## 8. Deploying

The same backend code ships to any of these — pick one:

- **Vercel** — `vercel.json` already routes `/api/*` → `api/index.ts`. Just set env vars in the
  Vercel dashboard and deploy.
- **Netlify** — see `netlify.toml` / `NetlifyGuideModal.tsx` in-app guide.
- **Node / Cloud Run** — `npm run build && npm start` runs `dist/server.cjs` (bundled `server.ts`).

In all cases, set `GEMINI_API_KEY` and optionally `GROQ_API_KEY` as environment variables — the
app works without them, just with less varied phrasing (§5).
