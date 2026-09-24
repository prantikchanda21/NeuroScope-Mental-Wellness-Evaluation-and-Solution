# NeuroScope — AI-Assisted Mental Health Evaluation & Self-Reflection

NeuroScope is a **web-based mental-health self-reflection and screening application** built with React, TypeScript, Vite, Express, and browser-side machine-learning models.

The project is designed around a simple idea:

> Instead of giving someone a single generic score, ask better questions, understand the language in their answers, adapt the next question to what they just said, look for safety signals, and turn the result into understandable next steps.

It combines a **20-question adaptive assessment**, local clinical-style scoring, browser-side NLP, semantic retrieval, research-grounded explanations, AI-assisted report generation, multilingual UI translation, history tracking, interactive exercises, and crisis-support guidance.

**Important:** NeuroScope is a screening/self-reflection tool, not a medical device and not a replacement for a licensed mental-health professional. Its scores and AI-generated explanations should not be treated as a diagnosis.

---

## ✨ What NeuroScope Actually Does

At a high level, a user:

1. Opens NeuroScope.
2. Optionally signs in locally or with Google.
3. Chooses an assessment length:
   - **5 questions** — quick check-in
   - **10 questions** — balanced assessment
   - **20 questions** — full adaptive screening
4. Answers questions using free text or quick prompts.
5. NeuroScope analyzes each answer while the assessment is happening.
6. The next question is selected based on the user's previous answer instead of following one completely fixed list.
7. A separate safety/risk layer continuously watches for potentially serious signals.
8. At the end, the app combines the answers, dimensional scoring, semantic analysis, research retrieval, and AI/clinical reasoning.
9. The user receives a readable results dashboard with:
   - overall interpretation
   - dimension-by-dimension scores
   - personalized suggestions
   - research-grounded explanations
   - safety guidance when appropriate
   - optional history/progress information
10. The user can continue with interactive tools such as breathing, grounding, journaling, timers, and daily protocols.

The important architectural principle is that **AI is not the only thing deciding what happens**. The application has deterministic local engines for sentiment, adaptive selection, clinical-style scoring, risk detection, and research retrieval. External AI is used primarily for natural-language generation and richer explanations.

---

# 🧠 Core Architecture

A simplified view of the system looks like this:

```text
                    ┌───────────────────────────┐
                    │        React UI           │
                    │  Components + Tailwind    │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Assessment Controller   │
                    │     src/App.tsx            │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌───────────────────┐       ┌───────────────────┐
          │ Adaptive Engine   │       │ Semantic Engine   │
          │ Question ranking  │       │ NLP / embeddings  │
          └─────────┬─────────┘       └─────────┬─────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │    Risk Engine      │
                       │ Safety + escalation │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Clinical Engine     │
                       │ Local scoring       │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Research Retrieval  │
                       │ Semantic / lexical   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Express API / AI Layer    │
                    │ Gemini + Groq fallbacks   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Assessment Results View   │
                    │ Report + recommendations  │
                    └───────────────────────────┘
```

---

# 🗂️ Project Structure

```text
project/
│
├── src/
│   ├── components/
│   │   ├── AssessmentResultsView.tsx
│   │   ├── ClinicalSummaryReportModal.tsx
│   │   ├── DailyProtocolTracker.tsx
│   │   ├── DynamicSolutionCard.tsx
│   │   ├── HeaderNav.tsx
│   │   ├── ImmediateCrisisCard.tsx
│   │   ├── InteractiveNeuroTool.tsx
│   │   ├── LanguageSwitcher.tsx
│   │   ├── LongitudinalHistoryModal.tsx
│   │   ├── LoginPage.tsx
│   │   ├── NeurochemicalSimulatorModal.tsx
│   │   ├── NeuralVisualizer.tsx
│   │   ├── QuestionCard.tsx
│   │   ├── ThoughtDefusionModal.tsx
│   │   └── ...
│   │
│   ├── data/
│   │   ├── questions.ts
│   │   └── researchCorpus.ts
│   │
│   ├── i18n/
│   │   ├── languages.ts
│   │   ├── autoTranslate.ts
│   │   └── LanguageProvider.tsx
│   │
│   ├── utils/
│   │   ├── adaptiveEngine.ts
│   │   ├── audio.ts
│   │   ├── authStorage.ts
│   │   ├── clinicalEngine.ts
│   │   ├── dynamicFeelingSolutions.ts
│   │   ├── dynamicFollowUpResponses.ts
│   │   ├── googleAuth.ts
│   │   ├── historyStorage.ts
│   │   ├── researchKnowledge.ts
│   │   ├── researchRetrieval.ts
│   │   ├── riskEngine.ts
│   │   ├── semanticEngine.ts
│   │   └── zipExport.ts
│   │
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   └── types.ts
│
├── public/
│   └── landscapes/
│
├── netlify/
│   └── functions/
│       └── api.ts
│
├── scripts/
│   └── build-zip.js
│
├── server-app.ts
├── server.ts
├── translate-service.ts
├── vite.config.ts
├── netlify.toml
├── package.json
├── tsconfig.json
└── .env.example
```

---

# 🧩 Main Technologies

| Technology | Why it is used |
|---|---|
| **React 19** | UI and component architecture |
| **TypeScript** | Type-safe application logic |
| **Vite** | Frontend development/build tooling |
| **Tailwind CSS 4** | Styling and responsive UI |
| **Express** | API server |
| **Gemini** | AI-generated assessments, reports, solutions and language assistance |
| **Groq** | Alternative AI provider/fallback |
| **Transformers.js** | Runs NLP models directly in the browser |
| **ONNX** | Efficient browser inference for the NLP models |
| **Lucide React** | UI icons |
| **Motion** | UI animation |
| **Web Audio API** | Ambient/neural feedback sounds |
| **Netlify Functions** | Serverless deployment of the API |
| **localStorage** | Browser-local accounts, history, streaks and translation cache |

---

# 🧪 The Assessment System

## Question pool

The application contains a **50-question pool** distributed across five areas:

1. Mood and Emotional State
2. Daily Functioning and Physical Well-being
3. Thought Patterns and Perception of Reality
4. Stress, Coping, and Impulse Control
5. Social Connections and Relationships

The full assessment asks **20 questions**, but it does not simply show the first 20 questions.

The adaptive engine selects questions from the larger pool.

This gives the application room to react to what the user is actually saying.

---

# 🔀 How Adaptive Question Selection Works

The adaptive engine is in:

```text
src/utils/adaptiveEngine.ts
```

The important functions are:

```text
selectNextQuestion()
selectNextQuestionAsync()
irRelevance()
computeNDCG()
analyzeAnswerSentiment()
analyzeAnswerSemantics()
```

Conceptually:

```text
Previous answer
      │
      ▼
Understand answer
      │
      ├── Sentiment
      ├── Keywords
      ├── Semantic embedding
      └── Category relevance
      │
      ▼
Score remaining questions
      │
      ├── Relevance
      ├── Diversity
      ├── Previous coverage
      └── Ranking quality
      │
      ▼
Choose next question
```

### Why this matters

Imagine a user says:

> "I have been sleeping badly and my mind keeps racing at night."

A static questionnaire may immediately ask an unrelated relationship question.

NeuroScope can instead increase the relevance of questions connected to:

- sleep
- anxiety
- racing thoughts
- physical stress
- daily functioning

The goal is not to "guess the diagnosis." The goal is to make the next question **more contextually useful**.

---

# 📐 IR Relevance and NDCG

The project uses information-retrieval concepts rather than randomly selecting every question.

### IR relevance

A candidate question is compared with the user's previous answer.

The comparison can use:

- keyword overlap
- semantic similarity
- category/theme information

The relevance value is represented as:

```text
0 → unrelated
1 → highly relevant
```

### NDCG

The project also calculates **Normalized Discounted Cumulative Gain (NDCG)** to represent how well the selected questions are ranked relative to the relevance estimates.

This is useful because an adaptive system should not only select *a* relevant question — it should prefer questions that are highly relevant near the top of the ranking.

---

# 🤖 Browser-Side NLP

One of the more technical parts of NeuroScope is that several NLP operations happen **inside the user's browser**.

The implementation lives primarily in:

```text
src/utils/semanticEngine.ts
```

The project uses Transformers.js and ONNX-compatible models.

## Models

### Sentiment

```text
Xenova/twitter-roberta-base-sentiment-latest
```

Used to estimate the emotional valence of an answer.

### Emotion classification

```text
nicky48/emotion-english-distilroberta-base-ONNX
```

The classifier works with seven broad emotion classes:

```text
anger
disgust
fear
joy
neutral
sadness
surprise
```

### Embeddings

```text
Xenova/all-MiniLM-L6-v2
```

Used to convert text into vectors.

Those vectors allow the application to compare meanings rather than only exact words.

For example:

```text
"I feel my chest tighten before presentations"
```

can be semantically related to content discussing:

```text
anticipatory anxiety / threat response
```

even when the wording is different.

---

# 🧠 Why There Is Also a Lexicon Engine

Machine-learning models can fail to load.

Reasons include:

- slow internet
- blocked model CDN
- unsupported browser
- WebAssembly limitations
- model download failure
- timeout

NeuroScope therefore has a deterministic fallback.

The adaptive engine contains an offline sentiment lexicon.

For example, terms associated with:

```text
hopelessness
worthlessness
panic
dread
freeze
shutdown
calm
hope
safe
grounded
```

can contribute to the local sentiment reading.

This gives the application a graceful degradation path:

```text
Transformer available
        ↓
Use transformer + lexicon

Transformer unavailable
        ↓
Use lexicon / deterministic analysis
```

The app therefore does not become completely unusable just because a model failed to load.

---

# 🚨 Dynamic Risk Engine

Safety is intentionally separated from ordinary sentiment analysis.

The main implementation is:

```text
src/utils/riskEngine.ts
```

This is important because:

> "negative sentiment" is not the same thing as "immediate safety risk."

The risk engine considers multiple signals.

## Signals include

### 1. Explicit risk markers

Examples include language referring to:

- self-harm
- wanting to die
- ending one's life
- not wanting to live
- severe hopelessness

### 2. Semantic similarity

The semantic engine can compare the answer against a bank of high-risk language examples.

### 3. Tone

The application tracks whether the emotional tone is becoming more distressed.

### 4. Rate of change

A user's answers are considered over time rather than only independently.

For example:

```text
Answer 1 → mildly distressed
Answer 2 → distressed
Answer 3 → severe distress
```

A deteriorating pattern can contribute additional risk weight.

### 5. Emotion distribution

Sustained fear, sadness, anger or disgust can contribute to the risk picture.

### 6. Mandatory safety question

The assessment includes safety-focused logic that is treated differently from ordinary questions.

---

# 📊 Risk Levels

The dynamic engine uses:

```text
low
elevated
high
critical
```

The internal risk representation is approximately:

```text
0 ─────────────────────────────── 100
│                                  │
Low       Elevated      High     Critical
```

The exact score is not intended to represent a clinical probability.

It is an **application safety signal** used to decide when the UI should provide stronger support and crisis guidance.

---

# 🏥 Local Clinical-Style Engine

The local scoring system lives in:

```text
src/utils/clinicalEngine.ts
```

Its main functions include:

```text
checkImmediateRisk()
analyzeAnswersLocally()
getAlternateProtocols()
```

It evaluates answers across the five assessment dimensions.

Each dimension produces a score from:

```text
0 → 100
```

where a higher value represents better functioning in the application's scoring model.

Statuses include:

```text
Optimal
Mild Strain
Moderate Concern
Severe Strain
```

These labels are application-level screening labels, **not psychiatric diagnoses**.

---

# 🔬 Research-Grounded Retrieval

NeuroScope does not simply ask an LLM:

> "Give me some mental-health advice."

Instead, it has a local research corpus:

```text
src/data/researchCorpus.ts
```

and a retrieval system:

```text
src/utils/researchRetrieval.ts
```

The retrieval layer supports two paths.

## Primary path: semantic retrieval

```text
User answer
    ↓
MiniLM embedding
    ↓
Cosine similarity
    ↓
Rank research passages
    ↓
Select relevant passages
```

## Fallback path: lexical retrieval

If embeddings are unavailable:

```text
User answer
    ↓
Tokenize
    ↓
Remove stop words
    ↓
Keyword overlap
    ↓
Theme bonus
    ↓
Rank passages
```

The system normally selects up to **4 relevant passages**.

---

# 📚 Research Themes

The project contains research-grounded material covering topics such as:

- emotional distress
- depression-related symptoms
- anxiety
- anger
- neuroception
- nervous-system regulation
- coping
- avoidance
- trauma-informed care
- social support
- co-regulation

The project documentation identifies sources including work by:

- Stephen Porges
- Pilkonis and colleagues
- Gibbons and colleagues
- SAMHSA

The research corpus is used to ground explanations and solution rationales. It should not be interpreted as proof that a particular user has a condition discussed in a source.

---

# 🧾 How the Final Result Is Produced

The result is not simply:

```text
answers → AI → result
```

Instead, it is closer to:

```text
                         User answers
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
     Local scoring       Semantic analysis     Risk engine
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    Research retrieval
                              │
                              ▼
                       AI generation
                              │
                              ▼
                     Structured result
                              │
                              ▼
                  AssessmentResultsView
```

The resulting `AssessmentResult` contains information such as:

```text
overallVerdict
severityLevel
verdictSummary
dimensionalScores
personalizedSolutions
motivationalMessage
safetyAlert
providerUsed
riskAssessment
timestamp
```

---

# 🧠 AI Provider Architecture

The server supports two major external AI providers:

```text
Gemini
Groq
```

The main implementation is in:

```text
server-app.ts
```

## Gemini

Gemini is used for structured AI generation and language tasks.

The server has model fallback logic so if one configured Gemini model fails, another supported model can be attempted.

## Groq

Groq provides another AI route and fallback.

The project can use models such as:

```text
llama-3.3-70b-versatile
llama3-70b-8192
mixtral-8x7b-32768
openai/gpt-oss-120b
```

The exact model availability can change over time, so production deployments should verify current provider model names.

---

# 🔁 AI + Local Fallback Philosophy

The application is intentionally layered.

For example:

```text
Gemini
   ↓ fails?
Groq
   ↓ fails?
Local clinical / deterministic engine
```

Similarly:

```text
Semantic transformer
   ↓ unavailable?
Lexicon + keyword retrieval
```

This means the application is designed to degrade gracefully rather than having one single point of failure.

---

# 🌍 Multilingual System

The application supports **140 languages** according to its current language registry:

```text
40 Indian languages
100 international languages
```

The main files are:

```text
src/i18n/languages.ts
src/i18n/autoTranslate.ts
src/i18n/LanguageProvider.tsx
```

## How translation works

The UI is originally maintained in English.

When a user selects another language:

```text
Rendered UI
    ↓
Collect visible text
    ↓
Batch translation request
    ↓
/api/translate
    ↓
Gemini / Groq / Google fallback
    ↓
Translated UI
```

A `MutationObserver` watches for newly rendered content, which is important because React can create new elements after the initial translation pass.

The original English text is retained so switching back to English does not require translating the translated version.

Translations are cached in `localStorage`.

---

# 🔐 Authentication

NeuroScope currently has two authentication approaches.

## Email/password

Implemented in:

```text
src/utils/authStorage.ts
```

The current system is **browser-local**.

User records are stored in:

```text
localStorage
```

Passwords are processed with:

```text
SHA-256 + random salt
```

before being stored.

### Important security limitation

This is **not a production-grade authentication backend**.

There is:

- no server-side user database
- no server-side password verification
- no cross-device synchronization
- no centralized session management

For a real production healthcare application, authentication should be replaced with a proper identity/backend system.

---

# 🔑 Google Sign-In

Google authentication is implemented through:

```text
src/utils/googleAuth.ts
```

It uses Google Identity Services.

The browser receives the Google credential and the current implementation decodes the credential client-side.

The source code itself documents the important limitation:

> A production deployment should send the credential to the backend and verify it server-side before trusting identity information.

So Google login should be considered a prototype-level implementation until server-side token verification is added.

---

# 💾 Data Storage

NeuroScope currently uses browser storage for several pieces of data.

## Assessment history

```text
neuroscope_assessment_history_v1
```

Up to the latest 15 assessment entries are retained per local user scope.

## Daily protocol streaks

```text
neuroscope_protocol_streaks_v1
```

Tracks completed activities and streak information.

## Authentication

```text
neuroscope_auth_users_v1
neuroscope_auth_session_v1
```

## Translation cache

Translations are cached by language so repeated UI translations can be faster.

### Important privacy implication

Because the current architecture uses browser `localStorage`, users should understand that this is **device/browser-local storage**, not encrypted healthcare-grade storage.

Clearing site data can remove local information.

The current project should therefore not be marketed as storing protected medical records.

---

# 🎯 Personalized Solutions

Solution generation is handled through modules such as:

```text
src/utils/dynamicFeelingSolutions.ts
src/utils/dynamicFollowUpResponses.ts
```

Solutions can have types such as:

```text
Quick 2-min
Daily Routine
Mindset
Professional
```

They can also reference interactive tools:

```text
breathing
timer
grounding
journal
```

The application attempts to match recommendations to the user's detected themes instead of presenting the exact same advice to everyone.

---

# 🧘 Interactive Features

The project includes several non-assessment experiences.

Examples include:

- breathing exercises
- grounding
- thought defusion
- journaling
- daily protocol tracking
- neurochemical simulation/visualization
- neural visualizations
- ambient audio feedback
- longitudinal history
- clinical summary report
- crisis-support card

These are implemented as independent React components so they can be changed without rewriting the assessment engine.

---

# 🔊 Audio System

Audio is implemented in:

```text
src/utils/audio.ts
```

It uses the browser's:

```text
Web Audio API
```

No external audio asset is required for the generated tones.

The engine includes effects such as:

- thinking chime
- heartbeat pulse
- completion fanfare
- gentle chime

The audio context is initialized lazily because browsers generally restrict autoplay until a user interaction occurs.

---

# 🎨 Visual System

The application contains animated visual components including:

```text
NeuralVisualizer
HelixWaveEffect
SpiralVortexEffect
RealisticGreeneryLandscape
```

These are presentation layers. They are intentionally separated from the assessment/scoring logic.

This separation makes it possible to redesign the UI without changing how answers are evaluated.

---

# 🌐 Backend/API Routes

The Express application is defined in:

```text
server-app.ts
```

Current routes include:

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | API/provider health information |
| `/api/languages` | GET | Language capability information |
| `/api/translate` | POST | Batch UI translation |
| `/api/assess` | POST | Main assessment generation |
| `/api/feeling-solution` | POST | Dynamic feeling/solution generation |
| `/api/solution-followup` | POST | Follow-up conversation around a solution |
| `/api/reassess` | POST | Reassessment/custom revision |
| `/api/upload-artwork` | POST | Artwork upload |
| `/api/download-zip` | GET | Generated project ZIP download |

The backend accepts JSON through Express with a request body limit of:

```text
10 MB
```

---

# 🖥️ Local Development

## Prerequisites

Install:

- Node.js
- npm

A modern Node.js version is recommended because the project uses current React, Vite, TypeScript and server tooling.

---

## 1. Clone the project

```bash
git clone <your-repository-url>
cd project
```

If you received the ZIP instead, extract it and enter the `project` directory:

```bash
cd project
```

---

## 2. Install dependencies

```bash
npm install
```

The repository also contains:

```text
package-lock.json
bun.lock
```

The project is primarily documented around npm.

---

## 3. Configure environment variables

Create:

```text
.env
```

or:

```text
.env.local
```

Example:

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
VITE_GOOGLE_CLIENT_ID=your_google_client_id
```

Optional:

```env
TRANSLATE_MODELS=gemini-3.8-flash,gemini-2.5-flash,gemini-flash-latest
TRANSLATION_CACHE_DIR=.cache/translations
```

### What each variable does

| Variable | Required? | Purpose |
|---|---:|---|
| `GEMINI_API_KEY` | Recommended | Gemini AI generation |
| `GROQ_API_KEY` | Optional | Groq AI fallback/provider |
| `VITE_GOOGLE_CLIENT_ID` | Optional | Google Sign-In |
| `TRANSLATE_MODELS` | Optional | Translation model preference order |
| `TRANSLATION_CACHE_DIR` | Optional | Translation cache location |

Do **not** commit real API keys to Git.

---

# ▶️ Run the App

Start development mode:

```bash
npm run dev
```

The local server runs on:

```text
http://localhost:3000
```

The development server combines:

```text
Express
+
Vite middleware
```

so frontend and API development can happen together.

---

# 🏗️ Production Build

Run:

```bash
npm run build
```

The build performs three major tasks:

1. Builds the Vite frontend.
2. Bundles the Node/Express server using esbuild.
3. Runs the project's ZIP-building script.

Then start the bundled server:

```bash
npm start
```

---

# 🧪 Type Checking

Run:

```bash
npm run lint
```

This executes:

```bash
tsc --noEmit
```

It checks the TypeScript project without generating JavaScript files.

---

# 📦 Available npm Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Starts the local development server |
| `npm run build` | Builds frontend + server bundle + ZIP |
| `npm start` | Runs the production server bundle |
| `npm run preview` | Runs Vite's production preview |
| `npm run lint` | Runs TypeScript checking |
| `npm run clean` | Removes generated build artifacts |

---

# ☁️ Netlify Deployment

The project already contains:

```text
netlify.toml
netlify/functions/api.ts
```

The intended Netlify architecture is:

```text
Browser
   │
   ├── Static frontend → Netlify
   │
   └── /api/* → Netlify Function
                       │
                       ▼
                  Express app
```

The Netlify function wraps the same Express application used by the normal Node server.

This is important because there should be **one API implementation**, not separate logic for local and Netlify environments.

---

## Netlify Build Configuration

The project currently uses:

```toml
[build]
command = "npm run build"
publish = "dist"
functions = "netlify/functions"
```

The API redirect maps:

```text
/api/*
```

to:

```text
/.netlify/functions/api/*
```

The redirect must remain before the catch-all frontend redirect.

---

# 🚀 Other Deployment Options

Because `server.ts` is a normal Node/Express entry point, the application can also be adapted for:

- Cloud Run
- Render
- Railway
- Fly.io
- VPS/server
- other Node-compatible hosting

The important requirement is that the deployment can run the Express server and provide the required environment variables.

---

# 🔄 Local vs Serverless Architecture

## Local / Node server

```text
server.ts
    ↓
server-app.ts
    ↓
Express
    ↓
API routes
```

## Netlify

```text
Netlify Function
    ↓
netlify/functions/api.ts
    ↓
server-app.ts
    ↓
Express routes
```

The shared `server-app.ts` is deliberate: it prevents the AI/API logic from being duplicated.

---

# 🧑‍💻 Developer Guide: Where to Change Things

## Want to add/edit questions?

Open:

```text
src/data/questions.ts
```

Each question contains information such as:

```ts
{
  id: 1,
  category: "...",
  categoryIndex: 0,
  question: "...",
  rationale: "...",
  placeholder: "...",
  keywords: [...],
  quickPrompts: [...],
  yesMeansConcern: true
}
```

### Important fields

`keywords`

Used by the adaptive/retrieval logic.

`quickPrompts`

Short answers shown to the user.

`rationale`

Explains why the question exists.

`yesMeansConcern`

Helps interpret short literal yes/no responses.

`mandatory`

Can force a safety-critical question into an assessment.

---

## Want to change scoring?

Look at:

```text
src/utils/clinicalEngine.ts
```

---

## Want to change adaptive behavior?

Look at:

```text
src/utils/adaptiveEngine.ts
```

---

## Want to change safety/risk behavior?

Look at:

```text
src/utils/riskEngine.ts
```

---

## Want to change NLP models?

Look at:

```text
src/utils/semanticEngine.ts
```

The model identifiers are defined together in:

```ts
SEMANTIC_MODELS
```

---

## Want to change research retrieval?

Look at:

```text
src/data/researchCorpus.ts
src/utils/researchRetrieval.ts
src/utils/researchKnowledge.ts
```

---

## Want to change AI prompts/API behavior?

Look at:

```text
server-app.ts
```

This is where the Gemini/Groq calls, structured response generation, translation endpoint, and API routes live.

---

## Want to change authentication?

Look at:

```text
src/utils/authStorage.ts
src/utils/googleAuth.ts
```

---

## Want to change history/streak behavior?

Look at:

```text
src/utils/historyStorage.ts
```

---

## Want to add a new language?

Start with:

```text
src/i18n/languages.ts
```

Then check:

```text
src/i18n/autoTranslate.ts
src/i18n/LanguageProvider.tsx
```

---

# 🧱 Type System

The main shared interfaces are in:

```text
src/types.ts
```

Some of the most important types are:

```text
Question
SentimentResult
EmotionClassification
SemanticAnalysis
RiskAssessment
AdaptiveSelectionMeta
AnswerRecord
DimensionScore
SolutionItem
AssessmentResult
```

Keeping these types centralized makes it much easier to understand how data moves through the application.

For example:

```text
Question
   ↓
AnswerRecord
   ↓
SemanticAnalysis
   ↓
RiskAssessment
   ↓
AssessmentResult
```

---

# 🛡️ Failure Handling and Graceful Degradation

The project has several fallback layers.

## NLP failure

```text
Transformer
   ↓
Lexicon
```

## Research retrieval failure

```text
Semantic retrieval
   ↓
Keyword retrieval
```

## AI provider failure

```text
Gemini
   ↓
Groq
   ↓
Local/deterministic logic where applicable
```

## Translation failure

The translation service can fall back to a keyless Google translation endpoint, while failed/partial translations are not intentionally treated as successful cached translations.

The goal throughout the codebase is:

> A missing optional service should reduce intelligence or convenience, not completely break the application.

---

# 🔒 Security and Privacy Notes

This section is especially important if you plan to turn NeuroScope into a real product.

## Current prototype limitations

The current architecture is **not sufficient for handling regulated clinical data in a production healthcare environment**.

In particular:

### 1. Browser-local authentication

Accounts are stored in `localStorage`.

### 2. Client-side Google credential decoding

Google identity should be verified server-side before being trusted in production.

### 3. Assessment history is browser-local

History is not stored in a secure centralized database.

### 4. AI APIs receive assessment information

Depending on the route and configuration, user answers may be sent to the configured AI provider.

You should review the provider's current data-handling and retention policies before using real sensitive information.

### 5. No formal clinical compliance layer

The project does not by itself establish compliance with frameworks or regulations such as:

- HIPAA
- GDPR
- India's DPDP Act
- medical-device regulations
- institutional clinical governance

Compliance depends on the complete deployed system, organization, jurisdiction, contracts, data flows, and operational controls.

---

# 🏥 If You Want to Productionize It

A production-grade version should consider adding:

```text
Real authentication service
        ↓
Secure backend session/token handling
        ↓
Encrypted database
        ↓
Audit logging
        ↓
Consent management
        ↓
Data retention/deletion controls
        ↓
Server-side identity verification
        ↓
Provider data-processing agreements
        ↓
Monitoring + abuse protection
        ↓
Clinical safety review
```

Also consider:

- rate limiting
- CSRF protection where applicable
- strict CORS policy
- input validation
- server-side schema validation
- encrypted secrets management
- structured logging without sensitive answer content
- database encryption
- automated tests
- accessibility testing
- model/version pinning
- prompt-injection defenses
- human clinical review of safety flows

---

# ⚠️ Clinical Safety Philosophy

NeuroScope should never communicate:

```text
"You have depression."
"You have anxiety."
"You are clinically safe."
```

based solely on its own score.

A better framing is:

```text
"Your responses suggest..."
"Your answers indicate elevated distress..."
"This screening result is not a diagnosis..."
"Consider speaking with a qualified professional..."
```

If a user expresses immediate danger or self-harm intent, the application should prioritize:

```text
Immediate safety guidance
+
Crisis resources
+
Encouraging contact with a trusted person/professional
```

rather than continuing as if it were an ordinary wellness questionnaire.

---

# 🧪 Testing Checklist

Before submitting or deploying a change, run:

```bash
npm run lint
npm run build
```

Then manually test:

### Assessment

- [ ] 5-question mode
- [ ] 10-question mode
- [ ] 20-question mode
- [ ] Quick prompts
- [ ] Free-text answers
- [ ] Adaptive question changes
- [ ] Back/forward navigation if applicable
- [ ] Final results

### NLP

- [ ] Models load
- [ ] Model loading fallback works
- [ ] Offline/slow model behavior
- [ ] Sentiment output
- [ ] Emotion output
- [ ] Semantic similarity

### Safety

- [ ] Normal answer
- [ ] Distressed answer
- [ ] Explicit crisis-language test in a controlled development environment
- [ ] Safety card
- [ ] Crisis resources
- [ ] Risk escalation behavior

### AI

- [ ] Gemini available
- [ ] Gemini unavailable
- [ ] Groq available
- [ ] Both unavailable
- [ ] Invalid/malformed AI response

### Localization

- [ ] English
- [ ] Indian language
- [ ] International language
- [ ] RTL language
- [ ] Dynamic modal translation
- [ ] Switching back to English

### Storage

- [ ] New account
- [ ] Sign in
- [ ] Sign out
- [ ] Assessment history
- [ ] Streak tracking
- [ ] Browser storage cleared

---

# 🧭 A Beginner-Friendly Mental Model

If you're new to the codebase, don't try to understand every file at once.

Start here:

```text
1. src/App.tsx
        ↓
2. src/types.ts
        ↓
3. src/data/questions.ts
        ↓
4. src/utils/adaptiveEngine.ts
        ↓
5. src/utils/riskEngine.ts
        ↓
6. src/utils/clinicalEngine.ts
        ↓
7. src/utils/semanticEngine.ts
        ↓
8. server-app.ts
        ↓
9. components/AssessmentResultsView.tsx
```

Think of the application as five layers:

### Layer 1 — Interface

"What does the user see?"

```text
components/
```

### Layer 2 — Assessment content

"What are we asking?"

```text
data/questions.ts
```

### Layer 3 — Intelligence

"How do we understand and score answers?"

```text
adaptiveEngine.ts
semanticEngine.ts
clinicalEngine.ts
riskEngine.ts
```

### Layer 4 — Knowledge

"Where do explanations come from?"

```text
researchCorpus.ts
researchRetrieval.ts
researchKnowledge.ts
```

### Layer 5 — Generation

"How do we turn structured information into natural language?"

```text
server-app.ts
Gemini / Groq
```

Once these five layers make sense, most of the repository becomes much easier to navigate.

---

# 🔬 A More Technical Mental Model

For an advanced developer, the core pipeline can be represented as:

```text
x_t = user's answer at time t

S_t = sentiment(x_t)
E_t = emotion(x_t)
V_t = embedding(x_t)

R_t = semanticRisk(V_t)
T_t = toneTrend(S_1 ... S_t)

Q_{t+1} = adaptiveRank(
    remainingQuestions,
    x_t,
    S_t,
    V_t,
    categoryCoverage,
    informationRetrieval
)

Risk_t = f(
    explicitMarkers,
    R_t,
    S_t,
    E_t,
    T_t,
    safetyResponse
)

D = clinicalDimensionScoring(allAnswers)

P = retrieveResearch(
    answerThemes,
    semanticSimilarity,
    lexicalFallback
)

Result = generate(
    answers,
    D,
    Risk_t,
    P
)
```

The important architectural idea is that **different signals are kept separate**.

Sentiment is not risk.

Risk is not diagnosis.

Research retrieval is not diagnosis.

AI generation is not the sole source of truth.

That separation makes the system easier to reason about and safer to modify.

---

# 🧰 Common Development Problems

## "The NLP models are taking a long time"

The semantic models are downloaded into the browser and loaded through Transformers.js.

The first run can therefore take longer.

Once loaded/cached, subsequent use can be faster.

---

## "The AI result isn't appearing"

Check:

```text
/api/health
```

Then verify:

```env
GEMINI_API_KEY
GROQ_API_KEY
```

Also inspect the browser's network tab for:

```text
/api/assess
```

---

## "Google Sign-In isn't showing"

Check:

```env
VITE_GOOGLE_CLIENT_ID
```

The Google OAuth client should be configured for the correct web origin.

---

## "Translations aren't appearing"

Check:

```text
/api/languages
/api/translate
```

Then inspect the browser network tab.

The application has multiple translation fallbacks, so a completely broken translation usually points to a network/API/runtime problem rather than a missing UI string.

---

## "History disappeared"

Remember that history currently lives in browser `localStorage`.

It can disappear if:

- site data was cleared
- browser storage was reset
- a different browser/device is being used
- the storage bucket changed

This is expected behavior for the current prototype architecture.

---

# 📈 Future Improvements

The architecture leaves room for several meaningful upgrades.

## Backend persistence

Move from:

```text
localStorage
```

to:

```text
PostgreSQL / Supabase / Firebase / other secure database
```

with proper access control.

## Real authentication

Replace browser-local authentication with:

```text
OAuth/OIDC
+
server-side token verification
+
secure session handling
```

## Clinical validation

The scoring engine should be validated against appropriate clinical instruments and reviewed by qualified professionals before making clinical claims.

## Better observability

Add:

- structured server logs
- error tracking
- latency monitoring
- model failure metrics
- API usage monitoring

without logging sensitive user responses unnecessarily.

## Automated testing

Add:

```text
unit tests
integration tests
API tests
component tests
end-to-end tests
accessibility tests
```

## Model governance

Pin model versions and document:

```text
model name
model version
download source
license
intended use
known limitations
evaluation results
```

---

# 📜 Research-Grounded Update

The repository includes:

```text
RESEARCH_GROUNDED_ADDITIONS.md
```

This documents the research-driven expansion of the question pool and related lexicon/keyword updates.

The update adds research-grounded questions and vocabulary while keeping the existing scoring architecture.

This is a useful file to read if you want to understand **why particular question wording and keywords exist**.

---

# 🤝 Contributing

If you want to extend NeuroScope:

1. Create a feature branch.
2. Make the smallest isolated change possible.
3. Keep shared types updated.
4. Avoid mixing UI changes with scoring changes unless necessary.
5. Run:

```bash
npm run lint
npm run build
```

6. Test the affected flow manually.
7. Document changes that affect scoring, risk, research grounding, or data handling.

For changes to safety logic, also document:

- what signal changed
- why it changed
- what false-positive/false-negative behavior is expected
- how the user-facing safety flow changes

---

# 📄 License / Third-Party Models

Before public/commercial distribution, review the licenses and terms for every dependency, model, dataset, API provider, font, image, and other external asset included in the project.

In particular, verify the current licensing/usage terms for the Hugging Face models and external AI providers used by your deployment.

---

# ❤️ Final Note

NeuroScope is best understood as an **AI-assisted self-reflection and screening experience with a layered technical architecture**.

Its strongest architectural idea is not simply "AI generates a mental-health report."

It is the combination of:

```text
Adaptive questioning
        +
Browser-side NLP
        +
Deterministic safety logic
        +
Local clinical-style scoring
        +
Semantic research retrieval
        +
AI-assisted natural-language generation
        +
Multilingual UI
        +
Interactive self-regulation tools
```

That separation makes the project easier to understand, easier to debug, and easier to evolve.

If you are a beginner, start with `App.tsx → questions.ts → adaptiveEngine.ts`.

If you are an advanced developer, start with `types.ts → semanticEngine.ts → riskEngine.ts → researchRetrieval.ts → server-app.ts`.

Either way, the central question to keep in mind while reading the code is:

> **What information is being produced at this stage, and which layer is responsible for making the next decision?**

Once that question is clear, the rest of NeuroScope becomes much easier to follow.
