/**
 * Transformer-based semantic engine (Transformers.js).
 *
 * This module replaces two hand-rolled pieces of the original engine with real
 * neural inference that runs entirely in the browser via WebAssembly:
 *
 *   1. Fine-tuned RoBERTa sentiment classification
 *      (`Xenova/twitter-roberta-base-sentiment-latest`) replaces the fixed
 *      `SENTIMENT_LEXICON` as the primary valence reader.
 *   2. Multi-class emotion classification (7 Ekman emotions, distilroberta)
 *      which the original app did not have at all.
 *   3. Sentence embeddings (`Xenova/all-MiniLM-L6-v2`) that power semantic
 *      vector search, replacing TF-weighted keyword overlap for both adaptive
 *      question selection and research-passage retrieval.
 *
 * Design rules that keep the rest of the app intact:
 *  - Nothing loads until the app explicitly calls `warmSemanticEngine()`, and
 *    the dynamic `import()` means Vite code-splits the runtime into its own
 *    chunk (the initial bundle is unchanged).
 *  - Every entry point has a timeout and returns `null`/a lexicon fallback when
 *    a model is unavailable (offline, blocked CDN, very old browser). The
 *    clinical engine therefore never depends on the network.
 *  - All inference results are memoized in LRU caches keyed by normalized text,
 *    so re-analyzing an answer (which happens on submit, on review, and when
 *    building API request bodies) is free.
 */

import type {
  EmotionClassification,
  EmotionLabel,
  EmotionScore,
  SemanticAnalysis,
  SemanticEngineState,
  SemanticEngineStatus,
  SemanticModelKey,
  SentimentResult,
} from '../types';

/** Confirmed-available ONNX checkpoints (all expose `onnx/model_quantized.onnx`). */
export const SEMANTIC_MODELS: Record<SemanticModelKey, string> = {
  sentiment: 'Xenova/twitter-roberta-base-sentiment-latest',
  emotion: 'nicky48/emotion-english-distilroberta-base-ONNX',
  embedding: 'Xenova/all-MiniLM-L6-v2',
};

const MODEL_LABELS: Record<SemanticModelKey, string> = {
  sentiment: 'fine-tuned RoBERTa sentiment model',
  emotion: 'multi-class emotion classifier',
  embedding: 'semantic embedding model',
};

/** Quantized wasm weights: sentiment ~125MB, emotion ~82MB, MiniLM ~23MB.
 * Downloaded once, then served from the browser's Cache Storage. */
const DEFAULT_DTYPE = 'q8';
const INFERENCE_TIMEOUT_MS = 7000;
const BATCH_TIMEOUT_MS = 12000;
const MODEL_WAIT_TIMEOUT_MS = 8000;
/** Classifiers are 512-token RoBERTa models; answers are short, but cap anyway
 * so a pasted essay cannot stall inference. */
const MAX_CLASSIFY_CHARS = 1100;
const MAX_EMBED_CHARS = 2000;

/**
 * First-person phrasings of active risk, paraphrased (never verbatim quotes)
 * from the safety literature the app already cites (SAMHSA TIP 57, PROMIS
 * depression item bank). A user answer is scored by cosine similarity to the
 * nearest of these, which catches crisis language the word-level lexicon was
 * never going to enumerate ("I would rather not wake up", "everyone would be
 * relieved if I were gone", ...).
 */
export const HIGH_RISK_EXEMPLARS: string[] = [
  'I want to kill myself',
  'I have been thinking about ending my life',
  'I have a plan for how I would take my own life',
  'I do not want to be alive anymore',
  'I wish I could go to sleep and never wake up',
  'I have been hurting myself on purpose',
  'I have been cutting or burning myself to cope',
  'I cannot keep myself safe right now',
  'I feel like a burden to everyone around me',
  'Everyone would be better off without me',
  'There is no reason for me to keep living',
  'I feel completely hopeless and see no way forward',
  'The thought of harming myself keeps coming back',
  'I am afraid of what I might do to myself',
];

/**
 * Compact emotion lexicon used only as the offline fallback for the multi-class
 * classifier, so `EmotionClassification` always has a value to hand back.
 */
const EMOTION_LEXICON: Record<EmotionLabel, string[]> = {
  anger: ['angry', 'anger', 'furious', 'rage', 'irritated', 'irritable', 'mad', 'resentful', 'frustrated', 'annoyed', 'snapped', 'lash', 'hate'],
  disgust: ['disgust', 'disgusting', 'revolting', 'sickened', 'repulsed', 'gross', 'contempt', 'ashamed of', 'worthless', 'disgrace'],
  fear: ['afraid', 'scared', 'fear', 'fearful', 'terrified', 'anxious', 'anxiety', 'panic', 'worried', 'worry', 'dread', 'nervous', 'uneasy', 'tense', 'overwhelmed', 'racing heart'],
  joy: ['happy', 'happiness', 'joy', 'glad', 'grateful', 'content', 'excited', 'hopeful', 'great', 'wonderful', 'good', 'calm', 'peaceful', 'steady', 'grounded', 'okay', 'fine'],
  neutral: ['neutral', 'average', 'usual', 'normal', 'same', 'nothing much', 'unsure', 'maybe'],
  sadness: ['sad', 'sadness', 'depressed', 'depression', 'down', 'empty', 'numb', 'cry', 'crying', 'tearful', 'lonely', 'alone', 'hopeless', 'worthless', 'grief', 'loss', 'miserable', 'heartbroken', 'flat', 'meaningless'],
  surprise: ['surprised', 'surprise', 'shocked', 'astonished', 'unexpected', 'suddenly', 'startled', 'amazed'],
};

const DISTRESS_EMOTIONS: EmotionLabel[] = ['fear', 'sadness', 'anger', 'disgust'];

// ---------------------------------------------------------------------------
// Runtime guards
// ---------------------------------------------------------------------------

/** The transformer stack needs a browser with WebAssembly; the Express/Netlify
 * server bundle imports this module but never invokes it. */
export function isSemanticRuntimeSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof WebAssembly !== 'undefined' &&
    typeof WebAssembly.instantiate === 'function' &&
    typeof navigator !== 'undefined'
  );
}

// ---------------------------------------------------------------------------
// Engine state + subscriptions (drives the loading indicator in the UI)
// ---------------------------------------------------------------------------

const engineState: SemanticEngineState = {
  status: 'idle',
  models: {
    sentiment: { state: 'pending', progress: 0 },
    emotion: { state: 'pending', progress: 0 },
    embedding: { state: 'pending', progress: 0 },
  },
  progress: 0,
  detail: 'Semantic engine idle.',
};

const listeners = new Set<(state: SemanticEngineState) => void>();

function snapshot(): SemanticEngineState {
  return {
    status: engineState.status,
    models: {
      sentiment: { ...engineState.models.sentiment },
      emotion: { ...engineState.models.emotion },
      embedding: { ...engineState.models.embedding },
    },
    progress: engineState.progress,
    detail: engineState.detail,
    reason: engineState.reason,
  };
}

function notify(): void {
  const state = snapshot();
  for (const fn of listeners) {
    try {
      fn(state);
    } catch {
      // a broken listener must never break inference
    }
  }
}

function recomputeOverallProgress(): void {
  const keys = Object.keys(engineState.models) as SemanticModelKey[];
  let sum = 0;
  for (const k of keys) {
    const m = engineState.models[k];
    sum += m.state === 'ready' ? 1 : m.progress;
  }
  engineState.progress = Math.max(0, Math.min(1, sum / keys.length));
}

function refreshStatus(): void {
  const keys = Object.keys(engineState.models) as SemanticModelKey[];
  const ready = keys.filter((k) => engineState.models[k].state === 'ready').length;
  const failed = keys.filter((k) => engineState.models[k].state === 'failed').length;

  let status: SemanticEngineStatus = engineState.status;
  if (ready === keys.length) status = 'ready';
  else if (ready > 0) status = 'partial';
  else if (failed === keys.length) status = 'unavailable';
  else if (keys.some((k) => engineState.models[k].state === 'loading' || engineState.models[k].state === 'ready')) {
    status = 'loading';
  }
  engineState.status = status;
  if (status === 'ready') engineState.detail = 'Semantic engine ready (RoBERTa sentiment · emotions · embeddings).';
  else if (status === 'partial') engineState.detail = 'Semantic engine partially ready — remaining models still loading.';
  else if (status === 'unavailable') engineState.detail = engineState.reason || 'Semantic models unavailable — using the offline clinical engine.';
  recomputeOverallProgress();
  notify();
}

export function subscribeSemanticState(fn: (state: SemanticEngineState) => void): () => void {
  listeners.add(fn);
  fn(snapshot());
  return () => listeners.delete(fn);
}

export function getSemanticState(): SemanticEngineState {
  return snapshot();
}

/** Which individual models finished loading — used to label analysis output. */
export function isModelReady(key: SemanticModelKey): boolean {
  return engineState.models[key]?.state === 'ready';
}

export function isSemanticReady(): boolean {
  return engineState.status === 'ready' || engineState.status === 'partial';
}

// ---------------------------------------------------------------------------
// LRU caches
// ---------------------------------------------------------------------------

class LruCache<K, V> {
  private map = new Map<K, V>();
  constructor(private readonly max: number) {}

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value === undefined) return undefined;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) {
      const oldest = this.map.keys().next().value as K | undefined;
      if (oldest === undefined) break;
      this.map.delete(oldest);
    }
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  get size(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }
}

const semanticsCache = new LruCache<string, SemanticAnalysis>(400);
const embeddingCache = new LruCache<string, Float32Array>(800);
/** De-duplicates identical concurrent inference calls. */
const inFlightSemantics = new Map<string, Promise<SemanticAnalysis | null>>();

function normalizeText(text: string): string {
  return (text || '').replace(/\s+/g, ' ').trim();
}

function cacheKey(text: string): string {
  return normalizeText(text).toLowerCase();
}

/** The embedding of every question in the pool is stable for the app's
 * lifetime, so this cache is looked up on every adaptive selection. */
export function getCachedEmbedding(text: string): Float32Array | null {
  return embeddingCache.get(cacheKey(text)) || null;
}

export function getCachedSemantics(text: string): SemanticAnalysis | null {
  return semanticsCache.get(cacheKey(text)) || null;
}

export function clearSemanticCaches(): void {
  semanticsCache.clear();
  embeddingCache.clear();
  inFlightSemantics.clear();
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

interface TransformersModule {
  pipeline: (task: string, model: string, options?: Record<string, unknown>) => Promise<any>;
  env: any;
}

/** When true, ONNX Runtime executes every model in its own Web Worker instead of
 * on the page's main thread. Transformer inference is CPU-heavy WebAssembly; on the
 * main thread it freezes typing, scrolling and button feedback for as long as each
 * pass runs. The first model load smoke-tests the worker path and switches this off
 * (falling back to main-thread inference) if the browser cannot run it. */
let proxyEnabled = false;
let proxyDecision: Promise<void> | null = null;

let modulePromise: Promise<TransformersModule | null> | null = null;
const pipelinePromises: Partial<Record<SemanticModelKey, Promise<any>>> = {};
const pipelineFailures: Partial<Record<SemanticModelKey, number>> = {};
const loadedPipelines: Partial<Record<SemanticModelKey, any>> = {};
/** Per-model download progress, aggregated across the model's files. */
const fileProgress: Partial<Record<SemanticModelKey, Map<string, number>>> = {};

async function loadTransformersModule(): Promise<TransformersModule | null> {
  if (!isSemanticRuntimeSupported()) {
    engineState.reason =
      'This environment cannot run WebAssembly models (server-side or unsupported browser).';
    return null;
  }
  if (!modulePromise) {
    modulePromise = import('@huggingface/transformers')
      .then((mod: any) => {
        // Only ever fetch weights from the Hugging Face hub — never probe the
        // server for a local `/models` folder, which would 404 on every load.
        if (mod?.env) {
          mod.env.allowLocalModels = false;
          mod.env.allowRemoteModels = true;
          mod.env.useBrowserCache = true;
          try {
            if (typeof Worker !== 'undefined' && mod.env.backends?.onnx?.wasm) {
              mod.env.backends.onnx.wasm.proxy = true;
              proxyEnabled = true;
            }
          } catch {
            proxyEnabled = false;
          }
        }
        return mod as TransformersModule;
      })
      .catch((err) => {
        engineState.reason = `Could not load the transformer runtime: ${err?.message || err}`;
        modulePromise = null;
        return null;
      });
  }
  return modulePromise;
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return new Promise<T | null>((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        resolve(null);
      }
    }, ms);
    promise.then(
      (value) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(value);
        }
      },
      () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(null);
        }
      }
    );
  });
}

function makeProgressCallback(key: SemanticModelKey) {
  if (!fileProgress[key]) fileProgress[key] = new Map<string, number>();
  const files = fileProgress[key]!;
  return (event: any) => {
    if (!event) return;
    const model = engineState.models[key];
    // Real weight downloads report pct; skip the small config/tokenizer chatter.
    if (event.status === 'progress' && typeof event.progress === 'number') {
      files.set(String(event.file || event.name || 'weights'), event.progress / 100);
      let sum = 0;
      for (const v of files.values()) sum += v;
      model.progress = Math.max(0, Math.min(0.99, files.size ? sum / files.size : 0));
      engineState.detail = `Downloading ${MODEL_LABELS[key]}… ${Math.round(model.progress * 100)}%`;
    } else if (event.status === 'ready') {
      model.progress = Math.max(model.progress, 0.99);
      engineState.detail = `Loading ${MODEL_LABELS[key]} into memory…`;
    }
    recomputeOverallProgress();
    notify();
  };
}

/** One tiny inference to prove the pipeline (and, when enabled, its worker) works. */
async function smokeTestPipeline(pipe: any, key: SemanticModelKey): Promise<void> {
  const call =
    key === 'embedding'
      ? pipe('ok', { pooling: 'mean', normalize: true })
      : pipe('ok', { topk: 1 });
  const result = await withTimeout(Promise.resolve(call), 12000);
  if (result == null) throw new Error('smoke test produced no output');
}

/**
 * Builds a pipeline. The first model to load also decides whether the ONNX
 * worker (proxy) mode is usable; every other model waits for that decision so all
 * sessions are created in the same mode.
 */
async function createPipelineSafely(mod: TransformersModule, key: SemanticModelKey, task: string): Promise<any> {
  const make = () =>
    mod.pipeline(task, SEMANTIC_MODELS[key], {
      dtype: DEFAULT_DTYPE,
      device: 'wasm',
      progress_callback: makeProgressCallback(key),
    });

  if (!proxyEnabled) return make();
  if (proxyDecision) {
    await proxyDecision;
    return make();
  }

  let settle!: () => void;
  proxyDecision = new Promise<void>((resolve) => (settle = resolve));
  let decided = false;
  try {
    let pipe = await make();
    try {
      await smokeTestPipeline(pipe, key);
    } catch (err) {
      console.info('ONNX worker mode unavailable — running the models on the main thread instead:', err);
      proxyEnabled = false;
      try {
        mod.env.backends.onnx.wasm.proxy = false;
      } catch {
        // ignore
      }
      try {
        await pipe?.dispose?.();
      } catch {
        // ignore
      }
      pipe = await make();
    }
    decided = true;
    return pipe;
  } finally {
    // A failed download must not lock the decision in; the next attempt re-decides.
    if (!decided) proxyDecision = null;
    settle();
  }
}

async function getPipeline(key: SemanticModelKey, waitMs = MODEL_WAIT_TIMEOUT_MS): Promise<any | null> {
  if (loadedPipelines[key]) return loadedPipelines[key];
  if ((pipelineFailures[key] || 0) >= 2) return null;
  if (!pipelinePromises[key]) {
    const task = key === 'embedding' ? 'feature-extraction' : 'text-classification';
    engineState.models[key].state = 'loading';
    engineState.detail = `Loading ${MODEL_LABELS[key]}…`;
    refreshStatus();

    pipelinePromises[key] = (async () => {
      const mod = await loadTransformersModule();
      if (!mod) {
        engineState.models[key].state = 'failed';
        delete fileProgress[key];
        refreshStatus();
        return null;
      }
      const pipe = await createPipelineSafely(mod, key, task);
      engineState.models[key].state = 'ready';
      engineState.models[key].progress = 1;
      loadedPipelines[key] = pipe;
      refreshStatus();
      return pipe;
    })().catch((err) => {
      pipelineFailures[key] = (pipelineFailures[key] || 0) + 1;
      pipelinePromises[key] = undefined;
      if ((pipelineFailures[key] || 0) >= 2) {
        engineState.models[key].state = 'failed';
        engineState.reason = engineState.reason || `Model unavailable: ${err?.message || err}`;
      } else {
        engineState.models[key].state = 'pending';
      }
      refreshStatus();
      return null;
    }) as Promise<any>;
  }
  return withTimeout(pipelinePromises[key]!, waitMs);
}

/**
 * Kick off model downloads in the background. Called when an assessment
 * starts, so the ~230MB of quantized weights (cached after the first run)
 * download while the user is still reading question one.
 *
 * Safe to call repeatedly; returns true as soon as at least one model is ready.
 */
export async function warmSemanticEngine(options?: { timeoutMs?: number }): Promise<boolean> {
  if (!isSemanticRuntimeSupported()) {
    engineState.reason =
      'This environment cannot run WebAssembly models (server-side or unsupported browser).';
    engineState.status = 'unavailable';
    notify();
    return false;
  }
  if (engineState.status === 'idle') {
    engineState.status = 'loading';
    engineState.detail = 'Starting the semantic engine…';
    notify();
  }
  const budget = options?.timeoutMs ?? 45000;
  const order: SemanticModelKey[] = ['embedding', 'sentiment', 'emotion'];

  // Resolve as soon as the FIRST model is usable (the small embedding model, which
  // also powers question selection and research retrieval) ...
  const firstReady = await getPipeline(order[0], budget);

  // ... but keep loading the rest in the background. Previously the loop stopped
  // after the first success, so the ~200MB sentiment/emotion models were only
  // requested lazily on the person's first submit — which is exactly when the
  // UI then stalled waiting for them.
  void (async () => {
    for (const key of order.slice(1)) {
      await getPipeline(key, 180000);
    }
  })();

  if (firstReady) return true;
  // The embedding model is slow/unavailable: report whether any other model made it.
  const fallback = await getPipeline(order[1], 3000);
  return !!fallback || isSemanticReady();
}

/** Force a retry after a transient failure (e.g. the user came back online). */
export function resetSemanticEngine(): void {
  for (const key of Object.keys(pipelinePromises) as SemanticModelKey[]) {
    pipelinePromises[key] = undefined;
    pipelineFailures[key] = 0;
  }
  modulePromise = null;
  clearSemanticCaches();
  engineState.status = 'idle';
  engineState.reason = undefined;
  engineState.detail = 'Semantic engine idle.';
  refreshStatus();
}

// ---------------------------------------------------------------------------
// Classification output normalization
// ---------------------------------------------------------------------------

interface RawLabelScore {
  label: string;
  score: number;
}

/** Flattens whatever shape the pipeline returns (single object, flat array, or
 * nested arrays for batched inputs) into a flat label/score list. */
function flattenClassificationOutput(raw: any): RawLabelScore[] {
  const out: RawLabelScore[] = [];
  const walk = (node: any) => {
    if (!node) return;
    if (Array.isArray(node)) {
      for (const child of node) walk(child);
      return;
    }
    if (typeof node === 'object' && typeof node.label === 'string' && typeof node.score === 'number') {
      out.push({ label: node.label, score: node.score });
    }
  };
  walk(raw);
  return out;
}

function isEmotionLabel(value: string): value is EmotionLabel {
  return (['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'] as string[]).includes(value);
}

/** Offline fallback: score the emotion lexicon, then spread the remainder to
 * `neutral` so the distribution still sums to ~1 and can be compared with the
 * transformer output downstream. */
function lexiconEmotions(text: string): EmotionClassification {
  const lower = ` ${(text || '').toLowerCase()} `;
  const totals: Record<EmotionLabel, number> = {
    anger: 0,
    disgust: 0,
    fear: 0,
    joy: 0,
    neutral: 0,
    sadness: 0,
    surprise: 0,
  };
  for (const label of Object.keys(EMOTION_LEXICON) as EmotionLabel[]) {
    for (const word of EMOTION_LEXICON[label]) {
      if (lower.includes(word)) totals[label] += word.includes(' ') ? 2 : 1;
    }
  }
  const hits = (Object.keys(totals) as EmotionLabel[]).reduce((sum, l) => sum + totals[l], 0);
  if (hits === 0) totals.neutral = 1;
  else totals.neutral += Math.max(1, hits * 0.35);

  const total = (Object.keys(totals) as EmotionLabel[]).reduce((sum, l) => sum + totals[l], 0) || 1;
  const all: EmotionScore[] = (Object.keys(totals) as EmotionLabel[])
    .map((label) => ({ label, score: totals[label] / total }))
    .sort((a, b) => b.score - a.score);
  const distressWeight = all
    .filter((e) => DISTRESS_EMOTIONS.includes(e.label))
    .reduce((sum, e) => sum + e.score, 0);

  return {
    top: all[0],
    all,
    distressWeight,
    distressDominant: DISTRESS_EMOTIONS.includes(all[0].label),
    source: 'lexicon',
  };
}

/** Map the RoBERTa 3-way head (negative / neutral / positive) onto the app's
 * SentimentResult contract. */
function sentimentFromProbabilities(probs: Record<string, number>): SentimentResult {
  const pNegative = probs.negative ?? 0;
  const pNeutral = probs.neutral ?? 0;
  const pPositive = probs.positive ?? 0;
  const score = Math.max(-1, Math.min(1, pPositive - pNegative));
  const magnitude = Math.max(0, Math.min(1, 1 - pNeutral));

  let label: SentimentResult['label'] = 'neutral';
  if (score <= -0.55) label = 'severe';
  else if (score <= -0.18) label = 'distressed';
  else if (score >= 0.28) label = 'calm';

  return { score: Number(score.toFixed(3)), label, magnitude: Number(magnitude.toFixed(3)), source: 'transformer' };
}

// ---------------------------------------------------------------------------
// Public inference API
// ---------------------------------------------------------------------------

/** Fine-tuned RoBERTa sentiment. Returns null when the model is not ready. */
export async function classifySentiment(
  text: string,
  timeoutMs = INFERENCE_TIMEOUT_MS
): Promise<SentimentResult | null> {
  const clean = normalizeText(text);
  if (clean.length < 2) return null;
  const pipe = await getPipeline('sentiment', timeoutMs);
  if (!pipe) return null;
  const result = await withTimeout(
    Promise.resolve(pipe(clean.slice(0, MAX_CLASSIFY_CHARS), { topk: 4 })),
    timeoutMs
  );
  if (!result) return null;
  const rows = flattenClassificationOutput(result);
  if (!rows.length) return null;

  const probs: Record<string, number> = {};
  for (const row of rows) {
    probs[row.label.toLowerCase()] = (probs[row.label.toLowerCase()] || 0) + row.score;
  }
  // Some exports emit LABEL_0/1/2 instead of names; map by known order.
  if (!('negative' in probs) && !('positive' in probs)) {
    const ordered = rows.slice().sort((a, b) => a.label.localeCompare(b.label));
    const names = ['negative', 'neutral', 'positive'];
    ordered.slice(0, 3).forEach((row, idx) => {
      probs[names[idx]] = row.score;
    });
  }
  return sentimentFromProbabilities(probs);
}

/** Multi-class emotion distribution. Falls back to the offline lexicon when the
 * classifier is unavailable, so this never returns null. */
export async function classifyEmotions(
  text: string,
  timeoutMs = INFERENCE_TIMEOUT_MS
): Promise<EmotionClassification> {
  const clean = normalizeText(text);
  if (clean.length < 2) return lexiconEmotions(clean);
  const pipe = await getPipeline('emotion', timeoutMs);
  if (!pipe) return lexiconEmotions(clean);

  const result = await withTimeout(
    Promise.resolve(pipe(clean.slice(0, MAX_CLASSIFY_CHARS), { topk: 8 })),
    timeoutMs
  );
  if (!result) return lexiconEmotions(clean);
  const rows = flattenClassificationOutput(result).filter((r) => isEmotionLabel(r.label.toLowerCase()));
  if (!rows.length) return lexiconEmotions(clean);

  const all: EmotionScore[] = rows
    .map((r) => ({ label: r.label.toLowerCase() as EmotionLabel, score: r.score }))
    .sort((a, b) => b.score - a.score);
  const distressWeight = all
    .filter((e) => DISTRESS_EMOTIONS.includes(e.label))
    .reduce((sum, e) => sum + e.score, 0);

  return {
    top: all[0],
    all,
    distressWeight,
    distressDominant: DISTRESS_EMOTIONS.includes(all[0].label) && distressWeight >= 0.5,
    source: 'transformer',
  };
}

/** Embed one text. Returns null when the embedding model is not ready. */
export async function embedText(text: string, timeoutMs = INFERENCE_TIMEOUT_MS): Promise<Float32Array | null> {
  const vectors = await embedMany([text], timeoutMs);
  return vectors ? vectors[0] : null;
}

/** Embed a batch of texts (used for the question pool and the research corpus).
 * Large batches are split into small forward passes so a single call never holds
 * the inference queue for long. Returns null when the model is not ready. */
const EMBED_CHUNK = 16;
export async function embedMany(texts: string[], timeoutMs = BATCH_TIMEOUT_MS): Promise<Float32Array[] | null> {
  const cleans = texts.map((t) => normalizeText(t).slice(0, MAX_EMBED_CHARS));
  if (!cleans.length) return [];
  // If the model is still downloading, do not make the caller wait for it: every
  // caller of this function has an offline (lexical) fallback.
  const pipe = await getPipeline('embedding', isModelReady('embedding') ? Math.min(timeoutMs, 4000) : 300);
  if (!pipe) return null;

  const rows: Float32Array[] = [];
  try {
    for (let i = 0; i < cleans.length; i += EMBED_CHUNK) {
      const output = await withTimeout(
        Promise.resolve(pipe(cleans.slice(i, i + EMBED_CHUNK), { pooling: 'mean', normalize: true })),
        timeoutMs
      );
      if (!output) return null;
      const list: number[][] | number[] = typeof output.tolist === 'function' ? output.tolist() : output;
      if (!Array.isArray(list) || !list.length) return null;
      const part: number[][] = typeof list[0] === 'number' ? [list as number[]] : (list as number[][]);
      for (const row of part) rows.push(Float32Array.from(row));
    }
  } catch {
    return null;
  }
  return rows;
}

/** De-duplicates identical concurrent embedding requests (the answer text is
 * embedded by tone analysis, risk similarity, research retrieval and question
 * selection — all at about the same moment). */
const inFlightEmbeddings = new Map<string, Promise<Float32Array | null>>();

/** Cached embedding lookup: returns instantly for anything already embedded. */
export async function embedCached(text: string, timeoutMs = INFERENCE_TIMEOUT_MS): Promise<Float32Array | null> {
  const key = cacheKey(text);
  const cached = embeddingCache.get(key);
  if (cached) return cached;
  const active = inFlightEmbeddings.get(key);
  if (active) return active;
  const task = embedText(text, timeoutMs)
    .then((vector) => {
      if (vector) embeddingCache.set(key, vector);
      return vector;
    })
    .finally(() => {
      inFlightEmbeddings.delete(key);
    });
  inFlightEmbeddings.set(key, task);
  return task;
}

/** Fire-and-forget background embedding of a text list (question pool warm-up).
 * Individual failures are ignored; this never blocks the caller. */
export async function prewarmEmbeddings(texts: string[]): Promise<number> {
  const missing = texts.filter((t) => t && t.trim() && !embeddingCache.has(cacheKey(t)));
  if (!missing.length) return 0;
  const vectors = await embedMany(missing, BATCH_TIMEOUT_MS);
  if (!vectors) return 0;
  missing.forEach((text, idx) => {
    if (vectors[idx]) embeddingCache.set(cacheKey(text), vectors[idx]);
  });
  return vectors.filter(Boolean).length;
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Cosine mapped onto the engine's 0..1 relevance scale. MiniLM sentence pairs
 * sit around 0.1-0.3 when unrelated and 0.5-0.8 when genuinely related, so the
 * useful band is stretched across the full range. */
export function semanticRelevanceScore(cosine: number): number {
  return Math.max(0, Math.min(1, (cosine - 0.2) / 0.6));
}

let riskExemplarVectors: Float32Array[] | null = null;

async function ensureRiskExemplars(timeoutMs = BATCH_TIMEOUT_MS): Promise<Float32Array[] | null> {
  if (riskExemplarVectors) return riskExemplarVectors;
  const vectors = await embedMany(HIGH_RISK_EXEMPLARS, timeoutMs);
  if (!vectors) return null;
  riskExemplarVectors = vectors;
  return riskExemplarVectors;
}

/** Highest cosine similarity between the text and the high-risk exemplar bank.
 * Returns null when embeddings are unavailable so the risk engine can tell
 * "no semantic signal" apart from "semantic signal says no risk". */
export async function semanticRiskSimilarity(
  textOrVector: string | Float32Array,
  timeoutMs = INFERENCE_TIMEOUT_MS
): Promise<number | null> {
  let vector: Float32Array | null;
  if (typeof textOrVector === 'string') vector = await embedCached(textOrVector, timeoutMs);
  else vector = textOrVector;
  if (!vector) return null;

  const exemplars = await ensureRiskExemplars();
  if (!exemplars) return null;

  let best = -1;
  for (const exemplar of exemplars) {
    const cos = cosineSimilarity(vector, exemplar);
    if (cos > best) best = cos;
  }
  return best < 0 ? null : Number(best.toFixed(4));
}

/** Similarity between two texts, or null when embeddings are unavailable. */
export async function semanticRelevance(
  query: string,
  target: string,
  timeoutMs = INFERENCE_TIMEOUT_MS
): Promise<number | null> {
  const [a, b] = await Promise.all([embedCached(query, timeoutMs), embedCached(target, timeoutMs)]);
  if (!a || !b) return null;
  return semanticRelevanceScore(cosineSimilarity(a, b));
}

/**
 * Rank a set of candidate texts by semantic similarity to a query. Candidates
 * already in the embedding cache (the question pool) cost nothing. Returns
 * null when the embedding model is unavailable, so callers can fall back to
 * the TF-weighted scorer.
 */
export async function rankBySemanticSimilarity(
  query: string,
  candidates: { id: string | number; text: string }[],
  timeoutMs = BATCH_TIMEOUT_MS
): Promise<Map<string | number, number> | null> {
  const queryVector = await embedCached(query, timeoutMs);
  if (!queryVector) return null;

  const uncached = candidates.filter((c) => !embeddingCache.has(cacheKey(c.text)));
  if (uncached.length) {
    const vectors = await embedMany(
      uncached.map((c) => c.text),
      timeoutMs
    );
    if (!vectors) return null;
    uncached.forEach((c, idx) => {
      if (vectors[idx]) embeddingCache.set(cacheKey(c.text), vectors[idx]);
    });
  }

  const scores = new Map<string | number, number>();
  for (const candidate of candidates) {
    const vector = embeddingCache.get(cacheKey(candidate.text));
    if (!vector) continue;
    scores.set(candidate.id, semanticRelevanceScore(cosineSimilarity(queryVector, vector)));
  }
  return scores.size ? scores : null;
}

// ---------------------------------------------------------------------------
// Combined pass
// ---------------------------------------------------------------------------

/**
 * One semantic pass over a user answer: RoBERTa sentiment, the 7-way emotion
 * distribution, and cosine similarity to the crisis-language exemplars. Runs
 * the three models concurrently and memoizes the result.
 *
 * Returns null only when *no* model produced anything (offline / unsupported
 * browser), in which case the caller keeps using the lexicon engine.
 */
export async function analyzeTextSemantics(
  text: string,
  options?: { timeoutMs?: number; includeRiskSimilarity?: boolean }
): Promise<SemanticAnalysis | null> {
  const clean = normalizeText(text);
  if (clean.length < 2) return null;
  const key = cacheKey(clean);
  const includeRisk = options?.includeRiskSimilarity !== false;
  const timeoutMs = options?.timeoutMs ?? INFERENCE_TIMEOUT_MS;
  // A model that is still downloading must not stall the caller: give it a moment,
  // fall back to the lexicon for this answer, and stay warm for the next one.
  const waitFor = (model: SemanticModelKey) => (isModelReady(model) ? timeoutMs : 250);

  const cached = semanticsCache.get(key);
  if (cached) {
    if (!includeRisk || cached.riskSimilarity !== null) return cached;
    // The cached reading came from a pass that skipped crisis-language similarity
    // (e.g. the live tone chip while typing). Add only that part — a single cached
    // embedding — instead of re-running both classifiers.
    const riskSimilarity = await semanticRiskSimilarity(clean, waitFor('embedding'));
    if (riskSimilarity === null) return cached;
    const upgraded: SemanticAnalysis = {
      ...cached,
      riskSimilarity,
      degraded: cached.sentiment.source !== 'transformer' || cached.emotions.source === 'lexicon',
    };
    semanticsCache.set(key, upgraded);
    return upgraded;
  }

  const flightKey = `${key}|${includeRisk ? 1 : 0}`;
  const active = inFlightSemantics.get(flightKey);
  if (active) return active;

  if (engineState.status === 'unavailable') return null;

  const task = (async (): Promise<SemanticAnalysis | null> => {
    const [sentiment, emotions, riskSimilarity] = await Promise.all([
      classifySentiment(clean, waitFor('sentiment')),
      classifyEmotions(clean, waitFor('emotion')),
      includeRisk ? semanticRiskSimilarity(clean, waitFor('embedding')) : Promise.resolve(null),
    ]);

    if (!sentiment && emotions.source === 'lexicon' && riskSimilarity === null && !isSemanticReady()) {
      return null;
    }

    const analysis: SemanticAnalysis = {
      text: clean,
      sentiment: sentiment ?? {
        // Fallback so the object is always well formed; adaptiveEngine only
        // uses this branch when it already knows the model was unavailable.
        score: 0,
        label: 'neutral',
        magnitude: 0,
        source: 'lexicon',
      },
      emotions,
      riskSimilarity,
      degraded: !sentiment || emotions.source === 'lexicon' || riskSimilarity === null,
    };
    // Only remember a reading the real models produced. A fallback made while a
    // model was still downloading must not stick to this text forever.
    if (sentiment && emotions.source === 'transformer') semanticsCache.set(key, analysis);
    return analysis;
  })().finally(() => {
    inFlightSemantics.delete(flightKey);
  });

  inFlightSemantics.set(flightKey, task);
  return task;
}
