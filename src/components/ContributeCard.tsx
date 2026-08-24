/**
 * ContributeCard.tsx
 *
 * "Pool" mode: lend this tab's GPU to a MeshGPU coordinator so colleagues can
 * use it through the coordinator's OpenAI-compatible endpoint.
 *
 * Every contributing tab holds a complete model and serves one request at a
 * time. Throughput scales with the number of people who leave the tab open,
 * and a tab closing costs at most one retry — unlike layer sharding, which
 * needs every stage present for any request to complete.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { MeshWorker } from '../engine/mesh-worker';
import type { MeshStatus } from '../engine/mesh-worker';
import { IdlePolicy } from '../engine/idle-policy';
import type { IdleState } from '../engine/idle-policy';
import { WEBLLM_MODELS, WebLLMRuntime } from '../engine/web-llm';

/** Remember the coordinator across reloads so rejoining is one click. */
const STORE_KEY = 'meshgpu.contribute';

interface StoredSettings {
  coordinatorUrl: string;
  token: string;
  label: string;
  modelId: string;
  pauseOnBattery: boolean;
  pauseWhenActive: boolean;
}

function defaultCoordinatorUrl(): string {
  // Served by the coordinator itself? Then it is this origin. During
  // `npm run dev` the page is on :5173, so point at the usual coordinator port.
  if (typeof window === 'undefined') return '';
  const { origin, port } = window.location;
  return port === '5173' || port === '3000' ? 'http://localhost:8080' : origin;
}

function loadSettings(): StoredSettings {
  const fallback: StoredSettings = {
    coordinatorUrl: defaultCoordinatorUrl(),
    token: '',
    label: 'my browser',
    modelId: WEBLLM_MODELS[0].id,
    pauseOnBattery: true,
    pauseWhenActive: false,
  };

  try {
    const stored = JSON.parse(window.localStorage.getItem(STORE_KEY) ?? '{}') as Partial<StoredSettings>;
    const merged = { ...fallback, ...stored };
    // A token in the join link always wins — that is what the link is for.
    const fromUrl = new URLSearchParams(window.location.search).get('token');
    if (fromUrl) merged.token = fromUrl;
    return merged;
  } catch {
    return fallback;
  }
}

export function ContributeCard() {
  const [settings, setSettings] = useState<StoredSettings>(loadSettings);
  const [status, setStatus] = useState<MeshStatus>('offline');
  const [statusDetail, setStatusDetail] = useState<string>('');
  const [contributing, setContributing] = useState(false);
  const [idle, setIdle] = useState<IdleState>({
    paused: true,
    reason: 'contributing is off',
    onBattery: false,
    userActive: false,
  });

  const [modelState, setModelState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadText, setLoadText] = useState('');
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [served, setServed] = useState(0);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const runtimeRef = useRef<WebLLMRuntime | null>(null);
  const meshRef = useRef<MeshWorker | null>(null);
  const policyRef = useRef<IdlePolicy | null>(null);
  const idleRef = useRef(idle);
  const loadedRef = useRef<string | null>(null);

  idleRef.current = idle;
  loadedRef.current = loadedModelId;

  const pushLog = useCallback((line: string) => {
    setLog((prev) => [...prev.slice(-49), `${new Date().toLocaleTimeString()} ${line}`]);
  }, []);

  // Persist settings so a reload does not mean re-entering the key.
  useEffect(() => {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(settings));
    } catch {
      // Private browsing — the mesh still works, it just will not remember.
    }
  }, [settings]);

  // One idle policy for the lifetime of the card.
  useEffect(() => {
    const policy = new IdlePolicy({
      enabled: false,
      pauseOnBattery: settings.pauseOnBattery,
      pauseWhenActive: settings.pauseWhenActive,
    });
    policyRef.current = policy;
    const unsubscribe = policy.subscribe(setIdle);
    return () => {
      unsubscribe();
      policy.dispose();
      policyRef.current = null;
    };
    // Intentionally mount-only: preference changes are pushed via update().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    policyRef.current?.update({
      enabled: contributing,
      pauseOnBattery: settings.pauseOnBattery,
      pauseWhenActive: settings.pauseWhenActive,
    });
  }, [contributing, settings.pauseOnBattery, settings.pauseWhenActive]);

  // Any change to readiness or pause state is news the coordinator needs.
  useEffect(() => {
    meshRef.current?.announce();
  }, [idle.paused, loadedModelId]);

  const loadModel = useCallback(async () => {
    const runtime = runtimeRef.current ?? new WebLLMRuntime();
    runtimeRef.current = runtime;

    setModelState('loading');
    setLoadProgress(0);
    setLoadedModelId(null);
    pushLog(`loading ${settings.modelId} on a worker thread…`);

    try {
      await runtime.load(settings.modelId, (progress) => {
        setLoadProgress(progress.progress);
        setLoadText(progress.text);
      });
      setModelState('ready');
      setLoadedModelId(settings.modelId);
      pushLog(`${settings.modelId} ready`);
    } catch (err) {
      setModelState('error');
      const message = err instanceof Error ? err.message : String(err);
      setLoadText(message);
      pushLog(`load failed: ${message}`);
    }
  }, [settings.modelId, pushLog]);

  const connect = useCallback(() => {
    if (settings.token.trim().length === 0) {
      setStatus('error');
      setStatusDetail('enter the API key printed by the coordinator');
      return;
    }

    meshRef.current?.stop();
    const mesh = new MeshWorker({
      coordinatorUrl: settings.coordinatorUrl.trim() || undefined,
      token: settings.token.trim(),
      label: settings.label.trim() || 'browser',
      getEngine: () => runtimeRef.current?.currentEngine ?? null,
      getModelId: () => loadedRef.current,
      isPaused: () => idleRef.current.paused,
      callbacks: {
        onStatus: (next, detail) => {
          setStatus(next);
          setStatusDetail(detail ?? '');
        },
        onJobStart: (jobId) => {
          setActiveJob(jobId);
          pushLog(`serving ${jobId}`);
        },
        onJobEnd: (jobId, outcome) => {
          setActiveJob(null);
          if (outcome === 'done') setServed((count) => count + 1);
          pushLog(`${jobId} ${outcome}`);
        },
        onLog: pushLog,
      },
    });
    meshRef.current = mesh;
    mesh.start();
    setContributing(true);
  }, [settings.coordinatorUrl, settings.token, settings.label, pushLog]);

  const disconnect = useCallback(() => {
    meshRef.current?.stop();
    meshRef.current = null;
    setContributing(false);
    setActiveJob(null);
    pushLog('left the mesh');
  }, [pushLog]);

  useEffect(
    () => () => {
      meshRef.current?.stop();
      void runtimeRef.current?.dispose();
    },
    [],
  );

  const ready = modelState === 'ready' && loadedModelId !== null;
  const serving = contributing && status === 'connected' && ready && !idle.paused;

  const stateLine = useMemo(() => {
    if (!contributing) return 'Not contributing';
    if (status === 'connecting') return 'Connecting to the coordinator…';
    if (status === 'error') return statusDetail || 'Connection problem';
    if (status === 'offline') return statusDetail ? `Offline — ${statusDetail}` : 'Offline';
    if (!ready) return 'Connected — load a model to start serving';
    if (idle.paused) return `Connected — holding off (${idle.reason})`;
    if (activeJob) return 'Serving a request';
    return 'Connected and waiting for work';
  }, [contributing, status, statusDetail, ready, idle.paused, idle.reason, activeJob]);

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium text-zinc-300">Contribute to a Mesh (Pool mode)</h2>
        <span
          className={`inline-flex items-center gap-2 whitespace-nowrap rounded-full border px-3 py-1 text-xs font-medium ${
            serving
              ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
              : contributing
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                : 'border-zinc-700 bg-zinc-800/50 text-zinc-400'
          }`}
        >
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              serving ? 'bg-emerald-400' : contributing ? 'bg-amber-400' : 'bg-zinc-600'
            }`}
          />
          {stateLine}
        </span>
      </div>

      <p className="mb-4 text-xs leading-relaxed text-zinc-500">
        Your GPU runs a whole model and answers requests other people send to the coordinator's
        OpenAI-compatible endpoint. The model runs on a worker thread, so this tab stays
        responsive. Nothing leaves your local network.
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Coordinator URL">
          <input
            type="url"
            value={settings.coordinatorUrl}
            onChange={(event) =>
              setSettings((prev) => ({ ...prev, coordinatorUrl: event.target.value }))
            }
            placeholder="http://mesh.local:8080"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500"
          />
        </Field>
        <Field label="API key">
          <input
            type="password"
            value={settings.token}
            onChange={(event) => setSettings((prev) => ({ ...prev, token: event.target.value }))}
            placeholder="printed when the coordinator starts"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500"
          />
        </Field>
        <Field label="Show up as">
          <input
            type="text"
            value={settings.label}
            onChange={(event) => setSettings((prev) => ({ ...prev, label: event.target.value }))}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500"
          />
        </Field>
        <Field label="Model to serve">
          <select
            value={settings.modelId}
            onChange={(event) => setSettings((prev) => ({ ...prev, modelId: event.target.value }))}
            disabled={modelState === 'loading'}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500 disabled:opacity-50"
          >
            {WEBLLM_MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} · {model.approxSize}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          onClick={loadModel}
          disabled={modelState === 'loading' || loadedModelId === settings.modelId}
          className="rounded-lg border border-emerald-700 px-4 py-2 text-sm text-emerald-300 transition hover:bg-emerald-950 disabled:opacity-50"
        >
          {modelState === 'loading'
            ? `Loading… ${(loadProgress * 100).toFixed(0)}%`
            : loadedModelId === settings.modelId
              ? 'Model ready'
              : 'Load model'}
        </button>

        {contributing ? (
          <button
            onClick={disconnect}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 transition hover:bg-zinc-800"
          >
            Stop contributing
          </button>
        ) : (
          <button
            onClick={connect}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500"
          >
            Start contributing
          </button>
        )}

        <label className="flex items-center gap-2 text-xs text-zinc-400">
          <input
            type="checkbox"
            checked={settings.pauseOnBattery}
            onChange={(event) =>
              setSettings((prev) => ({ ...prev, pauseOnBattery: event.target.checked }))
            }
            className="accent-emerald-500"
          />
          Pause on battery
        </label>

        <label className="flex items-center gap-2 text-xs text-zinc-400">
          <input
            type="checkbox"
            checked={settings.pauseWhenActive}
            onChange={(event) =>
              setSettings((prev) => ({ ...prev, pauseWhenActive: event.target.checked }))
            }
            className="accent-emerald-500"
          />
          Pause while I'm using this device
        </label>
      </div>

      {modelState === 'loading' && (
        <div className="mt-3">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-500 transition-[width]"
              style={{ width: `${Math.max(2, loadProgress * 100)}%` }}
            />
          </div>
          <p className="mt-1.5 truncate text-xs text-zinc-500">{loadText}</p>
        </div>
      )}

      {modelState === 'error' && (
        <p className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/5 px-3 py-2 text-xs text-rose-300">
          {loadText}
        </p>
      )}

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
        <Stat label="Requests served" value={String(served)} />
        <Stat label="Serving now" value={activeJob ? 'yes' : 'no'} />
        <Stat label="Power" value={idle.onBattery ? 'battery' : 'mains'} />
      </div>

      <pre className="mt-3 max-h-32 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-500">
        {log.length > 0 ? log.join('\n') : '// not connected'}
      </pre>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-zinc-500">{label}</span>
      {children}
    </label>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="mt-1 text-lg font-medium text-zinc-100">{value}</div>
    </div>
  );
}
