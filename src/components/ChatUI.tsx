/**
 * ChatUI.tsx
 *
 * Local inference card: loads a quantized model with @mlc-ai/web-llm and
 * streams a conversation on this tab's own GPU. This is the single-node
 * baseline; the P2P layer-sharded pipeline supersedes it.
 */

import { useEffect, useRef, useState } from 'react';

import { WEBLLM_MODELS, WebLLMRuntime, isModelCached } from '../engine/web-llm';
import type { ChatMessage, WebLLMProgress, WebLLMStatus } from '../engine/web-llm';

export function ChatUI() {
  const [runtime] = useState(() => new WebLLMRuntime());
  const [modelId, setModelId] = useState(WEBLLM_MODELS[0].id);
  const [status, setStatus] = useState<WebLLMStatus>('idle');
  const [progress, setProgress] = useState<WebLLMProgress | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [cached, setCached] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  const ready = status === 'ready';

  useEffect(() => {
    return () => {
      void runtime.dispose();
    };
  }, [runtime]);

  useEffect(() => {
    let cancelled = false;
    isModelCached(modelId).then((value) => {
      if (!cancelled) setCached(value);
    });
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const load = async (): Promise<void> => {
    setStatus('loading');
    setProgress(null);
    try {
      await runtime.load(modelId, setProgress);
      setStatus('ready');
    } catch {
      setStatus('error');
    }
  };

  const send = async (): Promise<void> => {
    const text = input.trim();
    if (!text || busy || !ready) return;

    const history: ChatMessage[] = [...messages, { role: 'user', content: text }];
    setMessages([...history, { role: 'assistant', content: '' }]);
    setInput('');
    setBusy(true);

    try {
      await runtime.generate(history, {
        onToken: (delta) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { role: 'assistant', content: last.content + delta };
            return next;
          });
        },
      });
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: 'assistant',
          content: `[error] ${err instanceof Error ? err.message : String(err)}`,
        };
        return next;
      });
    } finally {
      setBusy(false);
    }
  };

  const model = WEBLLM_MODELS.find((entry) => entry.id === modelId) ?? WEBLLM_MODELS[0];

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
      <h2 className="text-sm font-medium text-zinc-300">Local Inference (WebLLM)</h2>
      <p className="mt-1 text-xs text-zinc-500">
        Run a quantized model end-to-end on this tab&apos;s GPU — the single-node baseline for the
        P2P pipeline.
      </p>

      <div className="mt-3 flex flex-wrap items-end gap-4">
        <label className="block text-sm">
          <span className="text-zinc-400">Model</span>
          <select
            value={modelId}
            disabled={status === 'loading' || busy}
            onChange={(event) => setModelId(event.target.value)}
            className="mt-1 w-56 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500 disabled:opacity-50"
          >
            {WEBLLM_MODELS.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={load}
          disabled={status === 'loading'}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50"
        >
          {status === 'loading' ? 'Loading…' : ready ? 'Reload' : 'Load model'}
        </button>

        <span className="text-xs text-zinc-600">
          {model.approxSize}
          {cached ? ' · cached ✓' : ''}
        </span>
      </div>

      {status === 'loading' && progress ? (
        <div className="mt-3">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full bg-emerald-500 transition-all"
              style={{ width: `${Math.round(progress.progress * 100)}%` }}
            />
          </div>
          <p className="mt-1 truncate text-xs text-zinc-500">{progress.text}</p>
        </div>
      ) : null}

      {status === 'error' ? (
        <p className="mt-3 text-xs text-rose-400">
          {runtime.currentError ?? 'Failed to load model'}
        </p>
      ) : null}

      <div className="mt-4 flex h-72 flex-col gap-2 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
        {messages.length === 0 ? (
          <p className="text-xs text-zinc-600">
            Load a model, then start chatting. The first run downloads weights and compiles shaders,
            so it is the slowest.
          </p>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                message.role === 'user'
                  ? 'self-end bg-emerald-900/60 text-emerald-100'
                  : 'self-start bg-zinc-800 text-zinc-200'
              }`}
            >
              <p className="whitespace-pre-wrap break-words">{message.content || '…'}</p>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>

      <div className="mt-3 flex gap-2">
        <input
          value={input}
          disabled={!ready || busy}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void send();
          }}
          placeholder={ready ? 'Ask something…' : 'Load a model to chat'}
          className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500 disabled:opacity-50"
        />
        <button
          onClick={send}
          disabled={!ready || busy || input.trim().length === 0}
          className="rounded-lg border border-emerald-700 px-4 py-2 text-sm text-emerald-300 transition hover:bg-emerald-950 disabled:opacity-50"
        >
          {busy ? '…' : 'Send'}
        </button>
      </div>
    </section>
  );
}
