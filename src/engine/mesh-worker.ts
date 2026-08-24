/**
 * mesh-worker.ts
 *
 * Connects this browser tab to a MeshGPU coordinator and runs the chat jobs it
 * sends. This is the "Pool" path: every contributing tab holds a whole model
 * and serves one request at a time, so throughput scales with the number of
 * people who leave the tab open, and a tab closing costs at most one retry.
 *
 * Unlike the sharded pipeline, nothing here is peer-to-peer — prompts travel
 * client -> coordinator -> this tab, all inside the LAN.
 */

import type { MLCEngineInterface } from '@mlc-ai/web-llm';

import type { ChatMessage } from './web-llm';

export type MeshStatus = 'offline' | 'connecting' | 'connected' | 'error';

export interface MeshJobPayload {
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  seed?: number;
}

interface JobMessage {
  type: 'job';
  jobId: string;
  model: string;
  payload: MeshJobPayload;
}

type Inbound =
  | { type: 'welcome'; workerId: string; heartbeatMs: number }
  | JobMessage
  | { type: 'cancel'; jobId: string };

export interface MeshWorkerCallbacks {
  onStatus?: (status: MeshStatus, detail?: string) => void;
  onJobStart?: (jobId: string) => void;
  onJobEnd?: (jobId: string, outcome: 'done' | 'error' | 'cancelled') => void;
  onLog?: (line: string) => void;
}

export interface MeshWorkerOptions {
  /** Coordinator base URL. Defaults to the origin serving this page. */
  coordinatorUrl?: string;
  token: string;
  /** Name shown in the coordinator's worker list. */
  label: string;
  /** Supplies the loaded engine, or null when no model is ready. */
  getEngine: () => MLCEngineInterface | null;
  /** Model id this tab currently serves, or null. */
  getModelId: () => string | null;
  /** Whether the idle policy currently permits work. */
  isPaused: () => boolean;
  callbacks?: MeshWorkerCallbacks;
}

const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

/** Build the `/mesh` WebSocket URL from an http(s) coordinator origin. */
export function meshSocketUrl(base: string, token: string, label: string): string {
  const url = new URL('/mesh', base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('token', token);
  url.searchParams.set('label', label);
  return url.toString();
}

export class MeshWorker {
  private readonly options: MeshWorkerOptions;
  private readonly callbacks: MeshWorkerCallbacks;

  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private stopped = false;
  private activeJobId: string | null = null;
  private cancelledJobs = new Set<string>();

  constructor(options: MeshWorkerOptions) {
    this.options = options;
    this.callbacks = options.callbacks ?? {};
  }

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  get busy(): boolean {
    return this.activeJobId !== null;
  }

  start(): void {
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    this.clearReconnect();
    this.socket?.close(1000, 'worker stopped');
    this.socket = null;
    this.callbacks.onStatus?.('offline');
  }

  /**
   * Tell the coordinator what this tab can serve right now. Safe to call on
   * every model change and every idle-policy transition.
   */
  announce(): void {
    if (!this.connected) return;
    this.send({
      type: 'status',
      model: this.options.getModelId(),
      paused: this.options.isPaused() || this.options.getEngine() === null,
    });
  }

  private open(): void {
    if (this.stopped) return;
    this.clearReconnect();

    const base = this.options.coordinatorUrl ?? window.location.origin;
    let url: string;
    try {
      url = meshSocketUrl(base, this.options.token, this.options.label);
    } catch {
      this.callbacks.onStatus?.('error', `"${base}" is not a valid coordinator URL`);
      return;
    }

    this.callbacks.onStatus?.('connecting');
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.callbacks.onStatus?.('connected');
      this.announce();
    };

    socket.onmessage = (event) => {
      let message: Inbound;
      try {
        message = JSON.parse(String(event.data)) as Inbound;
      } catch {
        return;
      }
      void this.handle(message);
    };

    socket.onerror = () => {
      this.callbacks.onStatus?.('error', 'could not reach the coordinator');
    };

    socket.onclose = (event) => {
      this.socket = null;
      this.activeJobId = null;
      if (this.stopped) return;

      // 4401 is our own "bad token" code — retrying cannot fix it.
      if (event.code === 4401) {
        this.callbacks.onStatus?.('error', 'the coordinator rejected this key');
        return;
      }
      this.callbacks.onStatus?.('offline', event.reason || 'connection closed');
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer !== null) return;
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.reconnectAttempts, RECONNECT_MAX_MS);
    this.reconnectAttempts += 1;
    this.callbacks.onLog?.(`reconnecting in ${Math.round(delay / 1000)}s`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, delay);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private send(message: object): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  private async handle(message: Inbound): Promise<void> {
    switch (message.type) {
      case 'welcome':
        this.callbacks.onLog?.(`joined the mesh as ${message.workerId}`);
        break;
      case 'cancel':
        this.cancelledJobs.add(message.jobId);
        if (this.activeJobId === message.jobId) {
          this.options.getEngine()?.interruptGenerate();
        }
        break;
      case 'job':
        await this.runJob(message);
        break;
      default:
        break;
    }
  }

  private async runJob(job: JobMessage): Promise<void> {
    const engine = this.options.getEngine();
    const modelId = this.options.getModelId();

    if (!engine || modelId !== job.model) {
      this.send({
        type: 'error',
        jobId: job.jobId,
        message: `this worker no longer has "${job.model}" loaded`,
      });
      this.announce();
      return;
    }

    this.activeJobId = job.jobId;
    this.callbacks.onJobStart?.(job.jobId);

    try {
      const stream = await engine.chat.completions.create({
        messages: job.payload.messages,
        stream: true,
        temperature: job.payload.temperature,
        top_p: job.payload.top_p,
        max_tokens: job.payload.max_tokens ?? 512,
        stop: job.payload.stop,
        seed: job.payload.seed,
      });

      for await (const chunk of stream) {
        if (this.cancelledJobs.has(job.jobId)) break;
        const delta = chunk.choices[0]?.delta?.content ?? '';
        if (delta.length > 0) this.send({ type: 'chunk', jobId: job.jobId, delta });
      }

      if (this.cancelledJobs.has(job.jobId)) {
        this.cancelledJobs.delete(job.jobId);
        this.callbacks.onJobEnd?.(job.jobId, 'cancelled');
      } else {
        this.send({ type: 'done', jobId: job.jobId });
        this.callbacks.onJobEnd?.(job.jobId, 'done');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.send({ type: 'error', jobId: job.jobId, message });
      this.callbacks.onJobEnd?.(job.jobId, 'error');
    } finally {
      this.activeJobId = null;
      // Conversation state must not leak between requests from different
      // people — every mesh job is its own independent conversation.
      await engine.resetChat().catch(() => undefined);
      this.announce();
    }
  }
}
