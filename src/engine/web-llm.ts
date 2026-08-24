/**
 * web-llm.ts
 *
 * Reference single-node inference runtime backed by @mlc-ai/web-llm. One
 * browser tab downloads a quantized model and runs it end-to-end on its local
 * WebGPU device via `MLCEngine`.
 *
 * Architecture note: WebLLM's public API runs a whole model through
 * `engine.chat.completions.create()` and does not expose per-layer execution,
 * so true layer-sharding across peers requires forking WebLLM / tvmjs
 * internals (out of MVP scope). This module validates the full WebGPU
 * inference path on one node — the baseline the sharded pipeline will replace.
 */

import type {
  ChatCompletionMessageParam,
  InitProgressReport,
  MLCEngine,
} from '@mlc-ai/web-llm';

export type WebLLMStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface WebLLMModel {
  /** web-llm model id (must be in prebuiltAppConfig or a custom appConfig). */
  id: string;
  name: string;
  approxSize: string;
}

export interface WebLLMProgress {
  /** 0..1 fraction of model initialization complete. */
  progress: number;
  text: string;
  timeElapsed: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export const WEBLLM_MODELS: readonly WebLLMModel[] = [
  { id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', name: 'Qwen2.5-1.5B (q4f16)', approxSize: '≈ 1.0 GB' },
  { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', name: 'Qwen2.5-3B (q4f16)', approxSize: '≈ 2.1 GB' },
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', name: 'Llama-3.2-1B (q4f16)', approxSize: '≈ 0.8 GB' },
];

/** Lazily fetch the (large) web-llm runtime so it stays out of the main bundle. */
function loadWebLLM(): Promise<typeof import('@mlc-ai/web-llm')> {
  return import('@mlc-ai/web-llm');
}

export class WebLLMRuntime {
  private engine: MLCEngine | null = null;
  private currentModelId: string | null = null;
  private status: WebLLMStatus = 'idle';
  private lastError: string | null = null;

  get isReady(): boolean {
    return this.status === 'ready' && this.engine !== null;
  }

  get currentStatus(): WebLLMStatus {
    return this.status;
  }

  get currentError(): string | null {
    return this.lastError;
  }

  get modelId(): string | null {
    return this.currentModelId;
  }

  /** Load a model (no-op if it is already loaded). Throws on failure. */
  async load(modelId: string, onProgress?: (progress: WebLLMProgress) => void): Promise<void> {
    if (this.isReady && this.currentModelId === modelId) return;
    await this.disposeEngine();

    this.status = 'loading';
    this.lastError = null;
    this.currentModelId = modelId;

    try {
      const webllm = await loadWebLLM();
      this.engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (report: InitProgressReport) => {
          onProgress?.({
            progress: report.progress,
            text: report.text,
            timeElapsed: report.timeElapsed,
          });
        },
      });
      this.status = 'ready';
    } catch (err) {
      this.status = 'error';
      this.lastError = err instanceof Error ? err.message : String(err);
      this.currentModelId = null;
      throw err;
    }
  }

  /**
   * Stream a chat completion. `onToken` is invoked with each text delta as it
   * is produced. Resolves with the full reply.
   */
  async generate(
    messages: readonly ChatMessage[],
    options: { onToken?: (delta: string) => void; maxTokens?: number } = {},
  ): Promise<string> {
    if (!this.engine) throw new Error('no model loaded');

    const params: ChatCompletionMessageParam[] = messages.map((message) => {
      if (message.role === 'system') return { role: 'system', content: message.content };
      if (message.role === 'assistant') return { role: 'assistant', content: message.content };
      return { role: 'user', content: message.content };
    });

    const stream = await this.engine.chat.completions.create({
      messages: params,
      stream: true,
      max_tokens: options.maxTokens ?? 256,
    });

    let full = '';
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      if (delta.length > 0) {
        full += delta;
        options.onToken?.(delta);
      }
    }
    return full;
  }

  /** Maximum storage-buffer size the loaded engine can allocate (bytes). */
  async getMaxStorageBufferBindingSize(): Promise<number | null> {
    return this.engine ? this.engine.getMaxStorageBufferBindingSize() : null;
  }

  async getGPUVendor(): Promise<string | null> {
    return this.engine ? this.engine.getGPUVendor() : null;
  }

  async dispose(): Promise<void> {
    await this.disposeEngine();
    this.status = 'idle';
    this.currentModelId = null;
    this.lastError = null;
  }

  private async disposeEngine(): Promise<void> {
    if (!this.engine) return;
    try {
      // `unload()` releases the GPU buffers holding the weights. `resetChat()`
      // only clears conversation state, so relying on it leaks the whole model
      // every time the user switches to a different one.
      await this.engine.unload();
    } catch {
      // Best-effort teardown — the engine is dropped regardless.
    }
    this.engine = null;
  }
}

/** Whether a model's weights are already downloaded in this browser. */
export async function isModelCached(modelId: string): Promise<boolean> {
  const webllm = await loadWebLLM();
  return webllm.hasModelInCache(modelId);
}
