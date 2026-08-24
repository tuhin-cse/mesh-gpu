/**
 * webllm-worker.ts
 *
 * Web Worker host for the WebLLM engine.
 *
 * Model loading, shader compilation and token generation all block whatever
 * thread they run on. Keeping them here means a machine can contribute to the
 * mesh while its owner keeps working — which is the whole premise of borrowing
 * idle office GPUs. Running the engine on the main thread instead makes the
 * lender's browser stutter, and a stuttering browser gets its tab closed.
 */

import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (event: MessageEvent): void => {
  handler.onmessage(event);
};
