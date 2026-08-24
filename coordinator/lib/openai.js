/**
 * openai.js
 *
 * Just enough of the OpenAI chat-completions contract that existing tools —
 * Cursor, Continue, the openai SDK, curl — can point at the mesh unmodified.
 * Anything a browser worker cannot honour is rejected explicitly rather than
 * accepted and ignored.
 */

import crypto from 'node:crypto';

/** Parameters forwarded to the worker; everything else is rejected. */
const SUPPORTED_PARAMS = ['temperature', 'top_p', 'max_tokens', 'stop', 'seed'];

/** Accepted but ignored, because they cannot change a single-stream result. */
const IGNORED_PARAMS = ['user', 'stream_options', 'n'];

export function newId(prefix) {
  return `${prefix}-${crypto.randomBytes(12).toString('hex')}`;
}

/**
 * Validate an incoming request body.
 *
 * @param {any} body
 * @returns {{ model: string, messages: object[], stream: boolean, params: object }}
 */
export function parseChatRequest(body) {
  if (typeof body !== 'object' || body === null) {
    throw badRequest('request body must be a JSON object');
  }
  if (typeof body.model !== 'string' || body.model.length === 0) {
    throw badRequest('"model" is required');
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw badRequest('"messages" must be a non-empty array');
  }

  const messages = body.messages.map((message, index) => {
    if (typeof message !== 'object' || message === null) {
      throw badRequest(`messages[${index}] must be an object`);
    }
    const { role, content } = message;
    if (role !== 'system' && role !== 'user' && role !== 'assistant') {
      throw badRequest(`messages[${index}].role must be system, user or assistant`);
    }
    if (typeof content !== 'string') {
      // Vision/tool content blocks would need a worker that can run them.
      throw badRequest(`messages[${index}].content must be a string`);
    }
    return { role, content };
  });

  if (body.n !== undefined && body.n !== 1) {
    throw badRequest('"n" other than 1 is not supported — the mesh streams one completion');
  }
  if (body.tools !== undefined || body.functions !== undefined) {
    throw badRequest('tool calling is not supported yet');
  }

  const params = {};
  for (const key of SUPPORTED_PARAMS) {
    if (body[key] !== undefined) params[key] = body[key];
  }
  for (const key of Object.keys(body)) {
    if (key === 'model' || key === 'messages' || key === 'stream') continue;
    if (SUPPORTED_PARAMS.includes(key) || IGNORED_PARAMS.includes(key)) continue;
    throw badRequest(`unsupported parameter "${key}"`);
  }

  return { model: body.model, messages, stream: body.stream === true, params };
}

/** One SSE frame of an in-progress completion. */
export function streamChunk(id, model, delta, created) {
  return {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices: [{ index: 0, delta: delta === null ? {} : { content: delta }, finish_reason: null }],
  };
}

/** The terminal SSE frame, carrying the finish reason. */
export function streamDone(id, model, created, finishReason = 'stop') {
  return {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  };
}

/** The single-shot (non-streaming) response body. */
export function completionBody(id, model, content, created, finishReason = 'stop') {
  return {
    id,
    object: 'chat.completion',
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: finishReason,
      },
    ],
    // Token accounting lives in the browser worker; the coordinator sees text
    // only, so reporting counts here would be invented.
    usage: null,
  };
}

/** OpenAI-shaped error envelope, which is what client SDKs parse. */
export function errorBody(message, type = 'invalid_request_error', code = null) {
  return { error: { message, type, code, param: null } };
}

export function badRequest(message) {
  return Object.assign(new Error(message), { status: 400, code: 'invalid_request_error' });
}
