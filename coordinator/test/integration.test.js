/**
 * End-to-end coverage of the coordinator: a real HTTP server, a real
 * WebSocket worker, and real OpenAI-shaped requests over the wire.
 *
 * The worker here is a stand-in that echoes canned tokens instead of running
 * WebLLM, which keeps the test honest about the routing, auth, streaming and
 * failover paths while needing no GPU and no model download.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import WebSocket from 'ws';

const PORT = 8123;
const TOKEN = 'test-token-value';
const BASE = `http://127.0.0.1:${PORT}`;
const MODEL = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';

let server;

/** A browser tab, minus the browser. */
class FakeWorker {
  constructor({ model = MODEL, reply = 'Hello from the mesh', dieOnJob = false } = {}) {
    this.model = model;
    this.reply = reply;
    this.dieOnJob = dieOnJob;
    this.jobsSeen = [];
    this.socket = new WebSocket(`ws://127.0.0.1:${PORT}/mesh?token=${TOKEN}&label=fake`);
  }

  ready() {
    return new Promise((resolve, reject) => {
      this.socket.on('error', reject);
      this.socket.on('close', (code) => reject(new Error(`worker socket closed: ${code}`)));
      this.socket.on('message', (raw) => {
        const message = JSON.parse(String(raw));

        if (message.type === 'welcome') {
          this.socket.send(JSON.stringify({ type: 'status', model: this.model, paused: false }));
          // Give the coordinator a tick to record the status before proceeding.
          setTimeout(resolve, 20);
          return;
        }

        if (message.type === 'job') {
          this.jobsSeen.push(message);
          if (this.dieOnJob) {
            this.socket.terminate();
            return;
          }
          for (const word of this.reply.split(' ')) {
            this.socket.send(JSON.stringify({ type: 'chunk', jobId: message.jobId, delta: `${word} ` }));
          }
          this.socket.send(JSON.stringify({ type: 'done', jobId: message.jobId }));
        }
      });
    });
  }

  close() {
    if (this.socket.readyState === WebSocket.OPEN) this.socket.close();
  }
}

function post(path, body, token = TOKEN) {
  return fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
}

/** Collect the text deltas out of an SSE chat-completions stream. */
async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let sawDone = false;
  let finishReason = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      const chunk = JSON.parse(payload);
      text += chunk.choices[0]?.delta?.content ?? '';
      if (chunk.choices[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
    }
  }

  return { text, sawDone, finishReason };
}

beforeAll(async () => {
  process.env.MESH_PORT = String(PORT);
  process.env.MESH_TOKEN = TOKEN;
  process.env.MESH_MDNS = 'off';
  process.env.MESH_JOB_TIMEOUT_MS = '5000';

  ({ server } = await import('../server.js'));
  if (!server.listening) {
    await new Promise((resolve) => server.once('listening', resolve));
  }
});

afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
});

describe('coordinator HTTP surface', () => {
  it('answers /healthz without a key', async () => {
    const response = await fetch(`${BASE}/healthz`);
    expect(response.status).toBe(200);
    expect((await response.json()).ok).toBe(true);
  });

  it('rejects /v1 requests with no key, a wrong key, or a truncated key', async () => {
    expect((await post('/v1/chat/completions', {}, null)).status).toBe(401);
    expect((await post('/v1/chat/completions', {}, 'wrong')).status).toBe(401);
    expect((await post('/v1/chat/completions', {}, TOKEN.slice(0, -1))).status).toBe(401);
  });

  it('returns an OpenAI-shaped error for a malformed body', async () => {
    const response = await post('/v1/chat/completions', { messages: [] });
    expect(response.status).toBe(400);
    expect((await response.json()).error.message).toMatch(/"model" is required/);
  });

  it('reports 503 when no worker has the model', async () => {
    const response = await post('/v1/chat/completions', {
      model: 'nothing-loaded',
      messages: [{ role: 'user', content: 'hi' }],
    });
    expect(response.status).toBe(503);
    expect((await response.json()).error.message).toMatch(/no worker/);
  });

  it('lists no models while the mesh is empty', async () => {
    const response = await fetch(`${BASE}/v1/models`, {
      headers: { authorization: `Bearer ${TOKEN}` },
    });
    expect((await response.json()).data).toEqual([]);
  });
});

describe('routing a request to a browser worker', () => {
  let worker;

  beforeAll(async () => {
    worker = new FakeWorker();
    await worker.ready();
  });

  afterAll(() => worker.close());

  it('advertises the worker’s model through /v1/models', async () => {
    const response = await fetch(`${BASE}/v1/models`, {
      headers: { authorization: `Bearer ${TOKEN}` },
    });
    const body = await response.json();
    expect(body.data.map((entry) => entry.id)).toEqual([MODEL]);
    expect(body.data[0].owned_by).toBe('meshgpu');
  });

  it('completes a non-streaming request end to end', async () => {
    const response = await post('/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'say hello' }],
    });

    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.object).toBe('chat.completion');
    expect(body.choices[0].message.content.trim()).toBe('Hello from the mesh');
    expect(body.choices[0].message.role).toBe('assistant');
    expect(body.model).toBe(MODEL);
  });

  it('forwards the prompt the client actually sent', async () => {
    worker.jobsSeen.length = 0;
    await post('/v1/chat/completions', {
      model: MODEL,
      messages: [
        { role: 'system', content: 'be terse' },
        { role: 'user', content: 'ping' },
      ],
      temperature: 0.1,
      max_tokens: 32,
    });

    expect(worker.jobsSeen).toHaveLength(1);
    const { payload } = worker.jobsSeen[0];
    expect(payload.messages).toEqual([
      { role: 'system', content: 'be terse' },
      { role: 'user', content: 'ping' },
    ]);
    expect(payload.temperature).toBe(0.1);
    expect(payload.max_tokens).toBe(32);
  });

  it('streams SSE frames terminated by [DONE]', async () => {
    const response = await post('/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'say hello' }],
      stream: true,
    });

    expect(response.headers.get('content-type')).toMatch(/text\/event-stream/);
    const { text, sawDone, finishReason } = await readStream(response);
    expect(text.trim()).toBe('Hello from the mesh');
    expect(sawDone).toBe(true);
    expect(finishReason).toBe('stop');
  });

  it('serves concurrent requests across two workers', async () => {
    const second = new FakeWorker({ reply: 'second worker here' });
    await second.ready();

    try {
      const [a, b] = await Promise.all([
        post('/v1/chat/completions', { model: MODEL, messages: [{ role: 'user', content: '1' }] }),
        post('/v1/chat/completions', { model: MODEL, messages: [{ role: 'user', content: '2' }] }),
      ]);

      const replies = [
        (await a.json()).choices[0].message.content.trim(),
        (await b.json()).choices[0].message.content.trim(),
      ].sort();

      // One request went to each worker rather than queueing behind one.
      expect(replies).toEqual(['Hello from the mesh', 'second worker here']);
    } finally {
      second.close();
    }
  });
});

describe('fault tolerance', () => {
  it('retries on a healthy worker when the chosen one dies mid-request', async () => {
    // The dying worker is registered first; both can serve the model, so the
    // request must survive whichever one the scheduler picks.
    const dying = new FakeWorker({ dieOnJob: true });
    await dying.ready();
    const healthy = new FakeWorker({ reply: 'recovered cleanly' });
    await healthy.ready();

    try {
      const response = await post('/v1/chat/completions', {
        model: MODEL,
        messages: [{ role: 'user', content: 'hi' }],
      });

      expect(response.status).toBe(200);
      const content = (await response.json()).choices[0].message.content.trim();
      // Either the healthy worker was chosen outright, or the dying one was
      // chosen and the job was transparently reassigned. Both end here.
      expect(content).toBe('recovered cleanly');
    } finally {
      dying.close();
      healthy.close();
    }
  });

  it('stops advertising a model once its worker disconnects', async () => {
    const temporary = new FakeWorker();
    await temporary.ready();

    const before = await (await fetch(`${BASE}/v1/models`, {
      headers: { authorization: `Bearer ${TOKEN}` },
    })).json();
    expect(before.data).toHaveLength(1);

    temporary.close();
    await new Promise((resolve) => setTimeout(resolve, 100));

    const after = await (await fetch(`${BASE}/v1/models`, {
      headers: { authorization: `Bearer ${TOKEN}` },
    })).json();
    expect(after.data).toEqual([]);
  });

  it('closes a worker socket that presents a bad token', async () => {
    const socket = new WebSocket(`ws://127.0.0.1:${PORT}/mesh?token=nope`);
    const code = await new Promise((resolve) => socket.on('close', resolve));
    expect(code).toBe(4401);
  });
});
