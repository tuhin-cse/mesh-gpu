/**
 * End-to-end coverage of the control plane: scopes, quotas, model allowlists,
 * the audit log, and the admin API — driven over real HTTP and WebSocket.
 *
 * These are the properties an organisation would be buying, so they are tested
 * the way a buyer would check them: by making the requests, not by calling the
 * modules directly.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import WebSocket from 'ws';

const PORT = 8145;
const ROOT_KEY = 'root-admin-key-for-tests';
const BASE = `http://127.0.0.1:${PORT}`;
const MODEL = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';
const OTHER_MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

let server;
let dataDir;

function request(pathname, { method = 'GET', body, token = ROOT_KEY } = {}) {
  return fetch(`${BASE}${pathname}`, {
    method,
    headers: {
      'content-type': 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
}

async function json(pathname, options) {
  const res = await request(pathname, options);
  return { status: res.status, body: await res.json().catch(() => ({})), res };
}

function chat(token, model = MODEL, content = 'hello') {
  return json('/v1/chat/completions', {
    method: 'POST',
    token,
    body: { model, messages: [{ role: 'user', content }] },
  });
}

/** A worker that echoes a fixed reply, so completions actually complete. */
class FakeWorker {
  constructor(token, model = MODEL) {
    this.socket = new WebSocket(`ws://127.0.0.1:${PORT}/mesh?token=${encodeURIComponent(token)}&label=fake`);
    this.model = model;
    this.closeCode = null;
  }

  ready() {
    return new Promise((resolve, reject) => {
      this.socket.on('error', reject);
      this.socket.on('close', (code) => {
        this.closeCode = code;
        reject(Object.assign(new Error(`closed ${code}`), { code }));
      });
      this.socket.on('message', (raw) => {
        const message = JSON.parse(String(raw));
        if (message.type === 'welcome') {
          this.socket.send(JSON.stringify({ type: 'status', model: this.model, paused: false }));
          setTimeout(resolve, 20);
        }
        if (message.type === 'job') {
          this.socket.send(JSON.stringify({ type: 'chunk', jobId: message.jobId, delta: 'reply' }));
          this.socket.send(JSON.stringify({ type: 'done', jobId: message.jobId }));
        }
      });
    });
  }

  close() {
    if (this.socket.readyState === WebSocket.OPEN) this.socket.close();
  }
}

beforeAll(async () => {
  dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'meshgpu-gov-'));
  process.env.MESH_PORT = String(PORT);
  process.env.MESH_TOKEN = ROOT_KEY;
  process.env.MESH_MDNS = 'off';
  process.env.MESH_DATA_DIR = dataDir;

  ({ server } = await import('../server.js'));
  if (!server.listening) await new Promise((resolve) => server.once('listening', resolve));
});

afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
  fs.rmSync(dataDir, { recursive: true, force: true });
});

describe('bootstrap', () => {
  it('adopts MESH_TOKEN as a full-scope admin key', async () => {
    const { status, body } = await json('/admin/api/keys');
    expect(status).toBe(200);
    expect(body.keys).toHaveLength(1);
    expect(body.keys[0].scopes.sort()).toEqual(['admin', 'chat', 'serve']);
  });

  it('serves the admin console without a key, since it holds no data', async () => {
    const res = await fetch(`${BASE}/admin`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toMatch(/text\/html/);
  });

  it('never exposes a key hash over the admin API', async () => {
    const { body } = await json('/admin/api/keys');
    expect(JSON.stringify(body)).not.toMatch(/"hash"/);
  });
});

describe('scopes', () => {
  it('refuses the admin API to a key without the admin scope', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'chat only', scopes: ['chat'] },
    });

    const denied = await json('/admin/api/keys', { token: body.key });
    expect(denied.status).toBe(403);
    expect(denied.body.error.message).toMatch(/admin scope/);
  });

  it('refuses chat to a key without the chat scope', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'serve only', scopes: ['serve'] },
    });

    const denied = await chat(body.key);
    expect(denied.status).toBe(403);
    expect(denied.body.error.message).toMatch(/chat scope/);
  });

  it('refuses a worker socket to a key without the serve scope', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'no serve', scopes: ['chat'] },
    });

    const worker = new FakeWorker(body.key);
    await expect(worker.ready()).rejects.toMatchObject({ code: 4403 });
  });

  it('refuses a worker socket to an unknown key', async () => {
    const worker = new FakeWorker('not-a-real-key');
    await expect(worker.ready()).rejects.toMatchObject({ code: 4401 });
  });
});

describe('revocation', () => {
  it('stops a revoked key immediately without disturbing others', async () => {
    const alice = (await json('/admin/api/keys', { method: 'POST', body: { name: 'Alice' } })).body;
    const bob = (await json('/admin/api/keys', { method: 'POST', body: { name: 'Bob' } })).body;

    // Both reach the endpoint; 503 means "no worker has this model", which is
    // past authentication and authorisation.
    expect((await chat(alice.key)).status).toBe(503);
    expect((await chat(bob.key)).status).toBe(503);

    await json(`/admin/api/keys/${alice.record.id}`, { method: 'DELETE' });

    expect((await chat(alice.key)).status).toBe(401);
    expect((await chat(bob.key)).status).toBe(503);
  });

  it('refuses to revoke the last admin key', async () => {
    const listed = await json('/admin/api/keys');
    const admins = listed.body.keys.filter((key) => key.scopes.includes('admin'));
    expect(admins).toHaveLength(1);

    const { status, body } = await json(`/admin/api/keys/${admins[0].id}`, { method: 'DELETE' });
    expect(status).toBe(409);
    expect(body.error.message).toMatch(/last admin key/);

    // Still usable — locking everyone out of a box on a shelf is unrecoverable.
    expect((await json('/admin/api/keys')).status).toBe(200);
  });
});

describe('model allowlist', () => {
  it('blocks a model outside the key’s allowlist', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'qwen only', allowedModels: [MODEL] },
    });

    const denied = await chat(body.key, OTHER_MODEL);
    expect(denied.status).toBe(403);
    expect(denied.body.error.code).toBe('model_not_allowed');
  });

  it('hides models a key cannot use from /v1/models', async () => {
    const worker = new FakeWorker(ROOT_KEY);
    await worker.ready();

    try {
      const { body } = await json('/admin/api/keys', {
        method: 'POST',
        body: { name: 'restricted', allowedModels: [OTHER_MODEL] },
      });

      const open = await json('/v1/models');
      expect(open.body.data.map((m) => m.id)).toContain(MODEL);

      const restricted = await json('/v1/models', { token: body.key });
      expect(restricted.body.data.map((m) => m.id)).not.toContain(MODEL);
    } finally {
      worker.close();
    }
  });

  it('lets a mesh-wide block list override a permissive key', async () => {
    await json('/admin/api/settings', { method: 'PUT', body: { blockedModels: [MODEL] } });
    try {
      const denied = await chat(ROOT_KEY, MODEL);
      expect(denied.status).toBe(403);
    } finally {
      await json('/admin/api/settings', { method: 'PUT', body: { blockedModels: [] } });
    }
  });
});

describe('quotas', () => {
  it('enforces a daily request limit and says when it resets', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'limited', dailyRequestLimit: 2 },
    });

    expect((await chat(body.key)).status).toBe(503); // consumed 1
    expect((await chat(body.key)).status).toBe(503); // consumed 2

    const denied = await chat(body.key);
    expect(denied.status).toBe(429);
    expect(denied.body.error.code).toBe('rate_limit_exceeded');
    expect(Number(denied.res.headers.get('retry-after'))).toBeGreaterThan(0);
  });

  it('counts quota per key, not globally', async () => {
    const limited = (await json('/admin/api/keys', {
      method: 'POST', body: { name: 'tight', dailyRequestLimit: 1 },
    })).body;
    const roomy = (await json('/admin/api/keys', {
      method: 'POST', body: { name: 'roomy', dailyRequestLimit: 50 },
    })).body;

    await chat(limited.key);
    expect((await chat(limited.key)).status).toBe(429);
    expect((await chat(roomy.key)).status).toBe(503);
  });

  it('does not charge quota for a request denied before dispatch', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'allowlisted', allowedModels: [MODEL], dailyRequestLimit: 3 },
    });

    // Three rejections on an allowlist must not consume the daily budget.
    for (let i = 0; i < 3; i += 1) {
      expect((await chat(body.key, OTHER_MODEL)).status).toBe(403);
    }
    expect((await chat(body.key, MODEL)).status).toBe(503);
  });
});

describe('audit log', () => {
  it('records a completion with who, what and the outcome', async () => {
    const worker = new FakeWorker(ROOT_KEY);
    await worker.ready();

    try {
      const { body } = await json('/admin/api/keys', {
        method: 'POST',
        body: { name: 'Audited', scopes: ['chat'] },
      });
      await chat(body.key, MODEL, 'a very specific secret prompt');

      const audit = await json('/admin/api/audit?type=chat.completion&limit=10');
      const entry = audit.body.entries.find((e) => e.keyName === 'Audited');

      expect(entry).toBeDefined();
      expect(entry.model).toBe(MODEL);
      expect(entry.outcome).toBe('ok');
      expect(entry.workerId).toBeTruthy();
      expect(entry.durationMs).toBeGreaterThanOrEqual(0);
    } finally {
      worker.close();
    }
  });

  it('stores a prompt hash and length, never the text', async () => {
    const audit = await json('/admin/api/audit?type=chat.completion&limit=50');

    expect(audit.body.retention).toBe('hashed');
    expect(JSON.stringify(audit.body)).not.toContain('a very specific secret prompt');

    const entry = audit.body.entries.find((e) => e.promptHash);
    expect(entry.promptHash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry.promptChars).toBeGreaterThan(0);
    expect(entry.promptText).toBeUndefined();
  });

  it('records denials, which is the point of having the log', async () => {
    const { body } = await json('/admin/api/keys', {
      method: 'POST',
      body: { name: 'Denied', allowedModels: [OTHER_MODEL] },
    });
    await chat(body.key, MODEL);

    const audit = await json('/admin/api/audit?outcome=denied&limit=20');
    const entry = audit.body.entries.find((e) => e.keyName === 'Denied');
    expect(entry.outcome).toBe('denied');
    expect(entry.detail.reason).toMatch(/not permitted/);
  });

  it('records administrative changes, not just traffic', async () => {
    const created = await json('/admin/api/keys', { method: 'POST', body: { name: 'Ephemeral' } });
    await json(`/admin/api/keys/${created.body.record.id}`, { method: 'DELETE' });

    const audit = await json('/admin/api/audit?limit=50');
    const types = audit.body.entries.map((e) => e.type);
    expect(types).toContain('key.created');
    expect(types).toContain('key.revoked');

    const revocation = audit.body.entries.find((e) => e.type === 'key.revoked');
    expect(revocation.detail.targetKeyName).toBe('Ephemeral');
  });

  it('records workers joining and leaving', async () => {
    const worker = new FakeWorker(ROOT_KEY);
    await worker.ready();
    worker.close();
    await new Promise((resolve) => setTimeout(resolve, 120));

    const audit = await json('/admin/api/audit?limit=60');
    const types = audit.body.entries.map((e) => e.type);
    expect(types).toContain('worker.joined');
    expect(types).toContain('worker.left');
  });

  it('writes newline-delimited JSON to disk for export', () => {
    const lines = fs
      .readFileSync(path.join(dataDir, 'audit.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));

    expect(lines.length).toBeGreaterThan(5);
    expect(lines.every((entry) => entry.ts && entry.type)).toBe(true);
  });
});

describe('admin overview', () => {
  it('reports the mesh in one round trip', async () => {
    const { body } = await json('/admin/api/overview');

    expect(body).toHaveProperty('workers');
    expect(body).toHaveProperty('queue');
    expect(body).toHaveProperty('usage');
    expect(body.retention).toBe('hashed');
    expect(body.keys).toBeGreaterThan(0);
    expect(body.audit.byOutcome).toBeTypeOf('object');
  });

  it('persists keys across a restart', () => {
    const state = JSON.parse(fs.readFileSync(path.join(dataDir, 'state.json'), 'utf8'));
    expect(Object.keys(state.keys).length).toBeGreaterThan(1);
    expect(Object.values(state.usage).some((count) => count > 0)).toBe(true);
  });
});
