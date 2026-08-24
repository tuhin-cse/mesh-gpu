/**
 * MeshGPU coordinator.
 *
 * A small on-prem control plane for an office mesh. It:
 *   - serves the built frontend, so joining is "open this URL" rather than
 *     scanning a QR code;
 *   - accepts browser tabs as workers over WebSocket;
 *   - exposes an OpenAI-compatible endpoint and routes each request to a tab
 *     that is idle and has the model loaded;
 *   - advertises itself over mDNS so people can find it by name.
 *
 * What it does NOT do: run models, see GPU memory, or talk to the internet.
 * Prompts and completions pass through it on their way between an LAN client
 * and an LAN browser tab. By default nothing of their content is persisted —
 * the audit log records a hash and a length, not the text.
 */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

import { WorkerRegistry } from './lib/registry.js';
import { JobQueue } from './lib/queue.js';
import { resolveToken } from './lib/auth.js';
import { bearerFrom, readBody, sendJson } from './lib/http.js';
import { Store } from './lib/store.js';
import { Identity, SCOPES, hasScope, mayUseModel } from './lib/identity.js';
import { Quota } from './lib/quota.js';
import { AuditLog, RETENTION } from './lib/audit.js';
import { handleAdminApi } from './lib/admin.js';
import {
  completionBody,
  errorBody,
  newId,
  parseChatRequest,
  streamChunk,
  streamDone,
} from './lib/openai.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.resolve(HERE, '..', 'dist');

const PORT = Number(process.env.MESH_PORT ?? 8080);
const HOST = process.env.MESH_HOST ?? '0.0.0.0';
const JOB_TIMEOUT_MS = Number(process.env.MESH_JOB_TIMEOUT_MS ?? 120_000);
const MAX_QUEUE_DEPTH = Number(process.env.MESH_MAX_QUEUE ?? 64);
const HEARTBEAT_MS = 15_000;

const DATA_DIR = process.env.MESH_DATA_DIR ?? path.resolve(HERE, 'data');
const STATE_FILE = process.env.MESH_STATE_FILE ?? path.join(DATA_DIR, 'state.json');
const AUDIT_FILE = process.env.MESH_AUDIT_FILE ?? path.join(DATA_DIR, 'audit.jsonl');
const AUDIT_RETENTION = process.env.MESH_AUDIT_RETENTION ?? RETENTION.HASHED;

const store = new Store(STATE_FILE);
const identity = new Identity(store);
const quota = new Quota(store);
const audit = new AuditLog({ filePath: AUDIT_FILE, retention: AUDIT_RETENTION });

/**
 * Bootstrap an administrator.
 *
 * MESH_TOKEN, when set, is adopted as a full-scope key so existing setups keep
 * working. Otherwise, on a mesh with no admin key at all, one is generated and
 * printed — a coordinator that starts with no way in is useless, and one that
 * starts wide open is worse.
 */
const { token: BOOTSTRAP_TOKEN, generated: TOKEN_GENERATED } = resolveToken(process.env.MESH_TOKEN);
let bootstrapKey = null;
if (process.env.MESH_TOKEN) {
  identity.adopt(BOOTSTRAP_TOKEN, 'MESH_TOKEN');
  bootstrapKey = BOOTSTRAP_TOKEN;
} else if (!identity.hasAdmin()) {
  identity.adopt(BOOTSTRAP_TOKEN, 'bootstrap admin');
  bootstrapKey = BOOTSTRAP_TOKEN;
}
store.flush();

// Usage rows accumulate one per key per day; trim on startup.
quota.prunePastDays(32);

const registry = new WorkerRegistry();
const queue = new JobQueue({
  registry,
  maxQueueDepth: MAX_QUEUE_DEPTH,
  jobTimeoutMs: JOB_TIMEOUT_MS,
});

// ---------------------------------------------------------------------------
// Static frontend
// ---------------------------------------------------------------------------

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function serveStatic(req, res) {
  if (!fs.existsSync(DIST_DIR)) {
    return sendJson(res, 503, errorBody('frontend not built — run `npm run build` first', 'server_error'));
  }

  const requested = decodeURIComponent((req.url ?? '/').split('?')[0]);
  const relative = requested === '/' ? 'index.html' : requested.replace(/^\/+/, '');
  const resolved = path.resolve(DIST_DIR, relative);

  // Never serve outside dist/, whatever the path traversal attempt looks like.
  if (!resolved.startsWith(DIST_DIR + path.sep) && resolved !== path.join(DIST_DIR, 'index.html')) {
    return sendJson(res, 403, errorBody('forbidden', 'server_error'));
  }

  const filePath = fs.existsSync(resolved) && fs.statSync(resolved).isFile()
    ? resolved
    : path.join(DIST_DIR, 'index.html'); // SPA fallback

  const type = CONTENT_TYPES[path.extname(filePath)] ?? 'application/octet-stream';
  res.writeHead(200, { 'content-type': type, 'cache-control': 'no-cache' });
  fs.createReadStream(filePath).pipe(res);
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/** CORS so a page served from `npm run dev` can talk to a coordinator on :8080. */
function applyCors(res) {
  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'authorization, content-type, x-api-key');
  res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

async function handleChatCompletions(req, res, principal) {
  const body = await readBody(req);
  const { model, messages, stream, params } = parseChatRequest(body);

  const startedAt = Date.now();
  // Prompt text is handed to the audit log, which decides what — if anything —
  // of it is retained. By default that is a hash and a length.
  const promptText = messages.map((message) => `${message.role}: ${message.content}`).join('\n');

  // errorBody(message, type, code) — the second argument is the category, the
  // third the specific reason. Client SDKs branch on `code`.
  const deny = (status, message, type, code, extraHeaders = {}) => {
    audit.record({
      type: 'chat.completion',
      keyId: principal.id,
      keyName: principal.name,
      model,
      outcome: 'denied',
      prompt: promptText,
      detail: { reason: message },
    });
    sendJson(res, status, errorBody(message, type, code), extraHeaders);
  };

  if (!mayUseModel(principal, model, store.state.settings.blockedModels)) {
    return deny(
      403,
      `this key is not permitted to use "${model}"`,
      'permission_error',
      'model_not_allowed',
    );
  }

  const allowance = quota.check(principal);
  if (!allowance.ok) {
    return deny(429, allowance.reason, 'rate_limit_error', 'rate_limit_exceeded', {
      'retry-after': String(allowance.retryAfterSeconds),
    });
  }
  quota.consume(principal);
  identity.markUsed(principal.id);

  const id = newId('chatcmpl');
  const created = Math.floor(Date.now() / 1000);

  const finish = (outcome, completion, workerId, detail) => {
    audit.record({
      type: 'chat.completion',
      keyId: principal.id,
      keyName: principal.name,
      model,
      workerId: workerId ?? null,
      outcome,
      durationMs: Date.now() - startedAt,
      prompt: promptText,
      completion,
      detail,
    });
  };

  if (stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });

    const write = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`);
    write(streamChunk(id, model, null, created));

    let text = '';
    const job = queue.submit({
      id: newId('job'),
      model,
      payload: { messages, ...params },
      onChunk: (delta) => {
        text += delta;
        write(streamChunk(id, model, delta, created));
      },
      onDone: (info) => {
        write(streamDone(id, model, created));
        res.write('data: [DONE]\n\n');
        res.end();
        finish('ok', text, info.workerId);
      },
      onError: (error) => {
        write({ ...streamDone(id, model, created, 'error'), error: { message: error.message } });
        res.write('data: [DONE]\n\n');
        res.end();
        finish('error', text, null, { reason: error.message });
      },
    });

    res.on('close', () => queue.cancel(job.id));
    return;
  }

  let text = '';
  let servedBy = null;
  try {
    await new Promise((resolve, reject) => {
      const job = queue.submit({
        id: newId('job'),
        model,
        payload: { messages, ...params },
        onChunk: (delta) => {
          text += delta;
        },
        onDone: (info) => {
          servedBy = info.workerId;
          resolve(undefined);
        },
        onError: reject,
      });
      res.on('close', () => queue.cancel(job.id));
    });
  } catch (error) {
    finish('error', text, null, { reason: error.message });
    throw error;
  }

  finish('ok', text, servedBy);
  sendJson(res, 200, completionBody(id, model, text, created));
}

/**
 * Resolve the caller. Returns the key record, or null — callers must not
 * distinguish "unknown key" from "revoked key" in what they send back.
 */
function authenticate(req) {
  return identity.verify(bearerFrom(req));
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const route = url.pathname;

  applyCors(res);
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // Unauthenticated: liveness, and the admin page shell. The console is inert
  // HTML — every byte of data it shows comes from an authenticated fetch.
  if (route === '/healthz') {
    return sendJson(res, 200, { ok: true, workers: registry.size, ...queue.stats });
  }
  if (route === '/admin' || route === '/admin/') {
    return serveFile(res, path.join(HERE, 'public', 'admin.html'), 'text/html; charset=utf-8');
  }

  const needsAuth = route.startsWith('/v1/') || route.startsWith('/admin/') || route === '/status';
  let principal = null;
  if (needsAuth) {
    principal = authenticate(req);
    if (!principal) {
      return sendJson(res, 401, errorBody('missing or invalid API key', 'authentication_error'));
    }
  }

  if (route.startsWith('/admin/')) {
    if (!hasScope(principal, SCOPES.ADMIN)) {
      return sendJson(res, 403, errorBody('this key lacks the admin scope', 'permission_error'));
    }
    return handleAdminApi(
      { identity, quota, audit, store, registry, queue, principal },
      req,
      res,
      url,
    ).then((handled) => {
      if (!handled) sendJson(res, 404, errorBody(`no route for ${route}`));
    }).catch((error) => {
      if (res.headersSent) return res.end();
      sendJson(res, error.status ?? 500, errorBody(error.message, error.code ?? 'server_error'));
    });
  }

  if (route === '/status') {
    if (!hasScope(principal, SCOPES.ADMIN)) {
      return sendJson(res, 403, errorBody('this key lacks the admin scope', 'permission_error'));
    }
    return sendJson(res, 200, {
      workers: registry.snapshot(),
      models: registry.availableModels(),
      queue: queue.stats,
    });
  }

  if (route.startsWith('/v1/') && !hasScope(principal, SCOPES.CHAT)) {
    return sendJson(res, 403, errorBody('this key lacks the chat scope', 'permission_error'));
  }

  if (route === '/v1/models') {
    // Only advertise what this key could actually call.
    const usable = registry
      .availableModels()
      .filter((model) => mayUseModel(principal, model, store.state.settings.blockedModels));
    return sendJson(res, 200, {
      object: 'list',
      data: usable.map((id) => ({
        id,
        object: 'model',
        owned_by: 'meshgpu',
        created: Math.floor(Date.now() / 1000),
      })),
    });
  }

  if (route === '/v1/chat/completions') {
    if (req.method !== 'POST') {
      return sendJson(res, 405, errorBody('use POST', 'invalid_request_error'));
    }
    return handleChatCompletions(req, res, principal).catch((error) => {
      if (res.headersSent) return res.end();
      const status = error.status ?? 500;
      sendJson(res, status, errorBody(error.message, error.code ?? 'server_error'));
    });
  }

  if (req.method === 'GET') return serveStatic(req, res);
  sendJson(res, 404, errorBody(`no route for ${route}`, 'invalid_request_error'));
});

/** Send a single file, used for the admin console shell. */
function serveFile(res, filePath, contentType) {
  if (!fs.existsSync(filePath)) {
    return sendJson(res, 404, errorBody('admin console not installed', 'server_error'));
  }
  res.writeHead(200, { 'content-type': contentType, 'cache-control': 'no-cache' });
  fs.createReadStream(filePath).pipe(res);
}

// ---------------------------------------------------------------------------
// Worker WebSocket
// ---------------------------------------------------------------------------

const wss = new WebSocketServer({ server, path: '/mesh' });

wss.on('connection', (socket, req) => {
  const url = new URL(req.url ?? '/mesh', 'http://localhost');
  // Browsers cannot set headers on a WebSocket handshake, so the key comes in
  // the query string. It never leaves the LAN and is never written to the log.
  const principal = identity.verify(url.searchParams.get('token'));
  if (!principal) {
    socket.close(4401, 'invalid key');
    return;
  }
  if (!hasScope(principal, SCOPES.SERVE)) {
    // A key that may ask the mesh questions is not automatically a key that
    // may answer them for other people.
    socket.close(4403, 'this key lacks the serve scope');
    return;
  }

  const workerId = newId('worker');
  const label = (url.searchParams.get('label') ?? 'browser').slice(0, 60);
  identity.markUsed(principal.id);

  registry.register({
    id: workerId,
    label,
    maxConcurrent: Number(url.searchParams.get('concurrency') ?? 1),
    send: (message) => {
      if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(message));
    },
  });

  socket.isAlive = true;
  socket.on('pong', () => {
    socket.isAlive = true;
  });

  socket.send(JSON.stringify({ type: 'welcome', workerId, heartbeatMs: HEARTBEAT_MS }));
  log(`worker joined: ${label} [${principal.name}] — ${registry.size} on the mesh`);
  audit.record({
    type: 'worker.joined',
    keyId: principal.id,
    keyName: principal.name,
    workerId,
    detail: { label },
  });

  socket.on('message', (raw) => {
    let message;
    try {
      message = JSON.parse(String(raw));
    } catch {
      return;
    }

    switch (message.type) {
      case 'status': {
        // A worker announces the model it has loaded and whether its idle
        // policy currently permits work.
        registry.setModel(workerId, typeof message.model === 'string' ? message.model : null);
        registry.setPaused(workerId, message.paused === true);
        queue.pump();
        break;
      }
      case 'chunk':
        queue.chunk(message.jobId, String(message.delta ?? ''));
        break;
      case 'done':
        queue.complete(message.jobId);
        break;
      case 'error':
        queue.fail(message.jobId, new Error(String(message.message ?? 'worker error')));
        break;
      default:
        break;
    }
  });

  const drop = (reason) => {
    if (registry.remove(workerId)) {
      queue.releaseWorker(workerId);
      log(`worker left: ${label} (${reason}) — ${registry.size} on the mesh`);
      audit.record({
        type: 'worker.left',
        keyId: principal.id,
        keyName: principal.name,
        workerId,
        detail: { reason },
      });
    }
  };

  socket.on('close', () => drop('closed'));
  socket.on('error', () => drop('socket error'));
});

// Drop workers whose tab was suspended or whose machine slept.
const heartbeat = setInterval(() => {
  for (const socket of wss.clients) {
    if (socket.isAlive === false) {
      socket.terminate();
      continue;
    }
    socket.isAlive = false;
    socket.ping();
  }
}, HEARTBEAT_MS);
heartbeat.unref();

// ---------------------------------------------------------------------------
// mDNS advertisement (optional — the mesh works without it)
// ---------------------------------------------------------------------------

async function advertise() {
  if (process.env.MESH_MDNS === 'off') return null;
  try {
    const { Bonjour } = await import('bonjour-service');
    const bonjour = new Bonjour();
    const service = bonjour.publish({
      name: process.env.MESH_NAME ?? `MeshGPU (${os.hostname()})`,
      type: 'meshgpu',
      protocol: 'tcp',
      port: PORT,
      txt: { path: '/', version: '0.2.0' },
    });
    log(`advertising as _meshgpu._tcp on port ${PORT}`);
    return { bonjour, service };
  } catch {
    // bonjour-service is optional; the coordinator is perfectly usable without
    // discovery as long as people know the host.
    log('mDNS unavailable — share the URL below manually');
    return null;
  }
}

function localAddresses() {
  const addresses = [];
  for (const entries of Object.values(os.networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family === 'IPv4' && !entry.internal) addresses.push(entry.address);
    }
  }
  return addresses;
}

function log(message) {
  console.log(`[meshgpu] ${message}`);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

let advertisement = null;

server.listen(PORT, HOST, async () => {
  advertisement = await advertise();

  const addresses = localAddresses();
  const primary = addresses[0] ?? 'localhost';

  console.log('');
  log(`coordinator listening on http://${primary}:${PORT}`);
  log(`OpenAI endpoint:  http://${primary}:${PORT}/v1`);
  log(`Admin console:    http://${primary}:${PORT}/admin`);
  log(`Audit retention:  ${audit.retention}${audit.retention === RETENTION.FULL ? '  *** PROMPT TEXT IS BEING STORED ***' : ' (prompt text is not stored)'}`);
  log(`State file:       ${STATE_FILE}`);
  console.log('');
  if (bootstrapKey) {
    log(`Admin key:        ${bootstrapKey}${TOKEN_GENERATED ? '  (generated — set MESH_TOKEN to pin it)' : ''}`);
    console.log('');
    log('Share this link with colleagues to lend their GPU:');
    log(`  http://${primary}:${PORT}/?token=${bootstrapKey}`);
    log('Better: issue each person their own key in the admin console, so you');
    log('can see who used what and revoke one without rotating for everyone.');
  } else {
    log('Admin keys already exist — issue worker keys from the admin console.');
  }
  if (addresses.length > 1) {
    log(`Other addresses: ${addresses.slice(1).join(', ')}`);
  }
  if (!fs.existsSync(DIST_DIR)) {
    log('WARNING: dist/ not found — run `npm run build` so the join link works.');
  }
  console.log('');
});

function shutdown() {
  log('shutting down');
  clearInterval(heartbeat);
  store.close(); // flush any pending quota/key writes
  advertisement?.service?.stop?.();
  advertisement?.bonjour?.destroy?.();
  for (const socket of wss.clients) socket.close(1001, 'coordinator shutting down');
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export { server, registry, queue, identity, quota, audit, store };
