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
 * and an LAN browser tab, and are never persisted.
 */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

import { WorkerRegistry } from './lib/registry.js';
import { JobQueue } from './lib/queue.js';
import { bearerFrom, resolveToken, tokenMatches } from './lib/auth.js';
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

const { token: TOKEN, generated: TOKEN_GENERATED } = resolveToken(process.env.MESH_TOKEN);

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

function sendJson(res, status, body) {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function readBody(req, limitBytes = 4 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limitBytes) {
        reject(Object.assign(new Error('request body too large'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      if (raw.length === 0) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(Object.assign(new Error('request body is not valid JSON'), { status: 400 }));
      }
    });
    req.on('error', reject);
  });
}

/** CORS so a page served from `npm run dev` can talk to a coordinator on :8080. */
function applyCors(res) {
  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'authorization, content-type, x-api-key');
  res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

async function handleChatCompletions(req, res) {
  const body = await readBody(req);
  const { model, messages, stream, params } = parseChatRequest(body);

  const id = newId('chatcmpl');
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });

    const write = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`);
    // An empty leading delta gives the client its role immediately, matching
    // OpenAI's stream shape.
    write(streamChunk(id, model, null, created));

    const job = queue.submit({
      id: newId('job'),
      model,
      payload: { messages, ...params },
      onChunk: (delta) => write(streamChunk(id, model, delta, created)),
      onDone: () => {
        write(streamDone(id, model, created));
        res.write('data: [DONE]\n\n');
        res.end();
      },
      onError: (error) => {
        // Mid-stream there is no status code left to set, so the error is
        // delivered in-band; clients surface it as a truncated completion.
        write({ ...streamDone(id, model, created, 'error'), error: { message: error.message } });
        res.write('data: [DONE]\n\n');
        res.end();
      },
    });

    res.on('close', () => queue.cancel(job.id));
    return;
  }

  let text = '';
  await new Promise((resolve, reject) => {
    const job = queue.submit({
      id: newId('job'),
      model,
      payload: { messages, ...params },
      onChunk: (delta) => {
        text += delta;
      },
      onDone: () => resolve(undefined),
      onError: reject,
    });
    res.on('close', () => queue.cancel(job.id));
  });

  sendJson(res, 200, completionBody(id, model, text, created));
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

  // Unauthenticated: liveness only. Everything under /v1 needs the token.
  if (route === '/healthz') {
    return sendJson(res, 200, { ok: true, workers: registry.size, ...queue.stats });
  }

  if (route.startsWith('/v1/') || route === '/status') {
    if (!tokenMatches(bearerFrom(req), TOKEN)) {
      return sendJson(res, 401, errorBody('missing or invalid API key', 'authentication_error'));
    }
  }

  if (route === '/status') {
    return sendJson(res, 200, {
      workers: registry.snapshot(),
      models: registry.availableModels(),
      queue: queue.stats,
    });
  }

  if (route === '/v1/models') {
    return sendJson(res, 200, {
      object: 'list',
      data: registry.availableModels().map((id) => ({
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
    return handleChatCompletions(req, res).catch((error) => {
      if (res.headersSent) return res.end();
      const status = error.status ?? 500;
      sendJson(res, status, errorBody(error.message, error.code ?? 'server_error'));
    });
  }

  if (req.method === 'GET') return serveStatic(req, res);
  sendJson(res, 404, errorBody(`no route for ${route}`, 'invalid_request_error'));
});

// ---------------------------------------------------------------------------
// Worker WebSocket
// ---------------------------------------------------------------------------

const wss = new WebSocketServer({ server, path: '/mesh' });

wss.on('connection', (socket, req) => {
  const url = new URL(req.url ?? '/mesh', 'http://localhost');
  // Browsers cannot set headers on a WebSocket handshake, so the token comes
  // in the query string. It never leaves the LAN and is not logged.
  if (!tokenMatches(url.searchParams.get('token'), TOKEN)) {
    socket.close(4401, 'invalid token');
    return;
  }

  const workerId = newId('worker');
  const label = (url.searchParams.get('label') ?? 'browser').slice(0, 60);

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
  log(`worker joined: ${label} (${workerId}) — ${registry.size} on the mesh`);

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
  log(`API key:          ${TOKEN}${TOKEN_GENERATED ? '  (generated — set MESH_TOKEN to pin it)' : ''}`);
  console.log('');
  log('Share this link with colleagues to lend their GPU:');
  log(`  http://${primary}:${PORT}/?token=${TOKEN}`);
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
  advertisement?.service?.stop?.();
  advertisement?.bonjour?.destroy?.();
  for (const socket of wss.clients) socket.close(1001, 'coordinator shutting down');
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export { server, registry, queue };
