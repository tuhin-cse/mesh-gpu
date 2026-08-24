/**
 * MeshGPU signaling server.
 *
 * Zero-compute relay: it only brokers WebRTC offer/answer/ICE between peers in
 * the same room. It never sees model weights or tensor data.
 *
 * Message protocol (JSON over WebSocket):
 *
 *   client -> server
 *     { type: 'join',   roomId, peerId }
 *     { type: 'signal', targetPeerId, signal }   // relayed offer/answer/ice
 *     { type: 'leave' }
 *
 *   server -> client
 *     { type: 'welcome',     roomId, peerId, peers: string[], iceServers: RTCIceServer[] }
 *     { type: 'peer-joined', roomId, peerId }
 *     { type: 'peer-left',   roomId, peerId }
 *     { type: 'signal',      fromPeerId, signal }
 *     { type: 'error',       message }
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, WebSocket } from 'ws';

// Load .env (if present) so Cloudflare TURN credentials can be configured
// without exporting them in the shell. Values already in process.env win.
function loadEnvFile() {
  const envPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '.env');
  try {
    const text = fs.readFileSync(envPath, 'utf8');
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eq = trimmed.indexOf('=');
      if (eq === -1) continue;
      const key = trimmed.slice(0, eq).trim();
      const value = trimmed.slice(eq + 1).trim().replace(/^['"]|['"]$/g, '');
      if (key && process.env[key] === undefined) process.env[key] = value;
    }
  } catch {
    // No .env file — ignore.
  }
}

loadEnvFile();

const PORT = Number(process.env.PORT ?? 8080);
const HEARTBEAT_INTERVAL_MS = 30_000;

// Cloudflare Calls / TURN. Ephemeral credentials are generated server-side so
// the HMAC key never reaches the browser (see .env.example).
const CLOUDFLARE_TURN_KEY_ID = process.env.CLOUDFLARE_TURN_KEY_ID ?? '';
const CLOUDFLARE_TURN_KEY_SECRET = process.env.CLOUDFLARE_TURN_KEY_SECRET ?? '';
const TURN_TTL_SECONDS = 86400; // 24h

/**
 * Build the ICE server list handed to clients. Always includes public STUN;
 * adds Cloudflare TURN (relay) when credentials are configured.
 */
function buildIceServers() {
  const iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
  if (CLOUDFLARE_TURN_KEY_ID && CLOUDFLARE_TURN_KEY_SECRET) {
    const expiry = Math.floor(Date.now() / 1000) + TURN_TTL_SECONDS;
    const username = `${expiry}:${CLOUDFLARE_TURN_KEY_ID}`;
    const credential = crypto
      .createHmac('sha1', CLOUDFLARE_TURN_KEY_SECRET)
      .update(username)
      .digest('base64');
    iceServers.push({
      urls: [
        'turn:rtc.cloudflare.com:53?transport=udp',
        'turns:rtc.cloudflare.com:443?transport=tcp',
      ],
      username,
      credential,
    });
  }
  return iceServers;
}

/** @type {Map<import('ws').WebSocket, { peerId: string, roomId: string, alive: boolean }>} */
const clients = new Map();

/** @type {Map<string, Set<import('ws').WebSocket>>} */
const rooms = new Map();

const wss = new WebSocketServer({ port: PORT });

function send(ws, payload) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

/** Send a payload to every peer in a room except (optionally) one. */
function broadcast(roomId, payload, except = null) {
  const room = rooms.get(roomId);
  if (!room) return;
  for (const ws of room) {
    if (ws !== except && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }
}

function roomPeerIds(roomId) {
  const room = rooms.get(roomId);
  if (!room) return [];
  return [...room].map((ws) => clients.get(ws)?.peerId).filter(Boolean);
}

function handleJoin(ws, meta, msg) {
  const roomId = typeof msg.roomId === 'string' ? msg.roomId.trim() : '';
  const peerId = typeof msg.peerId === 'string' ? msg.peerId.trim() : '';

  if (!roomId || !peerId) {
    return send(ws, { type: 'error', message: 'join requires non-empty roomId and peerId' });
  }
  if (meta.roomId) {
    return send(ws, { type: 'error', message: 'already joined a room' });
  }

  const room = rooms.get(roomId) ?? new Set();
  const peerIdTaken = [...room].some((w) => clients.get(w)?.peerId === peerId);
  if (peerIdTaken) {
    return send(ws, { type: 'error', message: `peerId "${peerId}" is already in room "${roomId}"` });
  }

  meta.peerId = peerId;
  meta.roomId = roomId;
  room.add(ws);
  rooms.set(roomId, room);

  send(ws, {
    type: 'welcome',
    roomId,
    peerId,
    peers: roomPeerIds(roomId).filter((id) => id !== peerId),
    iceServers: buildIceServers(),
  });
  broadcast(roomId, { type: 'peer-joined', roomId, peerId }, ws);
  console.log(`[mesh-gpu] ${peerId} joined room "${roomId}" (${room.size} peers)`);
}

function handleSignal(ws, meta, msg) {
  if (!meta.roomId) {
    return send(ws, { type: 'error', message: 'join a room before signaling' });
  }

  const targetPeerId = typeof msg.targetPeerId === 'string' ? msg.targetPeerId : '';
  if (!targetPeerId || msg.signal === undefined) {
    return send(ws, { type: 'error', message: 'signal requires targetPeerId and a signal payload' });
  }

  const room = rooms.get(meta.roomId);
  if (!room) return;

  const target = [...room].find((w) => clients.get(w)?.peerId === targetPeerId);
  if (!target) {
    return send(ws, { type: 'error', message: `peer "${targetPeerId}" is not in the room` });
  }

  send(target, { type: 'signal', fromPeerId: meta.peerId, signal: msg.signal });
}

function handleMessage(ws, msg) {
  const meta = clients.get(ws);
  if (!meta) return;

  switch (msg.type) {
    case 'join':
      return handleJoin(ws, meta, msg);
    case 'signal':
      return handleSignal(ws, meta, msg);
    case 'leave':
      return leaveRoom(ws);
    default:
      return send(ws, { type: 'error', message: `unknown message type: ${String(msg.type)}` });
  }
}

function leaveRoom(ws) {
  const meta = clients.get(ws);
  if (!meta) return;

  const { roomId, peerId } = meta;
  const room = rooms.get(roomId);
  if (room) {
    room.delete(ws);
    if (room.size === 0) {
      rooms.delete(roomId);
    } else if (peerId) {
      broadcast(roomId, { type: 'peer-left', roomId, peerId });
      console.log(`[mesh-gpu] ${peerId} left room "${roomId}" (${room.size} peers)`);
    }
  }
  clients.delete(ws);
}

wss.on('connection', (ws) => {
  clients.set(ws, { peerId: '', roomId: '', alive: true });

  ws.on('pong', () => {
    const meta = clients.get(ws);
    if (meta) meta.alive = true;
  });

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return send(ws, { type: 'error', message: 'invalid JSON' });
    }
    if (msg && typeof msg === 'object') {
      handleMessage(ws, msg);
    }
  });

  ws.on('close', () => leaveRoom(ws));
  ws.on('error', () => leaveRoom(ws));
});

// Heartbeat: drop dead connections so orphaned peers are detected quickly.
const heartbeat = setInterval(() => {
  for (const [ws, meta] of clients) {
    if (!meta.alive) {
      ws.terminate();
      continue;
    }
    meta.alive = false;
    ws.ping();
  }
}, HEARTBEAT_INTERVAL_MS);

wss.on('close', () => clearInterval(heartbeat));

console.log(`[mesh-gpu] signaling server listening on ws://0.0.0.0:${PORT}`);
console.log(
  `[mesh-gpu] TURN: ${CLOUDFLARE_TURN_KEY_ID ? 'Cloudflare TURN enabled' : 'disabled (set CLOUDFLARE_TURN_KEY_ID / CLOUDFLARE_TURN_KEY_SECRET)'}`,
);
