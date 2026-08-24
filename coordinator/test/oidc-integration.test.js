/**
 * Single sign-on driven end to end against a real local identity provider.
 *
 * A small HTTP server plays the IdP: it publishes a discovery document and a
 * JWKS, and mints tokens with a private key it keeps. The coordinator is
 * pointed at it with nothing stubbed, so this exercises discovery, key
 * fetching, signature verification, group-to-scope mapping and the audit trail
 * the same way a real deployment would.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import WebSocket from 'ws';

const PORT = 8162;
const IDP_PORT = 8163;
const ISSUER = `http://127.0.0.1:${IDP_PORT}`;
const AUDIENCE = 'meshgpu';
const ROOT_KEY = 'root-key-for-oidc-tests';
const BASE = `http://127.0.0.1:${PORT}`;
const MODEL = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';

let server;
let idp;
let dataDir;
let signingKey;

// ---------------------------------------------------------------------------
// A minimal identity provider
// ---------------------------------------------------------------------------

function startIdp() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  signingKey = { privateKey, jwk: { ...publicKey.export({ format: 'jwk' }), kid: 'idp-1', use: 'sig', alg: 'RS256' } };

  const httpServer = http.createServer((req, res) => {
    res.setHeader('content-type', 'application/json');
    if (req.url.startsWith('/.well-known/openid-configuration')) {
      res.end(JSON.stringify({ issuer: ISSUER, jwks_uri: `${ISSUER}/jwks` }));
      return;
    }
    if (req.url.startsWith('/jwks')) {
      res.end(JSON.stringify({ keys: [signingKey.jwk] }));
      return;
    }
    res.statusCode = 404;
    res.end('{}');
  });

  return new Promise((resolve) => httpServer.listen(IDP_PORT, '127.0.0.1', () => resolve(httpServer)));
}

const b64 = (obj) => Buffer.from(JSON.stringify(obj)).toString('base64url');

/** Mint a token as the IdP would after a user signs in. */
function issueToken({ sub = 'alice', email = 'alice@example.com', groups = [], ...overrides } = {}) {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: 'RS256', typ: 'JWT', kid: 'idp-1' };
  const payload = {
    iss: ISSUER, aud: AUDIENCE, sub, email, groups,
    iat: now, exp: now + 3600, ...overrides,
  };
  const data = `${b64(header)}.${b64(payload)}`;
  const signature = crypto.sign('sha256', Buffer.from(data), signingKey.privateKey);
  return `${data}.${signature.toString('base64url')}`;
}

function call(pathname, { method = 'GET', body, token } = {}) {
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
  const res = await call(pathname, options);
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

const chat = (token, model = MODEL, content = 'hello') =>
  json('/v1/chat/completions', {
    method: 'POST', token,
    body: { model, messages: [{ role: 'user', content }] },
  });

beforeAll(async () => {
  idp = await startIdp();

  dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'meshgpu-oidc-'));
  process.env.MESH_PORT = String(PORT);
  process.env.MESH_TOKEN = ROOT_KEY;
  process.env.MESH_MDNS = 'off';
  process.env.MESH_DATA_DIR = dataDir;
  process.env.MESH_OIDC_ISSUER = ISSUER;
  process.env.MESH_OIDC_AUDIENCE = AUDIENCE;
  process.env.MESH_OIDC_SCOPE_MAP = 'mesh-admins=admin,chat,serve;mesh-users=chat';
  process.env.MESH_OIDC_DEFAULT_SCOPES = 'chat';
  process.env.MESH_OIDC_DAILY_LIMIT = '5';

  ({ server } = await import('../server.js'));
  if (!server.listening) await new Promise((resolve) => server.once('listening', resolve));
});

afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
  await new Promise((resolve) => idp.close(resolve));
  fs.rmSync(dataDir, { recursive: true, force: true });
  for (const name of Object.keys(process.env)) {
    if (name.startsWith('MESH_OIDC_')) delete process.env[name];
  }
});

describe('signing in with an IdP token', () => {
  it('accepts a token, discovering the key set on its own', async () => {
    const { status } = await chat(issueToken({ groups: ['mesh-users'] }));
    // 503 = past authentication and authorisation, no worker has the model.
    expect(status).toBe(503);
  });

  it('still accepts API keys alongside OIDC', async () => {
    expect((await chat(ROOT_KEY)).status).toBe(503);
  });

  it('reports OIDC as enabled to the admin console', async () => {
    const { body } = await json('/admin/api/overview', { token: ROOT_KEY });
    expect(body.oidc.enabled).toBe(true);
    expect(body.oidc.issuer).toBe(ISSUER);
  });
});

describe('group membership decides what a person may do', () => {
  it('grants the admin API to a member of the admin group', async () => {
    const { status } = await json('/admin/api/overview', {
      token: issueToken({ sub: 'boss', groups: ['mesh-admins'] }),
    });
    expect(status).toBe(200);
  });

  it('refuses the admin API to an ordinary user', async () => {
    const { status, body } = await json('/admin/api/keys', {
      token: issueToken({ sub: 'alice', groups: ['mesh-users'] }),
    });
    expect(status).toBe(403);
    expect(body.error.message).toMatch(/admin scope/);
  });

  it('gives an unmapped group only the default scopes', async () => {
    const token = issueToken({ sub: 'carol', groups: ['accounts-payable'] });
    expect((await chat(token)).status).toBe(503);            // chat: allowed
    expect((await json('/admin/api/keys', { token })).status).toBe(403); // admin: not
  });

  it('lets a mesh-admin connect a worker, and refuses an ordinary user', async () => {
    const connect = (token) => new Promise((resolve) => {
      const socket = new WebSocket(`ws://127.0.0.1:${PORT}/mesh?token=${encodeURIComponent(token)}&label=oidc`);
      socket.on('message', (raw) => {
        if (JSON.parse(String(raw)).type === 'welcome') { socket.close(); resolve('accepted'); }
      });
      socket.on('close', (code) => resolve(code === 1000 ? 'accepted' : code));
      socket.on('error', () => {});
    });

    expect(await connect(issueToken({ sub: 'boss', groups: ['mesh-admins'] }))).toBe('accepted');
    expect(await connect(issueToken({ sub: 'alice', groups: ['mesh-users'] }))).toBe(4403);
  });
});

describe('rejecting bad tokens', () => {
  it('refuses an expired token', async () => {
    const now = Math.floor(Date.now() / 1000);
    const { status } = await chat(issueToken({ groups: ['mesh-users'], exp: now - 7200 }));
    expect(status).toBe(401);
  });

  it('refuses a token from another issuer', async () => {
    const { status } = await chat(issueToken({ iss: 'https://evil.example.com' }));
    expect(status).toBe(401);
  });

  it('refuses a token minted for a different application', async () => {
    const { status } = await chat(issueToken({ aud: 'some-other-app' }));
    expect(status).toBe(401);
  });

  it('refuses a token signed by a key the IdP never published', async () => {
    const rogue = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
    const now = Math.floor(Date.now() / 1000);
    const header = { alg: 'RS256', typ: 'JWT', kid: 'idp-1' };
    const payload = { iss: ISSUER, aud: AUDIENCE, sub: 'attacker', exp: now + 3600, groups: ['mesh-admins'] };
    const data = `${b64(header)}.${b64(payload)}`;
    const signature = crypto.sign('sha256', Buffer.from(data), rogue.privateKey).toString('base64url');

    const { status } = await json('/admin/api/overview', { token: `${data}.${signature}` });
    expect(status).toBe(401);
  });

  it('does not explain why a token failed', async () => {
    const { body } = await chat(issueToken({ iss: 'https://evil.example.com' }));
    // Telling a caller which check failed tells them about the configuration.
    expect(body.error.message).toBe('missing or invalid credentials');
  });
});

describe('quota and audit follow the person', () => {
  it('applies the configured OIDC quota per subject', async () => {
    const token = issueToken({ sub: 'heavy-user', email: 'heavy@example.com', groups: ['mesh-users'] });

    for (let i = 0; i < 5; i += 1) {
      expect((await chat(token)).status).toBe(503);
    }
    const denied = await chat(token);
    expect(denied.status).toBe(429);
    expect(denied.body.error.code).toBe('rate_limit_exceeded');
  });

  it('keeps counting the same person across a token refresh', async () => {
    // A fresh token for the same subject must not reset the quota, or the
    // limit would be meaningless to anyone who can sign in twice.
    const refreshed = issueToken({ sub: 'heavy-user', email: 'heavy@example.com', groups: ['mesh-users'] });
    expect((await chat(refreshed)).status).toBe(429);
  });

  it('records the person and how they signed in', async () => {
    await chat(issueToken({ sub: 'audited', email: 'audited@example.com', groups: ['mesh-users'] }));

    const { body } = await json('/admin/api/audit?limit=50', { token: ROOT_KEY });
    const entry = body.entries.find((e) => e.keyName === 'audited@example.com');

    expect(entry).toBeDefined();
    expect(entry.keyId).toBe('oidc:audited');
    expect(entry.detail.via).toBe('oidc');
  });

  it('distinguishes an API key from an IdP identity in the log', async () => {
    await chat(ROOT_KEY);
    const { body } = await json('/admin/api/audit?limit=50', { token: ROOT_KEY });
    const byKey = body.entries.find((e) => e.keyName === 'MESH_TOKEN');
    expect(byKey.detail.via).toBe('api-key');
  });

  it('never writes a token or a prompt into the audit log', async () => {
    const token = issueToken({ sub: 'secretive', groups: ['mesh-users'] });
    await chat(token, MODEL, 'a confidential question');

    const raw = fs.readFileSync(path.join(dataDir, 'audit.jsonl'), 'utf8');
    expect(raw).not.toContain(token);
    expect(raw).not.toContain('a confidential question');
  });
});
