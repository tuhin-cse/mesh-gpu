/**
 * auth.js
 *
 * One shared token guards both the HTTP API and the worker WebSocket. It is
 * generated at startup and printed unless MESH_TOKEN is set, so the default
 * posture is authenticated rather than open — an unauthenticated mesh on an
 * office LAN is a prompt-exfiltration hazard, not a convenience.
 *
 * This is deliberately modest: a shared secret on a trusted subnet. It is not
 * per-user identity, and it is not a substitute for SSO.
 */

import crypto from 'node:crypto';

export function resolveToken(envToken) {
  const token = (envToken ?? '').trim();
  if (token.length > 0) return { token, generated: false };
  // Six bytes of base32-ish text: short enough to type, long enough that
  // guessing it over a LAN is not worth anyone's afternoon.
  return { token: crypto.randomBytes(9).toString('base64url'), generated: true };
}

/**
 * Constant-time comparison so a token cannot be recovered by timing the
 * rejection.
 *
 * @param {string|null|undefined} candidate
 * @param {string} expected
 */
export function tokenMatches(candidate, expected) {
  if (typeof candidate !== 'string' || candidate.length === 0) return false;
  const a = Buffer.from(candidate);
  const b = Buffer.from(expected);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

/**
 * Pull a bearer token out of a request, accepting the header forms real
 * clients send.
 *
 * @param {import('node:http').IncomingMessage} req
 */
export function bearerFrom(req) {
  const header = req.headers.authorization;
  if (typeof header === 'string' && header.toLowerCase().startsWith('bearer ')) {
    return header.slice(7).trim();
  }
  const apiKey = req.headers['x-api-key'];
  if (typeof apiKey === 'string') return apiKey.trim();
  return null;
}
