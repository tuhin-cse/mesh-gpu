/**
 * identity.js
 *
 * Named API keys with scopes, replacing the single shared token.
 *
 * The shared token was adequate for a trusted LAN and inadequate for anything
 * with a compliance requirement: it cannot say who made a request, and
 * revoking one person means rotating for everyone. Named keys fix both, which
 * is the precondition for an audit log meaning anything.
 *
 * Keys are stored as SHA-256 hashes. A stolen state file therefore leaks who
 * has access, but not the credentials themselves.
 */

import crypto from 'node:crypto';

/** What a key is allowed to do. A key may hold several. */
export const SCOPES = Object.freeze({
  /** Call /v1/chat/completions and /v1/models. */
  CHAT: 'chat',
  /** Connect as a worker and serve jobs. */
  SERVE: 'serve',
  /** Manage keys, read the audit log, change settings. */
  ADMIN: 'admin',
});

const ALL_SCOPES = new Set(Object.values(SCOPES));
const KEY_PREFIX = 'mesh';

export function hashKey(plaintext) {
  return crypto.createHash('sha256').update(plaintext, 'utf8').digest('hex');
}

/** Generate a key. The plaintext is returned once and never stored. */
export function generateKey() {
  return `${KEY_PREFIX}_${crypto.randomBytes(24).toString('base64url')}`;
}

/**
 * Short, non-secret handle for a key, safe to show in logs and the console.
 * Derived from the hash so it is stable without revealing anything.
 */
export function keyIdFor(hash) {
  return hash.slice(0, 12);
}

export class Identity {
  /** @param {import('./store.js').Store} store */
  constructor(store) {
    this.store = store;
  }

  /**
   * Create a key. Returns `{ key, record }` where `key` is the only time the
   * plaintext exists — the caller must show it to the human immediately.
   *
   * @param {object} options
   * @param {string} options.name          Who or what this key is for.
   * @param {string[]} [options.scopes]
   * @param {string[]|null} [options.allowedModels]  null = any model.
   * @param {number} [options.dailyRequestLimit]     0 = unlimited.
   * @param {number} [options.requestsPerMinute]     0 = unlimited.
   */
  create({
    name,
    scopes = [SCOPES.CHAT],
    allowedModels = null,
    dailyRequestLimit = 0,
    requestsPerMinute = 0,
  }) {
    const trimmed = String(name ?? '').trim();
    if (trimmed.length === 0) throw badRequest('a key needs a name');
    if (trimmed.length > 80) throw badRequest('name must be 80 characters or fewer');

    const cleanScopes = normaliseScopes(scopes);
    if (cleanScopes.length === 0) throw badRequest('a key needs at least one scope');

    const plaintext = generateKey();
    const hash = hashKey(plaintext);
    const record = {
      id: keyIdFor(hash),
      hash,
      name: trimmed,
      scopes: cleanScopes,
      allowedModels: normaliseModels(allowedModels),
      dailyRequestLimit: nonNegativeInt(dailyRequestLimit),
      requestsPerMinute: nonNegativeInt(requestsPerMinute),
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
      revokedAt: null,
    };

    this.store.state.keys[record.id] = record;
    this.store.touch();
    return { key: plaintext, record: publicView(record) };
  }

  /**
   * Look up a key by its plaintext. Returns null for unknown or revoked keys —
   * the caller must not distinguish between the two in its response.
   */
  verify(plaintext) {
    if (typeof plaintext !== 'string' || plaintext.length === 0) return null;
    const hash = hashKey(plaintext);

    const candidate = Buffer.from(hash);
    for (const record of Object.values(this.store.state.keys)) {
      const stored = Buffer.from(String(record.hash ?? ''));
      // timingSafeEqual throws on a length mismatch, which only happens if the
      // state file was hand-edited. Skip rather than crash the coordinator.
      if (stored.length !== candidate.length) continue;
      if (crypto.timingSafeEqual(stored, candidate)) {
        if (record.revokedAt) return null;
        return record;
      }
    }
    return null;
  }

  /** Record that a key was used. Cheap enough to call per request. */
  markUsed(keyId) {
    const record = this.store.state.keys[keyId];
    if (!record) return;
    record.lastUsedAt = new Date().toISOString();
    this.store.touch();
  }

  /** Revoke a key. Returns false if it does not exist or was already revoked. */
  revoke(keyId) {
    const record = this.store.state.keys[keyId];
    if (!record || record.revokedAt) return false;
    record.revokedAt = new Date().toISOString();
    this.store.touch();
    return true;
  }

  /** Update the policy attached to a key. Scopes and limits only. */
  update(keyId, changes) {
    const record = this.store.state.keys[keyId];
    if (!record || record.revokedAt) return null;

    if (changes.scopes !== undefined) {
      const scopes = normaliseScopes(changes.scopes);
      if (scopes.length === 0) throw badRequest('a key needs at least one scope');
      record.scopes = scopes;
    }
    if (changes.allowedModels !== undefined) {
      record.allowedModels = normaliseModels(changes.allowedModels);
    }
    if (changes.dailyRequestLimit !== undefined) {
      record.dailyRequestLimit = nonNegativeInt(changes.dailyRequestLimit);
    }
    if (changes.requestsPerMinute !== undefined) {
      record.requestsPerMinute = nonNegativeInt(changes.requestsPerMinute);
    }
    if (changes.name !== undefined) {
      const trimmed = String(changes.name).trim();
      if (trimmed.length === 0) throw badRequest('a key needs a name');
      record.name = trimmed.slice(0, 80);
    }

    this.store.touch();
    return publicView(record);
  }

  get(keyId) {
    const record = this.store.state.keys[keyId];
    return record ? publicView(record) : null;
  }

  list({ includeRevoked = false } = {}) {
    return Object.values(this.store.state.keys)
      .filter((record) => includeRevoked || !record.revokedAt)
      .map(publicView)
      .sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1));
  }

  /** True when at least one usable admin key exists. */
  hasAdmin() {
    return Object.values(this.store.state.keys).some(
      (record) => !record.revokedAt && record.scopes.includes(SCOPES.ADMIN),
    );
  }

  /**
   * Adopt a pre-shared token as an admin key, for MESH_TOKEN compatibility and
   * for first-run bootstrap. Idempotent: re-running with the same token does
   * not create a second record.
   */
  adopt(plaintext, name) {
    const hash = hashKey(plaintext);
    const id = keyIdFor(hash);
    const existing = this.store.state.keys[id];
    if (existing) {
      existing.revokedAt = null;
      this.store.touch();
      return publicView(existing);
    }

    const record = {
      id,
      hash,
      name,
      scopes: [SCOPES.CHAT, SCOPES.SERVE, SCOPES.ADMIN],
      allowedModels: null,
      dailyRequestLimit: 0,
      requestsPerMinute: 0,
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
      revokedAt: null,
    };
    this.store.state.keys[id] = record;
    this.store.touch();
    return publicView(record);
  }
}

/** Whether a key may use a model, honouring both its allowlist and the block list. */
export function mayUseModel(record, model, blockedModels = []) {
  if (blockedModels.includes(model)) return false;
  if (record.allowedModels === null) return true;
  return record.allowedModels.includes(model);
}

export function hasScope(record, scope) {
  return Array.isArray(record?.scopes) && record.scopes.includes(scope);
}

/** The shape safe to return over the admin API — never includes the hash. */
function publicView(record) {
  const { hash, ...rest } = record;
  void hash;
  return { ...rest };
}

function normaliseScopes(scopes) {
  if (!Array.isArray(scopes)) return [];
  return [...new Set(scopes.filter((scope) => ALL_SCOPES.has(scope)))];
}

function normaliseModels(models) {
  if (models === null || models === undefined) return null;
  if (!Array.isArray(models)) throw badRequest('allowedModels must be an array or null');
  const cleaned = models.filter((model) => typeof model === 'string' && model.length > 0);
  return cleaned.length > 0 ? [...new Set(cleaned)] : null;
}

function nonNegativeInt(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) throw badRequest('limits must be zero or more');
  return Math.floor(number);
}

function badRequest(message) {
  return Object.assign(new Error(message), { status: 400, code: 'invalid_request_error' });
}
