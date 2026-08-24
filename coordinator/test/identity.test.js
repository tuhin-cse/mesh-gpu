import { describe, expect, it, beforeEach } from 'vitest';

import { Store } from '../lib/store.js';
import { Identity, SCOPES, hasScope, hashKey, mayUseModel } from '../lib/identity.js';

describe('Identity', () => {
  let identity;
  let store;

  beforeEach(() => {
    store = new Store(null); // in-memory
    identity = new Identity(store);
  });

  it('issues a usable key and returns the plaintext exactly once', () => {
    const { key, record } = identity.create({ name: 'Tuhin' });

    expect(key).toMatch(/^mesh_[A-Za-z0-9_-]+$/);
    expect(record.name).toBe('Tuhin');
    expect(record.scopes).toEqual([SCOPES.CHAT]);
    // The plaintext must never come back from a lookup.
    expect(JSON.stringify(identity.list())).not.toContain(key);
  });

  it('never stores the plaintext, only its hash', () => {
    const { key } = identity.create({ name: 'Tuhin' });
    const serialised = JSON.stringify(store.state);

    expect(serialised).not.toContain(key);
    expect(serialised).toContain(hashKey(key));
  });

  it('verifies a real key and rejects everything else', () => {
    const { key } = identity.create({ name: 'Tuhin' });

    expect(identity.verify(key)?.name).toBe('Tuhin');
    expect(identity.verify('mesh_wrong')).toBeNull();
    expect(identity.verify('')).toBeNull();
    expect(identity.verify(null)).toBeNull();
    expect(identity.verify(undefined)).toBeNull();
  });

  it('stops accepting a revoked key without disturbing the others', () => {
    const alice = identity.create({ name: 'Alice' });
    const bob = identity.create({ name: 'Bob' });

    expect(identity.revoke(alice.record.id)).toBe(true);

    expect(identity.verify(alice.key)).toBeNull();
    expect(identity.verify(bob.key)?.name).toBe('Bob');
    // Revoking one person must not require rotating for everyone — the whole
    // reason named keys replaced the shared token.
    expect(identity.list()).toHaveLength(1);
    expect(identity.list({ includeRevoked: true })).toHaveLength(2);
  });

  it('reports a second revoke as a no-op', () => {
    const { record } = identity.create({ name: 'Alice' });
    expect(identity.revoke(record.id)).toBe(true);
    expect(identity.revoke(record.id)).toBe(false);
    expect(identity.revoke('nonexistent')).toBe(false);
  });

  it('rejects keys with no name or no valid scope', () => {
    expect(() => identity.create({ name: '   ' })).toThrow(/needs a name/);
    expect(() => identity.create({ name: 'x', scopes: [] })).toThrow(/at least one scope/);
    expect(() => identity.create({ name: 'x', scopes: ['superuser'] })).toThrow(/at least one scope/);
  });

  it('keeps only recognised scopes and drops duplicates', () => {
    const { record } = identity.create({
      name: 'mixed',
      scopes: [SCOPES.CHAT, SCOPES.CHAT, 'nonsense', SCOPES.ADMIN],
    });
    expect(record.scopes).toEqual([SCOPES.CHAT, SCOPES.ADMIN]);
  });

  it('updates policy without reissuing the key', () => {
    const { key, record } = identity.create({ name: 'Alice' });

    identity.update(record.id, { dailyRequestLimit: 500, scopes: [SCOPES.CHAT, SCOPES.SERVE] });

    const verified = identity.verify(key);
    expect(verified.dailyRequestLimit).toBe(500);
    expect(hasScope(verified, SCOPES.SERVE)).toBe(true);
  });

  it('refuses negative limits', () => {
    const { record } = identity.create({ name: 'Alice' });
    expect(() => identity.update(record.id, { dailyRequestLimit: -1 })).toThrow(/zero or more/);
  });

  it('tracks last use', () => {
    const { record } = identity.create({ name: 'Alice' });
    expect(identity.get(record.id).lastUsedAt).toBeNull();

    identity.markUsed(record.id);
    expect(identity.get(record.id).lastUsedAt).not.toBeNull();
  });

  it('knows whether any admin key remains', () => {
    expect(identity.hasAdmin()).toBe(false);

    const admin = identity.create({ name: 'root', scopes: [SCOPES.ADMIN] });
    expect(identity.hasAdmin()).toBe(true);

    identity.revoke(admin.record.id);
    expect(identity.hasAdmin()).toBe(false);
  });

  it('adopts a pre-shared token idempotently', () => {
    const first = identity.adopt('MESH_TOKEN_VALUE', 'bootstrap');
    const second = identity.adopt('MESH_TOKEN_VALUE', 'bootstrap');

    expect(first.id).toBe(second.id);
    expect(identity.list()).toHaveLength(1);
    expect(identity.verify('MESH_TOKEN_VALUE')?.scopes).toContain(SCOPES.ADMIN);
  });

  it('un-revokes an adopted token, so MESH_TOKEN always works after a restart', () => {
    const adopted = identity.adopt('TOKEN', 'bootstrap');
    identity.revoke(adopted.id);
    expect(identity.verify('TOKEN')).toBeNull();

    identity.adopt('TOKEN', 'bootstrap');
    expect(identity.verify('TOKEN')).not.toBeNull();
  });
});

describe('mayUseModel', () => {
  const anyModel = { allowedModels: null };
  const restricted = { allowedModels: ['qwen-1.5b'] };

  it('allows anything when no allowlist is set', () => {
    expect(mayUseModel(anyModel, 'llama-3.2-1b')).toBe(true);
  });

  it('enforces an allowlist', () => {
    expect(mayUseModel(restricted, 'qwen-1.5b')).toBe(true);
    expect(mayUseModel(restricted, 'llama-3.2-1b')).toBe(false);
  });

  it('lets a mesh-wide block list override a permissive key', () => {
    expect(mayUseModel(anyModel, 'banned', ['banned'])).toBe(false);
    expect(mayUseModel({ allowedModels: ['banned'] }, 'banned', ['banned'])).toBe(false);
  });
});
