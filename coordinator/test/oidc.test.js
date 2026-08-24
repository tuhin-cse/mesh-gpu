/**
 * OIDC token verification, including the forgeries it has to refuse.
 *
 * Hand-rolled JWT verification is where a good number of CVEs live, so the
 * known attacks are tested explicitly rather than assumed away: `alg: none`,
 * algorithm confusion, key injection through the token header, replayed
 * expired tokens, and tokens minted for a different application.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import crypto from 'node:crypto';

import { OidcVerifier, OidcError, decodeSegments, parseScopeMap } from '../lib/oidc.js';

const ISSUER = 'https://idp.example.com';
const AUDIENCE = 'meshgpu';

// ---------------------------------------------------------------------------
// Test IdP
// ---------------------------------------------------------------------------

function makeRsaKey(kid) {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  return { kid, publicKey, privateKey, jwk: { ...publicKey.export({ format: 'jwk' }), kid, use: 'sig', alg: 'RS256' } };
}

function makeEcKey(kid) {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ec', { namedCurve: 'P-256' });
  return { kid, publicKey, privateKey, jwk: { ...publicKey.export({ format: 'jwk' }), kid, use: 'sig', alg: 'ES256' } };
}

const b64 = (obj) => Buffer.from(JSON.stringify(obj)).toString('base64url');

/** Sign a token with a real key. */
function sign(key, { alg = 'RS256', header = {}, claims = {} } = {}) {
  const now = Math.floor(Date.now() / 1000);
  const fullHeader = { alg, typ: 'JWT', kid: key.kid, ...header };
  const payload = {
    iss: ISSUER,
    aud: AUDIENCE,
    sub: 'user-123',
    email: 'alice@example.com',
    iat: now,
    exp: now + 3600,
    ...claims,
  };
  const data = `${b64(fullHeader)}.${b64(payload)}`;
  const options = alg.startsWith('ES') ? { key: key.privateKey, dsaEncoding: 'ieee-p1363' } : key.privateKey;
  const hash = { RS256: 'sha256', ES256: 'sha256', RS384: 'sha384' }[alg] ?? 'sha256';
  const signature = crypto.sign(hash, Buffer.from(data), options);
  return `${data}.${signature.toString('base64url')}`;
}

/** Token with an arbitrary signature string — for forgery tests. */
function forge({ header, claims, signature = '' }) {
  const now = Math.floor(Date.now() / 1000);
  return `${b64(header)}.${b64({ iss: ISSUER, aud: AUDIENCE, sub: 'attacker', exp: now + 3600, ...claims })}.${signature}`;
}

function verifierFor(keys, overrides = {}) {
  const jwks = { keys: keys.map((k) => k.jwk) };
  return new OidcVerifier({
    issuer: ISSUER,
    audience: AUDIENCE,
    staticJwks: jwks,
    scopeMap: { 'mesh-admins': ['admin', 'chat', 'serve'], 'mesh-users': ['chat'] },
    ...overrides,
  });
}

// ---------------------------------------------------------------------------

describe('decodeSegments', () => {
  it('rejects anything that is not a compact JWS', () => {
    for (const bad of ['', 'a.b', 'a.b.c.d', 42, null]) {
      expect(() => decodeSegments(bad)).toThrow(OidcError);
    }
  });

  it('rejects segments that are not JSON', () => {
    expect(() => decodeSegments('bm90LWpzb24.bm90LWpzb24.sig')).toThrow(/not valid JSON/);
  });
});

describe('happy path', () => {
  let key;
  let verifier;

  beforeEach(() => {
    key = makeRsaKey('key-1');
    verifier = verifierFor([key]);
  });

  it('accepts a well-formed token and builds a principal', async () => {
    const principal = await verifier.verify(sign(key, { claims: { groups: ['mesh-users'] } }));

    expect(principal.id).toBe('oidc:user-123');
    expect(principal.name).toBe('alice@example.com');
    expect(principal.scopes).toEqual(['chat']);
    expect(principal.via).toBe('oidc');
  });

  it('maps group membership onto mesh scopes', async () => {
    const principal = await verifier.verify(sign(key, { claims: { groups: ['mesh-admins'] } }));
    expect(principal.scopes.sort()).toEqual(['admin', 'chat', 'serve']);
  });

  it('unions the scopes of every matching group', async () => {
    const principal = await verifier.verify(
      sign(key, { claims: { groups: ['mesh-users', 'mesh-admins', 'unrelated'] } }),
    );
    expect(principal.scopes.sort()).toEqual(['admin', 'chat', 'serve']);
  });

  it('falls back to default scopes when no group matches', async () => {
    const principal = await verifier.verify(sign(key, { claims: { groups: ['finance'] } }));
    expect(principal.scopes).toEqual(['chat']);
  });

  it('gives the same principal id across token refreshes, so quota follows the person', async () => {
    const first = await verifier.verify(sign(key));
    const second = await verifier.verify(sign(key, { claims: { iat: Math.floor(Date.now() / 1000) + 1 } }));
    expect(first.id).toBe(second.id);
  });

  it('namespaces the id so an IdP subject cannot collide with an API key', async () => {
    const principal = await verifier.verify(sign(key, { claims: { sub: 'abc123' } }));
    expect(principal.id).toBe('oidc:abc123');
  });

  it('accepts ES256 when configured', async () => {
    const ec = makeEcKey('ec-1');
    const ecVerifier = verifierFor([ec], { allowedAlgorithms: ['ES256'] });
    const principal = await ecVerifier.verify(sign(ec, { alg: 'ES256' }));
    expect(principal.subject).toBe('user-123');
  });

  it('accepts an array audience containing this mesh', async () => {
    const principal = await verifier.verify(sign(key, { claims: { aud: ['other-app', AUDIENCE] } }));
    expect(principal.subject).toBe('user-123');
  });
});

describe('forgery resistance', () => {
  let key;
  let verifier;

  beforeEach(() => {
    key = makeRsaKey('key-1');
    verifier = verifierFor([key]);
  });

  it('rejects alg: none', async () => {
    const token = forge({ header: { alg: 'none', kid: 'key-1' } });
    await expect(verifier.verify(token)).rejects.toThrow(/algorithm none is not accepted/);
  });

  it('rejects the HS256 algorithm-confusion attack', async () => {
    // The attack: re-sign with HMAC using the RSA *public* key as the secret.
    // A verifier that picks its algorithm from the token header accepts it.
    const publicPem = key.publicKey.export({ type: 'spki', format: 'pem' });
    const header = { alg: 'HS256', typ: 'JWT', kid: 'key-1' };
    const now = Math.floor(Date.now() / 1000);
    const payload = { iss: ISSUER, aud: AUDIENCE, sub: 'attacker', exp: now + 3600, groups: ['mesh-admins'] };
    const data = `${b64(header)}.${b64(payload)}`;
    const mac = crypto.createHmac('sha256', publicPem).update(data).digest('base64url');

    await expect(verifier.verify(`${data}.${mac}`)).rejects.toThrow(/HS256 is not accepted/);
  });

  it('rejects a token whose header supplies its own key', async () => {
    // `jwk` and `jku` let a token nominate the key that verifies it, which is
    // a complete bypass. Their presence is fatal, not merely ignored.
    const attacker = makeRsaKey('key-1');
    const token = sign(attacker, { header: { jwk: attacker.jwk } });
    await expect(verifier.verify(token)).rejects.toThrow(/supply its own key/);

    const jkuToken = sign(attacker, { header: { jku: 'https://attacker.example/keys' } });
    await expect(verifier.verify(jkuToken)).rejects.toThrow(/supply its own key/);
  });

  it('rejects a signature made by a key the mesh does not trust', async () => {
    const attacker = makeRsaKey('key-1'); // same kid, different key
    await expect(verifier.verify(sign(attacker))).rejects.toThrow(/does not verify/);
  });

  it('rejects a tampered payload', async () => {
    const token = sign(key, { claims: { groups: ['mesh-users'] } });
    const [header, , signature] = token.split('.');
    const escalated = b64({
      iss: ISSUER, aud: AUDIENCE, sub: 'user-123',
      exp: Math.floor(Date.now() / 1000) + 3600, groups: ['mesh-admins'],
    });

    await expect(verifier.verify(`${header}.${escalated}.${signature}`)).rejects.toThrow(/does not verify/);
  });

  it('rejects an unknown kid rather than trying every key', async () => {
    const token = sign({ ...key, kid: 'rotated-out' });
    await expect(verifier.verify(token)).rejects.toThrow(/no trusted key matches/);
  });

  it('refuses to guess when a token has no kid and several keys exist', async () => {
    const multi = verifierFor([key, makeRsaKey('key-2')]);
    const token = sign(key, { header: { kid: undefined } });
    await expect(multi.verify(token)).rejects.toThrow(/no trusted key matches/);
  });

  it('will not verify an RSA algorithm against an EC key', async () => {
    const ec = makeEcKey('key-1');
    const confused = verifierFor([ec]); // EC key published under an RS256 kid
    await expect(verifier.verify(sign(key))).resolves.toBeTruthy();
    await expect(confused.verify(sign(key))).rejects.toThrow(/not usable for this algorithm/);
  });

  it('ignores encryption keys when selecting a verification key', async () => {
    const encOnly = { ...key, jwk: { ...key.jwk, use: 'enc' } };
    const encVerifier = verifierFor([encOnly]);
    await expect(encVerifier.verify(sign(key))).rejects.toThrow(/no trusted key matches/);
  });
});

describe('claim validation', () => {
  let key;
  let verifier;

  beforeEach(() => {
    key = makeRsaKey('key-1');
    verifier = verifierFor([key]);
  });

  it('rejects an expired token', async () => {
    const now = Math.floor(Date.now() / 1000);
    await expect(verifier.verify(sign(key, { claims: { exp: now - 3600 } }))).rejects.toThrow(/expired/);
  });

  it('rejects a token with no expiry at all', async () => {
    const token = sign(key, { claims: { exp: undefined } });
    await expect(verifier.verify(token)).rejects.toThrow(/no expiry/);
  });

  it('allows a little clock skew, but not a lot', async () => {
    const now = Math.floor(Date.now() / 1000);
    const slightlyStale = sign(key, { claims: { exp: now - 30 } });
    await expect(verifier.verify(slightlyStale)).resolves.toBeTruthy();

    const properlyExpired = sign(key, { claims: { exp: now - 600 } });
    await expect(verifier.verify(properlyExpired)).rejects.toThrow(/expired/);
  });

  it('rejects a token from another issuer', async () => {
    const token = sign(key, { claims: { iss: 'https://evil.example.com' } });
    await expect(verifier.verify(token)).rejects.toThrow(/not this mesh's issuer/);
  });

  it('rejects a token minted for a different application', async () => {
    const token = sign(key, { claims: { aud: 'some-other-app' } });
    await expect(verifier.verify(token)).rejects.toThrow(/audience does not include this mesh/);
  });

  it('rejects a token that is not valid yet', async () => {
    const now = Math.floor(Date.now() / 1000);
    await expect(verifier.verify(sign(key, { claims: { nbf: now + 600 } }))).rejects.toThrow(/not valid yet/);
  });

  it('rejects a token with no subject', async () => {
    await expect(verifier.verify(sign(key, { claims: { sub: undefined } }))).rejects.toThrow(/no subject/);
  });

  it('treats a trailing slash on the issuer as the same issuer', async () => {
    const trailing = verifierFor([key], { issuer: `${ISSUER}/` });
    await expect(trailing.verify(sign(key))).resolves.toBeTruthy();
  });
});

describe('JWKS retrieval', () => {
  it('fetches and caches the key set', async () => {
    const key = makeRsaKey('key-1');
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => ({ keys: [key.jwk] }) }));

    const verifier = new OidcVerifier({
      issuer: ISSUER, audience: AUDIENCE,
      jwksUri: `${ISSUER}/jwks`, fetchImpl,
    });

    await verifier.verify(sign(key));
    await verifier.verify(sign(key));
    expect(fetchImpl).toHaveBeenCalledTimes(1); // cached
  });

  it('discovers jwks_uri when it is not configured', async () => {
    const key = makeRsaKey('key-1');
    const fetchImpl = vi.fn(async (url) => {
      if (String(url).includes('openid-configuration')) {
        return { ok: true, json: async () => ({ jwks_uri: `${ISSUER}/keys` }) };
      }
      return { ok: true, json: async () => ({ keys: [key.jwk] }) };
    });

    const verifier = new OidcVerifier({ issuer: ISSUER, audience: AUDIENCE, fetchImpl });
    await expect(verifier.verify(sign(key))).resolves.toBeTruthy();
    expect(fetchImpl.mock.calls[0][0]).toContain('/.well-known/openid-configuration');
  });

  it('picks up a rotated key by refetching once', async () => {
    const oldKey = makeRsaKey('old');
    const newKey = makeRsaKey('new');
    let published = [oldKey.jwk];
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => ({ keys: published }) }));

    let clock = Date.now();
    const verifier = new OidcVerifier({
      issuer: ISSUER, audience: AUDIENCE, jwksUri: `${ISSUER}/jwks`,
      fetchImpl, now: () => clock,
    });

    await verifier.verify(sign(oldKey));
    published = [newKey.jwk];
    clock += 20_000; // past the minimum refetch interval

    await expect(verifier.verify(sign(newKey))).resolves.toBeTruthy();
    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1);
  });

  it('does not refetch on every junk kid, which would be a DoS through us', async () => {
    const key = makeRsaKey('key-1');
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => ({ keys: [key.jwk] }) }));

    let clock = Date.now();
    const verifier = new OidcVerifier({
      issuer: ISSUER, audience: AUDIENCE, jwksUri: `${ISSUER}/jwks`,
      fetchImpl, now: () => clock,
    });

    await verifier.verify(sign(key));
    const before = fetchImpl.mock.calls.length;

    for (let i = 0; i < 20; i += 1) {
      await verifier.verify(sign({ ...key, kid: `junk-${i}` })).catch(() => {});
    }
    expect(fetchImpl.mock.calls.length).toBe(before);
  });

  it('serves a stale cache rather than locking everyone out when the IdP blips', async () => {
    const key = makeRsaKey('key-1');
    let healthy = true;
    const fetchImpl = vi.fn(async () => {
      if (!healthy) throw new Error('IdP unreachable');
      return { ok: true, json: async () => ({ keys: [key.jwk] }) };
    });

    let clock = Date.now();
    const verifier = new OidcVerifier({
      issuer: ISSUER, audience: AUDIENCE, jwksUri: `${ISSUER}/jwks`,
      fetchImpl, now: () => clock, cacheTtlMs: 1000,
    });

    await verifier.verify(sign(key));
    healthy = false;
    clock += 60_000; // cache is stale and the IdP is down

    await expect(verifier.verify(sign(key))).resolves.toBeTruthy();
  });

  it('fails cleanly on a cold start with an unreachable IdP', async () => {
    const fetchImpl = vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) }));
    const verifier = new OidcVerifier({
      issuer: ISSUER, audience: AUDIENCE, jwksUri: `${ISSUER}/jwks`, fetchImpl,
    });
    const key = makeRsaKey('key-1');
    await expect(verifier.verify(sign(key))).rejects.toThrow(/JWKS fetch failed/);
  });

  it('works entirely offline from a static key set', async () => {
    const key = makeRsaKey('key-1');
    const fetchImpl = vi.fn();
    const verifier = verifierFor([key], { fetchImpl });

    await expect(verifier.verify(sign(key))).resolves.toBeTruthy();
    // An air-gapped mesh cannot reach an IdP; it must never try.
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});

describe('configuration', () => {
  it('parses a group-to-scope map', () => {
    expect(parseScopeMap('admins=admin,chat;users=chat')).toEqual({
      admins: ['admin', 'chat'],
      users: ['chat'],
    });
    expect(parseScopeMap('')).toEqual({});
    expect(parseScopeMap(undefined)).toEqual({});
  });

  it('is off unless an issuer is configured', () => {
    expect(OidcVerifier.fromEnv({})).toBeNull();
    expect(OidcVerifier.fromEnv({ MESH_OIDC_ISSUER: ISSUER })).toBeInstanceOf(OidcVerifier);
  });

  it('reads scopes, algorithms and limits from the environment', () => {
    const verifier = OidcVerifier.fromEnv({
      MESH_OIDC_ISSUER: ISSUER,
      MESH_OIDC_AUDIENCE: 'meshgpu,other',
      MESH_OIDC_SCOPE_MAP: 'admins=admin',
      MESH_OIDC_DEFAULT_SCOPES: 'chat,serve',
      MESH_OIDC_ALGORITHMS: 'RS256,ES256',
      MESH_OIDC_DAILY_LIMIT: '250',
    });

    expect(verifier.audiences).toEqual(['meshgpu', 'other']);
    expect(verifier.scopeMap).toEqual({ admins: ['admin'] });
    expect(verifier.defaultScopes).toEqual(['chat', 'serve']);
    expect(verifier.dailyRequestLimit).toBe(250);
  });

  it('refuses to start with no usable algorithm', () => {
    expect(() => new OidcVerifier({ issuer: ISSUER, allowedAlgorithms: ['HS256'] }))
      .toThrow(/no usable OIDC signature algorithms/);
  });

  it('distinguishes a JWT from a MeshGPU API key', () => {
    expect(OidcVerifier.looksLikeJwt('a.b.c')).toBe(true);
    expect(OidcVerifier.looksLikeJwt('mesh_abc123')).toBe(false);
    expect(OidcVerifier.looksLikeJwt('mesh_a.b.c')).toBe(false);
    expect(OidcVerifier.looksLikeJwt(null)).toBe(false);
  });
});
