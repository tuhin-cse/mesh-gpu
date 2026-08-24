/**
 * oidc.js
 *
 * Accepts JWT bearer tokens issued by an organisation's identity provider, so
 * access follows the directory instead of a key someone pasted into Slack.
 * This is the difference between "we have API keys" and "we integrate with
 * your IdP", which is the sentence a procurement conversation turns on.
 *
 * Verification is done directly against Node's crypto rather than pulling in a
 * JWT library, because the coordinator is meant to be droppable onto a machine
 * with no internet and a small dependency tree is part of that. Hand-rolled
 * JWT verification is also where a good number of CVEs live, so the specific
 * attacks are named at the checks that stop them, and each has a test.
 *
 * Air-gapped meshes cannot reach an IdP's JWKS endpoint. `staticJwks` accepts
 * a key set from a file, exported from the IdP and carried across by hand.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';

/**
 * Signature algorithms this verifier will accept.
 *
 * Asymmetric only, deliberately. Allowing an HMAC algorithm alongside RSA
 * enables the classic confusion attack: an attacker re-signs a token with
 * HS256 using the *public* key as the shared secret, and a verifier that
 * picks its algorithm from the token's own header accepts it.
 */
const ALGORITHMS = Object.freeze({
  RS256: { hash: 'sha256', kty: 'RSA' },
  RS384: { hash: 'sha384', kty: 'RSA' },
  RS512: { hash: 'sha512', kty: 'RSA' },
  PS256: { hash: 'sha256', kty: 'RSA', padding: crypto.constants.RSA_PKCS1_PSS_PADDING },
  PS384: { hash: 'sha384', kty: 'RSA', padding: crypto.constants.RSA_PKCS1_PSS_PADDING },
  PS512: { hash: 'sha512', kty: 'RSA', padding: crypto.constants.RSA_PKCS1_PSS_PADDING },
  ES256: { hash: 'sha256', kty: 'EC', dsaEncoding: 'ieee-p1363' },
  ES384: { hash: 'sha384', kty: 'EC', dsaEncoding: 'ieee-p1363' },
  ES512: { hash: 'sha512', kty: 'EC', dsaEncoding: 'ieee-p1363' },
});

const DEFAULT_ALLOWED = ['RS256', 'ES256'];
const DEFAULT_CACHE_TTL_MS = 10 * 60 * 1000;
const DEFAULT_SKEW_SECONDS = 60;
/** Refuse to refetch JWKS more often than this, so a bad token cannot be a DoS. */
const MIN_REFETCH_INTERVAL_MS = 10_000;

export class OidcError extends Error {
  constructor(message, code = 'invalid_token') {
    super(message);
    this.code = code;
  }
}

/** Split a compact JWS without trusting any of it yet. */
export function decodeSegments(token) {
  if (typeof token !== 'string') throw new OidcError('token must be a string');
  const parts = token.split('.');
  if (parts.length !== 3) throw new OidcError('not a compact JWS (expected three segments)');

  const [headerB64, payloadB64, signatureB64] = parts;
  let header;
  let payload;
  try {
    header = JSON.parse(Buffer.from(headerB64, 'base64url').toString('utf8'));
    payload = JSON.parse(Buffer.from(payloadB64, 'base64url').toString('utf8'));
  } catch {
    throw new OidcError('token header or payload is not valid JSON');
  }
  if (typeof header !== 'object' || header === null) throw new OidcError('bad token header');
  if (typeof payload !== 'object' || payload === null) throw new OidcError('bad token payload');

  return {
    header,
    payload,
    signature: Buffer.from(signatureB64, 'base64url'),
    signedData: Buffer.from(`${headerB64}.${payloadB64}`, 'utf8'),
  };
}

export class OidcVerifier {
  /**
   * @param {object} options
   * @param {string} options.issuer            Expected `iss`, matched exactly.
   * @param {string|string[]} options.audience Expected `aud`.
   * @param {string} [options.jwksUri]         Skips discovery when provided.
   * @param {object} [options.staticJwks]      Key set from a file (air-gapped).
   * @param {string} [options.groupsClaim]     Claim holding group membership.
   * @param {Record<string,string[]>} [options.scopeMap]  group -> mesh scopes.
   * @param {string[]} [options.defaultScopes] Scopes when no group matches.
   * @param {string[]} [options.allowedAlgorithms]
   * @param {number} [options.clockSkewSeconds]
   * @param {number} [options.dailyRequestLimit]
   * @param {number} [options.requestsPerMinute]
   * @param {string[]|null} [options.allowedModels]
   * @param {typeof fetch} [options.fetchImpl] Injectable for tests.
   * @param {() => number} [options.now]
   */
  constructor(options) {
    if (!options.issuer) throw new Error('OIDC needs an issuer');

    this.issuer = String(options.issuer).replace(/\/$/, '');
    this.audiences = toArray(options.audience);
    this.jwksUri = options.jwksUri ?? null;
    this.groupsClaim = options.groupsClaim ?? 'groups';
    this.scopeMap = options.scopeMap ?? {};
    this.defaultScopes = options.defaultScopes ?? ['chat'];
    this.clockSkewSeconds = options.clockSkewSeconds ?? DEFAULT_SKEW_SECONDS;
    this.cacheTtlMs = options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
    this.dailyRequestLimit = options.dailyRequestLimit ?? 0;
    this.requestsPerMinute = options.requestsPerMinute ?? 0;
    this.allowedModels = options.allowedModels ?? null;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
    this.now = options.now ?? (() => Date.now());

    const requested = options.allowedAlgorithms ?? DEFAULT_ALLOWED;
    this.allowedAlgorithms = requested.filter((alg) => Object.hasOwn(ALGORITHMS, alg));
    if (this.allowedAlgorithms.length === 0) {
      throw new Error('no usable OIDC signature algorithms configured');
    }

    this.staticJwks = options.staticJwks ?? null;
    this.cachedKeys = this.staticJwks ? indexJwks(this.staticJwks) : null;
    this.cachedAt = this.staticJwks ? Infinity : 0;
    this.lastFetchAttempt = 0;
    this.inFlight = null;
  }

  /** Build a verifier from environment variables, or null if OIDC is off. */
  static fromEnv(env = process.env) {
    if (!env.MESH_OIDC_ISSUER) return null;

    let staticJwks = null;
    if (env.MESH_OIDC_JWKS_FILE) {
      staticJwks = JSON.parse(fs.readFileSync(env.MESH_OIDC_JWKS_FILE, 'utf8'));
    }

    return new OidcVerifier({
      issuer: env.MESH_OIDC_ISSUER,
      audience: env.MESH_OIDC_AUDIENCE ? env.MESH_OIDC_AUDIENCE.split(',').map((a) => a.trim()) : [],
      jwksUri: env.MESH_OIDC_JWKS_URI || undefined,
      staticJwks,
      groupsClaim: env.MESH_OIDC_GROUPS_CLAIM || 'groups',
      scopeMap: parseScopeMap(env.MESH_OIDC_SCOPE_MAP),
      defaultScopes: (env.MESH_OIDC_DEFAULT_SCOPES ?? 'chat').split(',').map((s) => s.trim()).filter(Boolean),
      allowedAlgorithms: env.MESH_OIDC_ALGORITHMS
        ? env.MESH_OIDC_ALGORITHMS.split(',').map((a) => a.trim())
        : undefined,
      dailyRequestLimit: Number(env.MESH_OIDC_DAILY_LIMIT ?? 0) || 0,
      requestsPerMinute: Number(env.MESH_OIDC_PER_MINUTE ?? 0) || 0,
    });
  }

  /** True when this looks like a JWT rather than a MeshGPU API key. */
  static looksLikeJwt(token) {
    return typeof token === 'string' && token.split('.').length === 3 && !token.startsWith('mesh_');
  }

  /**
   * Verify a token and return a principal shaped like an API key record, so
   * quota, allowlist and audit code paths need no special case.
   *
   * Throws OidcError with a specific reason; the caller decides how much of
   * that to reveal.
   */
  async verify(token) {
    const { header, payload, signature, signedData } = decodeSegments(token);

    // `alg: none` and every symmetric algorithm are rejected here, before any
    // key lookup. This is the check that stops the two classic forgeries.
    if (!this.allowedAlgorithms.includes(header.alg)) {
      throw new OidcError(`signature algorithm ${header.alg ?? 'none'} is not accepted`);
    }
    const spec = ALGORITHMS[header.alg];

    // A token must never be allowed to supply its own verification key. `jwk`
    // and `jku` in the header are exactly that, so their presence is fatal
    // rather than merely ignored.
    if (header.jwk !== undefined || header.jku !== undefined) {
      throw new OidcError('token header attempts to supply its own key');
    }

    const key = await this.keyFor(header.kid, spec.kty);
    if (!key) throw new OidcError(`no trusted key matches kid "${header.kid ?? '(none)'}"`);

    const verifyOptions = { key };
    if (spec.padding) verifyOptions.padding = spec.padding;
    if (spec.dsaEncoding) verifyOptions.dsaEncoding = spec.dsaEncoding;

    let signatureValid = false;
    try {
      signatureValid = crypto.verify(spec.hash, signedData, verifyOptions, signature);
    } catch {
      signatureValid = false;
    }
    if (!signatureValid) throw new OidcError('signature does not verify');

    // Claims are only trusted after the signature has been checked.
    this.assertClaims(payload);
    return this.principalFrom(payload);
  }

  assertClaims(payload) {
    const nowSeconds = Math.floor(this.now() / 1000);
    const skew = this.clockSkewSeconds;

    const issuer = String(payload.iss ?? '').replace(/\/$/, '');
    if (issuer !== this.issuer) {
      throw new OidcError(`token issuer "${payload.iss}" is not this mesh's issuer`);
    }

    if (this.audiences.length > 0) {
      const tokenAudiences = toArray(payload.aud);
      const matches = tokenAudiences.some((aud) => this.audiences.includes(aud));
      if (!matches) {
        // Without this a token minted for a different application in the same
        // tenant would be accepted here.
        throw new OidcError('token audience does not include this mesh');
      }
    }

    if (payload.exp === undefined) throw new OidcError('token has no expiry');
    if (typeof payload.exp !== 'number' || nowSeconds > payload.exp + skew) {
      throw new OidcError('token has expired');
    }
    if (payload.nbf !== undefined && typeof payload.nbf === 'number' && nowSeconds + skew < payload.nbf) {
      throw new OidcError('token is not valid yet');
    }
    if (payload.iat !== undefined && typeof payload.iat === 'number' && payload.iat > nowSeconds + skew) {
      throw new OidcError('token was issued in the future');
    }
    if (typeof payload.sub !== 'string' || payload.sub.length === 0) {
      throw new OidcError('token has no subject');
    }
  }

  /** Map verified claims onto the principal shape the rest of the server uses. */
  principalFrom(payload) {
    const groups = toArray(payload[this.groupsClaim]).filter((g) => typeof g === 'string');

    const scopes = new Set();
    for (const group of groups) {
      for (const scope of this.scopeMap[group] ?? []) scopes.add(scope);
    }
    if (scopes.size === 0) for (const scope of this.defaultScopes) scopes.add(scope);

    const name =
      payload.email ?? payload.preferred_username ?? payload.name ?? payload.sub;

    return {
      // Namespaced so an IdP subject can never collide with an API key id,
      // and so quota counters follow the person across token refreshes.
      id: `oidc:${payload.sub}`,
      name: String(name),
      scopes: [...scopes],
      allowedModels: this.allowedModels,
      dailyRequestLimit: this.dailyRequestLimit,
      requestsPerMinute: this.requestsPerMinute,
      via: 'oidc',
      subject: payload.sub,
      groups,
    };
  }

  /** Resolve a signing key, refetching once if the kid is unknown. */
  async keyFor(kid, expectedKty) {
    let keys = await this.keys();
    let jwk = pickKey(keys, kid);

    // An unknown kid usually means the IdP rotated. Refetch, but not more
    // often than MIN_REFETCH_INTERVAL_MS, or a token with a junk kid becomes
    // a way to hammer the IdP through us.
    if (!jwk && !this.staticJwks && this.now() - this.lastFetchAttempt > MIN_REFETCH_INTERVAL_MS) {
      this.cachedAt = 0;
      keys = await this.keys();
      jwk = pickKey(keys, kid);
    }
    if (!jwk) return null;

    // The key's own type must match what the algorithm requires — the second
    // half of the defence against algorithm confusion.
    if (jwk.kty !== expectedKty) {
      throw new OidcError(`key ${kid} is a ${jwk.kty} key, not usable for this algorithm`);
    }

    try {
      return crypto.createPublicKey({ key: jwk, format: 'jwk' });
    } catch {
      throw new OidcError(`key ${kid} could not be parsed`);
    }
  }

  /** The current key set, fetched and cached. */
  async keys() {
    if (this.cachedKeys && this.now() - this.cachedAt < this.cacheTtlMs) return this.cachedKeys;
    if (this.inFlight) return this.inFlight;

    this.inFlight = this.fetchKeys()
      .then((keys) => {
        this.cachedKeys = keys;
        this.cachedAt = this.now();
        return keys;
      })
      .catch((error) => {
        // Serve from a stale cache rather than locking everyone out because
        // the IdP blipped. Only a cold start with no cache actually fails.
        if (this.cachedKeys) return this.cachedKeys;
        throw error;
      })
      .finally(() => {
        this.inFlight = null;
      });

    return this.inFlight;
  }

  async fetchKeys() {
    this.lastFetchAttempt = this.now();
    const uri = this.jwksUri ?? (await this.discoverJwksUri());
    const response = await this.fetchImpl(uri, { headers: { accept: 'application/json' } });
    if (!response.ok) throw new OidcError(`JWKS fetch failed (${response.status})`, 'idp_unreachable');
    return indexJwks(await response.json());
  }

  async discoverJwksUri() {
    const url = `${this.issuer}/.well-known/openid-configuration`;
    const response = await this.fetchImpl(url, { headers: { accept: 'application/json' } });
    if (!response.ok) {
      throw new OidcError(`OIDC discovery failed (${response.status})`, 'idp_unreachable');
    }
    const document = await response.json();
    if (!document.jwks_uri) throw new OidcError('discovery document has no jwks_uri', 'idp_unreachable');
    this.jwksUri = document.jwks_uri;
    return document.jwks_uri;
  }
}

/** Keep only keys usable for signature verification. */
function indexJwks(jwks) {
  const keys = Array.isArray(jwks?.keys) ? jwks.keys : [];
  return keys.filter((jwk) => {
    if (typeof jwk !== 'object' || jwk === null) return false;
    // `use: enc` keys are for encryption; accepting them would let an
    // encryption key validate a signature.
    if (jwk.use !== undefined && jwk.use !== 'sig') return false;
    return jwk.kty === 'RSA' || jwk.kty === 'EC';
  });
}

/**
 * Select a key by kid. A token with no kid is only resolvable when the key set
 * is unambiguous — guessing among several keys is how a rotated-out key stays
 * usable long after it should have stopped working.
 */
function pickKey(keys, kid) {
  if (kid) return keys.find((jwk) => jwk.kid === kid) ?? null;
  return keys.length === 1 ? keys[0] : null;
}

/** `group=scope,scope;group=scope` -> { group: [scopes] } */
export function parseScopeMap(raw) {
  if (!raw) return {};
  const map = {};
  for (const entry of String(raw).split(';')) {
    const [group, scopes] = entry.split('=');
    if (!group || !scopes) continue;
    map[group.trim()] = scopes.split(',').map((s) => s.trim()).filter(Boolean);
  }
  return map;
}

function toArray(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}
