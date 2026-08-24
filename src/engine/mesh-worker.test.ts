import { describe, expect, it } from 'vitest';

import { meshSocketUrl } from './mesh-worker';

describe('meshSocketUrl', () => {
  it('upgrades http to ws and targets /mesh', () => {
    const url = new URL(meshSocketUrl('http://mesh.local:8080', 'secret', 'Tuhin'));
    expect(url.protocol).toBe('ws:');
    expect(url.host).toBe('mesh.local:8080');
    expect(url.pathname).toBe('/mesh');
  });

  it('upgrades https to wss', () => {
    expect(meshSocketUrl('https://mesh.local', 't', 'l').startsWith('wss://')).toBe(true);
  });

  it('carries the token and label as query parameters', () => {
    // Browsers cannot set headers on a WebSocket handshake, so the token has
    // to ride in the URL. It stays on the LAN and is never logged.
    const url = new URL(meshSocketUrl('http://host:8080', 'tok en/+', 'Ann & Bob'));
    expect(url.searchParams.get('token')).toBe('tok en/+');
    expect(url.searchParams.get('label')).toBe('Ann & Bob');
  });

  it('discards any path on the base URL', () => {
    expect(new URL(meshSocketUrl('http://host:8080/some/page', 't', 'l')).pathname).toBe('/mesh');
  });

  it('throws on a base URL it cannot parse', () => {
    expect(() => meshSocketUrl('not a url', 't', 'l')).toThrow();
  });
});
