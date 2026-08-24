import { afterEach, describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { AuditLog, RETENTION, hashText } from '../lib/audit.js';
import { Store } from '../lib/store.js';

const temps = [];
function tempFile(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'meshgpu-test-'));
  temps.push(dir);
  return path.join(dir, name);
}

afterEach(() => {
  while (temps.length) fs.rmSync(temps.pop(), { recursive: true, force: true });
});

const SECRET = 'the patient record says 12345';

describe('AuditLog privacy', () => {
  it('records a hash and a length, never the prompt, by default', () => {
    const log = new AuditLog({ filePath: null });
    const entry = log.record({ type: 'chat.completion', prompt: SECRET, completion: 'reply' });

    expect(entry.promptHash).toBe(hashText(SECRET));
    expect(entry.promptChars).toBe(SECRET.length);
    expect(entry.promptText).toBeUndefined();
    expect(JSON.stringify(entry)).not.toContain('patient');
  });

  it('keeps identical prompts comparable without revealing either', () => {
    const log = new AuditLog({ filePath: null });
    const a = log.record({ type: 'chat.completion', prompt: SECRET });
    const b = log.record({ type: 'chat.completion', prompt: SECRET });
    const c = log.record({ type: 'chat.completion', prompt: 'something else' });

    expect(a.promptHash).toBe(b.promptHash);
    expect(a.promptHash).not.toBe(c.promptHash);
  });

  it('records nothing about content in "none" mode', () => {
    const log = new AuditLog({ filePath: null, retention: RETENTION.NONE });
    const entry = log.record({ type: 'chat.completion', prompt: SECRET });

    expect(entry.promptHash).toBeUndefined();
    expect(entry.promptChars).toBeUndefined();
    expect(entry.promptText).toBeUndefined();
    // Who and what still recorded — that is the point of keeping the log.
    expect(entry.type).toBe('chat.completion');
  });

  it('stores text only when full retention is explicitly chosen', () => {
    const log = new AuditLog({ filePath: null, retention: RETENTION.FULL });
    const entry = log.record({ type: 'chat.completion', prompt: SECRET, completion: 'reply' });

    expect(entry.promptText).toBe(SECRET);
    expect(entry.completionText).toBe('reply');
  });

  it('stamps the retention mode into every entry, so a reader can tell', () => {
    expect(new AuditLog({ filePath: null }).record({ type: 'x' }).retention).toBe('hashed');
    expect(
      new AuditLog({ filePath: null, retention: RETENTION.FULL }).record({ type: 'x' }).retention,
    ).toBe('full');
  });

  it('falls back to hashed for an unrecognised retention setting', () => {
    const log = new AuditLog({ filePath: null, retention: 'everything' });
    expect(log.retention).toBe(RETENTION.HASHED);
  });
});

describe('AuditLog persistence', () => {
  it('appends newline-delimited JSON', () => {
    const file = tempFile('audit.jsonl');
    const log = new AuditLog({ filePath: file });

    log.record({ type: 'chat.completion', keyId: 'k1', model: 'qwen' });
    log.record({ type: 'key.created', keyId: 'k2' });

    const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]).type).toBe('chat.completion');
    expect(JSON.parse(lines[1]).type).toBe('key.created');
  });

  it('rotates once the file passes its size cap', () => {
    const file = tempFile('audit.jsonl');
    const log = new AuditLog({ filePath: file, maxBytes: 500 });

    for (let i = 0; i < 40; i += 1) {
      log.record({ type: 'chat.completion', keyId: `k${i}`, model: 'qwen' });
    }

    const rotated = fs.readdirSync(path.dirname(file)).filter((n) => n.startsWith('audit.jsonl.'));
    expect(rotated.length).toBeGreaterThan(0);
    expect(fs.statSync(file).size).toBeLessThan(500 + 400);
  });

  it('keeps serving when the log file cannot be written', () => {
    const log = new AuditLog({ filePath: null });
    log.filePath = '/proc/definitely/not/writable/audit.jsonl';

    // An unwritable audit log must not throw into the request path.
    expect(() => log.record({ type: 'chat.completion' })).not.toThrow();
    expect(log.lastWriteError).toBeTruthy();
  });
});

describe('AuditLog querying', () => {
  function seeded() {
    const log = new AuditLog({ filePath: null });
    log.record({ type: 'chat.completion', keyId: 'a', keyName: 'Alice', model: 'qwen', outcome: 'ok' });
    log.record({ type: 'chat.completion', keyId: 'b', keyName: 'Bob', model: 'qwen', outcome: 'ok' });
    log.record({ type: 'chat.completion', keyId: 'a', keyName: 'Alice', model: 'llama', outcome: 'denied' });
    log.record({ type: 'key.created', keyId: 'c' });
    return log;
  }

  it('returns newest first', () => {
    const entries = seeded().tail();
    expect(entries[0].type).toBe('key.created');
    expect(entries).toHaveLength(4);
  });

  it('filters by key, type and outcome', () => {
    const log = seeded();
    expect(log.tail({ keyId: 'a' })).toHaveLength(2);
    expect(log.tail({ type: 'chat.completion' })).toHaveLength(3);
    expect(log.tail({ outcome: 'denied' })).toHaveLength(1);
  });

  it('caps the number of entries returned', () => {
    const log = new AuditLog({ filePath: null });
    for (let i = 0; i < 50; i += 1) log.record({ type: 'chat.completion' });
    expect(log.tail({ limit: 10 })).toHaveLength(10);
    expect(log.tail({ limit: 99_999 }).length).toBeLessThanOrEqual(1000);
  });

  it('bounds the in-memory tail so a busy mesh cannot exhaust memory', () => {
    const log = new AuditLog({ filePath: null, memoryLimit: 20 });
    for (let i = 0; i < 200; i += 1) log.record({ type: 'chat.completion' });
    expect(log.recent).toHaveLength(20);
  });

  it('summarises completions by outcome, model and key', () => {
    const summary = seeded().summary();
    expect(summary.byOutcome).toEqual({ ok: 2, denied: 1 });
    expect(summary.byModel).toEqual({ qwen: 2, llama: 1 });
    expect(summary.byKey).toEqual({ Alice: 2, Bob: 1 });
  });
});

describe('Store durability', () => {
  it('round-trips state through a file', () => {
    const file = tempFile('state.json');
    const first = new Store(file);
    first.state.keys.k1 = { id: 'k1', name: 'Alice' };
    first.touch();
    first.close();

    expect(new Store(file).state.keys.k1.name).toBe('Alice');
  });

  it('writes atomically, leaving no temp file behind', () => {
    const file = tempFile('state.json');
    const store = new Store(file);
    store.state.usage['k:2026-08-25'] = 5;
    store.touch();
    store.close();

    const siblings = fs.readdirSync(path.dirname(file));
    expect(siblings.filter((n) => n.includes('.tmp-'))).toHaveLength(0);
  });

  it('starts clean and quarantines a corrupt file rather than refusing to boot', () => {
    const file = tempFile('state.json');
    fs.writeFileSync(file, '{ this is not json');

    const store = new Store(file);
    expect(store.state.keys).toEqual({});

    const quarantined = fs.readdirSync(path.dirname(file)).filter((n) => n.includes('.corrupt-'));
    expect(quarantined).toHaveLength(1);
  });

  it('fills in defaults for fields an older state file predates', () => {
    const file = tempFile('state.json');
    fs.writeFileSync(file, JSON.stringify({ version: 1, keys: { k1: { id: 'k1' } } }));

    const store = new Store(file);
    expect(store.state.usage).toEqual({});
    expect(store.state.settings.blockedModels).toEqual([]);
  });
});
