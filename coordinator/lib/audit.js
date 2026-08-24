/**
 * audit.js
 *
 * Append-only record of who asked the mesh for what.
 *
 * The privacy default is the whole point. A compliance buyer needs to prove
 * that a request happened, by whom, against which model — and usually needs to
 * prove that prompt text was *not* retained. So the default records a SHA-256
 * of the prompt plus its length: enough to demonstrate two requests were
 * identical, or to match a known prompt if one is ever disclosed, without the
 * log itself becoming the leak it is supposed to guard against.
 *
 * Full-text capture exists for teams whose obligations require it, and it is
 * opt-in, announced loudly at startup, and stamped into every record so a
 * reader can tell which mode produced it.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export const RETENTION = Object.freeze({
  /** Store a hash and a character count. The default. */
  HASHED: 'hashed',
  /** Store nothing about content at all. */
  NONE: 'none',
  /** Store prompt and completion text verbatim. Opt-in. */
  FULL: 'full',
});

/** Rotate once the active file passes this, so it never grows unbounded. */
const DEFAULT_MAX_BYTES = 32 * 1024 * 1024;

export function hashText(text) {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
}

export class AuditLog {
  /**
   * @param {object} options
   * @param {string|null} options.filePath  JSONL destination. Null = memory only.
   * @param {string} [options.retention]    One of RETENTION.
   * @param {number} [options.maxBytes]
   * @param {number} [options.memoryLimit]  Recent entries kept for the console.
   */
  constructor({ filePath, retention = RETENTION.HASHED, maxBytes = DEFAULT_MAX_BYTES, memoryLimit = 500 }) {
    this.filePath = filePath;
    this.retention = Object.values(RETENTION).includes(retention) ? retention : RETENTION.HASHED;
    this.maxBytes = maxBytes;
    this.memoryLimit = memoryLimit;
    /** Ring buffer so the console can show a tail without reading the file. */
    this.recent = [];

    if (this.filePath) {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    }
  }

  /**
   * Write one event.
   *
   * @param {object} event
   * @param {string} event.type       e.g. 'chat.completion', 'key.created'
   * @param {string} [event.keyId]
   * @param {string} [event.keyName]
   * @param {string} [event.model]
   * @param {string} [event.workerId]
   * @param {string} [event.outcome]  'ok' | 'error' | 'denied'
   * @param {string} [event.prompt]   Content, handled per the retention mode.
   * @param {string} [event.completion]
   * @param {object} [event.detail]   Small, non-sensitive extras.
   */
  record(event) {
    const entry = {
      ts: new Date().toISOString(),
      id: crypto.randomBytes(8).toString('hex'),
      retention: this.retention,
      type: event.type,
      keyId: event.keyId ?? null,
      keyName: event.keyName ?? null,
      model: event.model ?? null,
      workerId: event.workerId ?? null,
      outcome: event.outcome ?? 'ok',
      durationMs: event.durationMs ?? null,
      ...this.contentFields(event),
    };

    if (event.detail && typeof event.detail === 'object') entry.detail = event.detail;

    this.recent.push(entry);
    if (this.recent.length > this.memoryLimit) this.recent.shift();

    this.append(entry);
    return entry;
  }

  /** Turn prompt/completion into whatever the retention mode permits. */
  contentFields(event) {
    const prompt = typeof event.prompt === 'string' ? event.prompt : null;
    const completion = typeof event.completion === 'string' ? event.completion : null;

    if (this.retention === RETENTION.NONE) return {};

    if (this.retention === RETENTION.FULL) {
      return {
        promptChars: prompt?.length ?? null,
        completionChars: completion?.length ?? null,
        promptText: prompt,
        completionText: completion,
      };
    }

    return {
      promptChars: prompt?.length ?? null,
      completionChars: completion?.length ?? null,
      promptHash: prompt === null ? null : hashText(prompt),
    };
  }

  append(entry) {
    if (!this.filePath) return;
    try {
      this.rotateIfNeeded();
      fs.appendFileSync(this.filePath, `${JSON.stringify(entry)}\n`, { mode: 0o600 });
    } catch {
      // An unwritable audit log must not take the mesh down. It is surfaced
      // through /admin/health instead, where an operator will see it.
      this.lastWriteError = new Date().toISOString();
    }
  }

  rotateIfNeeded() {
    if (!this.filePath || !fs.existsSync(this.filePath)) return;
    if (fs.statSync(this.filePath).size < this.maxBytes) return;
    fs.renameSync(this.filePath, `${this.filePath}.${Date.now()}`);
  }

  /** Most recent entries, newest first, optionally filtered. */
  tail({ limit = 100, keyId = null, type = null, outcome = null } = {}) {
    let entries = [...this.recent].reverse();
    if (keyId) entries = entries.filter((entry) => entry.keyId === keyId);
    if (type) entries = entries.filter((entry) => entry.type === type);
    if (outcome) entries = entries.filter((entry) => entry.outcome === outcome);
    return entries.slice(0, Math.max(1, Math.min(1000, limit)));
  }

  /** Counts for the console: totals by outcome, model and key. */
  summary() {
    const byOutcome = {};
    const byModel = {};
    const byKey = {};
    for (const entry of this.recent) {
      if (entry.type !== 'chat.completion') continue;
      byOutcome[entry.outcome] = (byOutcome[entry.outcome] ?? 0) + 1;
      if (entry.model) byModel[entry.model] = (byModel[entry.model] ?? 0) + 1;
      if (entry.keyName) byKey[entry.keyName] = (byKey[entry.keyName] ?? 0) + 1;
    }
    return { byOutcome, byModel, byKey, retained: this.recent.length };
  }
}
