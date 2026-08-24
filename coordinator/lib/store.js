/**
 * store.js
 *
 * Durable state for the control plane: API keys, quota counters, settings.
 *
 * A JSON file rather than a database, deliberately. This runs on whatever
 * machine an office had spare, often with no internet and no operator — an
 * install step that says "first, provision Postgres" is an install step that
 * does not happen. One file is also one thing to back up and one thing to
 * carry across an air gap.
 *
 * Writes are atomic (temp file, then rename) so a crash mid-write leaves the
 * previous state intact rather than a truncated file.
 */

import fs from 'node:fs';
import path from 'node:path';

const CURRENT_VERSION = 1;

function emptyState() {
  return {
    version: CURRENT_VERSION,
    keys: {},
    /** Daily request counters, keyed by `${keyId}:${YYYY-MM-DD}`. */
    usage: {},
    settings: {
      /** Models no key may use, whatever its own allowlist says. */
      blockedModels: [],
    },
  };
}

export class Store {
  /**
   * @param {string|null} filePath  Where to persist. Null keeps state in
   *   memory only, which is what the tests use.
   */
  constructor(filePath) {
    this.filePath = filePath;
    this.state = emptyState();
    this.dirty = false;
    this.flushTimer = null;
    this.load();
  }

  load() {
    if (!this.filePath || !fs.existsSync(this.filePath)) return;
    try {
      const parsed = JSON.parse(fs.readFileSync(this.filePath, 'utf8'));
      if (parsed && typeof parsed === 'object') {
        // Merge rather than replace, so a file written by an older version
        // that lacks a newer field still loads with sane defaults.
        this.state = {
          ...emptyState(),
          ...parsed,
          settings: { ...emptyState().settings, ...(parsed.settings ?? {}) },
        };
      }
    } catch {
      // A corrupt file must not stop the mesh from starting. Keep the broken
      // copy for inspection rather than overwriting it silently.
      try {
        fs.renameSync(this.filePath, `${this.filePath}.corrupt-${Date.now()}`);
      } catch {
        // Best effort — if we cannot rename it we still start clean.
      }
      this.state = emptyState();
    }
  }

  /** Mark state changed and schedule a write. */
  touch() {
    this.dirty = true;
    if (!this.filePath || this.flushTimer) return;
    // Coalesce bursts: a busy mesh should not write the file per request.
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.flush();
    }, 250);
    this.flushTimer.unref?.();
  }

  /** Write immediately. Safe to call when nothing has changed. */
  flush() {
    if (!this.filePath || !this.dirty) return;

    const directory = path.dirname(this.filePath);
    fs.mkdirSync(directory, { recursive: true });

    const temp = `${this.filePath}.tmp-${process.pid}`;
    fs.writeFileSync(temp, JSON.stringify(this.state, null, 2), { mode: 0o600 });
    fs.renameSync(temp, this.filePath);
    this.dirty = false;
  }

  close() {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush();
  }
}
