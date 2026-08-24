/**
 * quota.js
 *
 * Per-key request limits.
 *
 * Two windows, because they stop different things. The per-minute limit stops
 * a runaway script from starving everyone else off a mesh whose capacity is
 * somebody's laptop. The daily limit is the one an administrator actually
 * reasons about when handing out access.
 *
 * Daily counters live in the store so they survive a restart — a quota you can
 * reset by bouncing the process is not a quota. Per-minute state is in memory,
 * which is the right trade: losing it on restart forgives at most 60 seconds.
 */

/** Rolling window for the short limit. */
const MINUTE_MS = 60_000;

export class Quota {
  /**
   * @param {import('./store.js').Store} store
   * @param {() => number} [now]  Injectable clock for tests.
   */
  constructor(store, now = () => Date.now()) {
    this.store = store;
    this.now = now;
    /** @type {Map<string, number[]>} keyId -> recent request timestamps */
    this.recent = new Map();
  }

  /** `YYYY-MM-DD` in UTC, so a mesh spanning time zones agrees on "today". */
  today() {
    return new Date(this.now()).toISOString().slice(0, 10);
  }

  usageFor(keyId) {
    return this.store.state.usage[`${keyId}:${this.today()}`] ?? 0;
  }

  /**
   * Check a key against its limits without consuming anything.
   *
   * @returns {{ ok: true } | { ok: false, reason: string, retryAfterSeconds: number }}
   */
  check(record) {
    const daily = record.dailyRequestLimit ?? 0;
    if (daily > 0 && this.usageFor(record.id) >= daily) {
      return {
        ok: false,
        reason: `daily limit of ${daily} requests reached`,
        retryAfterSeconds: this.secondsUntilUtcMidnight(),
      };
    }

    const perMinute = record.requestsPerMinute ?? 0;
    if (perMinute > 0) {
      const stamps = this.prune(record.id);
      if (stamps.length >= perMinute) {
        const oldest = stamps[0];
        const waitMs = Math.max(0, oldest + MINUTE_MS - this.now());
        return {
          ok: false,
          reason: `rate limit of ${perMinute} requests per minute reached`,
          retryAfterSeconds: Math.max(1, Math.ceil(waitMs / 1000)),
        };
      }
    }

    return { ok: true };
  }

  /** Record one request against a key. Call only after `check` passes. */
  consume(record) {
    const dayKey = `${record.id}:${this.today()}`;
    this.store.state.usage[dayKey] = (this.store.state.usage[dayKey] ?? 0) + 1;
    this.store.touch();

    if ((record.requestsPerMinute ?? 0) > 0) {
      const stamps = this.prune(record.id);
      stamps.push(this.now());
    }
  }

  /**
   * Drop usage rows older than `days`. Without this the store grows by one row
   * per key per day forever.
   */
  prunePastDays(days = 32) {
    const cutoff = new Date(this.now() - days * 86_400_000).toISOString().slice(0, 10);
    let removed = 0;
    for (const dayKey of Object.keys(this.store.state.usage)) {
      const date = dayKey.slice(dayKey.lastIndexOf(':') + 1);
      if (date < cutoff) {
        delete this.store.state.usage[dayKey];
        removed += 1;
      }
    }
    if (removed > 0) this.store.touch();
    return removed;
  }

  /** Usage for every key over the last `days`, for the admin console. */
  recentUsage(days = 7) {
    const out = {};
    const earliest = new Date(this.now() - (days - 1) * 86_400_000).toISOString().slice(0, 10);
    for (const [dayKey, count] of Object.entries(this.store.state.usage)) {
      const split = dayKey.lastIndexOf(':');
      const keyId = dayKey.slice(0, split);
      const date = dayKey.slice(split + 1);
      if (date < earliest) continue;
      out[keyId] = (out[keyId] ?? 0) + count;
    }
    return out;
  }

  /** Drop timestamps outside the rolling minute and return the live list. */
  prune(keyId) {
    const cutoff = this.now() - MINUTE_MS;
    const stamps = (this.recent.get(keyId) ?? []).filter((stamp) => stamp > cutoff);
    this.recent.set(keyId, stamps);
    return stamps;
  }

  secondsUntilUtcMidnight() {
    const now = new Date(this.now());
    const midnight = Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate() + 1,
      0, 0, 0, 0,
    );
    return Math.max(1, Math.ceil((midnight - this.now()) / 1000));
  }
}
