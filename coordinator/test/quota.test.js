import { beforeEach, describe, expect, it } from 'vitest';

import { Store } from '../lib/store.js';
import { Quota } from '../lib/quota.js';

const DAY = 86_400_000;

describe('Quota', () => {
  let store;
  let clock;
  let quota;

  beforeEach(() => {
    store = new Store(null);
    clock = Date.UTC(2026, 7, 25, 12, 0, 0);
    quota = new Quota(store, () => clock);
  });

  const key = (overrides = {}) => ({
    id: 'k1',
    dailyRequestLimit: 0,
    requestsPerMinute: 0,
    ...overrides,
  });

  it('allows everything when no limits are set', () => {
    const unlimited = key();
    for (let i = 0; i < 200; i += 1) {
      expect(quota.check(unlimited).ok).toBe(true);
      quota.consume(unlimited);
    }
  });

  it('enforces a daily limit and reports when it resets', () => {
    const limited = key({ dailyRequestLimit: 3 });

    for (let i = 0; i < 3; i += 1) {
      expect(quota.check(limited).ok).toBe(true);
      quota.consume(limited);
    }

    const denied = quota.check(limited);
    expect(denied.ok).toBe(false);
    expect(denied.reason).toMatch(/daily limit of 3/);
    // 12:00 UTC, so midnight is 12 hours away.
    expect(denied.retryAfterSeconds).toBe(12 * 3600);
  });

  it('resets the daily limit at UTC midnight', () => {
    const limited = key({ dailyRequestLimit: 2 });
    quota.consume(limited);
    quota.consume(limited);
    expect(quota.check(limited).ok).toBe(false);

    clock += DAY;
    expect(quota.check(limited).ok).toBe(true);
  });

  it('enforces a rolling per-minute limit', () => {
    const limited = key({ requestsPerMinute: 3 });

    for (let i = 0; i < 3; i += 1) {
      expect(quota.check(limited).ok).toBe(true);
      quota.consume(limited);
    }

    const denied = quota.check(limited);
    expect(denied.ok).toBe(false);
    expect(denied.reason).toMatch(/3 requests per minute/);
    expect(denied.retryAfterSeconds).toBeGreaterThan(0);
    expect(denied.retryAfterSeconds).toBeLessThanOrEqual(60);
  });

  it('rolls the minute window rather than resetting on the minute boundary', () => {
    const limited = key({ requestsPerMinute: 2 });

    quota.consume(limited);
    clock += 30_000;
    quota.consume(limited);
    expect(quota.check(limited).ok).toBe(false);

    // 31s later the first request ages out; the second has not.
    clock += 31_000;
    expect(quota.check(limited).ok).toBe(true);
    quota.consume(limited);
    expect(quota.check(limited).ok).toBe(false);
  });

  it('applies whichever limit binds first', () => {
    const limited = key({ dailyRequestLimit: 10, requestsPerMinute: 2 });

    quota.consume(limited);
    quota.consume(limited);
    expect(quota.check(limited).reason).toMatch(/per minute/);

    clock += 61_000;
    expect(quota.check(limited).ok).toBe(true);
  });

  it('counts each key separately', () => {
    const alice = key({ id: 'alice', dailyRequestLimit: 1 });
    const bob = key({ id: 'bob', dailyRequestLimit: 1 });

    quota.consume(alice);
    expect(quota.check(alice).ok).toBe(false);
    expect(quota.check(bob).ok).toBe(true);
  });

  it('survives a restart, because a resettable quota is not a quota', () => {
    const limited = key({ dailyRequestLimit: 2 });
    quota.consume(limited);
    quota.consume(limited);

    // Same store, brand new Quota — as if the process had been bounced.
    const afterRestart = new Quota(store, () => clock);
    expect(afterRestart.check(limited).ok).toBe(false);
  });

  it('prunes usage rows older than the retention window', () => {
    const limited = key({ dailyRequestLimit: 5 });
    quota.consume(limited);

    clock += 40 * DAY;
    quota.consume(limited);

    expect(Object.keys(store.state.usage)).toHaveLength(2);
    expect(quota.prunePastDays(32)).toBe(1);
    expect(Object.keys(store.state.usage)).toHaveLength(1);
  });

  it('totals recent usage per key for the console', () => {
    const alice = key({ id: 'alice' });
    quota.consume(alice);
    quota.consume(alice);
    clock += DAY;
    quota.consume(alice);

    expect(quota.recentUsage(7).alice).toBe(3);
  });
});
