import { describe, expect, it, vi } from 'vitest';

import { WorkerRegistry } from '../lib/registry.js';

function add(registry, id, model, options = {}) {
  registry.register({ id, send: vi.fn(), ...options });
  registry.setModel(id, model);
  return registry.get(id);
}

describe('WorkerRegistry', () => {
  it('does not hand work to a worker that has not loaded a model', () => {
    const registry = new WorkerRegistry();
    registry.register({ id: 'w1', send: vi.fn() });
    expect(registry.claim('qwen')).toBeNull();

    registry.setModel('w1', 'qwen');
    expect(registry.claim('qwen')?.id).toBe('w1');
  });

  it('only matches the exact model requested', () => {
    const registry = new WorkerRegistry();
    add(registry, 'w1', 'qwen-1.5b');
    expect(registry.claim('qwen-3b')).toBeNull();
  });

  it('skips paused workers but keeps them connected', () => {
    const registry = new WorkerRegistry();
    add(registry, 'w1', 'qwen');
    registry.setPaused('w1', true);

    expect(registry.claim('qwen')).toBeNull();
    expect(registry.canServe('qwen')).toBe(false);
    expect(registry.size).toBe(1);

    registry.setPaused('w1', false);
    expect(registry.claim('qwen')?.id).toBe('w1');
  });

  it('respects each worker concurrency limit', () => {
    const registry = new WorkerRegistry();
    add(registry, 'w1', 'qwen', { maxConcurrent: 2 });

    expect(registry.claim('qwen')?.id).toBe('w1');
    expect(registry.claim('qwen')?.id).toBe('w1');
    expect(registry.claim('qwen')).toBeNull();

    registry.release('w1');
    expect(registry.claim('qwen')?.id).toBe('w1');
  });

  it('spreads load least-loaded first, then round-robins', () => {
    const registry = new WorkerRegistry();
    add(registry, 'a', 'qwen');
    add(registry, 'b', 'qwen');
    add(registry, 'c', 'qwen');

    const first = [registry.claim('qwen').id, registry.claim('qwen').id, registry.claim('qwen').id];
    expect(new Set(first).size).toBe(3); // every worker got exactly one job

    // Free two of them; the next claims should go to those, not the busy one.
    registry.release('a');
    registry.release('b');
    const next = [registry.claim('qwen').id, registry.claim('qwen').id];
    expect(new Set(next)).toEqual(new Set(['a', 'b']));
  });

  it('prefers the least recently used worker when load is equal', () => {
    const registry = new WorkerRegistry();
    add(registry, 'a', 'qwen');
    add(registry, 'b', 'qwen');

    const first = registry.claim('qwen').id;
    registry.release(first);
    const second = registry.claim('qwen').id;

    expect(second).not.toBe(first);
  });

  it('reports only models a live, unpaused worker can serve', () => {
    const registry = new WorkerRegistry();
    add(registry, 'a', 'qwen');
    add(registry, 'b', 'llama');
    expect(registry.availableModels()).toEqual(['llama', 'qwen']);

    registry.setPaused('b', true);
    expect(registry.availableModels()).toEqual(['qwen']);

    registry.remove('a');
    expect(registry.availableModels()).toEqual([]);
  });

  it('counts completions but not failures', () => {
    const registry = new WorkerRegistry();
    add(registry, 'a', 'qwen');

    registry.claim('qwen');
    registry.release('a', true);
    registry.claim('qwen');
    registry.release('a', false);

    expect(registry.get('a').completed).toBe(1);
    expect(registry.get('a').inFlight).toBe(0);
  });

  it('never drives in-flight below zero on a duplicate release', () => {
    const registry = new WorkerRegistry();
    add(registry, 'a', 'qwen');
    registry.release('a');
    registry.release('a');
    expect(registry.get('a').inFlight).toBe(0);
  });
});
