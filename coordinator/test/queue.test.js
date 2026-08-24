import { beforeEach, describe, expect, it, vi } from 'vitest';

import { JobQueue } from '../lib/queue.js';
import { WorkerRegistry } from '../lib/registry.js';

/** A worker that records every message the coordinator sends it. */
function fakeWorker(registry, id, model, options = {}) {
  const sent = [];
  registry.register({ id, send: (message) => sent.push(message), ...options });
  registry.setModel(id, model);
  return { id, sent, jobs: () => sent.filter((m) => m.type === 'job') };
}

function makeJob(id, model, sink) {
  return {
    id,
    model,
    payload: { messages: [{ role: 'user', content: 'hi' }] },
    onChunk: (delta) => sink.chunks.push(delta),
    onDone: (info) => sink.done.push(info),
    onError: (error) => sink.errors.push(error.message),
  };
}

describe('JobQueue', () => {
  let registry;
  let queue;
  let sink;

  beforeEach(() => {
    registry = new WorkerRegistry();
    queue = new JobQueue({ registry, jobTimeoutMs: 0 });
    sink = { chunks: [], done: [], errors: [] };
  });

  it('rejects a model no worker has loaded', () => {
    expect(() => queue.submit(makeJob('j1', 'missing', sink))).toThrow(/no worker/);
    try {
      queue.submit(makeJob('j2', 'missing', sink));
    } catch (error) {
      expect(error.status).toBe(503);
    }
  });

  it('dispatches straight to a free worker and streams back', () => {
    const worker = fakeWorker(registry, 'w1', 'qwen');
    queue.submit(makeJob('j1', 'qwen', sink));

    expect(worker.jobs()).toHaveLength(1);
    expect(worker.jobs()[0].jobId).toBe('j1');

    queue.chunk('j1', 'Hel');
    queue.chunk('j1', 'lo');
    queue.complete('j1');

    expect(sink.chunks.join('')).toBe('Hello');
    expect(sink.done).toHaveLength(1);
    expect(registry.get('w1').inFlight).toBe(0);
  });

  it('queues work when every worker is busy and drains on completion', () => {
    const worker = fakeWorker(registry, 'w1', 'qwen');
    queue.submit(makeJob('j1', 'qwen', sink));
    queue.submit(makeJob('j2', 'qwen', sink));

    expect(worker.jobs()).toHaveLength(1);
    expect(queue.stats).toEqual({ waiting: 1, active: 1 });

    queue.complete('j1');
    expect(worker.jobs()).toHaveLength(2);
    expect(queue.stats).toEqual({ waiting: 0, active: 1 });
  });

  it('spreads concurrent requests across the mesh', () => {
    const a = fakeWorker(registry, 'a', 'qwen');
    const b = fakeWorker(registry, 'b', 'qwen');

    queue.submit(makeJob('j1', 'qwen', sink));
    queue.submit(makeJob('j2', 'qwen', sink));

    expect(a.jobs()).toHaveLength(1);
    expect(b.jobs()).toHaveLength(1);
    expect(queue.stats.waiting).toBe(0);
  });

  it('reassigns a job when its worker disappears before any output', () => {
    const a = fakeWorker(registry, 'a', 'qwen');
    const b = fakeWorker(registry, 'b', 'qwen');

    queue.submit(makeJob('j1', 'qwen', sink));
    const original = a.jobs().length === 1 ? a : b;
    const survivor = original === a ? b : a;

    registry.remove(original.id);
    queue.releaseWorker(original.id);

    expect(survivor.jobs()).toHaveLength(1);
    expect(sink.errors).toEqual([]); // the client never saw a failure

    queue.chunk('j1', 'recovered');
    queue.complete('j1');
    expect(sink.chunks.join('')).toBe('recovered');
  });

  it('fails rather than retries once tokens have reached the client', () => {
    fakeWorker(registry, 'a', 'qwen');
    fakeWorker(registry, 'b', 'qwen');
    queue.submit(makeJob('j1', 'qwen', sink));

    queue.chunk('j1', 'partial output');
    queue.fail('j1', new Error('worker died'));

    // Retrying would duplicate text the client has already rendered.
    expect(sink.errors).toEqual(['worker died']);
    expect(sink.chunks).toEqual(['partial output']);
  });

  it('gives up after the attempt limit instead of looping forever', () => {
    fakeWorker(registry, 'a', 'qwen');
    const limited = new JobQueue({ registry, jobTimeoutMs: 0, maxAttempts: 2 });
    limited.submit(makeJob('j1', 'qwen', sink));

    limited.fail('j1', new Error('first failure'));
    expect(sink.errors).toEqual([]); // retried

    limited.fail('j1', new Error('second failure'));
    expect(sink.errors).toEqual(['second failure']);
  });

  it('rejects new work once the queue is full', () => {
    fakeWorker(registry, 'a', 'qwen');
    const shallow = new JobQueue({ registry, jobTimeoutMs: 0, maxQueueDepth: 1 });

    shallow.submit(makeJob('j1', 'qwen', sink)); // dispatched
    shallow.submit(makeJob('j2', 'qwen', sink)); // waiting

    expect(() => shallow.submit(makeJob('j3', 'qwen', sink))).toThrow(/saturated/);
    try {
      shallow.submit(makeJob('j4', 'qwen', sink));
    } catch (error) {
      expect(error.status).toBe(429);
    }
  });

  it('cancels a queued job without touching any worker', () => {
    const worker = fakeWorker(registry, 'a', 'qwen');
    queue.submit(makeJob('j1', 'qwen', sink));
    queue.submit(makeJob('j2', 'qwen', sink));

    queue.cancel('j2');
    queue.complete('j1');

    expect(worker.jobs().map((job) => job.jobId)).toEqual(['j1']);
  });

  it('tells the worker to stop when an active job is cancelled', () => {
    const worker = fakeWorker(registry, 'a', 'qwen');
    queue.submit(makeJob('j1', 'qwen', sink));
    queue.cancel('j1');

    expect(worker.sent.some((m) => m.type === 'cancel' && m.jobId === 'j1')).toBe(true);
    expect(registry.get('a').inFlight).toBe(0);
  });

  it('times out a worker that goes silent, then retries elsewhere', () => {
    vi.useFakeTimers();
    try {
      const a = fakeWorker(registry, 'a', 'qwen');
      const b = fakeWorker(registry, 'b', 'qwen');
      const timed = new JobQueue({ registry, jobTimeoutMs: 5_000 });

      timed.submit(makeJob('j1', 'qwen', sink));
      const stalled = a.jobs().length === 1 ? a : b;
      const survivor = stalled === a ? b : a;

      vi.advanceTimersByTime(5_001);

      expect(survivor.jobs()).toHaveLength(1);
      expect(sink.errors).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not time out a slow generation that is still streaming', () => {
    vi.useFakeTimers();
    try {
      fakeWorker(registry, 'a', 'qwen');
      const timed = new JobQueue({ registry, jobTimeoutMs: 5_000, maxAttempts: 1 });
      timed.submit(makeJob('j1', 'qwen', sink));

      for (let i = 0; i < 10; i += 1) {
        vi.advanceTimersByTime(4_000);
        timed.chunk('j1', 'token ');
      }
      vi.advanceTimersByTime(4_000);

      expect(sink.errors).toEqual([]);
      timed.complete('j1');
      expect(sink.done).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores messages for jobs it has never heard of', () => {
    fakeWorker(registry, 'a', 'qwen');
    expect(() => {
      queue.chunk('ghost', 'x');
      queue.complete('ghost');
      queue.fail('ghost', new Error('x'));
      queue.cancel('ghost');
    }).not.toThrow();
  });
});
