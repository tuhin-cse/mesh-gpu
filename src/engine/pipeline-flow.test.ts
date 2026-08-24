/**
 * Multi-stage pipeline behaviour over the in-memory mock transport.
 *
 * These are the paths that used to hang or lose data silently: a dropped or
 * unforwardable tensor, a stage that throws, and several sequences in flight
 * at once. Nothing here needs WebRTC or a GPU.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import { MockSignaling } from './mock-transport';
import { PipelineNode } from './p2p-pipeline';
import type { StageExecutor, StageRange } from './p2p-pipeline';

let created: PipelineNode[] = [];

afterEach(() => {
  for (const node of created) node.close();
  created = [];
});

/** Two nodes wired through the mock signaling bus, staged head and tail. */
async function twoStageMesh(options: {
  headExecutor?: StageExecutor;
  tailExecutor?: StageExecutor;
  maxConcurrent?: number;
  runTimeoutMs?: number;
} = {}) {
  const roomId = `room-${Math.random().toString(36).slice(2)}`;

  const make = (peerId: string, executor?: StageExecutor) => {
    const signaling = new MockSignaling({ roomId, peerId, delayMs: 0 });
    const node = new PipelineNode({
      signaling,
      executor,
      maxConcurrent: options.maxConcurrent,
      runTimeoutMs: options.runTimeoutMs,
      createPeerConnection: (remotePeerId) => signaling.createPeerConnection(remotePeerId),
    });
    created.push(node);
    return node;
  };

  // "head" sorts before "tail", so the deterministic initiator rule holds.
  const head = make('head', options.headExecutor);
  const tail = make('tail', options.tailExecutor);

  await head.join();
  await tail.join();
  await waitFor(() => head.peers.length === 1 && tail.peers.length === 1);

  head.setStage({ start: 0, end: 8 });
  tail.setStage({ start: 8, end: 16 });
  // Stage announcements cross the control channel; wait for both to land.
  await waitFor(() => head.getTopology().length === 1 && tail.getTopology().length === 1);

  return { head, tail };
}

async function waitFor(predicate: () => boolean, timeoutMs = 4000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('timed out waiting for a condition');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

/** Adds a fixed value to every element, so a stage's effect is observable. */
class AddExecutor implements StageExecutor {
  constructor(private readonly amount: number) {}

  execute(_stage: StageRange, input: Float32Array): Float32Array {
    return input.map((value) => value + this.amount);
  }

  finalize(_stage: StageRange, hidden: Float32Array) {
    // Report the first element so a test can assert what reached the tail.
    return { token: hidden[0] };
  }
}

describe('two-stage pipeline', () => {
  it('carries a hidden state through both stages to a token', async () => {
    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(1),
      tailExecutor: new AddExecutor(10),
    });

    // 5 -> head adds 1 -> 6 -> tail adds 10 -> 16
    const token = await head.run(new Float32Array([5, 5, 5, 5]), [4]);
    expect(token).toBe(16);
  });

  it('preserves values across the f16 wire format', async () => {
    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: new AddExecutor(0),
    });
    // 0.5 is exact in half precision, so this must survive untouched.
    expect(await head.run(new Float32Array([0.5, 0.25]), [2])).toBe(0.5);
  });

  it('chunks and reassembles a tensor far larger than one DataChannel message', async () => {
    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: new AddExecutor(0),
    });

    // 400k elements = 800 KB in f16, well past a single-message limit.
    const values = new Float32Array(400_000).fill(2);
    values[0] = 7;
    expect(await head.run(values, [400_000])).toBe(7);
  });

  it('runs several sequences concurrently rather than one at a time', async () => {
    let peakConcurrent = 0;
    let inStage = 0;

    const slowTail: StageExecutor = {
      async execute(_stage, input) {
        inStage += 1;
        peakConcurrent = Math.max(peakConcurrent, inStage);
        await new Promise((resolve) => setTimeout(resolve, 25));
        inStage -= 1;
        return input;
      },
      finalize: (_stage, hidden) => ({ token: hidden[0] }),
    };

    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: slowTail,
      maxConcurrent: 4,
    });

    const tokens = await Promise.all(
      [1, 2, 3, 4].map((value) => head.run(new Float32Array([value]), [1])),
    );

    expect(tokens.sort()).toEqual([1, 2, 3, 4]);
    // The point of microbatching: the tail worked on more than one at a time.
    expect(peakConcurrent).toBeGreaterThan(1);
  });

  it('keeps concurrent sequences from being mixed up with each other', async () => {
    const { head } = await twoStageMesh({
      headExecutor: {
        // Random delays deliberately reorder completion relative to submission.
        async execute(_stage, input) {
          await new Promise((resolve) => setTimeout(resolve, Math.random() * 20));
          return input;
        },
      },
      tailExecutor: new AddExecutor(0),
      maxConcurrent: 8,
    });

    const inputs = [11, 22, 33, 44, 55, 66];
    const tokens = await Promise.all(
      inputs.map((value) => head.run(new Float32Array([value]), [1])),
    );

    // Each promise must resolve with its own sequence's result, in order.
    expect(tokens).toEqual(inputs);
  });

  it('admits no more than maxConcurrent sequences at once', async () => {
    let started = 0;
    let peak = 0;

    const { head } = await twoStageMesh({
      headExecutor: {
        async execute(_stage, input) {
          started += 1;
          peak = Math.max(peak, started);
          await new Promise((resolve) => setTimeout(resolve, 15));
          started -= 1;
          return input;
        },
      },
      tailExecutor: new AddExecutor(0),
      maxConcurrent: 2,
    });

    await Promise.all(
      [1, 2, 3, 4, 5, 6].map((value) => head.run(new Float32Array([value]), [1])),
    );
    expect(peak).toBeLessThanOrEqual(2);
  });
});

describe('failure handling', () => {
  it('rejects the run when a downstream stage throws, instead of hanging', async () => {
    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: {
        execute() {
          throw new Error('tail exploded');
        },
      },
    });

    await expect(head.run(new Float32Array([1]), [1])).rejects.toThrow(/tail exploded/);
  });

  it('rejects the run when the next stage rejects the tensor as non-finite', async () => {
    const { head } = await twoStageMesh({
      headExecutor: {
        execute: () => new Float32Array([Number.NaN, 1]),
      },
      tailExecutor: new AddExecutor(0),
    });

    await expect(head.run(new Float32Array([1, 1]), [2])).rejects.toThrow(/non-finite|NaN/);
  });

  it('times out a sequence whose next stage never answers', async () => {
    const { head, tail } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: {
        // Never resolves — models a wedged or vanished stage.
        execute: () => new Promise<Float32Array>(() => {}),
      },
      runTimeoutMs: 200,
    });
    expect(tail).toBeDefined();

    await expect(head.run(new Float32Array([1]), [1])).rejects.toThrow(/produced no token/);
  });

  it('frees the concurrency slot after a timeout so the node keeps working', async () => {
    const failing = vi.fn(() => new Promise<Float32Array>(() => {}));
    const { head } = await twoStageMesh({
      headExecutor: new AddExecutor(0),
      tailExecutor: { execute: failing },
      maxConcurrent: 1,
      runTimeoutMs: 150,
    });

    await expect(head.run(new Float32Array([1]), [1])).rejects.toThrow();
    expect(head.inFlightCount).toBe(0);

    // A second run must be admitted rather than blocked behind the dead one.
    await expect(head.run(new Float32Array([2]), [1])).rejects.toThrow();
    expect(failing).toHaveBeenCalledTimes(2);
  });
});

describe('single-node pipeline', () => {
  it('finalizes locally when there is no next stage', async () => {
    const signaling = new MockSignaling({ roomId: 'solo', peerId: 'only', delayMs: 0 });
    const node = new PipelineNode({
      signaling,
      executor: new AddExecutor(3),
      createPeerConnection: (remotePeerId) => signaling.createPeerConnection(remotePeerId),
    });
    created.push(node);

    await node.join();
    node.setStage({ start: 0, end: 16 });

    expect(await node.run(new Float32Array([1, 1]), [2])).toBe(4);
  });

  it('refuses to start a forward pass from a middle stage', async () => {
    const signaling = new MockSignaling({ roomId: 'mid', peerId: 'only', delayMs: 0 });
    const node = new PipelineNode({
      signaling,
      createPeerConnection: (remotePeerId) => signaling.createPeerConnection(remotePeerId),
    });
    created.push(node);

    await node.join();
    node.setStage({ start: 8, end: 16 });

    await expect(node.run(new Float32Array([1]), [1])).rejects.toThrow(/first pipeline stage/);
  });
});
