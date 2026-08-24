import { describe, expect, it } from 'vitest';

import { computeAssignments } from './scheduler';
import type { PeerCapacity } from './scheduler';

/**
 * Deterministic PRNG so a failing property test reproduces exactly from its
 * seed instead of vanishing on the next run.
 */
function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function peers(entries: Array<[string, number, number]>): Map<string, PeerCapacity> {
  return new Map(entries.map(([id, capacity, weight]) => [id, { capacity, weight }]));
}

/** Layer ranges, ordered by start, for assertion convenience. */
function ranges(result: Map<string, { start: number; end: number }>) {
  return [...result.values()].sort((a, b) => a.start - b.start);
}

function totalAssigned(result: Map<string, { start: number; end: number }>): number {
  return [...result.values()].reduce((sum, stage) => sum + (stage.end - stage.start), 0);
}

describe('computeAssignments — invariants', () => {
  it('produces contiguous, non-overlapping ranges starting at layer 0', () => {
    const result = computeAssignments(28, peers([['a', 20, 1], ['b', 20, 1], ['c', 20, 1]]));
    const sorted = ranges(result);

    expect(sorted[0].start).toBe(0);
    for (let i = 1; i < sorted.length; i += 1) {
      expect(sorted[i].start).toBe(sorted[i - 1].end);
    }
  });

  it('never assigns a peer more layers than its memory capacity allows', () => {
    const result = computeAssignments(28, peers([['a', 2, 5], ['b', 40, 1]]));
    expect(result.get('a')!.end - result.get('a')!.start).toBeLessThanOrEqual(2);
  });

  it('covers every layer when the peers collectively have the capacity', () => {
    const result = computeAssignments(28, peers([['a', 14, 1], ['b', 14, 1]]));
    expect(totalAssigned(result)).toBe(28);
  });

  it('leaves the tail unassigned when capacity falls short of the model', () => {
    const result = computeAssignments(28, peers([['a', 4, 1], ['b', 4, 1]]));
    expect(totalAssigned(result)).toBe(8);
  });

  it('gives a 2x faster peer roughly twice the layers at equal capacity', () => {
    const result = computeAssignments(30, peers([['fast', 30, 2], ['slow', 30, 1]]));
    const fast = result.get('fast')!.end - result.get('fast')!.start;
    const slow = result.get('slow')!.end - result.get('slow')!.start;
    expect(fast).toBe(20);
    expect(slow).toBe(10);
  });

  it('excludes peers that cannot host anything', () => {
    const result = computeAssignments(10, peers([['a', 10, 1], ['zero', 0, 1]]));
    expect(result.has('zero')).toBe(false);
    expect(totalAssigned(result)).toBe(10);
  });

  it('returns an empty assignment when no peer can host', () => {
    expect(computeAssignments(28, peers([['a', 0, 1]])).size).toBe(0);
    expect(computeAssignments(28, new Map()).size).toBe(0);
  });

  it('is deterministic — the same input always yields the same output', () => {
    const input = peers([['c', 7, 1.5], ['a', 11, 0.5], ['b', 3, 3]]);
    const first = computeAssignments(21, input);
    for (let i = 0; i < 25; i += 1) {
      expect([...computeAssignments(21, input)]).toEqual([...first]);
    }
  });

  it('is independent of the insertion order of the peer map', () => {
    const forward = computeAssignments(24, peers([['a', 9, 1], ['b', 9, 2], ['c', 9, 1]]));
    const reverse = computeAssignments(24, peers([['c', 9, 1], ['b', 9, 2], ['a', 9, 1]]));
    expect([...forward].sort()).toEqual([...reverse].sort());
  });
});

describe('computeAssignments — randomised properties', () => {
  it('holds every invariant across 2000 random topologies', () => {
    const random = rng(0xC0FFEE);

    for (let iteration = 0; iteration < 2000; iteration += 1) {
      const totalLayers = 1 + Math.floor(random() * 80);
      const peerCount = 1 + Math.floor(random() * 8);

      const entries: Array<[string, number, number]> = [];
      for (let i = 0; i < peerCount; i += 1) {
        entries.push([
          `peer-${String.fromCharCode(97 + i)}`,
          Math.floor(random() * 40),
          Math.round(random() * 900) / 100, // 0.00 .. 9.00
        ]);
      }

      const input = peers(entries);
      const result = computeAssignments(totalLayers, input);
      const context = `seed iteration ${iteration}: ${totalLayers} layers, ${JSON.stringify(entries)}`;

      // 1. Contiguous from zero, no gaps, no overlaps.
      const sorted = ranges(result);
      let cursor = 0;
      for (const stage of sorted) {
        expect(stage.start, context).toBe(cursor);
        expect(stage.end, context).toBeGreaterThan(stage.start);
        cursor = stage.end;
      }

      // 2. Never over-commits a peer's memory.
      for (const [peerId, stage] of result) {
        const capacity = input.get(peerId)!.capacity;
        expect(stage.end - stage.start, `${context} / ${peerId}`).toBeLessThanOrEqual(capacity);
      }

      // 3. Assigns exactly as many layers as the mesh can hold, never more.
      const usable = [...input.values()]
        .filter((peer) => peer.weight > 0)
        .reduce((sum, peer) => sum + peer.capacity, 0);
      expect(totalAssigned(result), context).toBe(Math.min(totalLayers, usable));

      // 4. No zero-width stages leak into the topology — `nextHop()` matches on
      //    `stage.start === this.stage.end`, so an empty range would alias.
      for (const stage of sorted) {
        expect(stage.end - stage.start, context).toBeGreaterThan(0);
      }
    }
  });
});
