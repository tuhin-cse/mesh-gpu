/**
 * Mathematical properties of the reference layer implementation.
 *
 * The GPU tests check that the kernels agree with this file. That is only
 * worth something if this file is itself correct, so these tests check the
 * maths against properties that hold independently: a normalisation that
 * actually normalises, a rotation that preserves length, attention that
 * reduces to the identity in the cases where it must.
 */

import { describe, expect, it } from 'vitest';

import {
  TEST_CONFIG,
  applyRoPE,
  attention,
  createKVCache,
  forwardLayer,
  kvHeadSize,
  layerWeightBytes,
  matVec,
  randomWeights,
  rmsNorm,
  seededRandom,
  silu,
  validateConfig,
} from './transformer';
import type { TransformerConfig } from './transformer';

function close(a: number, b: number, tolerance = 1e-5): boolean {
  return Math.abs(a - b) <= tolerance;
}

/** Both .fill() and .map() widen the buffer type, so build these by hand. */
function filled(length: number, value: number): Float32Array {
  const out = new Float32Array(length);
  out.fill(value);
  return out;
}

function randomVector(length: number, next: () => number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) out[i] = next() * 2 - 1;
  return out;
}

describe('rmsNorm', () => {
  it('normalises to unit root-mean-square when the weight is all ones', () => {
    const input = new Float32Array([3, -4, 5, -6]);
    const out = rmsNorm(input, new Float32Array([1, 1, 1, 1]), 0);

    let sumSquares = 0;
    for (const value of out) sumSquares += value * value;
    expect(close(Math.sqrt(sumSquares / out.length), 1, 1e-4)).toBe(true);
  });

  it('applies the learned per-channel scale', () => {
    const input = new Float32Array([1, 1, 1, 1]);
    const out = rmsNorm(input, new Float32Array([2, 4, 6, 8]), 0);
    // RMS of the input is 1, so each output is just its weight.
    expect([...out].map((v) => Math.round(v))).toEqual([2, 4, 6, 8]);
  });

  it('is scale invariant, unlike a plain sum', () => {
    const weight = new Float32Array([1, 1, 1, 1]);
    const small = rmsNorm(new Float32Array([1, 2, 3, 4]), weight, 0);
    const large = rmsNorm(new Float32Array([100, 200, 300, 400]), weight, 0);
    for (let i = 0; i < 4; i += 1) expect(close(small[i], large[i], 1e-4)).toBe(true);
  });

  it('does not divide by zero on an all-zero input', () => {
    const out = rmsNorm(new Float32Array(4), new Float32Array([1, 1, 1, 1]), 1e-6);
    expect([...out].every(Number.isFinite)).toBe(true);
  });
});

describe('matVec', () => {
  it('computes row-major dot products', () => {
    // [[1,2],[3,4],[5,6]] @ [1,10] = [21, 43, 65]
    const matrix = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect([...matVec(matrix, new Float32Array([1, 10]), 3, 2)]).toEqual([21, 43, 65]);
  });

  it('reproduces the input under an identity matrix', () => {
    const identity = new Float32Array(9);
    identity[0] = 1;
    identity[4] = 1;
    identity[8] = 1;
    expect([...matVec(identity, new Float32Array([7, 8, 9]), 3, 3)]).toEqual([7, 8, 9]);
  });
});

describe('applyRoPE', () => {
  it('is the identity at position 0', () => {
    const vector = new Float32Array([1, 2, 3, 4]);
    const before = [...vector];
    applyRoPE(vector, 1, 4, 0, 10_000);
    for (let i = 0; i < 4; i += 1) expect(close(vector[i], before[i])).toBe(true);
  });

  it('preserves the norm of every rotated pair', () => {
    const vector = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const normBefore = Math.hypot(...vector);
    applyRoPE(vector, 2, 4, 7, 10_000);
    expect(close(Math.hypot(...vector), normBefore, 1e-4)).toBe(true);
  });

  it('rotates the first pair by exactly the position angle', () => {
    // At i = 0 the frequency is 1, so the angle is the position itself.
    const vector = new Float32Array([1, 0]);
    applyRoPE(vector, 1, 2, 1, 10_000);
    expect(close(vector[0], Math.cos(1), 1e-6)).toBe(true);
    expect(close(vector[1], Math.sin(1), 1e-6)).toBe(true);
  });

  it('gives different positions different rotations', () => {
    const atFive = new Float32Array([1, 0, 1, 0]);
    const atSix = new Float32Array([1, 0, 1, 0]);
    applyRoPE(atFive, 1, 4, 5, 10_000);
    applyRoPE(atSix, 1, 4, 6, 10_000);
    expect(close(atFive[0], atSix[0])).toBe(false);
  });
});

describe('silu', () => {
  it('is zero at zero and approaches the identity for large positive input', () => {
    expect(silu(0)).toBe(0);
    expect(close(silu(20), 20, 1e-4)).toBe(true);
  });

  it('is small and negative for large negative input', () => {
    expect(silu(-20)).toBeLessThan(0);
    expect(silu(-20)).toBeGreaterThan(-1e-6);
  });
});

describe('attention', () => {
  const config: TransformerConfig = { ...TEST_CONFIG, numHeads: 2, numKVHeads: 2, headDim: 4, hiddenSize: 8 };

  it('returns the single cached value when there is only one position', () => {
    const cache = createKVCache(config);
    const width = kvHeadSize(config);
    const value = new Float32Array(width);
    for (let i = 0; i < width; i += 1) value[i] = i + 1;
    cache.keys.set(filled(width, 0.5), 0);
    cache.values.set(value, 0);
    cache.length = 1;

    const out = attention(filled(config.numHeads * config.headDim, 1), cache, config);
    // Softmax over one score is 1, so the output is that value verbatim.
    for (let i = 0; i < out.length; i += 1) expect(close(out[i], value[i], 1e-5)).toBe(true);
  });

  it('returns the common value when every cached value is identical', () => {
    const cache = createKVCache(config);
    const width = kvHeadSize(config);
    for (let t = 0; t < 5; t += 1) {
      cache.keys.set(randomVector(width, Math.random), t * width);
      cache.values.set(filled(width, 3), t * width);
    }
    cache.length = 5;

    // Whatever the attention weights are, they sum to 1 over identical values.
    const out = attention(filled(config.numHeads * config.headDim, 0.2), cache, config);
    for (const value of out) expect(close(value, 3, 1e-4)).toBe(true);
  });

  it('concentrates on the position whose key matches the query', () => {
    const cache = createKVCache(config);
    const width = kvHeadSize(config);
    for (let t = 0; t < 3; t += 1) {
      cache.keys.set(filled(width, t === 1 ? 10 : -10), t * width);
      cache.values.set(filled(width, t), t * width);
    }
    cache.length = 3;

    const out = attention(filled(config.numHeads * config.headDim, 1), cache, config);
    // Position 1's key aligns with the query, so its value should dominate.
    for (const value of out) expect(close(value, 1, 1e-3)).toBe(true);
  });

  it('routes grouped query heads to the right KV head', () => {
    // 4 query heads over 2 KV heads: heads 0,1 read KV 0 and heads 2,3 read KV 1.
    const gqa: TransformerConfig = { ...TEST_CONFIG, numHeads: 4, numKVHeads: 2, headDim: 4, hiddenSize: 16 };
    const cache = createKVCache(gqa);
    const width = kvHeadSize(gqa);

    const values = new Float32Array(width);
    values.fill(7, 0, gqa.headDim); // KV head 0
    values.fill(9, gqa.headDim); // KV head 1
    cache.keys.set(filled(width, 1), 0);
    cache.values.set(values, 0);
    cache.length = 1;

    const out = attention(filled(gqa.numHeads * gqa.headDim, 1), cache, gqa);
    for (let head = 0; head < 4; head += 1) {
      const expected = head < 2 ? 7 : 9;
      for (let d = 0; d < gqa.headDim; d += 1) {
        expect(close(out[head * gqa.headDim + d], expected, 1e-4)).toBe(true);
      }
    }
  });

  it('does not overflow on scores large enough to break a naive softmax', () => {
    const cache = createKVCache(config);
    const width = kvHeadSize(config);
    for (let t = 0; t < 3; t += 1) {
      cache.keys.set(filled(width, 500), t * width);
      cache.values.set(filled(width, t + 1), t * width);
    }
    cache.length = 3;

    const out = attention(filled(config.numHeads * config.headDim, 500), cache, config);
    expect([...out].every(Number.isFinite)).toBe(true);
  });
});

describe('forwardLayer', () => {
  it('grows the KV cache by exactly one position per token', () => {
    const weights = randomWeights(TEST_CONFIG, 1);
    const cache = createKVCache(TEST_CONFIG);
    let hidden = filled(TEST_CONFIG.hiddenSize, 0.1);

    for (let t = 0; t < 5; t += 1) {
      hidden = forwardLayer(TEST_CONFIG, weights, hidden, cache, t);
      expect(cache.length).toBe(t + 1);
    }
  });

  it('produces finite output of the right width', () => {
    const weights = randomWeights(TEST_CONFIG, 2);
    const cache = createKVCache(TEST_CONFIG);
    const random = seededRandom(3);
    const input = randomVector(TEST_CONFIG.hiddenSize, random);

    const out = forwardLayer(TEST_CONFIG, weights, input, cache, 0);
    expect(out.length).toBe(TEST_CONFIG.hiddenSize);
    expect([...out].every(Number.isFinite)).toBe(true);
  });

  it('ignores position while only one token is cached', () => {
    // Softmax over a single score is 1 whatever the score, so the output is
    // that position's value verbatim and RoPE cannot change it. Worth pinning
    // down: it means a one-token test can never catch a RoPE bug.
    const weights = randomWeights(TEST_CONFIG, 4);
    const input = filled(TEST_CONFIG.hiddenSize, 0.3);

    const atZero = forwardLayer(TEST_CONFIG, weights, input, createKVCache(TEST_CONFIG), 0);
    const atFive = forwardLayer(TEST_CONFIG, weights, input, createKVCache(TEST_CONFIG), 5);

    for (let i = 0; i < atZero.length; i += 1) {
      expect(close(atZero[i], atFive[i], 1e-5)).toBe(true);
    }
  });

  it('depends on the distance between query and key once the cache has history', () => {
    const weights = randomWeights(TEST_CONFIG, 4);
    // Constant-fill inputs would defeat this: rmsNorm is scale invariant, so
    // fill(0.3) and fill(0.7) normalise to the same vector and produce
    // identical values for attention to weight.
    const random = seededRandom(77);
    const first = randomVector(TEST_CONFIG.hiddenSize, random);
    const second = randomVector(TEST_CONFIG.hiddenSize, random);

    // Same history, same query — only the query's distance from it differs.
    const near = createKVCache(TEST_CONFIG);
    forwardLayer(TEST_CONFIG, weights, first, near, 0);
    const atOne = forwardLayer(TEST_CONFIG, weights, second, near, 1);

    const far = createKVCache(TEST_CONFIG);
    forwardLayer(TEST_CONFIG, weights, first, far, 0);
    const atTwenty = forwardLayer(TEST_CONFIG, weights, second, far, 20);

    let maxDiff = 0;
    for (let i = 0; i < atOne.length; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(atOne[i] - atTwenty[i]));
    }
    expect(maxDiff).toBeGreaterThan(1e-5);
  });

  it('refuses to write past the end of the cache', () => {
    const tiny = { ...TEST_CONFIG, maxSeqLen: 2 };
    const weights = randomWeights(tiny, 5);
    const cache = createKVCache(tiny);
    let hidden = filled(tiny.hiddenSize, 0.1);

    hidden = forwardLayer(tiny, weights, hidden, cache, 0);
    hidden = forwardLayer(tiny, weights, hidden, cache, 1);
    expect(() => forwardLayer(tiny, weights, hidden, cache, 2)).toThrow(/cache is full/);
  });
});

describe('config validation', () => {
  it('accepts the test config', () => {
    expect(() => validateConfig(TEST_CONFIG)).not.toThrow();
  });

  it('rejects a head count that does not divide evenly into KV heads', () => {
    expect(() => validateConfig({ ...TEST_CONFIG, numHeads: 5, numKVHeads: 2 })).toThrow(/multiple of/);
  });

  it('rejects a hidden size that disagrees with heads times headDim', () => {
    expect(() => validateConfig({ ...TEST_CONFIG, hiddenSize: 100 })).toThrow(/must equal hiddenSize/);
  });

  it('rejects an odd head dimension, which RoPE cannot pair', () => {
    expect(() => validateConfig({ ...TEST_CONFIG, headDim: 15, hiddenSize: 60 })).toThrow(/must be even/);
  });
});

describe('weight sizing', () => {
  it('reports bytes matching the actual arrays', () => {
    const weights = randomWeights(TEST_CONFIG, 6);
    const actual = Object.values(weights).reduce((sum, array) => sum + array.byteLength, 0);
    expect(layerWeightBytes(TEST_CONFIG)).toBe(actual);
  });

  it('is reproducible for a given seed', () => {
    expect([...randomWeights(TEST_CONFIG, 11).qProj]).toEqual([...randomWeights(TEST_CONFIG, 11).qProj]);
  });
});
