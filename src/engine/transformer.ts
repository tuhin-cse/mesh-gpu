/**
 * transformer.ts
 *
 * Model configuration, weight layout, and a plain-TypeScript reference
 * implementation of one Llama/Qwen-style decoder layer.
 *
 * This exists to be *obviously correct* rather than fast. The WebGPU kernels in
 * transformer-gpu.ts are checked against it, which is the only practical way to
 * tell a working attention kernel from one that produces plausible-looking
 * numbers. Nothing here runs in the hot path.
 *
 * Layer structure (pre-norm, RoPE, grouped-query attention, SwiGLU):
 *
 *   h  = rmsNorm(x, inputNorm)
 *   q,k,v = h @ Wq, h @ Wk, h @ Wv        (k and v are narrower under GQA)
 *   q,k   = rope(q, k, position)
 *   append k,v to the KV cache
 *   a  = attention(q, kCache, vCache) @ Wo
 *   x  = x + a
 *   h2 = rmsNorm(x, postAttnNorm)
 *   x  = x + (silu(h2 @ Wgate) * (h2 @ Wup)) @ Wdown
 */

export interface TransformerConfig {
  hiddenSize: number;
  numHeads: number;
  /** Key/value heads. Fewer than numHeads means grouped-query attention. */
  numKVHeads: number;
  headDim: number;
  /** Inner width of the SwiGLU feed-forward block. */
  ffnHiddenSize: number;
  rmsNormEps: number;
  /** RoPE base frequency. 10000 for Llama 2, 1e6 for Qwen2.5 and Llama 3. */
  ropeTheta: number;
  /** Positions the KV cache is sized for. */
  maxSeqLen: number;
}

/**
 * Weights for one layer. Projection matrices are row-major
 * `[outFeatures][inFeatures]`, so `out[i] = dot(W[i], in)` — the layout
 * safetensors uses for these tensors, and the one both backends index against.
 */
export interface LayerWeights {
  inputNorm: Float32Array;      // [hiddenSize]
  qProj: Float32Array;          // [numHeads * headDim][hiddenSize]
  kProj: Float32Array;          // [numKVHeads * headDim][hiddenSize]
  vProj: Float32Array;          // [numKVHeads * headDim][hiddenSize]
  oProj: Float32Array;          // [hiddenSize][numHeads * headDim]
  postAttnNorm: Float32Array;   // [hiddenSize]
  gateProj: Float32Array;       // [ffnHiddenSize][hiddenSize]
  upProj: Float32Array;         // [ffnHiddenSize][hiddenSize]
  downProj: Float32Array;       // [hiddenSize][ffnHiddenSize]
}

/** Per-sequence attention state. One of these per concurrent sequence. */
export interface KVCache {
  /** [maxSeqLen][numKVHeads * headDim], laid out flat. */
  keys: Float32Array;
  values: Float32Array;
  /** Positions written so far. */
  length: number;
}

export function attentionHeadSize(config: TransformerConfig): number {
  return config.numHeads * config.headDim;
}

export function kvHeadSize(config: TransformerConfig): number {
  return config.numKVHeads * config.headDim;
}

export function createKVCache(config: TransformerConfig): KVCache {
  const width = kvHeadSize(config);
  return {
    keys: new Float32Array(config.maxSeqLen * width),
    values: new Float32Array(config.maxSeqLen * width),
    length: 0,
  };
}

/** Bytes one layer's weights occupy at f32, for capacity planning. */
export function layerWeightBytes(config: TransformerConfig): number {
  const h = config.hiddenSize;
  const a = attentionHeadSize(config);
  const kv = kvHeadSize(config);
  const f = config.ffnHiddenSize;
  const elements = h + a * h + kv * h + kv * h + h * a + h + f * h + f * h + h * f;
  return elements * 4;
}

/**
 * Validate a config before any buffer is sized from it. A mismatch here shows
 * up as silently wrong numbers rather than an error, so it is worth catching.
 */
export function validateConfig(config: TransformerConfig): void {
  const problems: string[] = [];
  if (config.numHeads % config.numKVHeads !== 0) {
    problems.push(`numHeads (${config.numHeads}) must be a multiple of numKVHeads (${config.numKVHeads})`);
  }
  if (config.numHeads * config.headDim !== config.hiddenSize) {
    problems.push(
      `numHeads * headDim (${config.numHeads * config.headDim}) must equal hiddenSize (${config.hiddenSize})`,
    );
  }
  if (config.headDim % 2 !== 0) {
    problems.push(`headDim (${config.headDim}) must be even — RoPE rotates pairs`);
  }
  if (problems.length > 0) throw new Error(`invalid transformer config: ${problems.join('; ')}`);
}

// ---------------------------------------------------------------------------
// Reference kernels
// ---------------------------------------------------------------------------

export function rmsNorm(input: Float32Array, weight: Float32Array, eps: number): Float32Array {
  let sumSquares = 0;
  for (let i = 0; i < input.length; i += 1) sumSquares += input[i] * input[i];
  const scale = 1 / Math.sqrt(sumSquares / input.length + eps);

  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) out[i] = input[i] * scale * weight[i];
  return out;
}

/** out[i] = dot(matrix row i, input). Matrix is [outFeatures][inFeatures]. */
export function matVec(
  matrix: Float32Array,
  input: Float32Array,
  outFeatures: number,
  inFeatures: number,
): Float32Array {
  const out = new Float32Array(outFeatures);
  for (let row = 0; row < outFeatures; row += 1) {
    let sum = 0;
    const base = row * inFeatures;
    for (let col = 0; col < inFeatures; col += 1) sum += matrix[base + col] * input[col];
    out[row] = sum;
  }
  return out;
}

/**
 * Rotary position embedding, applied in place.
 *
 * Rotates the (2i, 2i+1) pairs within each head by an angle that grows with
 * position and shrinks with i, which is what lets attention scores depend on
 * relative rather than absolute position.
 */
export function applyRoPE(
  vector: Float32Array,
  numHeads: number,
  headDim: number,
  position: number,
  theta: number,
): void {
  const halfDim = headDim / 2;
  for (let head = 0; head < numHeads; head += 1) {
    const base = head * headDim;
    for (let i = 0; i < halfDim; i += 1) {
      const freq = 1 / theta ** ((2 * i) / headDim);
      const angle = position * freq;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const a = vector[base + 2 * i];
      const b = vector[base + 2 * i + 1];
      vector[base + 2 * i] = a * cos - b * sin;
      vector[base + 2 * i + 1] = a * sin + b * cos;
    }
  }
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/**
 * Grouped-query attention over the cache, for one query position.
 *
 * Every query head reads the key/value head it is grouped under, so a model
 * with 4 query heads and 2 KV heads has heads 0,1 sharing KV head 0.
 */
export function attention(
  query: Float32Array,
  cache: KVCache,
  config: TransformerConfig,
): Float32Array {
  const { numHeads, numKVHeads, headDim } = config;
  const groupSize = numHeads / numKVHeads;
  const kvWidth = kvHeadSize(config);
  const scale = 1 / Math.sqrt(headDim);
  const out = new Float32Array(numHeads * headDim);

  for (let head = 0; head < numHeads; head += 1) {
    const kvHead = Math.floor(head / groupSize);
    const qBase = head * headDim;
    const kvBase = kvHead * headDim;

    const scores = new Float32Array(cache.length);
    let maxScore = -Infinity;
    for (let t = 0; t < cache.length; t += 1) {
      let dot = 0;
      const keyBase = t * kvWidth + kvBase;
      for (let d = 0; d < headDim; d += 1) dot += query[qBase + d] * cache.keys[keyBase + d];
      const score = dot * scale;
      scores[t] = score;
      if (score > maxScore) maxScore = score;
    }

    // Subtract the max before exponentiating, or long contexts overflow.
    let sumExp = 0;
    for (let t = 0; t < cache.length; t += 1) {
      const value = Math.exp(scores[t] - maxScore);
      scores[t] = value;
      sumExp += value;
    }

    const inverse = sumExp > 0 ? 1 / sumExp : 0;
    for (let t = 0; t < cache.length; t += 1) {
      const weight = scores[t] * inverse;
      if (weight === 0) continue;
      const valueBase = t * kvWidth + kvBase;
      for (let d = 0; d < headDim; d += 1) {
        out[qBase + d] += weight * cache.values[valueBase + d];
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// One decoder layer
// ---------------------------------------------------------------------------

/**
 * Run a single token through one decoder layer, appending to the KV cache.
 * Returns the new hidden state. `cache` is mutated.
 */
export function forwardLayer(
  config: TransformerConfig,
  weights: LayerWeights,
  hidden: Float32Array,
  cache: KVCache,
  position: number,
): Float32Array {
  const h = config.hiddenSize;
  const attnWidth = attentionHeadSize(config);
  const kvWidth = kvHeadSize(config);

  if (cache.length >= config.maxSeqLen) {
    throw new Error(`KV cache is full at ${config.maxSeqLen} positions`);
  }

  const normed = rmsNorm(hidden, weights.inputNorm, config.rmsNormEps);

  const query = matVec(weights.qProj, normed, attnWidth, h);
  const key = matVec(weights.kProj, normed, kvWidth, h);
  const value = matVec(weights.vProj, normed, kvWidth, h);

  applyRoPE(query, config.numHeads, config.headDim, position, config.ropeTheta);
  applyRoPE(key, config.numKVHeads, config.headDim, position, config.ropeTheta);

  const slot = cache.length * kvWidth;
  cache.keys.set(key, slot);
  cache.values.set(value, slot);
  cache.length += 1;

  const attended = attention(query, cache, config);
  const projected = matVec(weights.oProj, attended, h, attnWidth);

  const afterAttention = new Float32Array(h);
  for (let i = 0; i < h; i += 1) afterAttention[i] = hidden[i] + projected[i];

  const normed2 = rmsNorm(afterAttention, weights.postAttnNorm, config.rmsNormEps);
  const gate = matVec(weights.gateProj, normed2, config.ffnHiddenSize, h);
  const up = matVec(weights.upProj, normed2, config.ffnHiddenSize, h);

  const activated = new Float32Array(config.ffnHiddenSize);
  for (let i = 0; i < config.ffnHiddenSize; i += 1) activated[i] = silu(gate[i]) * up[i];

  const down = matVec(weights.downProj, activated, h, config.ffnHiddenSize);

  const out = new Float32Array(h);
  for (let i = 0; i < h; i += 1) out[i] = afterAttention[i] + down[i];
  return out;
}

// ---------------------------------------------------------------------------
// Deterministic weights for testing
// ---------------------------------------------------------------------------

/** Small xorshift PRNG so both backends can be fed byte-identical weights. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x1_0000_0000;
  };
}

/**
 * Random weights at a realistic scale. Not a trained model — enough to verify
 * that two implementations of the same maths agree.
 */
export function randomWeights(config: TransformerConfig, seed = 42): LayerWeights {
  const random = seededRandom(seed);
  const h = config.hiddenSize;
  const a = attentionHeadSize(config);
  const kv = kvHeadSize(config);
  const f = config.ffnHiddenSize;

  // Roughly 1/sqrt(fan_in), which keeps activations from exploding across
  // layers the way uniform [-1,1) weights would.
  const fill = (length: number, scale: number): Float32Array => {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i += 1) out[i] = (random() * 2 - 1) * scale;
    return out;
  };
  const ones = (length: number): Float32Array => {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i += 1) out[i] = 0.8 + random() * 0.4;
    return out;
  };

  const hScale = 1 / Math.sqrt(h);
  const fScale = 1 / Math.sqrt(f);
  const aScale = 1 / Math.sqrt(a);

  return {
    inputNorm: ones(h),
    qProj: fill(a * h, hScale),
    kProj: fill(kv * h, hScale),
    vProj: fill(kv * h, hScale),
    oProj: fill(h * a, aScale),
    postAttnNorm: ones(h),
    gateProj: fill(f * h, hScale),
    upProj: fill(f * h, hScale),
    downProj: fill(h * f, fScale),
  };
}

/** A small but structurally complete config: GQA with 2 query heads per KV head. */
export const TEST_CONFIG: TransformerConfig = {
  hiddenSize: 64,
  numHeads: 4,
  numKVHeads: 2,
  headDim: 16,
  ffnHiddenSize: 128,
  rmsNormEps: 1e-6,
  ropeTheta: 10_000,
  maxSeqLen: 64,
};
