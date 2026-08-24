/**
 * layer.ts
 *
 * WGSL compute kernels for one decoder layer.
 *
 * These are written for clarity and verifiability, not peak throughput: one
 * thread per output element, straightforward loops, no tiling or subgroup
 * tricks. Every kernel is checked against the reference implementation in
 * transformer.ts, and an attention kernel that is subtly wrong produces
 * plausible numbers rather than an error — so correctness comes first and
 * optimisation is a later, separately verifiable step.
 *
 * Every kernel shares one uniform block so the host binds a single params
 * buffer everywhere.
 */

/** Fields must match `writeParams` in transformer-gpu.ts, in order. */
export const PARAMS_STRUCT = /* wgsl */ `
struct Params {
  hiddenSize    : u32,
  numHeads      : u32,
  numKVHeads    : u32,
  headDim       : u32,
  ffnHiddenSize : u32,
  attnWidth     : u32,   // numHeads * headDim
  kvWidth       : u32,   // numKVHeads * headDim
  seqLen        : u32,   // cache positions in use, including this token
  position      : u32,   // this token's position
  groupSize     : u32,   // numHeads / numKVHeads
  ropeTheta     : f32,
  rmsNormEps    : f32,
}
@group(0) @binding(0) var<uniform> params : Params;
`;

/** Workgroup size used by every kernel; dispatches are ceil(n / 64). */
export const WORKGROUP_SIZE = 64;

/**
 * RMS normalisation.
 *
 * A single workgroup reduces the sum of squares in shared memory, then every
 * thread scales its own elements. One workgroup keeps the reduction correct
 * without a second dispatch; hidden sizes here are small enough that the lost
 * parallelism does not matter.
 */
export const RMSNORM_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read>       input  : array<f32>;
@group(0) @binding(2) var<storage, read>       weight : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

var<workgroup> partial : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local : vec3<u32>) {
  let lane = local.x;
  let n = params.hiddenSize;

  var sum = 0.0;
  var i = lane;
  loop {
    if (i >= n) { break; }
    let v = input[i];
    sum = sum + v * v;
    i = i + 64u;
  }
  partial[lane] = sum;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  let scale = inverseSqrt(partial[0] / f32(n) + params.rmsNormEps);

  var j = lane;
  loop {
    if (j >= n) { break; }
    output[j] = input[j] * scale * weight[j];
    j = j + 64u;
  }
}
`;

/**
 * Matrix-vector product: `out[row] = dot(matrix[row], input)`.
 *
 * The matrix is row-major [outFeatures][inFeatures], matching both the
 * reference implementation and the safetensors layout for these tensors.
 */
export const MATVEC_WGSL = /* wgsl */ `
// No Params block here on purpose: \`layout: 'auto'\` derives the bind group
// layout from what the shader actually references, so declaring an unused
// uniform makes every bind group that binds it invalid.
struct Dims { outFeatures : u32, inFeatures : u32, _pad0 : u32, _pad1 : u32 }
@group(0) @binding(0) var<uniform> dims : Dims;
@group(0) @binding(1) var<storage, read>       matrix : array<f32>;
@group(0) @binding(2) var<storage, read>       input  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  if (row >= dims.outFeatures) { return; }

  let base = row * dims.inFeatures;
  var sum = 0.0;
  for (var col = 0u; col < dims.inFeatures; col = col + 1u) {
    sum = sum + matrix[base + col] * input[col];
  }
  output[row] = sum;
}
`;

/**
 * Rotary position embedding, in place.
 *
 * One thread per (head, pair). `headCount` is passed separately because
 * queries have numHeads and keys have numKVHeads.
 */
export const ROPE_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
struct RopeDims { headCount : u32, _pad0 : u32, _pad1 : u32, _pad2 : u32 }
@group(0) @binding(1) var<uniform> dims : RopeDims;
@group(0) @binding(2) var<storage, read_write> vector : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let halfDim = params.headDim / 2u;
  let total = dims.headCount * halfDim;
  if (gid.x >= total) { return; }

  let head = gid.x / halfDim;
  let pair = gid.x % halfDim;

  let exponent = (2.0 * f32(pair)) / f32(params.headDim);
  let freq = 1.0 / pow(params.ropeTheta, exponent);
  let angle = f32(params.position) * freq;
  let c = cos(angle);
  let s = sin(angle);

  let base = head * params.headDim + pair * 2u;
  let a = vector[base];
  let b = vector[base + 1u];
  vector[base]      = a * c - b * s;
  vector[base + 1u] = a * s + b * c;
}
`;

/** Append this token's key and value into the cache at `seqLen - 1`. */
export const KV_APPEND_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read>       key      : array<f32>;
@group(0) @binding(2) var<storage, read>       value    : array<f32>;
@group(0) @binding(3) var<storage, read_write> keyCache : array<f32>;
@group(0) @binding(4) var<storage, read_write> valCache : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.kvWidth) { return; }
  let slot = (params.seqLen - 1u) * params.kvWidth + gid.x;
  keyCache[slot] = key[gid.x];
  valCache[slot] = value[gid.x];
}
`;

/**
 * Attention scores: one thread per (head, cached position).
 *
 * Grouped-query attention means head h reads KV head h / groupSize, so several
 * query heads share one key/value stream.
 */
export const ATTN_SCORES_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read>       query    : array<f32>;
@group(0) @binding(2) var<storage, read>       keyCache : array<f32>;
@group(0) @binding(3) var<storage, read_write> scores   : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = params.numHeads * params.seqLen;
  if (gid.x >= total) { return; }

  let head = gid.x / params.seqLen;
  let pos  = gid.x % params.seqLen;
  let kvHead = head / params.groupSize;

  let qBase = head * params.headDim;
  let kBase = pos * params.kvWidth + kvHead * params.headDim;

  var dot = 0.0;
  for (var d = 0u; d < params.headDim; d = d + 1u) {
    dot = dot + query[qBase + d] * keyCache[kBase + d];
  }
  scores[gid.x] = dot * inverseSqrt(f32(params.headDim));
}
`;

/**
 * Softmax over each head's scores. One workgroup per head, max-subtracted so
 * long contexts cannot overflow exp().
 */
export const ATTN_SOFTMAX_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read_write> scores : array<f32>;

var<workgroup> reduce : array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) group : vec3<u32>,
  @builtin(local_invocation_id) local : vec3<u32>,
) {
  let head = group.x;
  let lane = local.x;
  let n = params.seqLen;
  let base = head * n;

  // Pass 1 — maximum.
  // Not f32::MIN: the literal 3.4028235e38 rounds *above* f32 max and fails
  // to parse. Attention scores never come near this, so it is a safe floor.
  var best = -1e30;
  var i = lane;
  loop {
    if (i >= n) { break; }
    best = max(best, scores[base + i]);
    i = i + 64u;
  }
  reduce[lane] = best;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { reduce[lane] = max(reduce[lane], reduce[lane + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let maxScore = reduce[0];
  workgroupBarrier();

  // Pass 2 — exponentiate and sum.
  var sum = 0.0;
  var j = lane;
  loop {
    if (j >= n) { break; }
    let e = exp(scores[base + j] - maxScore);
    scores[base + j] = e;
    sum = sum + e;
    j = j + 64u;
  }
  reduce[lane] = sum;
  workgroupBarrier();

  stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { reduce[lane] = reduce[lane] + reduce[lane + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let total = reduce[0];
  let inv = select(0.0, 1.0 / total, total > 0.0);

  // Pass 3 — normalise.
  var k = lane;
  loop {
    if (k >= n) { break; }
    scores[base + k] = scores[base + k] * inv;
    k = k + 64u;
  }
}
`;

/** Weighted sum of cached values: one thread per (head, dimension). */
export const ATTN_OUTPUT_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read>       scores   : array<f32>;
@group(0) @binding(2) var<storage, read>       valCache : array<f32>;
@group(0) @binding(3) var<storage, read_write> output   : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.attnWidth) { return; }

  let head = gid.x / params.headDim;
  let dim  = gid.x % params.headDim;
  let kvHead = head / params.groupSize;
  let scoreBase = head * params.seqLen;

  var sum = 0.0;
  for (var pos = 0u; pos < params.seqLen; pos = pos + 1u) {
    let vIndex = pos * params.kvWidth + kvHead * params.headDim + dim;
    sum = sum + scores[scoreBase + pos] * valCache[vIndex];
  }
  output[gid.x] = sum;
}
`;

/** Elementwise `out[i] = a[i] + b[i]`, used for both residual connections. */
export const ADD_WGSL = /* wgsl */ `
// No Params block — see the note on MATVEC_WGSL.
struct Dims { count : u32, _pad0 : u32, _pad1 : u32, _pad2 : u32 }
@group(0) @binding(0) var<uniform> dims : Dims;
@group(0) @binding(1) var<storage, read>       a      : array<f32>;
@group(0) @binding(2) var<storage, read>       b      : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= dims.count) { return; }
  output[gid.x] = a[gid.x] + b[gid.x];
}
`;

/** SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`. */
export const SWIGLU_WGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(1) var<storage, read>       gate   : array<f32>;
@group(0) @binding(2) var<storage, read>       up     : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.ffnHiddenSize) { return; }
  let g = gate[gid.x];
  output[gid.x] = (g / (1.0 + exp(-g))) * up[gid.x];
}
`;
