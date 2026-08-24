/**
 * transformer-gpu.ts
 *
 * Executes decoder layers on WebGPU, and adapts them to the pipeline's
 * StageExecutor interface so a peer can host a real contiguous layer range
 * instead of the identity function.
 *
 * Scope, stated plainly: this runs the layer maths correctly on the GPU and is
 * verified element-by-element against the reference implementation. It does
 * not yet load pretrained weights — there is no safetensors reader and no
 * tokenizer here — so it cannot serve a real model on its own. What it proves
 * is that the compute path and the transport can carry a genuine forward pass
 * between peers.
 */

import {
  ADD_WGSL,
  ATTN_OUTPUT_WGSL,
  ATTN_SCORES_WGSL,
  ATTN_SOFTMAX_WGSL,
  KV_APPEND_WGSL,
  MATVEC_WGSL,
  RMSNORM_WGSL,
  ROPE_WGSL,
  SWIGLU_WGSL,
  WORKGROUP_SIZE,
} from './wgsl/layer';
import {
  attentionHeadSize,
  kvHeadSize,
  layerWeightBytes,
  validateConfig,
} from './transformer';
import type { LayerWeights, TransformerConfig } from './transformer';
import type { ExecContext, StageExecutor, StageRange } from './p2p-pipeline';

const PARAMS_FLOATS = 12;
const PARAMS_BYTES = PARAMS_FLOATS * 4;
/** Uniform buffers must be at least 16-byte aligned; these hold 4 u32 slots. */
const DIMS_BYTES = 16;

const STORAGE = 0x0080; // GPUBufferUsage.STORAGE
const UNIFORM = 0x0040; // GPUBufferUsage.UNIFORM
const COPY_DST = 0x0008;
const COPY_SRC = 0x0004;
const MAP_READ = 0x0001;

/**
 * TypeScript types a Float32Array over ArrayBufferLike, which includes
 * SharedArrayBuffer, while `writeBuffer` insists on ArrayBuffer. The views
 * here are always plain-backed, so this narrows rather than lies.
 */
function asSource(view: Float32Array | Uint32Array): GPUAllowSharedBufferSource {
  return view as unknown as GPUAllowSharedBufferSource;
}

function dispatchCount(items: number): number {
  return Math.max(1, Math.ceil(items / WORKGROUP_SIZE));
}

interface LayerBuffers {
  inputNorm: GPUBuffer;
  qProj: GPUBuffer;
  kProj: GPUBuffer;
  vProj: GPUBuffer;
  oProj: GPUBuffer;
  postAttnNorm: GPUBuffer;
  gateProj: GPUBuffer;
  upProj: GPUBuffer;
  downProj: GPUBuffer;
  /** One KV cache per layer, per sequence. Keyed by sequence in `caches`. */
}

interface SequenceCache {
  keys: GPUBuffer[];
  values: GPUBuffer[];
  length: number;
}

/**
 * Holds compiled pipelines, weights and KV caches for a contiguous range of
 * layers. One instance per peer stage.
 */
export class GpuTransformerStage implements StageExecutor {
  private readonly device: GPUDevice;
  private readonly config: TransformerConfig;
  private readonly layers: LayerBuffers[] = [];
  private readonly caches = new Map<number, SequenceCache>();

  private readonly pipelines: Record<string, GPUComputePipeline>;
  private readonly params: GPUBuffer;
  private readonly dims: Map<string, GPUBuffer> = new Map();
  private readonly scratch: {
    hidden: GPUBuffer;
    normed: GPUBuffer;
    query: GPUBuffer;
    key: GPUBuffer;
    value: GPUBuffer;
    scores: GPUBuffer;
    attnOut: GPUBuffer;
    projected: GPUBuffer;
    residual: GPUBuffer;
    normed2: GPUBuffer;
    gate: GPUBuffer;
    up: GPUBuffer;
    activated: GPUBuffer;
    down: GPUBuffer;
    output: GPUBuffer;
    readback: GPUBuffer;
    upload: GPUBuffer;
  };

  private destroyed = false;

  private constructor(device: GPUDevice, config: TransformerConfig) {
    this.device = device;
    this.config = config;

    const make = (source: string): GPUComputePipeline =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: source }), entryPoint: 'main' },
      });

    this.pipelines = {
      rmsNorm: make(RMSNORM_WGSL),
      matVec: make(MATVEC_WGSL),
      rope: make(ROPE_WGSL),
      kvAppend: make(KV_APPEND_WGSL),
      attnScores: make(ATTN_SCORES_WGSL),
      attnSoftmax: make(ATTN_SOFTMAX_WGSL),
      attnOutput: make(ATTN_OUTPUT_WGSL),
      add: make(ADD_WGSL),
      swiglu: make(SWIGLU_WGSL),
    };

    this.params = device.createBuffer({ size: PARAMS_BYTES, usage: UNIFORM | COPY_DST });

    const h = config.hiddenSize;
    const a = attentionHeadSize(config);
    const kv = kvHeadSize(config);
    const f = config.ffnHiddenSize;
    const maxScores = config.numHeads * config.maxSeqLen;

    const storage = (elements: number, extra = 0): GPUBuffer =>
      device.createBuffer({ size: elements * 4, usage: STORAGE | COPY_DST | COPY_SRC | extra });

    this.scratch = {
      hidden: storage(h),
      normed: storage(h),
      query: storage(a),
      key: storage(kv),
      value: storage(kv),
      scores: storage(maxScores),
      attnOut: storage(a),
      projected: storage(h),
      residual: storage(h),
      normed2: storage(h),
      gate: storage(f),
      up: storage(f),
      activated: storage(f),
      down: storage(h),
      output: storage(h),
      readback: device.createBuffer({ size: h * 4, usage: COPY_DST | MAP_READ }),
      upload: device.createBuffer({ size: h * 4, usage: STORAGE | COPY_DST | COPY_SRC }),
    };
  }

  /**
   * Build a stage that hosts `weights.length` consecutive layers.
   *
   * `stage` is informational here — the executor holds exactly the layers it
   * was given, and the pipeline decides which range they correspond to.
   */
  static create(
    device: GPUDevice,
    config: TransformerConfig,
    weights: readonly LayerWeights[],
  ): GpuTransformerStage {
    validateConfig(config);
    const stage = new GpuTransformerStage(device, config);
    for (const layer of weights) stage.addLayer(layer);
    return stage;
  }

  get layerCount(): number {
    return this.layers.length;
  }

  /** Approximate device memory held by this stage's weights. */
  get weightBytes(): number {
    return this.layers.length * layerWeightBytes(this.config);
  }

  private addLayer(weights: LayerWeights): void {
    const upload = (data: Float32Array): GPUBuffer => {
      const buffer = this.device.createBuffer({
        size: Math.max(4, data.byteLength),
        usage: STORAGE | COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, asSource(data));
      return buffer;
    };

    this.layers.push({
      inputNorm: upload(weights.inputNorm),
      qProj: upload(weights.qProj),
      kProj: upload(weights.kProj),
      vProj: upload(weights.vProj),
      oProj: upload(weights.oProj),
      postAttnNorm: upload(weights.postAttnNorm),
      gateProj: upload(weights.gateProj),
      upProj: upload(weights.upProj),
      downProj: upload(weights.downProj),
    });
  }

  // -------------------------------------------------------------------------
  // StageExecutor
  // -------------------------------------------------------------------------

  /**
   * Run every layer this stage holds over one token's hidden state.
   *
   * Each sequence gets its own KV cache, so concurrent sequences do not read
   * one another's history — the bug that would make microbatching produce
   * quietly wrong text rather than an error.
   */
  async execute(
    _stage: StageRange,
    input: Float32Array,
    context: ExecContext,
  ): Promise<Float32Array> {
    this.assertLive();
    if (input.length !== this.config.hiddenSize) {
      throw new Error(
        `stage expects a ${this.config.hiddenSize}-element hidden state, got ${input.length}`,
      );
    }

    const cache = this.cacheFor(context.seqId);
    if (cache.length >= this.config.maxSeqLen) {
      throw new Error(`KV cache for sequence ${context.seqId} is full`);
    }
    cache.length += 1;

    this.writeParams(cache.length, context.tokenIndex);
    this.device.queue.writeBuffer(this.scratch.hidden, 0, asSource(input));

    for (let index = 0; index < this.layers.length; index += 1) {
      this.encodeLayer(index, cache);
    }

    return this.readHidden();
  }

  /** Release a sequence's KV cache once it will not be extended again. */
  releaseSequence(seqId: number): void {
    const cache = this.caches.get(seqId);
    if (!cache) return;
    for (const buffer of cache.keys) buffer.destroy();
    for (const buffer of cache.values) buffer.destroy();
    this.caches.delete(seqId);
  }

  get activeSequences(): number {
    return this.caches.size;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    for (const seqId of [...this.caches.keys()]) this.releaseSequence(seqId);
    for (const layer of this.layers) {
      for (const buffer of Object.values(layer)) (buffer as GPUBuffer).destroy();
    }
    for (const buffer of Object.values(this.scratch)) buffer.destroy();
    for (const buffer of this.dims.values()) buffer.destroy();
    this.params.destroy();
  }

  // -------------------------------------------------------------------------
  // Encoding
  // -------------------------------------------------------------------------

  private encodeLayer(index: number, cache: SequenceCache): void {
    const layer = this.layers[index];
    const encoder = this.device.createCommandEncoder();
    const h = this.config.hiddenSize;
    const a = attentionHeadSize(this.config);
    const kv = kvHeadSize(this.config);
    const f = this.config.ffnHiddenSize;
    const s = this.scratch;

    // Layer 0 reads the uploaded hidden state; later layers read the previous
    // layer's output, which was copied back into `hidden`.
    this.rmsNorm(encoder, s.hidden, layer.inputNorm, s.normed);

    this.matVec(encoder, 'q', layer.qProj, s.normed, s.query, a, h);
    this.matVec(encoder, 'k', layer.kProj, s.normed, s.key, kv, h);
    this.matVec(encoder, 'v', layer.vProj, s.normed, s.value, kv, h);

    this.rope(encoder, s.query, this.config.numHeads);
    this.rope(encoder, s.key, this.config.numKVHeads);

    this.kvAppend(encoder, cache, index);
    this.attention(encoder, cache, index);

    this.matVec(encoder, 'o', layer.oProj, s.attnOut, s.projected, h, a);
    this.add(encoder, s.hidden, s.projected, s.residual, h);

    this.rmsNorm(encoder, s.residual, layer.postAttnNorm, s.normed2);
    this.matVec(encoder, 'gate', layer.gateProj, s.normed2, s.gate, f, h);
    this.matVec(encoder, 'up', layer.upProj, s.normed2, s.up, f, h);
    this.swiglu(encoder, s.gate, s.up, s.activated);
    this.matVec(encoder, 'down', layer.downProj, s.activated, s.down, h, f);
    this.add(encoder, s.residual, s.down, s.output, h);

    // Feed this layer's output into the next one.
    encoder.copyBufferToBuffer(s.output, 0, s.hidden, 0, h * 4);
    this.device.queue.submit([encoder.finish()]);
  }

  private rmsNorm(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    weight: GPUBuffer,
    output: GPUBuffer,
  ): void {
    const pipeline = this.pipelines.rmsNorm;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: weight } },
          { binding: 3, resource: { buffer: output } },
        ],
      }),
    );
    pass.dispatchWorkgroups(1); // one workgroup: the reduction is shared-memory
    pass.end();
  }

  private matVec(
    encoder: GPUCommandEncoder,
    key: string,
    matrix: GPUBuffer,
    input: GPUBuffer,
    output: GPUBuffer,
    outFeatures: number,
    inFeatures: number,
  ): void {
    const dims = this.dimsBuffer(`matvec:${key}`, [outFeatures, inFeatures, 0, 0]);
    const pipeline = this.pipelines.matVec;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dims } },
          { binding: 1, resource: { buffer: matrix } },
          { binding: 2, resource: { buffer: input } },
          { binding: 3, resource: { buffer: output } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(outFeatures));
    pass.end();
  }

  private rope(encoder: GPUCommandEncoder, vector: GPUBuffer, headCount: number): void {
    const dims = this.dimsBuffer(`rope:${headCount}`, [headCount, 0, 0, 0]);
    const pipeline = this.pipelines.rope;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: dims } },
          { binding: 2, resource: { buffer: vector } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount((headCount * this.config.headDim) / 2));
    pass.end();
  }

  private kvAppend(encoder: GPUCommandEncoder, cache: SequenceCache, layer: number): void {
    const pipeline = this.pipelines.kvAppend;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: this.scratch.key } },
          { binding: 2, resource: { buffer: this.scratch.value } },
          { binding: 3, resource: { buffer: cache.keys[layer] } },
          { binding: 4, resource: { buffer: cache.values[layer] } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(kvHeadSize(this.config)));
    pass.end();
  }

  private attention(encoder: GPUCommandEncoder, cache: SequenceCache, layer: number): void {
    const { numHeads } = this.config;
    const s = this.scratch;

    const scoresPipeline = this.pipelines.attnScores;
    let pass = encoder.beginComputePass();
    pass.setPipeline(scoresPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: scoresPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: s.query } },
          { binding: 2, resource: { buffer: cache.keys[layer] } },
          { binding: 3, resource: { buffer: s.scores } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(numHeads * cache.length));
    pass.end();

    const softmaxPipeline = this.pipelines.attnSoftmax;
    pass = encoder.beginComputePass();
    pass.setPipeline(softmaxPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: softmaxPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: s.scores } },
        ],
      }),
    );
    pass.dispatchWorkgroups(numHeads); // one workgroup per head
    pass.end();

    const outputPipeline = this.pipelines.attnOutput;
    pass = encoder.beginComputePass();
    pass.setPipeline(outputPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: outputPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: s.scores } },
          { binding: 2, resource: { buffer: cache.values[layer] } },
          { binding: 3, resource: { buffer: s.attnOut } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(attentionHeadSize(this.config)));
    pass.end();
  }

  private add(
    encoder: GPUCommandEncoder,
    a: GPUBuffer,
    b: GPUBuffer,
    output: GPUBuffer,
    count: number,
  ): void {
    const dims = this.dimsBuffer(`add:${count}`, [count, 0, 0, 0]);
    const pipeline = this.pipelines.add;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dims } },
          { binding: 1, resource: { buffer: a } },
          { binding: 2, resource: { buffer: b } },
          { binding: 3, resource: { buffer: output } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(count));
    pass.end();
  }

  private swiglu(
    encoder: GPUCommandEncoder,
    gate: GPUBuffer,
    up: GPUBuffer,
    output: GPUBuffer,
  ): void {
    const pipeline = this.pipelines.swiglu;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: gate } },
          { binding: 2, resource: { buffer: up } },
          { binding: 3, resource: { buffer: output } },
        ],
      }),
    );
    pass.dispatchWorkgroups(dispatchCount(this.config.ffnHiddenSize));
    pass.end();
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private cacheFor(seqId: number): SequenceCache {
    const existing = this.caches.get(seqId);
    if (existing) return existing;

    const width = kvHeadSize(this.config) * this.config.maxSeqLen * 4;
    const keys: GPUBuffer[] = [];
    const values: GPUBuffer[] = [];
    for (let i = 0; i < this.layers.length; i += 1) {
      keys.push(this.device.createBuffer({ size: width, usage: STORAGE | COPY_DST | COPY_SRC }));
      values.push(this.device.createBuffer({ size: width, usage: STORAGE | COPY_DST | COPY_SRC }));
    }

    const cache: SequenceCache = { keys, values, length: 0 };
    this.caches.set(seqId, cache);
    return cache;
  }

  /** Uniform blocks are tiny and reused every token, so they are cached by key. */
  private dimsBuffer(key: string, values: [number, number, number, number]): GPUBuffer {
    let buffer = this.dims.get(key);
    if (!buffer) {
      buffer = this.device.createBuffer({ size: DIMS_BYTES, usage: UNIFORM | COPY_DST });
      this.dims.set(key, buffer);
    }
    this.device.queue.writeBuffer(buffer, 0, asSource(new Uint32Array(values)));
    return buffer;
  }

  /** Field order must match the Params struct in wgsl/layer.ts exactly. */
  private writeParams(seqLen: number, position: number): void {
    const data = new ArrayBuffer(PARAMS_BYTES);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    const c = this.config;

    u32[0] = c.hiddenSize;
    u32[1] = c.numHeads;
    u32[2] = c.numKVHeads;
    u32[3] = c.headDim;
    u32[4] = c.ffnHiddenSize;
    u32[5] = attentionHeadSize(c);
    u32[6] = kvHeadSize(c);
    u32[7] = seqLen;
    u32[8] = position;
    u32[9] = c.numHeads / c.numKVHeads;
    f32[10] = c.ropeTheta;
    f32[11] = c.rmsNormEps;

    this.device.queue.writeBuffer(this.params, 0, data);
  }

  private async readHidden(): Promise<Float32Array> {
    const bytes = this.config.hiddenSize * 4;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.scratch.output, 0, this.scratch.readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);

    await this.scratch.readback.mapAsync(1 /* GPUMapMode.READ */);
    const copy = new Float32Array(this.scratch.readback.getMappedRange().slice(0));
    this.scratch.readback.unmap();
    return copy;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error('this GPU stage has been destroyed');
  }
}
