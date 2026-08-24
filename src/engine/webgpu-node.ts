/**
 * webgpu-node.ts
 *
 * MeshGPU — WebGPU capability inspector and per-node layer allocator.
 *
 * WebGPU does NOT expose a standard "dedicated VRAM size" field. This module
 * therefore:
 *   1. Queries the GPUAdapter for its limits + features (the things WebGPU
 *      *does* expose reliably: maxBufferSize, maxStorageBufferBindingSize,
 *      compute workgroup limits, f16/subgroup feature support, …).
 *   2. Estimates VRAM by parsing `GPUAdapterInfo.description` against a curated
 *      device database (Apple Silicon, NVIDIA, AMD, Intel).
 *   3. Falls back to `navigator.deviceMemory` for unified-memory systems.
 *   4. Uses the estimate to compute how many transformer layers this node can
 *      host for a given model profile (pipeline-parallelism stage sizing).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AdapterInfoSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

/** WebGPU limits relevant to storing + executing transformer tensors. */
export interface GPULimitsSummary {
  maxBufferSize: number | null;
  maxStorageBufferBindingSize: number | null;
  maxStorageBuffersPerShaderStage: number | null;
  maxComputeWorkgroupStorageSize: number | null;
  maxComputeWorkgroupSizeX: number | null;
  maxComputeWorkgroupSizeY: number | null;
  maxComputeWorkgroupSizeZ: number | null;
  maxComputeInvocationsPerWorkgroup: number | null;
  maxComputeWorkgroupsPerDimension: number | null;
  maxBindGroups: number | null;
  maxTextureDimension2D: number | null;
}

export type VRAMEstimateSource = 'description-parse' | 'unified-memory-heuristic' | 'none';

export interface WebGPUInfo {
  /** `navigator.gpu` exists in this browser context. */
  gpuApiPresent: boolean;
  /** A GPU adapter could be requested. */
  adapterAcquired: boolean;
  adapterInfo: AdapterInfoSummary | null;
  limits: GPULimitsSummary | null;
  /** ML-relevant features (f16 shaders, subgroups, …) the adapter reports. */
  mlRelevantFeatures: GPUFeatureName[];
  /** Estimated dedicated/unified memory pool in bytes, or null if unknown. */
  estimatedVRAMBytes: number | null;
  vramSource: VRAMEstimateSource;
  /** Human-readable explanation of how the VRAM estimate was derived. */
  vramNote: string;
  /** Live device (created with raised buffer limits) — null if creation failed. */
  device: GPUDevice | null;
  error: string | null;
}

/** A quantized model whose layers can be sharded across peers. */
export interface ModelProfile {
  id: string;
  name: string;
  paramCount: number;
  layerCount: number;
  /** Bytes per parameter: 2 = fp16/bf16, 1 = int8, 0.5 = int4. */
  bytesPerParam: number;
}

export interface LayerAllocation {
  profile: ModelProfile;
  /** Full fp16 (or quantized) model size in bytes. */
  totalModelVRAMBytes: number;
  /** Approximate weights size for a single transformer layer. */
  bytesPerLayer: number;
  /** Bytes left for the model after reserving workspace headroom. */
  usableVRAMBytes: number | null;
  reservedForWorkspaceBytes: number;
  /** Number of transformer layers this node can host. */
  hostableLayers: number;
  reason: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const GiB = 1024 ** 3;
export const MiB = 1024 ** 2;

/**
 * ML-relevant WebGPU features MeshGPU probes for. f16 shaders roughly halve
 * register pressure/bandwidth vs f32; subgroups enable cross-lane matmul
 * reductions. Kept as `string` for forward-compatibility with lib.dom builds
 * that predate the newest feature names.
 */
export const ML_RELEVANT_FEATURES: readonly string[] = [
  'shader-f16',
  'shader-f64',
  'subgroups',
  'subgroups-f16',
  'float32-filterable',
];

/**
 * Curated model registry. Layer counts reflect the model's transformer blocks;
 * weights are fp16 unless quantized.
 */
export const MODEL_PROFILES: readonly ModelProfile[] = [
  {
    id: 'qwen2.5-7b-instruct',
    name: 'Qwen2.5-7B-Instruct',
    paramCount: 7_615_949_824, // 7.6B params
    layerCount: 28,
    bytesPerParam: 2,
  },
  {
    id: 'qwen2.5-3b-instruct',
    name: 'Qwen2.5-3B-Instruct',
    paramCount: 3_087_000_000,
    layerCount: 36,
    bytesPerParam: 2,
  },
  {
    id: 'qwen2.5-1.5b-instruct',
    name: 'Qwen2.5-1.5B-Instruct',
    paramCount: 1_536_000_000,
    layerCount: 28,
    bytesPerParam: 2,
  },
  {
    id: 'llama-3.2-3b-instruct',
    name: 'Llama-3.2-3B-Instruct',
    paramCount: 3_212_000_000,
    layerCount: 28,
    bytesPerParam: 2,
  },
];

// ---------------------------------------------------------------------------
// VRAM estimation database
//
// Browsers do not expose VRAM size. We parse `GPUAdapterInfo.description`
// (e.g. "NVIDIA GeForce RTX 4090", "Apple M2 Pro") against this table.
// Entries are order-sensitive: more specific names come first so that, e.g.,
// "M1 Max" matches before "M1". Apple Silicon values are the *base* unified
// memory configs (conservative).
// ---------------------------------------------------------------------------

interface VRAMEntry {
  /** Regex tested case-insensitively against the adapter description. */
  pattern: RegExp;
  vramBytes: number;
  note: string;
}

const VRAM_DATABASE: readonly VRAMEntry[] = [
  // Apple Silicon (unified memory — base configs, conservative)
  { pattern: /\bm1 ultra\b/i, vramBytes: 64 * GiB, note: 'Apple M1 Ultra (64 GB base)' },
  { pattern: /\bm1 max\b/i, vramBytes: 32 * GiB, note: 'Apple M1 Max (32 GB base)' },
  { pattern: /\bm1 pro\b/i, vramBytes: 16 * GiB, note: 'Apple M1 Pro (16 GB base)' },
  { pattern: /\bm1\b/i, vramBytes: 8 * GiB, note: 'Apple M1 (8 GB base)' },
  { pattern: /\bm2 ultra\b/i, vramBytes: 64 * GiB, note: 'Apple M2 Ultra (64 GB base)' },
  { pattern: /\bm2 max\b/i, vramBytes: 32 * GiB, note: 'Apple M2 Max (32 GB base)' },
  { pattern: /\bm2 pro\b/i, vramBytes: 16 * GiB, note: 'Apple M2 Pro (16 GB base)' },
  { pattern: /\bm2\b/i, vramBytes: 8 * GiB, note: 'Apple M2 (8 GB base)' },
  { pattern: /\bm3 ultra\b/i, vramBytes: 96 * GiB, note: 'Apple M3 Ultra (96 GB base)' },
  { pattern: /\bm3 max\b/i, vramBytes: 36 * GiB, note: 'Apple M3 Max (36 GB base)' },
  { pattern: /\bm3 pro\b/i, vramBytes: 18 * GiB, note: 'Apple M3 Pro (18 GB base)' },
  { pattern: /\bm3\b/i, vramBytes: 8 * GiB, note: 'Apple M3 (8 GB base)' },
  { pattern: /\bm4 ultra\b/i, vramBytes: 96 * GiB, note: 'Apple M4 Ultra (96 GB base)' },
  { pattern: /\bm4 max\b/i, vramBytes: 36 * GiB, note: 'Apple M4 Max (36 GB base)' },
  { pattern: /\bm4 pro\b/i, vramBytes: 24 * GiB, note: 'Apple M4 Pro (24 GB base)' },
  { pattern: /\bm4\b/i, vramBytes: 16 * GiB, note: 'Apple M4 (16 GB base)' },

  // NVIDIA GeForce / RTX
  { pattern: /\brtx 5090\b/i, vramBytes: 32 * GiB, note: 'NVIDIA RTX 5090' },
  { pattern: /\brtx 5080\b/i, vramBytes: 16 * GiB, note: 'NVIDIA RTX 5080' },
  { pattern: /\brtx 5070 ti\b/i, vramBytes: 16 * GiB, note: 'NVIDIA RTX 5070 Ti' },
  { pattern: /\brtx 5070\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 5070' },
  { pattern: /\brtx 5060 ti\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 5060 Ti (8 GB base)' },
  { pattern: /\brtx 5060\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 5060' },
  { pattern: /\brtx 4090\b/i, vramBytes: 24 * GiB, note: 'NVIDIA RTX 4090' },
  { pattern: /\brtx 4080 super\b/i, vramBytes: 16 * GiB, note: 'NVIDIA RTX 4080 SUPER' },
  { pattern: /\brtx 4080\b/i, vramBytes: 16 * GiB, note: 'NVIDIA RTX 4080' },
  { pattern: /\brtx 4070 ti super\b/i, vramBytes: 16 * GiB, note: 'NVIDIA RTX 4070 Ti SUPER' },
  { pattern: /\brtx 4070 ti\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 4070 Ti' },
  { pattern: /\brtx 4070 super\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 4070 SUPER' },
  { pattern: /\brtx 4070\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 4070' },
  { pattern: /\brtx 4060 ti\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 4060 Ti' },
  { pattern: /\brtx 4060\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 4060' },
  { pattern: /\brtx 3090 ti\b/i, vramBytes: 24 * GiB, note: 'NVIDIA RTX 3090 Ti' },
  { pattern: /\brtx 3090\b/i, vramBytes: 24 * GiB, note: 'NVIDIA RTX 3090' },
  { pattern: /\brtx 3080 ti\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 3080 Ti' },
  { pattern: /\brtx 3080\b/i, vramBytes: 10 * GiB, note: 'NVIDIA RTX 3080' },
  { pattern: /\brtx 3070 ti\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 3070 Ti' },
  { pattern: /\brtx 3070\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 3070' },
  { pattern: /\brtx 3060 ti\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 3060 Ti' },
  { pattern: /\brtx 3060\b/i, vramBytes: 12 * GiB, note: 'NVIDIA RTX 3060' },
  { pattern: /\brtx 3050\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 3050' },
  { pattern: /\brtx 2080 ti\b/i, vramBytes: 11 * GiB, note: 'NVIDIA RTX 2080 Ti' },
  { pattern: /\brtx 2080\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 2080' },
  { pattern: /\brtx 2070\b/i, vramBytes: 8 * GiB, note: 'NVIDIA RTX 2070' },
  { pattern: /\brtx 2060\b/i, vramBytes: 6 * GiB, note: 'NVIDIA RTX 2060' },
  { pattern: /\bgtx 1080 ti\b/i, vramBytes: 11 * GiB, note: 'NVIDIA GTX 1080 Ti' },
  { pattern: /\bgtx 1080\b/i, vramBytes: 8 * GiB, note: 'NVIDIA GTX 1080' },
  { pattern: /\bgtx 1070\b/i, vramBytes: 8 * GiB, note: 'NVIDIA GTX 1070' },
  { pattern: /\bgtx 1060\b/i, vramBytes: 6 * GiB, note: 'NVIDIA GTX 1060' },
  { pattern: /\bgtx 1660\b/i, vramBytes: 6 * GiB, note: 'NVIDIA GTX 1660' },
  { pattern: /\bgtx 1650\b/i, vramBytes: 4 * GiB, note: 'NVIDIA GTX 1650' },

  // NVIDIA datacenter
  { pattern: /\ba100\b/i, vramBytes: 40 * GiB, note: 'NVIDIA A100 (40 GB base)' },
  { pattern: /\ba6000\b/i, vramBytes: 48 * GiB, note: 'NVIDIA RTX A6000' },
  { pattern: /\bh100\b/i, vramBytes: 80 * GiB, note: 'NVIDIA H100' },
  { pattern: /\bv100\b/i, vramBytes: 16 * GiB, note: 'NVIDIA V100 (16 GB base)' },
  { pattern: /\bt4\b/i, vramBytes: 16 * GiB, note: 'NVIDIA T4' },

  // AMD Radeon
  { pattern: /\brx 9070 xt\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 9070 XT' },
  { pattern: /\brx 9070\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 9070' },
  { pattern: /\brx 7900 xtx\b/i, vramBytes: 24 * GiB, note: 'AMD Radeon RX 7900 XTX' },
  { pattern: /\brx 7900 xt\b/i, vramBytes: 20 * GiB, note: 'AMD Radeon RX 7900 XT' },
  { pattern: /\brx 7800 xt\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 7800 XT' },
  { pattern: /\brx 7700 xt\b/i, vramBytes: 12 * GiB, note: 'AMD Radeon RX 7700 XT' },
  { pattern: /\brx 7600 xt\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 7600 XT' },
  { pattern: /\brx 7600\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX 7600' },
  { pattern: /\brx 6900 xt\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 6900 XT' },
  { pattern: /\brx 6800 xt\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 6800 XT' },
  { pattern: /\brx 6800\b/i, vramBytes: 16 * GiB, note: 'AMD Radeon RX 6800' },
  { pattern: /\brx 6700 xt\b/i, vramBytes: 12 * GiB, note: 'AMD Radeon RX 6700 XT' },
  { pattern: /\brx 6600 xt\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX 6600 XT' },
  { pattern: /\brx 6600\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX 6600' },
  { pattern: /\brx 5700 xt\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX 5700 XT' },
  { pattern: /\bvega 64\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX Vega 64' },
  { pattern: /\brx 580\b/i, vramBytes: 8 * GiB, note: 'AMD Radeon RX 580' },

  // Intel Arc
  { pattern: /\barc a770\b/i, vramBytes: 16 * GiB, note: 'Intel Arc A770' },
  { pattern: /\barc a750\b/i, vramBytes: 8 * GiB, note: 'Intel Arc A750' },
  { pattern: /\barc a580\b/i, vramBytes: 8 * GiB, note: 'Intel Arc A580' },
];

export interface VRAMEstimate {
  vramBytes: number | null;
  source: VRAMEstimateSource;
  note: string;
}

/** Estimate the VRAM pool from the adapter description (and device-memory hint). */
export function estimateVRAM(adapterInfo: AdapterInfoSummary): VRAMEstimate {
  const description = adapterInfo.description.toLowerCase();

  for (const entry of VRAM_DATABASE) {
    if (entry.pattern.test(description)) {
      return { vramBytes: entry.vramBytes, source: 'description-parse', note: entry.note };
    }
  }

  // Unified-memory fallback for Apple / Intel integrated graphics: the GPU
  // shares system RAM, which `navigator.deviceMemory` approximates in GiB.
  // Some browsers leave `description` empty, so also sniff vendor/architecture.
  const vendor = adapterInfo.vendor.toLowerCase();
  const architecture = adapterInfo.architecture.toLowerCase();
  const isUnified =
    /\bapple\b/i.test(description) ||
    /\bintel\b/i.test(description) ||
    vendor.includes('apple') ||
    vendor.includes('intel') ||
    architecture.includes('metal');
  const deviceMemoryGiB = getDeviceMemoryGiB();
  if (isUnified && deviceMemoryGiB !== null) {
    return {
      vramBytes: deviceMemoryGiB * GiB,
      source: 'unified-memory-heuristic',
      note:
        `Unified memory fallback: navigator.deviceMemory ≈ ${deviceMemoryGiB} GiB. `
        + 'Browsers clamp this value at 8 GiB, so machines with more memory are under-reported.',
    };
  }

  return {
    vramBytes: null,
    source: 'none',
    note: 'No VRAM estimate available for this adapter',
  };
}

// ---------------------------------------------------------------------------
// Layer allocation
// ---------------------------------------------------------------------------

export interface LayerAllocationOptions {
  /** Fraction of VRAM reserved for activations, KV-cache and workspace. Default 0.2. */
  headroomRatio?: number;
}

/**
 * Compute how many contiguous transformer layers a node can host.
 *
 * Example: Qwen2.5-7B = 7.6B params → 15.2 GB in fp16 → 28 layers ≈ 543 MB/layer.
 * With 4 GB VRAM and 0% headroom that's ~7 layers; with the default 20% headroom
 * it's ~5. Treat headroom as safety margin for activations/KV-cache/workspace.
 */
export function allocateLayersForModel(
  profile: ModelProfile,
  vramBytes: number | null,
  options: LayerAllocationOptions = {},
): LayerAllocation {
  const headroomRatio = clamp(options.headroomRatio ?? 0.2, 0, 0.9);
  const totalModelVRAMBytes = profile.paramCount * profile.bytesPerParam;
  const bytesPerLayer = Math.ceil(totalModelVRAMBytes / profile.layerCount);

  if (vramBytes === null || vramBytes <= 0) {
    return {
      profile,
      totalModelVRAMBytes,
      bytesPerLayer,
      usableVRAMBytes: null,
      reservedForWorkspaceBytes: 0,
      hostableLayers: 0,
      reason: 'VRAM estimate unavailable — run on a WebGPU-capable device or assign a stage manually.',
    };
  }

  const reservedForWorkspaceBytes = Math.floor(vramBytes * headroomRatio);
  const usableVRAMBytes = vramBytes - reservedForWorkspaceBytes;
  const hostableLayers = Math.max(
    0,
    Math.min(profile.layerCount, Math.floor(usableVRAMBytes / bytesPerLayer)),
  );

  return {
    profile,
    totalModelVRAMBytes,
    bytesPerLayer,
    usableVRAMBytes,
    reservedForWorkspaceBytes,
    hostableLayers,
    reason: `${formatBytes(usableVRAMBytes)} usable after ${(headroomRatio * 100).toFixed(0)}% headroom `
      + `(${formatBytes(reservedForWorkspaceBytes)} reserved for activations / KV-cache / workspace).`,
  };
}

export function getModelProfile(id: string): ModelProfile | undefined {
  return MODEL_PROFILES.find((profile) => profile.id === id);
}

// ---------------------------------------------------------------------------
// WebGPU inspection
// ---------------------------------------------------------------------------

/** Extract the ML-relevant subset of an adapter's limits into a plain object. */
export function extractLimits(adapter: GPUAdapter): GPULimitsSummary {
  const limits = adapter.limits;
  return {
    maxBufferSize: limits.maxBufferSize ?? null,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize ?? null,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage ?? null,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize ?? null,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX ?? null,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY ?? null,
    maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ ?? null,
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup ?? null,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension ?? null,
    maxBindGroups: limits.maxBindGroups ?? null,
    maxTextureDimension2D: limits.maxTextureDimension2D ?? null,
  };
}

/**
 * Create a GPUDevice with the adapter's maximum buffer limits. The WebGPU spec
 * defaults maxBufferSize to 256 MiB, which is far too small to hold transformer
 * weights; requesting the adapter maximum is required before any real work.
 */
export async function createDevice(
  adapter: GPUAdapter,
  requiredFeatures: GPUFeatureName[] = [],
): Promise<GPUDevice> {
  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxBufferSize !== undefined) {
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  }
  if (adapter.limits.maxStorageBufferBindingSize !== undefined) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  return adapter.requestDevice({ requiredFeatures, requiredLimits });
}

/** Inspect the local WebGPU stack end-to-end. Never throws. */
export async function inspectWebGPU(): Promise<WebGPUInfo> {
  const base: WebGPUInfo = {
    gpuApiPresent: false,
    adapterAcquired: false,
    adapterInfo: null,
    limits: null,
    mlRelevantFeatures: [],
    estimatedVRAMBytes: null,
    vramSource: 'none',
    vramNote: '',
    device: null,
    error: null,
  };

  const gpu = getGPU();
  if (!gpu) {
    base.error =
      'WebGPU is not supported here. Use a Chromium/Edge build (or Firefox Nightly / '
      + 'Safari Technology Preview) with WebGPU enabled, served over https or localhost.';
    return base;
  }
  base.gpuApiPresent = true;

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (err) {
    base.error = `requestAdapter failed: ${errorMessage(err)}`;
    return base;
  }

  if (!adapter) {
    base.error = 'No WebGPU adapter available (GPU disabled or blocklisted).';
    return base;
  }
  base.adapterAcquired = true;

  base.adapterInfo = await readAdapterInfo(adapter);

  base.limits = extractLimits(adapter);
  base.mlRelevantFeatures = ML_RELEVANT_FEATURES.filter((feature) =>
    adapter.features.has(feature as GPUFeatureName),
  ) as GPUFeatureName[];

  const estimate = base.adapterInfo
    ? estimateVRAM(base.adapterInfo)
    : ({ vramBytes: null, source: 'none', note: 'Adapter info unavailable' } satisfies VRAMEstimate);
  base.estimatedVRAMBytes = estimate.vramBytes;
  base.vramSource = estimate.source;
  base.vramNote = estimate.note;

  // Create a live device with raised buffer limits so large tensors fit.
  base.device = await createDevice(adapter, base.mlRelevantFeatures).catch(() => null);

  return base;
}

/**
 * Lifecycle wrapper for a single browser tab's GPU node. Holds the inspected
 * capabilities + live device, and answers "how many layers can I host?".
 */
export class WebGPUNode {
  private constructor(public readonly info: WebGPUInfo) {}

  static async create(): Promise<WebGPUNode> {
    return new WebGPUNode(await inspectWebGPU());
  }

  get device(): GPUDevice | null {
    return this.info.device;
  }

  allocateFor(profileId: string, headroomRatio = 0.2): LayerAllocation | null {
    const profile = getModelProfile(profileId);
    if (!profile) return null;
    return allocateLayersForModel(profile, this.info.estimatedVRAMBytes, { headroomRatio });
  }

  dispose(): void {
    this.info.device?.destroy();
  }
}

// ---------------------------------------------------------------------------
// Tensor integrity + GPU micro-benchmark
// ---------------------------------------------------------------------------

export interface TensorSanitizeResult {
  ok: boolean;
  reason: string;
  nonFiniteCount: number;
}

/**
 * Lightweight guardrails for tensors arriving from remote peers: validates the
 * element count (against the decoded shape) and rejects NaN/Inf payloads before
 * they reach local WebGPU buffers.
 */
export function sanitizeTensor(data: Float32Array, expectedLength?: number): TensorSanitizeResult {
  if (expectedLength !== undefined && data.length !== expectedLength) {
    return {
      ok: false,
      reason: `size mismatch: got ${data.length} elements, expected ${expectedLength}`,
      nonFiniteCount: 0,
    };
  }

  let nonFinite = 0;
  for (let i = 0; i < data.length; i += 1) {
    if (!Number.isFinite(data[i])) nonFinite += 1;
  }

  if (nonFinite > 0) {
    return {
      ok: false,
      reason: `${nonFinite} non-finite (NaN/Inf) values`,
      nonFiniteCount: nonFinite,
    };
  }

  return { ok: true, reason: 'ok', nonFiniteCount: 0 };
}

const FLOPS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> buf: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  var acc = buf[i];
  for (var k = 0u; k < 2048u; k = k + 1u) {
    acc = fma(acc, vec4<f32>(1.0000001), vec4<f32>(0.0000001));
  }
  buf[i] = acc;
}
`;

const BANDWIDTH_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  dst[id.x] = src[id.x];
}
`;

export interface GpuBenchmarkResult {
  gflops: number;
  bandwidthGiBps: number;
  durationMs: number;
}

/**
 * Two-second pre-flight micro-benchmark: measures raw compute throughput via an
 * FMA kernel and memory bandwidth via a copy kernel. Used to weight pipeline
 * layer assignment by actual GPU speed (avoids the straggler bottleneck).
 */
export async function benchmarkWebGPU(
  device: GPUDevice,
  durationMs = 2000,
): Promise<GpuBenchmarkResult> {
  const started = performance.now();
  const gflops = await benchmarkFlops(device, durationMs);
  const bandwidthGiBps = await benchmarkBandwidth(device, durationMs);
  return { gflops, bandwidthGiBps, durationMs: performance.now() - started };
}

/** Relative throughput score; 1.0 = reference device. Clamped to [0.1, 10]. */
export function computeThroughputScore(benchmark: GpuBenchmarkResult): number {
  if (benchmark.gflops <= 0) return 1;
  const flopsScore = benchmark.gflops / 1000; // → TFLOPS
  const bwScore = benchmark.bandwidthGiBps / 100; // → ×100 GB/s
  return Math.max(0.1, Math.min(10, flopsScore * 0.7 + bwScore * 0.3));
}

async function benchmarkFlops(device: GPUDevice, durationMs: number): Promise<number> {
  const elements = 1 << 16; // 65536 vec4 = 1 MiB
  const iterations = 2048;
  const flopsPerDispatch = elements * 4 * 2 * iterations; // 4 lanes × 2 flops/FMA

  const module = device.createShaderModule({ code: FLOPS_WGSL });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const buffer = device.createBuffer({
    size: elements * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }],
  });

  const dispatch = (): void => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(elements / 256);
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  dispatch(); // warm-up + shader compilation
  await device.queue.onSubmittedWorkDone();

  const batch = 8;
  let dispatches = 0;
  const start = performance.now();
  let elapsed = 0;
  while (elapsed < durationMs) {
    for (let i = 0; i < batch; i += 1) dispatch();
    dispatches += batch;
    await device.queue.onSubmittedWorkDone();
    elapsed = performance.now() - start;
  }

  buffer.destroy();
  return (dispatches * flopsPerDispatch) / (Math.max(elapsed, 1) / 1000) / 1e9;
}

async function benchmarkBandwidth(device: GPUDevice, durationMs: number): Promise<number> {
  const elements = 1 << 18; // 262144 vec4 = 4 MiB per buffer
  const bytesPerPass = 2 * elements * 16; // read + write

  const module = device.createShaderModule({ code: BANDWIDTH_WGSL });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const src = device.createBuffer({
    size: elements * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const dst = device.createBuffer({
    size: elements * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
    ],
  });

  const dispatch = (): void => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(elements / 256);
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  dispatch(); // warm-up + shader compilation
  await device.queue.onSubmittedWorkDone();

  const batch = 8;
  let passes = 0;
  const start = performance.now();
  let elapsed = 0;
  while (elapsed < durationMs) {
    for (let i = 0; i < batch; i += 1) dispatch();
    passes += batch;
    await device.queue.onSubmittedWorkDone();
    elapsed = performance.now() - start;
  }

  src.destroy();
  dst.destroy();
  return (passes * bytesPerPass) / GiB / (Math.max(elapsed, 1) / 1000);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function formatBytes(bytes: number | null | undefined, decimals = 1): string {
  if (bytes === null || bytes === undefined || bytes === 0) return '—';
  if (bytes >= GiB) return `${(bytes / GiB).toFixed(decimals)} GiB`;
  if (bytes >= MiB) return `${(bytes / MiB).toFixed(decimals)} MiB`;
  return `${bytes.toFixed(0)} B`;
}

function getGPU(): GPU | null {
  if (typeof navigator === 'undefined') return null;
  const nav = navigator as Navigator & { gpu?: GPU };
  return nav.gpu ?? null;
}

function getDeviceMemoryGiB(): number | null {
  if (typeof navigator === 'undefined') return null;
  const nav = navigator as Navigator & { deviceMemory?: number };
  return typeof nav.deviceMemory === 'number' && nav.deviceMemory > 0 ? nav.deviceMemory : null;
}

interface AdapterInfoRuntime {
  info?: GPUAdapterInfo;
  requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
}

function toAdapterSummary(raw: GPUAdapterInfo): AdapterInfoSummary {
  return {
    vendor: raw.vendor,
    architecture: raw.architecture,
    device: raw.device,
    description: raw.description,
  };
}

/**
 * Read adapter info across API generations. The WebGPU spec moved from the
 * async `requestAdapterInfo()` (what Chromium ships today) to a synchronous
 * `info` attribute; support both so the inspector works now and later.
 */
async function readAdapterInfo(adapter: GPUAdapter): Promise<AdapterInfoSummary | null> {
  const runtime = adapter as unknown as AdapterInfoRuntime;

  if (runtime.info) {
    return toAdapterSummary(runtime.info);
  }

  if (typeof runtime.requestAdapterInfo === 'function') {
    const raw = await runtime.requestAdapterInfo.call(adapter).catch(() => null);
    return raw ? toAdapterSummary(raw) : null;
  }

  return null;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
