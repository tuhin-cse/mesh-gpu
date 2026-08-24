/**
 * gpu-budget.ts
 *
 * Measures how much GPU memory a browser tab will actually hand out.
 *
 * The previous approach regex-matched `GPUAdapterInfo.description` against a
 * hardcoded device table. Two things are wrong with that. Chrome commonly
 * returns empty strings for those fields, so the lookup misses entirely on
 * many machines. And even a hit answers the wrong question: an RTX 4090 has
 * 24 GB, but a tab will not get anywhere near that — `maxBufferSize` is
 * typically capped near 2 GiB and the per-process budget is well below the
 * card's capacity.
 *
 * So: allocate real buffers until allocation fails, then release them. The
 * number that comes back is the number that matters.
 */

export interface GpuBudget {
  /** Bytes successfully allocated at once, in buffers of `blockBytes`. */
  usableBytes: number;
  /** Largest single buffer this device permits. */
  maxBufferBytes: number;
  /** Buffers allocated before the device refused. */
  blocksAllocated: number;
  blockBytes: number;
  /** True if probing stopped at the cap rather than at a real failure. */
  hitCeiling: boolean;
  durationMs: number;
}

export interface ProbeOptions {
  /** Allocation granularity. Larger is faster but coarser. Default 256 MiB. */
  blockBytes?: number;
  /** Stop here even if allocation keeps succeeding. Default 32 GiB. */
  ceilingBytes?: number;
  /** Abandon the probe after this long. Default 5s. */
  budgetMs?: number;
}

const MiB = 1024 * 1024;
const GiB = 1024 * MiB;

/**
 * `GPUBufferUsage.STORAGE` from the WebGPU spec. Read the global when it
 * exists and fall back to the literal otherwise, so this module can be
 * imported — and its logic tested — outside a browser.
 */
const STORAGE_USAGE =
  (globalThis as { GPUBufferUsage?: { STORAGE: number } }).GPUBufferUsage?.STORAGE ?? 0x0080;

const DEFAULT_BLOCK_BYTES = 256 * MiB;
const DEFAULT_CEILING_BYTES = 32 * GiB;
const DEFAULT_BUDGET_MS = 5_000;

/**
 * Probe the device's real allocation limit.
 *
 * Every buffer is destroyed before returning, including on failure, so this
 * leaves no memory held. It is safe to run while other work is in flight, but
 * the answer will be lower — run it at startup for a clean reading.
 */
export async function probeGpuBudget(
  device: GPUDevice,
  options: ProbeOptions = {},
): Promise<GpuBudget> {
  const blockBytes = Math.max(MiB, options.blockBytes ?? DEFAULT_BLOCK_BYTES);
  const ceilingBytes = options.ceilingBytes ?? DEFAULT_CEILING_BYTES;
  const budgetMs = options.budgetMs ?? DEFAULT_BUDGET_MS;

  const started = performance.now();
  const maxBufferBytes = Math.min(blockBytes, device.limits.maxBufferSize ?? blockBytes);

  const buffers: GPUBuffer[] = [];
  let hitCeiling = false;

  // An uncaptured device error is how WebGPU reports an out-of-memory
  // allocation; without a scope it would surface as an unhandled console
  // error rather than something we can act on.
  try {
    for (;;) {
      if ((buffers.length + 1) * blockBytes > ceilingBytes) {
        hitCeiling = true;
        break;
      }
      if (performance.now() - started > budgetMs) {
        hitCeiling = true;
        break;
      }

      const buffer = await tryAllocate(device, maxBufferBytes);
      if (!buffer) break;
      buffers.push(buffer);
    }
  } finally {
    for (const buffer of buffers) {
      try {
        buffer.destroy();
      } catch {
        // Already lost with the device — nothing left to release.
      }
    }
  }

  return {
    usableBytes: buffers.length * maxBufferBytes,
    maxBufferBytes,
    blocksAllocated: buffers.length,
    blockBytes: maxBufferBytes,
    hitCeiling,
    durationMs: performance.now() - started,
  };
}

/**
 * Allocate one buffer, returning null if the device refuses.
 *
 * `createBuffer` can fail two ways: it may throw, or it may return a buffer
 * and report the failure asynchronously through an error scope. Both count.
 */
async function tryAllocate(device: GPUDevice, size: number): Promise<GPUBuffer | null> {
  device.pushErrorScope('out-of-memory');

  let buffer: GPUBuffer | null = null;
  try {
    buffer = device.createBuffer({ size, usage: STORAGE_USAGE });
  } catch {
    await device.popErrorScope().catch(() => null);
    return null;
  }

  const error = await device.popErrorScope().catch(() => null);
  if (error) {
    try {
      buffer.destroy();
    } catch {
      // Nothing to do — the allocation never really happened.
    }
    return null;
  }

  return buffer;
}

// ---------------------------------------------------------------------------
// Turning a budget into a layer count
// ---------------------------------------------------------------------------

export interface StageCapacity {
  /** Layers this node can host after reserving workspace. */
  hostableLayers: number;
  bytesPerLayer: number;
  usableBytes: number;
  reservedBytes: number;
  /** Layers a single buffer's worth of weights allows, as a sanity bound. */
  maxLayersPerBuffer: number;
  reason: string;
}

export interface CapacityInput {
  totalLayers: number;
  /** Whole-model weight size in bytes, at the quantization being served. */
  modelBytes: number;
  budget: Pick<GpuBudget, 'usableBytes' | 'maxBufferBytes'>;
  /** Share held back for activations, KV cache and scratch. Default 0.25. */
  headroomRatio?: number;
}

/**
 * Convert a measured budget into a layer count.
 *
 * Headroom defaults higher than the old 0.2 because the KV cache grows with
 * context length and is easy to underestimate — running out mid-generation
 * loses the whole sequence, while reserving a little too much only costs a
 * layer or two.
 */
export function capacityFromBudget(input: CapacityInput): StageCapacity {
  const headroomRatio = clamp(input.headroomRatio ?? 0.25, 0, 0.9);
  const bytesPerLayer = Math.ceil(input.modelBytes / Math.max(1, input.totalLayers));

  const reservedBytes = Math.floor(input.budget.usableBytes * headroomRatio);
  const usableBytes = Math.max(0, input.budget.usableBytes - reservedBytes);

  const byMemory = Math.floor(usableBytes / bytesPerLayer);
  const maxLayersPerBuffer = Math.max(1, Math.floor(input.budget.maxBufferBytes / bytesPerLayer));

  const hostableLayers = Math.max(0, Math.min(input.totalLayers, byMemory));

  return {
    hostableLayers,
    bytesPerLayer,
    usableBytes,
    reservedBytes,
    maxLayersPerBuffer,
    reason:
      hostableLayers === 0
        ? `measured ${formatBytes(usableBytes)} usable, but one layer needs `
          + `${formatBytes(bytesPerLayer)} — this node cannot host a stage of this model`
        : `${formatBytes(usableBytes)} measured usable after ${(headroomRatio * 100).toFixed(0)}% `
          + `headroom (${formatBytes(reservedBytes)} held for activations and KV cache)`,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatBytes(bytes: number): string {
  if (bytes >= GiB) return `${(bytes / GiB).toFixed(1)} GiB`;
  if (bytes >= MiB) return `${(bytes / MiB).toFixed(0)} MiB`;
  return `${bytes} B`;
}
