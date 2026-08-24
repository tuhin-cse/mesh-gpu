/**
 * tensor-wire.ts
 *
 * Binary wire format for hidden-state tensors moving between mesh peers.
 *
 * Version 2 adds the two things version 1 could not survive contact with a
 * real model:
 *
 *   - **Chunking.** A 2,048-token prefill at hidden dim 3,584 is 29 MB in one
 *     tensor. WebRTC DataChannels do not carry single messages of that size,
 *     so frames are split across a chunk index/count carried in the header and
 *     reassembled on arrival.
 *   - **f16.** Hidden states are activations, not weights; half precision is
 *     ample and halves everything on the wire.
 *
 * Header layout (32 bytes, little-endian):
 *
 *   0..3    uint32  magic ("MGPU")
 *   4       uint8   version (2)
 *   5       uint8   frame kind
 *   6       uint8   dtype
 *   7       uint8   rank
 *   8..11   uint32  seqId
 *   12..15  uint32  tokenIndex
 *   16..19  uint32  fromLayer
 *   20..23  uint32  toLayer
 *   24..27  uint32  this chunk's payload byte length
 *   28..29  uint16  chunk index
 *   30..31  uint16  chunk count
 *   32..    uint32[rank] shape dims
 *   ...     payload
 */

export const TENSOR_MAGIC = 0x4d475055;
export const TENSOR_VERSION = 2;
export const TENSOR_HEADER_BYTES = 32;

/**
 * Chunk payload ceiling. SCTP handles larger messages by fragmenting, but
 * browsers differ on where reliability degrades, and 64 KiB is comfortably
 * under every implementation's limit while keeping per-chunk overhead trivial.
 */
export const DEFAULT_MAX_CHUNK_BYTES = 64 * 1024;

/** Chunk index and count are uint16, so this is the hard ceiling per frame. */
export const MAX_CHUNKS = 65_535;

export enum TensorDType {
  F32 = 1,
  F16 = 2,
  I32 = 3,
  U8 = 4,
}

export enum TensorFrameKind {
  /** Hidden-state tensor flowing forward through the pipeline. */
  Forward = 1,
  /** Reserved for future binary token frames. */
  OutputToken = 2,
}

export interface TensorFrame {
  kind: TensorFrameKind;
  dtype: TensorDType;
  shape: number[];
  seqId: number;
  tokenIndex: number;
  fromLayer: number;
  toLayer: number;
  chunkIndex: number;
  chunkCount: number;
  /** This chunk's payload bytes. */
  data: ArrayBuffer;
}

export function bytesPerElement(dtype: TensorDType): number {
  switch (dtype) {
    case TensorDType.F32:
    case TensorDType.I32:
      return 4;
    case TensorDType.F16:
      return 2;
    case TensorDType.U8:
      return 1;
    default:
      throw new Error(`unknown dtype ${dtype}`);
  }
}

export function elementCount(shape: readonly number[]): number {
  return shape.reduce((total, dim) => total * dim, 1);
}

// ---------------------------------------------------------------------------
// Half precision
// ---------------------------------------------------------------------------

const scratch = new ArrayBuffer(4);
const scratchF32 = new Float32Array(scratch);
const scratchU32 = new Uint32Array(scratch);

/**
 * IEEE 754 binary32 -> binary16 with round-half-to-even, the same rounding
 * every GPU uses. Overflow saturates to infinity, and values too small for a
 * subnormal flush to zero.
 */
export function f32ToF16Bits(value: number): number {
  scratchF32[0] = value;
  const bits = scratchU32[0];

  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x007fffff;

  if (exponent === 0xff) {
    // Infinity, or NaN — keep it a NaN rather than letting it become infinity.
    return sign | 0x7c00 | (mantissa !== 0 ? 0x200 : 0);
  }

  const unbiased = exponent - 127 + 15;

  if (unbiased >= 31) return sign | 0x7c00; // overflows half range
  if (unbiased <= 0) {
    if (unbiased < -10) return sign; // smaller than the smallest subnormal
    mantissa |= 0x800000; // restore the implicit leading one
    const shift = 14 - unbiased;
    const truncated = mantissa >>> shift;
    const roundBit = (mantissa >>> (shift - 1)) & 1;
    const sticky = (mantissa & ((1 << (shift - 1)) - 1)) !== 0;
    return sign | (truncated + (roundBit && (sticky || (truncated & 1)) ? 1 : 0));
  }

  const truncated = (unbiased << 10) | (mantissa >>> 13);
  const roundBit = (mantissa >>> 12) & 1;
  const sticky = (mantissa & 0xfff) !== 0;
  // A carry out of the mantissa lands in the exponent, which is exactly right:
  // 0x7bff + 1 == 0x7c00 == infinity.
  return sign | (truncated + (roundBit && (sticky || (truncated & 1)) ? 1 : 0));
}

/** binary16 -> binary32. Exact: every half value is representable as a float. */
export function f16BitsToF32(half: number): number {
  const sign = (half & 0x8000) << 16;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x03ff;

  if (exponent === 0) {
    if (mantissa === 0) {
      scratchU32[0] = sign;
      return scratchF32[0];
    }
    // Subnormal half, normal float: shift until the leading one appears.
    let shifted = mantissa;
    let extra = 0;
    while ((shifted & 0x0400) === 0) {
      shifted <<= 1;
      extra += 1;
    }
    scratchU32[0] = sign | ((127 - 15 - extra + 1) << 23) | ((shifted & 0x03ff) << 13);
    return scratchF32[0];
  }

  if (exponent === 0x1f) {
    scratchU32[0] = sign | 0x7f800000 | (mantissa << 13);
    return scratchF32[0];
  }

  scratchU32[0] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return scratchF32[0];
}

export function f32ArrayToF16(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = f32ToF16Bits(values[i]);
  return out;
}

export function f16ArrayToF32(halves: Uint16Array): Float32Array {
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i += 1) out[i] = f16BitsToF32(halves[i]);
  return out;
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

export interface FrameMeta {
  seqId: number;
  tokenIndex: number;
  fromLayer: number;
  toLayer: number;
}

export interface EncodeOptions {
  /** Wire dtype. F16 halves bandwidth and is ample for activations. */
  dtype?: TensorDType.F32 | TensorDType.F16;
  maxChunkBytes?: number;
  kind?: TensorFrameKind;
}

/**
 * Serialise a hidden state into one or more transmittable chunks. Always
 * returns at least one buffer, even for an empty tensor, so the receiver
 * always learns the sequence exists.
 */
export function encodeTensor(
  shape: readonly number[],
  values: Float32Array,
  meta: FrameMeta,
  options: EncodeOptions = {},
): ArrayBuffer[] {
  const dtype = options.dtype ?? TensorDType.F16;
  const maxChunkBytes = Math.max(64, options.maxChunkBytes ?? DEFAULT_MAX_CHUNK_BYTES);
  const kind = options.kind ?? TensorFrameKind.Forward;

  const expected = elementCount(shape);
  if (values.length !== expected) {
    throw new Error(`shape [${shape}] implies ${expected} elements, got ${values.length}`);
  }

  const payload =
    dtype === TensorDType.F16
      ? new Uint8Array(f32ArrayToF16(values).buffer)
      : new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));

  const elementSize = bytesPerElement(dtype);
  // Never split a chunk mid-element; reassembly assumes element alignment.
  const perChunk = Math.max(elementSize, Math.floor(maxChunkBytes / elementSize) * elementSize);
  const chunkCount = Math.max(1, Math.ceil(payload.byteLength / perChunk));

  if (chunkCount > MAX_CHUNKS) {
    throw new Error(
      `tensor needs ${chunkCount} chunks, over the ${MAX_CHUNKS} limit — raise maxChunkBytes`,
    );
  }

  const frames: ArrayBuffer[] = [];
  for (let index = 0; index < chunkCount; index += 1) {
    const start = index * perChunk;
    const slice = payload.subarray(start, Math.min(start + perChunk, payload.byteLength));
    frames.push(
      encodeFrame({
        kind,
        dtype,
        shape: [...shape],
        ...meta,
        chunkIndex: index,
        chunkCount,
        data: slice.slice().buffer,
      }),
    );
  }
  return frames;
}

/** Serialise a single frame (one chunk) to bytes. */
export function encodeFrame(frame: TensorFrame): ArrayBuffer {
  const rank = frame.shape.length;
  if (rank > 255) throw new Error(`rank ${rank} exceeds the 255 the header can hold`);

  const headerBytes = TENSOR_HEADER_BYTES + rank * 4;
  const out = new ArrayBuffer(headerBytes + frame.data.byteLength);
  const view = new DataView(out);

  view.setUint32(0, TENSOR_MAGIC, true);
  view.setUint8(4, TENSOR_VERSION);
  view.setUint8(5, frame.kind);
  view.setUint8(6, frame.dtype);
  view.setUint8(7, rank);
  view.setUint32(8, frame.seqId >>> 0, true);
  view.setUint32(12, frame.tokenIndex >>> 0, true);
  view.setUint32(16, frame.fromLayer >>> 0, true);
  view.setUint32(20, frame.toLayer >>> 0, true);
  view.setUint32(24, frame.data.byteLength >>> 0, true);
  view.setUint16(28, frame.chunkIndex, true);
  view.setUint16(30, frame.chunkCount, true);

  for (let i = 0; i < rank; i += 1) {
    view.setUint32(TENSOR_HEADER_BYTES + i * 4, frame.shape[i] >>> 0, true);
  }

  new Uint8Array(out, headerBytes).set(new Uint8Array(frame.data));
  return out;
}

/** Parse a single frame. Throws on anything malformed rather than guessing. */
export function decodeFrame(buffer: ArrayBuffer): TensorFrame {
  if (buffer.byteLength < TENSOR_HEADER_BYTES) {
    throw new Error('tensor frame too small');
  }

  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== TENSOR_MAGIC) throw new Error('bad tensor frame magic');

  const version = view.getUint8(4);
  if (version !== TENSOR_VERSION) throw new Error(`unsupported tensor frame version ${version}`);

  const rank = view.getUint8(7);
  const headerBytes = TENSOR_HEADER_BYTES + rank * 4;
  if (buffer.byteLength < headerBytes) throw new Error('tensor frame truncated in its shape');

  const shape: number[] = [];
  for (let i = 0; i < rank; i += 1) {
    shape.push(view.getUint32(TENSOR_HEADER_BYTES + i * 4, true));
  }

  const payloadBytes = view.getUint32(24, true);
  if (headerBytes + payloadBytes !== buffer.byteLength) {
    throw new Error('tensor payload length mismatch');
  }

  const chunkIndex = view.getUint16(28, true);
  const chunkCount = view.getUint16(30, true);
  if (chunkCount === 0) throw new Error('chunk count must be at least 1');
  if (chunkIndex >= chunkCount) {
    throw new Error(`chunk ${chunkIndex} is out of range for a ${chunkCount}-chunk frame`);
  }

  return {
    kind: view.getUint8(5) as TensorFrameKind,
    dtype: view.getUint8(6) as TensorDType,
    shape,
    seqId: view.getUint32(8, true),
    tokenIndex: view.getUint32(12, true),
    fromLayer: view.getUint32(16, true),
    toLayer: view.getUint32(20, true),
    chunkIndex,
    chunkCount,
    data: buffer.slice(headerBytes),
  };
}

/** Read a fully-reassembled frame's payload as f32, whatever it travelled as. */
export function frameAsFloat32(frame: TensorFrame): Float32Array {
  if (frame.dtype === TensorDType.F32) return new Float32Array(frame.data.slice(0));
  if (frame.dtype === TensorDType.F16) {
    return f16ArrayToF32(new Uint16Array(frame.data.slice(0)));
  }
  throw new Error(`cannot read dtype ${frame.dtype} as f32`);
}

// ---------------------------------------------------------------------------
// Reassembly
// ---------------------------------------------------------------------------

interface Partial {
  frame: TensorFrame;
  chunks: Array<ArrayBuffer | undefined>;
  received: number;
  bytes: number;
  updatedAt: number;
}

export interface ReassemblerOptions {
  /** Concurrent partial frames to track before evicting the oldest. */
  maxPending?: number;
  /** Reject a frame whose payload would exceed this, before allocating it. */
  maxPayloadBytes?: number;
  /** Injectable clock so eviction can be tested without waiting. */
  now?: () => number;
}

/**
 * Collects chunks until a frame is whole.
 *
 * Chunks may arrive out of order — the tensor channel is unordered by design,
 * since strict ordering across independent sequences would let one slow
 * sequence stall every other one.
 */
export class FrameReassembler {
  private readonly pending = new Map<number, Partial>();
  /**
   * Sequence ids delivered recently. A retransmitted chunk can arrive after
   * its frame is already complete; without this it would open a fresh partial
   * that never fills and just waits to be evicted.
   */
  private readonly completed = new Set<number>();
  private readonly maxPending: number;
  private readonly maxPayloadBytes: number;
  private readonly now: () => number;

  constructor(options: ReassemblerOptions = {}) {
    this.maxPending = Math.max(1, options.maxPending ?? 32);
    this.maxPayloadBytes = options.maxPayloadBytes ?? 256 * 1024 * 1024;
    this.now = options.now ?? (() => Date.now());
  }

  /**
   * Feed in one decoded chunk. Returns the complete frame once every chunk has
   * arrived, otherwise null. Duplicates are ignored, which matters because a
   * retransmit may race the original.
   */
  push(frame: TensorFrame): TensorFrame | null {
    if (frame.chunkCount === 1) return frame;
    if (this.completed.has(frame.seqId)) return null; // late duplicate

    const expectedBytes = elementCount(frame.shape) * bytesPerElement(frame.dtype);
    if (expectedBytes > this.maxPayloadBytes) {
      throw new Error(
        `frame claims ${expectedBytes} bytes, over the ${this.maxPayloadBytes} limit`,
      );
    }

    let partial = this.pending.get(frame.seqId);
    if (!partial) {
      this.evictIfFull();
      partial = {
        frame,
        chunks: new Array<ArrayBuffer | undefined>(frame.chunkCount),
        received: 0,
        bytes: 0,
        updatedAt: this.now(),
      };
      this.pending.set(frame.seqId, partial);
    }

    // A peer restarting could reuse a sequence id with a different shape.
    // Trust the newest frame and start over rather than splicing two tensors.
    if (partial.frame.chunkCount !== frame.chunkCount) {
      partial.frame = frame;
      partial.chunks = new Array<ArrayBuffer | undefined>(frame.chunkCount);
      partial.received = 0;
      partial.bytes = 0;
    }

    if (partial.chunks[frame.chunkIndex] === undefined) {
      partial.chunks[frame.chunkIndex] = frame.data;
      partial.received += 1;
      partial.bytes += frame.data.byteLength;
    }
    partial.updatedAt = this.now();

    if (partial.received !== frame.chunkCount) return null;

    this.pending.delete(frame.seqId);
    this.rememberCompleted(frame.seqId);

    const merged = new Uint8Array(partial.bytes);
    let offset = 0;
    for (const chunk of partial.chunks) {
      if (!chunk) throw new Error('reassembly hole — chunk accounting is wrong');
      merged.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    return {
      ...partial.frame,
      chunkIndex: 0,
      chunkCount: 1,
      data: merged.buffer,
    };
  }

  /** Sequence ids with chunks still outstanding. */
  pendingSeqIds(): number[] {
    return [...this.pending.keys()];
  }

  /** Which chunks of a partial frame are still missing (for retransmit asks). */
  missingChunks(seqId: number): number[] {
    const partial = this.pending.get(seqId);
    if (!partial) return [];
    const missing: number[] = [];
    for (let i = 0; i < partial.chunks.length; i += 1) {
      if (partial.chunks[i] === undefined) missing.push(i);
    }
    return missing;
  }

  /** Drop partial frames untouched for longer than `maxAgeMs`. */
  evictStale(maxAgeMs: number): number[] {
    const cutoff = this.now() - maxAgeMs;
    const dropped: number[] = [];
    for (const [seqId, partial] of this.pending) {
      if (partial.updatedAt < cutoff) {
        this.pending.delete(seqId);
        dropped.push(seqId);
      }
    }
    return dropped;
  }

  forget(seqId: number): void {
    this.pending.delete(seqId);
    this.completed.delete(seqId);
  }

  clear(): void {
    this.pending.clear();
    this.completed.clear();
  }

  get size(): number {
    return this.pending.size;
  }

  private evictIfFull(): void {
    if (this.pending.size < this.maxPending) return;
    const oldest = this.pending.keys().next().value;
    if (oldest !== undefined) this.pending.delete(oldest);
  }

  /** Bounded memory: only the most recent completions are worth remembering. */
  private rememberCompleted(seqId: number): void {
    this.completed.add(seqId);
    if (this.completed.size > this.maxPending * 4) {
      const oldest = this.completed.keys().next().value;
      if (oldest !== undefined) this.completed.delete(oldest);
    }
  }
}
