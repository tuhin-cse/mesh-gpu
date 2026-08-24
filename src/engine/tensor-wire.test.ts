import { describe, expect, it } from 'vitest';

import {
  DEFAULT_MAX_CHUNK_BYTES,
  FrameReassembler,
  TENSOR_HEADER_BYTES,
  TENSOR_MAGIC,
  TensorDType,
  TensorFrameKind,
  decodeFrame,
  encodeFrame,
  encodeTensor,
  f16BitsToF32,
  f32ToF16Bits,
  frameAsFloat32,
} from './tensor-wire';

const META = { seqId: 7, tokenIndex: 3, fromLayer: 14, toLayer: 28 };

/** Encode then fully reassemble, the way the transport actually uses this. */
function roundTrip(shape: number[], values: Float32Array, options = {}) {
  const reassembler = new FrameReassembler();
  let complete = null;
  for (const buffer of encodeTensor(shape, values, META, options)) {
    complete = reassembler.push(decodeFrame(buffer));
  }
  if (!complete) throw new Error('frame never completed');
  return complete;
}

describe('half precision', () => {
  it('round-trips values that are exactly representable', () => {
    for (const value of [0, 1, -1, 0.5, -0.5, 2, 1024, -2048, 0.25, 65504, -65504]) {
      expect(f16BitsToF32(f32ToF16Bits(value))).toBe(value);
    }
  });

  it('preserves signed zero', () => {
    expect(Object.is(f16BitsToF32(f32ToF16Bits(-0)), -0)).toBe(true);
    expect(Object.is(f16BitsToF32(f32ToF16Bits(0)), 0)).toBe(true);
  });

  it('keeps infinities and NaN distinct', () => {
    expect(f16BitsToF32(f32ToF16Bits(Infinity))).toBe(Infinity);
    expect(f16BitsToF32(f32ToF16Bits(-Infinity))).toBe(-Infinity);
    expect(Number.isNaN(f16BitsToF32(f32ToF16Bits(NaN)))).toBe(true);
  });

  it('saturates to infinity above the half range instead of wrapping', () => {
    // 65520 is the round-half-to-even tie that becomes infinity.
    expect(f16BitsToF32(f32ToF16Bits(70000))).toBe(Infinity);
    expect(f16BitsToF32(f32ToF16Bits(-1e30))).toBe(-Infinity);
  });

  it('represents subnormals rather than flushing them to zero', () => {
    const smallestSubnormal = 2 ** -24;
    expect(f16BitsToF32(f32ToF16Bits(smallestSubnormal))).toBe(smallestSubnormal);
    expect(f16BitsToF32(f32ToF16Bits(2 ** -14))).toBe(2 ** -14); // smallest normal
  });

  it('flushes to zero only below the smallest subnormal', () => {
    expect(f16BitsToF32(f32ToF16Bits(1e-20))).toBe(0);
  });

  it('rounds half to even, matching GPU behaviour', () => {
    // Halfway between two representable halves near 1.0, step is 2^-10.
    const step = 2 ** -10;
    // 1 + step/2 ties; 1.0 has an even mantissa, so it rounds down to 1.
    expect(f16BitsToF32(f32ToF16Bits(1 + step / 2))).toBe(1);
    // 1 + step has an odd mantissa, so the next tie rounds up, away from it.
    expect(f16BitsToF32(f32ToF16Bits(1 + step + step / 2))).toBe(1 + 2 * step);
  });

  it('keeps activation-scale values within half precision tolerance', () => {
    // Hidden states sit roughly in [-10, 10]; relative error must stay under
    // 2^-11, the half-precision epsilon.
    let worst = 0;
    for (let i = 0; i < 20_000; i += 1) {
      const value = (Math.random() - 0.5) * 20;
      const error = Math.abs(f16BitsToF32(f32ToF16Bits(value)) - value) / Math.abs(value);
      worst = Math.max(worst, error);
    }
    expect(worst).toBeLessThan(2 ** -10);
  });
});

describe('encode / decode', () => {
  it('round-trips a small tensor in f32 exactly', () => {
    const values = new Float32Array([0.125, -0.5, 3.25, 1024]);
    const frame = roundTrip([1, 4], values, { dtype: TensorDType.F32 });
    expect([...frameAsFloat32(frame)]).toEqual([...values]);
  });

  it('carries every header field through unchanged', () => {
    const frame = roundTrip([2, 2], new Float32Array([1, 2, 3, 4]));
    expect(frame.seqId).toBe(7);
    expect(frame.tokenIndex).toBe(3);
    expect(frame.fromLayer).toBe(14);
    expect(frame.toLayer).toBe(28);
    expect(frame.shape).toEqual([2, 2]);
    expect(frame.kind).toBe(TensorFrameKind.Forward);
  });

  it('defaults to f16, halving the bytes on the wire', () => {
    const values = new Float32Array(3584); // one token of a 7B hidden state
    const asF16 = encodeTensor([1, 3584], values, META);
    const asF32 = encodeTensor([1, 3584], values, META, { dtype: TensorDType.F32 });

    const size = (frames: ArrayBuffer[]) =>
      frames.reduce((total, frame) => total + frame.byteLength, 0);

    expect(size(asF16)).toBeLessThan(size(asF32));
    expect(size(asF16)).toBeLessThan(8 * 1024); // ~7 KiB per token, not 14
  });

  it('rejects a value count that disagrees with the shape', () => {
    expect(() => encodeTensor([2, 3], new Float32Array(5), META)).toThrow(/implies 6 elements/);
  });

  it('rejects short buffers, bad magic and unknown versions', () => {
    expect(() => decodeFrame(new ArrayBuffer(8))).toThrow(/too small/);

    const bad = new ArrayBuffer(TENSOR_HEADER_BYTES);
    new DataView(bad).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeFrame(bad)).toThrow(/magic/);

    const wrongVersion = new ArrayBuffer(TENSOR_HEADER_BYTES);
    const view = new DataView(wrongVersion);
    view.setUint32(0, TENSOR_MAGIC, true);
    view.setUint8(4, 99);
    view.setUint16(30, 1, true);
    expect(() => decodeFrame(wrongVersion)).toThrow(/version 99/);
  });

  it('rejects a payload length that disagrees with the buffer', () => {
    const buffer = encodeTensor([2], new Float32Array([1, 2]), META)[0];
    new DataView(buffer).setUint32(24, 4096, true);
    expect(() => decodeFrame(buffer)).toThrow(/length mismatch/);
  });

  it('rejects an out-of-range chunk index', () => {
    const buffer = encodeTensor([2], new Float32Array([1, 2]), META)[0];
    new DataView(buffer).setUint16(28, 5, true); // index 5 of 1
    expect(() => decodeFrame(buffer)).toThrow(/out of range/);
  });

  it('does not alias the caller buffer', () => {
    const values = new Float32Array([1, 2, 3]);
    const frames = encodeTensor([3], values, META, { dtype: TensorDType.F32 });
    values[0] = 99;
    expect(frameAsFloat32(decodeFrame(frames[0]))[0]).toBe(1);
  });
});

describe('chunking', () => {
  it('keeps a single-token hidden state in one chunk', () => {
    expect(encodeTensor([1, 3584], new Float32Array(3584), META)).toHaveLength(1);
  });

  it('splits a prefill-sized tensor that no DataChannel would carry whole', () => {
    // 2048 tokens x 3584 hidden = 14.7 MB in f16 — the case that silently
    // failed before chunking existed.
    const elements = 2048 * 3584;
    const frames = encodeTensor([2048, 3584], new Float32Array(elements), META);

    expect(frames.length).toBeGreaterThan(200);
    for (const frame of frames) {
      expect(frame.byteLength).toBeLessThanOrEqual(DEFAULT_MAX_CHUNK_BYTES + TENSOR_HEADER_BYTES + 16);
    }
  });

  it('reassembles a chunked tensor to the exact original values', () => {
    const values = new Float32Array(50_000).map((_, i) => (i % 17) - 8); // exact in f16
    const frame = roundTrip([50_000], values, { maxChunkBytes: 4096 });
    expect([...frameAsFloat32(frame)]).toEqual([...values]);
  });

  it('reassembles correctly when chunks arrive out of order', () => {
    const values = new Float32Array(20_000).map((_, i) => (i % 9) - 4);
    const buffers = encodeTensor([20_000], values, META, { maxChunkBytes: 2048 });
    expect(buffers.length).toBeGreaterThan(5);

    const reassembler = new FrameReassembler();
    let complete = null;
    for (const buffer of [...buffers].reverse()) {
      complete = reassembler.push(decodeFrame(buffer)) ?? complete;
    }

    expect(complete).not.toBeNull();
    expect([...frameAsFloat32(complete!)]).toEqual([...values]);
  });

  it('ignores duplicate chunks from a retransmit racing the original', () => {
    const values = new Float32Array(8_000).map((_, i) => i % 5);
    const buffers = encodeTensor([8_000], values, META, { maxChunkBytes: 1024 });

    const reassembler = new FrameReassembler();
    let complete = null;
    for (const buffer of buffers) {
      complete = reassembler.push(decodeFrame(buffer)) ?? complete; // original
      complete = reassembler.push(decodeFrame(buffer)) ?? complete; // duplicate
    }

    expect(complete).not.toBeNull();
    expect([...frameAsFloat32(complete!)]).toEqual([...values]);
    // The late duplicate of the final chunk must not open a phantom partial.
    expect(reassembler.size).toBe(0);
  });

  it('never splits an element across two chunks', () => {
    // An odd byte budget must still land on f16 element boundaries.
    const frames = encodeTensor([1000], new Float32Array(1000), META, { maxChunkBytes: 101 });
    for (const buffer of frames) {
      const frame = decodeFrame(buffer);
      expect(frame.data.byteLength % 2).toBe(0);
    }
  });

  it('refuses a tensor needing more chunks than the header can index', () => {
    // chunkIndex/chunkCount are uint16. At the 64-byte floor that caps a frame
    // at ~4 MB, so a 6 MB tensor must be rejected rather than silently wrap.
    expect(() => encodeTensor([3_000_000], new Float32Array(3_000_000), META, { maxChunkBytes: 1 }))
      .toThrow(/over the 65535 limit/);
  });
});

describe('FrameReassembler housekeeping', () => {
  it('reports which chunks are still missing', () => {
    const values = new Float32Array(4000).map((_, i) => i % 3);
    const buffers = encodeTensor([4000], values, META, { maxChunkBytes: 1024 });

    const reassembler = new FrameReassembler();
    reassembler.push(decodeFrame(buffers[0]));
    reassembler.push(decodeFrame(buffers[2]));

    expect(reassembler.pendingSeqIds()).toEqual([META.seqId]);
    expect(reassembler.missingChunks(META.seqId)).toContain(1);
    expect(reassembler.missingChunks(META.seqId)).not.toContain(0);
  });

  it('evicts partial frames that stop arriving', () => {
    let clock = 1000;
    const reassembler = new FrameReassembler({ now: () => clock });
    const buffers = encodeTensor([4000], new Float32Array(4000), META, { maxChunkBytes: 1024 });

    reassembler.push(decodeFrame(buffers[0]));
    expect(reassembler.size).toBe(1);

    clock += 30_000;
    expect(reassembler.evictStale(10_000)).toEqual([META.seqId]);
    expect(reassembler.size).toBe(0);
  });

  it('caps how many partial frames it will hold at once', () => {
    const reassembler = new FrameReassembler({ maxPending: 3 });
    for (let seqId = 0; seqId < 10; seqId += 1) {
      const buffers = encodeTensor([4000], new Float32Array(4000), { ...META, seqId }, {
        maxChunkBytes: 1024,
      });
      reassembler.push(decodeFrame(buffers[0]));
    }
    expect(reassembler.size).toBeLessThanOrEqual(3);
  });

  it('refuses to allocate for an implausibly large declared frame', () => {
    const reassembler = new FrameReassembler({ maxPayloadBytes: 1024 });
    const frame = decodeFrame(
      encodeFrame({
        kind: TensorFrameKind.Forward,
        dtype: TensorDType.F16,
        shape: [10_000_000],
        ...META,
        chunkIndex: 0,
        chunkCount: 2,
        data: new ArrayBuffer(16),
      }),
    );
    expect(() => reassembler.push(frame)).toThrow(/over the 1024 limit/);
  });

  it('restarts cleanly if a sequence id is reused with a different shape', () => {
    const reassembler = new FrameReassembler();
    const first = encodeTensor([4000], new Float32Array(4000), META, { maxChunkBytes: 1024 });
    reassembler.push(decodeFrame(first[0]));

    const values = new Float32Array(2000).map((_, i) => i % 4);
    const second = encodeTensor([2000], values, META, { maxChunkBytes: 1024 });

    let complete = null;
    for (const buffer of second) complete = reassembler.push(decodeFrame(buffer)) ?? complete;

    expect(complete).not.toBeNull();
    expect([...frameAsFloat32(complete!)]).toEqual([...values]);
  });
});
