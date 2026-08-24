import { describe, expect, it } from 'vitest';

import {
  TENSOR_HEADER_BYTES,
  TENSOR_MAGIC,
  TensorCodec,
  TensorDType,
  TensorFrameKind,
  frameAsFloat32,
  makeForwardFrame,
} from './p2p-pipeline';
import { sanitizeTensor } from './webgpu-node';

describe('TensorCodec', () => {
  it('round-trips a forward frame byte-for-byte', () => {
    const data = new Float32Array([0.123, 0.456, -0.789, 3.5e-8]);
    const frame = makeForwardFrame([1, 4], data, {
      seqId: 7,
      tokenIndex: 42,
      fromLayer: 8,
      toLayer: 16,
    });

    const decoded = TensorCodec.decode(TensorCodec.encode(frame));

    expect(decoded.kind).toBe(TensorFrameKind.Forward);
    expect(decoded.dtype).toBe(TensorDType.F32);
    expect(decoded.shape).toEqual([1, 4]);
    expect(decoded.seqId).toBe(7);
    expect(decoded.tokenIndex).toBe(42);
    expect(decoded.fromLayer).toBe(8);
    expect(decoded.toLayer).toBe(16);
    expect([...frameAsFloat32(decoded)]).toEqual([...data]);
  });

  it('survives a realistic hidden-state payload', () => {
    // One token of Qwen2.5-7B hidden state: 3584 floats = 14 KiB on the wire.
    const hidden = new Float32Array(3584).map((_, i) => Math.sin(i) * 0.5);
    const frame = makeForwardFrame([1, 3584], hidden, {
      seqId: 1,
      tokenIndex: 0,
      fromLayer: 14,
      toLayer: 28,
    });

    const encoded = TensorCodec.encode(frame);
    expect(encoded.byteLength).toBe(TENSOR_HEADER_BYTES + 2 * 4 + 3584 * 4);
    expect([...frameAsFloat32(TensorCodec.decode(encoded))]).toEqual([...hidden]);
  });

  it('copies the payload rather than aliasing the caller buffer', () => {
    const data = new Float32Array([1, 2, 3]);
    const frame = makeForwardFrame([3], data, {
      seqId: 1,
      tokenIndex: 0,
      fromLayer: 0,
      toLayer: 1,
    });
    data[0] = 99;
    expect(frameAsFloat32(frame)[0]).toBe(1);
  });

  it('rejects a buffer smaller than the header', () => {
    expect(() => TensorCodec.decode(new ArrayBuffer(8))).toThrow(/too small/);
  });

  it('rejects a bad magic number', () => {
    const buffer = new ArrayBuffer(TENSOR_HEADER_BYTES);
    new DataView(buffer).setUint32(0, 0xdeadbeef, true);
    expect(() => TensorCodec.decode(buffer)).toThrow(/magic/);
  });

  it('rejects an unknown format version', () => {
    const buffer = new ArrayBuffer(TENSOR_HEADER_BYTES);
    const view = new DataView(buffer);
    view.setUint32(0, TENSOR_MAGIC, true);
    view.setUint8(4, 99);
    expect(() => TensorCodec.decode(buffer)).toThrow(/version 99/);
  });

  it('rejects a declared payload length that disagrees with the buffer', () => {
    const frame = makeForwardFrame([2], new Float32Array([1, 2]), {
      seqId: 1,
      tokenIndex: 0,
      fromLayer: 0,
      toLayer: 1,
    });
    const encoded = TensorCodec.encode(frame);
    new DataView(encoded).setUint32(24, 4096, true); // claim a much larger payload
    expect(() => TensorCodec.decode(encoded)).toThrow(/length mismatch/);
  });

  it('does not read past the buffer on a truncated frame', () => {
    const frame = makeForwardFrame([64], new Float32Array(64), {
      seqId: 1,
      tokenIndex: 0,
      fromLayer: 0,
      toLayer: 1,
    });
    const encoded = TensorCodec.encode(frame);
    expect(() => TensorCodec.decode(encoded.slice(0, encoded.byteLength - 16))).toThrow();
  });
});

describe('sanitizeTensor', () => {
  it('accepts a well-formed tensor of the expected length', () => {
    expect(sanitizeTensor(new Float32Array([1, 2, 3]), 3).ok).toBe(true);
  });

  it('rejects a length that disagrees with the decoded shape', () => {
    const result = sanitizeTensor(new Float32Array([1, 2, 3]), 4);
    expect(result.ok).toBe(false);
    expect(result.reason).toMatch(/size mismatch/);
  });

  it('rejects NaN and Infinity before they reach a GPU buffer', () => {
    const result = sanitizeTensor(new Float32Array([1, NaN, Infinity, -Infinity]));
    expect(result.ok).toBe(false);
    expect(result.nonFiniteCount).toBe(3);
  });
});
