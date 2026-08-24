import { describe, expect, it } from 'vitest';

import { capacityFromBudget, probeGpuBudget } from './gpu-budget';

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

/**
 * A GPUDevice stand-in that hands out buffers until a byte limit, then reports
 * out-of-memory the way a real device does. Lets the probe loop, the error
 * scope handling and the cleanup all be tested without a GPU.
 */
function fakeDevice(limitBytes: number, options: { throwOnFail?: boolean; maxBufferSize?: number } = {}) {
  let allocated = 0;
  let live = 0;
  const scopes: Array<'out-of-memory'> = [];
  let pendingError: object | null = null;

  const device = {
    limits: { maxBufferSize: options.maxBufferSize ?? 256 * MiB },
    pushErrorScope: (kind: 'out-of-memory') => {
      scopes.push(kind);
      pendingError = null;
    },
    popErrorScope: async () => {
      scopes.pop();
      const error = pendingError;
      pendingError = null;
      return error;
    },
    createBuffer: ({ size }: { size: number }) => {
      if (allocated + size > limitBytes) {
        if (options.throwOnFail) throw new Error('out of memory');
        pendingError = { message: 'out of memory' };
        return { destroy: () => {} };
      }
      allocated += size;
      live += 1;
      return {
        destroy: () => {
          allocated -= size;
          live -= 1;
        },
      };
    },
  } as unknown as GPUDevice;

  return { device, liveBuffers: () => live, allocated: () => allocated };
}

describe('probeGpuBudget', () => {
  it('measures how much the device will actually give out', async () => {
    const { device } = fakeDevice(1 * GiB, { maxBufferSize: 256 * MiB });
    const budget = await probeGpuBudget(device, { blockBytes: 256 * MiB });

    expect(budget.blocksAllocated).toBe(4);
    expect(budget.usableBytes).toBe(1 * GiB);
    expect(budget.hitCeiling).toBe(false);
  });

  it('releases every buffer it allocated', async () => {
    const probe = fakeDevice(1 * GiB, { maxBufferSize: 256 * MiB });
    await probeGpuBudget(probe.device, { blockBytes: 256 * MiB });

    expect(probe.liveBuffers()).toBe(0);
    expect(probe.allocated()).toBe(0);
  });

  it('handles a device that throws instead of using the error scope', async () => {
    const probe = fakeDevice(512 * MiB, { throwOnFail: true, maxBufferSize: 256 * MiB });
    const budget = await probeGpuBudget(probe.device, { blockBytes: 256 * MiB });

    expect(budget.blocksAllocated).toBe(2);
    expect(probe.liveBuffers()).toBe(0);
  });

  it('never requests a buffer larger than the device permits', async () => {
    const { device } = fakeDevice(4 * GiB, { maxBufferSize: 128 * MiB });
    // Ask for 1 GiB blocks; the device caps single buffers at 128 MiB.
    const budget = await probeGpuBudget(device, { blockBytes: 1 * GiB, ceilingBytes: 1 * GiB });

    expect(budget.maxBufferBytes).toBe(128 * MiB);
    expect(budget.blockBytes).toBe(128 * MiB);
  });

  it('stops at the ceiling on a device that would keep saying yes', async () => {
    const { device } = fakeDevice(Number.MAX_SAFE_INTEGER, { maxBufferSize: 256 * MiB });
    const budget = await probeGpuBudget(device, {
      blockBytes: 256 * MiB,
      ceilingBytes: 1 * GiB,
    });

    expect(budget.hitCeiling).toBe(true);
    expect(budget.usableBytes).toBeLessThanOrEqual(1 * GiB);
  });

  it('reports zero rather than failing when nothing can be allocated', async () => {
    const probe = fakeDevice(0, { maxBufferSize: 256 * MiB });
    const budget = await probeGpuBudget(probe.device, { blockBytes: 256 * MiB });

    expect(budget.usableBytes).toBe(0);
    expect(budget.blocksAllocated).toBe(0);
    expect(probe.liveBuffers()).toBe(0);
  });
});

describe('capacityFromBudget', () => {
  const qwen7b = { totalLayers: 28, modelBytes: 4.3 * GiB }; // 7.6B at 4-bit

  it('turns a measured budget into a layer count', () => {
    const capacity = capacityFromBudget({
      ...qwen7b,
      budget: { usableBytes: 2 * GiB, maxBufferBytes: 256 * MiB },
    });

    // 4.3 GiB / 28 layers ~= 157 MiB; 2 GiB minus 25% headroom ~= 1.5 GiB.
    expect(capacity.hostableLayers).toBeGreaterThan(5);
    expect(capacity.hostableLayers).toBeLessThan(12);
  });

  it('never claims more layers than the model has', () => {
    const capacity = capacityFromBudget({
      ...qwen7b,
      budget: { usableBytes: 64 * GiB, maxBufferBytes: 2 * GiB },
    });
    expect(capacity.hostableLayers).toBe(28);
  });

  it('reports zero, with a reason, when one layer will not fit', () => {
    const capacity = capacityFromBudget({
      ...qwen7b,
      budget: { usableBytes: 64 * MiB, maxBufferBytes: 64 * MiB },
    });

    expect(capacity.hostableLayers).toBe(0);
    expect(capacity.reason).toMatch(/cannot host a stage/);
  });

  it('reserves more for a larger headroom ratio', () => {
    const base = { ...qwen7b, budget: { usableBytes: 4 * GiB, maxBufferBytes: 256 * MiB } };
    const lean = capacityFromBudget({ ...base, headroomRatio: 0.1 });
    const cautious = capacityFromBudget({ ...base, headroomRatio: 0.5 });

    expect(cautious.hostableLayers).toBeLessThan(lean.hostableLayers);
    expect(cautious.reservedBytes).toBeGreaterThan(lean.reservedBytes);
  });

  it('clamps an absurd headroom ratio instead of producing nonsense', () => {
    const capacity = capacityFromBudget({
      ...qwen7b,
      budget: { usableBytes: 4 * GiB, maxBufferBytes: 256 * MiB },
      headroomRatio: 5,
    });
    expect(capacity.hostableLayers).toBeGreaterThanOrEqual(0);
    expect(capacity.reservedBytes).toBeLessThanOrEqual(4 * GiB);
  });
});
