import { test, expect, type Page } from '@playwright/test';

/**
 * Numerical verification of the WebGPU decoder-layer kernels.
 *
 * An attention kernel that is subtly wrong — a transposed index, a missing
 * scale, the wrong KV head under grouped-query attention — produces
 * plausible-looking numbers rather than an error. The only way to tell is to
 * run the same weights through an independent implementation and compare
 * element by element, which is what these tests do: reference maths in plain
 * TypeScript on the CPU, the real kernels on the GPU, same inputs.
 *
 * Skips itself if the machine running the suite has no WebGPU adapter.
 */

interface Stats {
  maxAbsDiff: number;
  /** Relative error, measured only where the reference value is not near zero. */
  maxRelDiff: number;
  /** Worst violation of |cpu - gpu| <= atol + rtol * |cpu|. <= 0 means agreement. */
  worstViolation: number;
  cpuFirst: number[];
  gpuFirst: number[];
  allFinite: boolean;
}

async function ready(page: Page): Promise<boolean> {
  await page.goto('/');
  await page.waitForFunction(
    () => Boolean((window as unknown as { meshTransformer?: unknown }).meshTransformer),
    undefined,
    { timeout: 30_000 },
  );
  return page.evaluate(async () => {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return false;
    return Boolean(await gpu.requestAdapter());
  });
}

/**
 * Run `tokenCount` tokens through `layerCount` layers on both backends and
 * report how far apart they end up.
 */
async function compare(
  page: Page,
  layerCount: number,
  tokenCount: number,
  tolerance: { atol: number; rtol: number },
): Promise<Stats> {
  return page.evaluate(
    async ({ layerCount, tokenCount, tolerance }) => {
      const api = (window as unknown as { meshTransformer: any }).meshTransformer;
      const config = api.TEST_CONFIG;

      const weights = Array.from({ length: layerCount }, (_, i) =>
        api.randomWeights(config, 1000 + i),
      );

      // --- reference ---
      const cpuCaches = weights.map(() => api.createKVCache(config));
      const random = api.seededRandom(7);
      const inputs: Float32Array[] = [];
      for (let t = 0; t < tokenCount; t += 1) {
        const token = new Float32Array(config.hiddenSize);
        for (let i = 0; i < token.length; i += 1) token[i] = random() * 2 - 1;
        inputs.push(token);
      }

      const cpuOutputs: Float32Array[] = [];
      for (let t = 0; t < tokenCount; t += 1) {
        let hidden = inputs[t];
        for (let l = 0; l < layerCount; l += 1) {
          hidden = api.forwardLayer(config, weights[l], hidden, cpuCaches[l], t);
        }
        cpuOutputs.push(hidden);
      }

      // --- GPU ---
      const adapter = await (navigator as any).gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const stage = api.GpuTransformerStage.create(device, config, weights);

      const gpuOutputs: Float32Array[] = [];
      for (let t = 0; t < tokenCount; t += 1) {
        gpuOutputs.push(
          await stage.execute({ start: 0, end: layerCount }, inputs[t], {
            seqId: 1,
            tokenIndex: t,
          }),
        );
      }
      stage.destroy();

      // --- compare ---
      let maxAbsDiff = 0;
      let maxRelDiff = 0;
      let worstViolation = -Infinity;
      let allFinite = true;

      for (let t = 0; t < tokenCount; t += 1) {
        for (let i = 0; i < config.hiddenSize; i += 1) {
          const cpu = cpuOutputs[t][i];
          const gpu = gpuOutputs[t][i];
          if (!Number.isFinite(gpu) || !Number.isFinite(cpu)) allFinite = false;

          const abs = Math.abs(cpu - gpu);
          maxAbsDiff = Math.max(maxAbsDiff, abs);

          // Relative error is meaningless as |cpu| approaches zero, so it is
          // only sampled where the reference value carries real magnitude.
          if (Math.abs(cpu) > 1e-2) {
            maxRelDiff = Math.max(maxRelDiff, abs / Math.abs(cpu));
          }

          // numpy's allclose convention, which handles both regimes at once.
          const allowed = tolerance.atol + tolerance.rtol * Math.abs(cpu);
          worstViolation = Math.max(worstViolation, abs - allowed);
        }
      }

      return {
        maxAbsDiff,
        maxRelDiff,
        worstViolation,
        cpuFirst: Array.from(cpuOutputs[tokenCount - 1].slice(0, 6)),
        gpuFirst: Array.from(gpuOutputs[tokenCount - 1].slice(0, 6)),
        allFinite,
      };
    },
    { layerCount, tokenCount, tolerance },
  );
}

/**
 * Why not exact equality: WGSL permits reduced-precision transcendentals, so
 * a GPU `exp` or `inverseSqrt` differs from JavaScript's in the last few bits,
 * and f32 sums accumulate in a different order. Observed disagreement on this
 * config is ~2e-5 absolute — four orders of magnitude below what any real
 * kernel bug produces, which showed up as ~1.5 during development.
 */
const SINGLE_LAYER_TOLERANCE = { atol: 1e-4, rtol: 1e-3 };
const MULTI_LAYER_TOLERANCE = { atol: 1e-3, rtol: 1e-2 };

test('WebGPU layer matches the reference implementation for a single token', async ({ page }) => {
  test.skip(!(await ready(page)), 'no WebGPU adapter on this machine');

  const stats = await compare(page, 1, 1, SINGLE_LAYER_TOLERANCE);

  expect(stats.allFinite, 'outputs must be finite').toBe(true);
  expect(stats.worstViolation, `cpu=${stats.cpuFirst} gpu=${stats.gpuFirst}`).toBeLessThanOrEqual(0);
});

test('WebGPU attention matches the reference across a growing KV cache', async ({ page }) => {
  test.skip(!(await ready(page)), 'no WebGPU adapter on this machine');

  // Eight tokens exercise softmax over a real cache, RoPE at several positions,
  // and grouped-query head mapping — the parts most likely to be wrong.
  const stats = await compare(page, 1, 8, SINGLE_LAYER_TOLERANCE);

  expect(stats.allFinite).toBe(true);
  expect(stats.worstViolation).toBeLessThanOrEqual(0);
  // Softmax over a real cache is where a wrong KV-head mapping would show up.
  expect(stats.maxRelDiff).toBeLessThan(1e-2);
});

test('WebGPU matches the reference across a multi-layer stage', async ({ page }) => {
  test.skip(!(await ready(page)), 'no WebGPU adapter on this machine');

  // Four layers, four tokens: errors compound across layers, so this is the
  // strictest of the three.
  const stats = await compare(page, 4, 4, MULTI_LAYER_TOLERANCE);

  expect(stats.allFinite).toBe(true);
  expect(stats.worstViolation).toBeLessThanOrEqual(0);
});

test('concurrent sequences keep separate KV caches', async ({ page }) => {
  test.skip(!(await ready(page)), 'no WebGPU adapter on this machine');

  const result = await page.evaluate(async () => {
    const api = (window as unknown as { meshTransformer: any }).meshTransformer;
    const config = api.TEST_CONFIG;
    const weights = [api.randomWeights(config, 99)];

    const adapter = await (navigator as any).gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const stage = api.GpuTransformerStage.create(device, config, weights);

    const token = new Float32Array(config.hiddenSize).fill(0.1);

    // Sequence A gets three tokens; sequence B gets one. If the caches were
    // shared, B's single token would attend over A's history and diverge from
    // a clean one-token run.
    await stage.execute({ start: 0, end: 1 }, token, { seqId: 1, tokenIndex: 0 });
    await stage.execute({ start: 0, end: 1 }, token, { seqId: 1, tokenIndex: 1 });
    await stage.execute({ start: 0, end: 1 }, token, { seqId: 1, tokenIndex: 2 });

    const freshB = await stage.execute({ start: 0, end: 1 }, token, { seqId: 2, tokenIndex: 0 });

    // A clean reference run of one token.
    const cache = api.createKVCache(config);
    const expected = api.forwardLayer(config, weights[0], token, cache, 0);

    let maxDiff = 0;
    for (let i = 0; i < config.hiddenSize; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(expected[i] - freshB[i]));
    }

    const active = stage.activeSequences;
    stage.releaseSequence(1);
    const afterRelease = stage.activeSequences;
    stage.destroy();

    return { maxDiff, active, afterRelease };
  });

  expect(result.active, 'two sequences should each hold a cache').toBe(2);
  expect(result.afterRelease, 'releasing one should free it').toBe(1);
  expect(result.maxDiff, 'sequence B must not see sequence A history').toBeLessThan(1e-4);
});
