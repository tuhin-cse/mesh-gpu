import { defineConfig } from 'vitest/config';

// Unit tests live next to the code they cover: `src/**/*.test.ts` for the
// browser client, `coordinator/test/*.test.js` for the control plane. The
// Playwright E2E suite owns `tests/` and runs separately via `npm run test:e2e`.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts', 'coordinator/test/**/*.test.js'],
    environment: 'node',
  },
});
