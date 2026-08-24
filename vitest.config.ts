import { defineConfig } from 'vitest/config';

// Unit tests live next to the code as `*.test.ts`. The Playwright E2E suite
// owns `tests/` and is run separately via `npm run test:e2e`.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
