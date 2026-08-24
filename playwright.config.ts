import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * MeshGPU staging E2E configuration.
 *
 * The frontend (Vite, port 3000) is auto-started before any test runs. There
 * is no signaling server anymore — WebRTC handshakes are driven entirely
 * client-side via the manual SDP/QR payload flow.
 */
export default defineConfig({
  testDir: path.join(__dirname, 'tests'),
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  reporter: [['list']],

  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        // Expose real IPs (not mDNS .local names) so host-candidate loopback
        // connects between the two isolated browser contexts.
        '--disable-features=WebRtcHideLocalIpsWithMdns',
        '--no-sandbox',
      ],
    },
  },

  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],

  webServer: [
    {
      command: 'npm run dev -- --port 3000 --strictPort',
      cwd: __dirname,
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
});
