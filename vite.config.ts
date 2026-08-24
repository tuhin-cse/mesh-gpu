import { fileURLToPath, URL } from 'node:url';

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative asset paths so the built app works from any sub-path (GitHub Pages).
  base: './',
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
  },
  build: {
    target: 'esnext',
  },
  // WebGPU compute kernels (and WebLLM) ship WASM + workers; keep workers as ES modules.
  worker: {
    format: 'es',
  },
});
