# Contributing to MeshGPU

Thanks for your interest in contributing! MeshGPU is a browser-native, fully
air-gapped WebGPU inference pooling project. This guide covers how to get set
up, run the tests, and submit changes.

## Code of Conduct

Be kind, constructive, and respectful. Assume good faith, keep discussions
technical, and remember that contributors span many time zones, languages, and
levels of experience.

## Before you start

- Search [open issues](../../issues) to see if your idea is already tracked.
- For anything larger than a typo, open an issue first to discuss the approach
  with maintainers — this saves you time if a change is out of scope.
- Read the [README](README.md) to understand the architecture and the
  air-gapped manual-handshake model.

## License & CLA

MeshGPU is licensed under the **AGPL-3.0-only** (see [LICENSE](LICENSE)).
Before we can merge your first pull request, you must agree to the
[Contributor License Agreement](CLA.md). The CLA is lightweight: you retain
ownership of your work, but you grant the project maintainer a broad,
irrevocable license to it — and, importantly, **assign copyright of your
contributed changes to the project maintainer**. This keeps the project able
to be relicensed or dual-licensed commercially in the future without chasing
down every contributor.

Add the following line to your PR description (or a `Signed-off-by` trailer in
your commit messages):

```text
I agree to the terms of the MeshGPU Contributor License Agreement (CLA.md).
```

## Development setup

Prerequisites: **Node.js ≥ 18**, npm, and a WebGPU-capable browser
(Chrome/Edge stable, Firefox Nightly, or Safari Technology Preview).

```bash
git clone https://github.com/<your-org>/mesh-gpu.git
cd mesh-gpu
npm install
npm run dev          # → http://localhost:5173
```

## Project structure

```
src/
├── engine/
│   ├── webgpu-node.ts      # GPUAdapter inspection + VRAM estimation
│   ├── manual-signaling.ts # Manual WebRTC SDP exchange (QR / Base64)
│   ├── signaling.ts        # Transport-agnostic signaling contracts
│   ├── p2p-pipeline.ts     # WebRTC DataChannel tensor streaming
│   ├── scheduler.ts        # Layer assignment & topology balancing
│   ├── mock-transport.ts   # In-memory RTC/DataChannel mock
│   └── web-llm.ts          # @mlc-ai/web-llm runtime
├── components/             # Topology graph, VRAM gauge, QR modal, chat UI
└── App.tsx
tests/                      # Playwright e2e (manual handshake + mock)
package.json
```

## Commands

| Command | Purpose |
| ------- | ------- |
| `npm run dev` | Start the Vite dev server |
| `npm run typecheck` | `tsc --noEmit` for the app and node configs |
| `npm run build` | Typecheck + production build |
| `npm run test:e2e:manual` | Manual-handshake + MockTransport tensor streaming tests |
| `npm run test:e2e` | Run all Playwright tests |

The test suite uses Playwright with headless Chromium. Install the browser once:

```bash
npx playwright install chromium
```

## Pull request checklist

- [ ] Branch from `main` and keep commits focused.
- [ ] Run `npm run typecheck` and `npm run test:e2e:manual` locally.
- [ ] Keep the code strictly typed (`tsconfig.json` enables `strict`,
      `noUnusedLocals`, `noUnusedParameters`).
- [ ] Add or update tests for behavioral changes.
- [ ] Update `README.md` if the user-facing workflow changes.
- [ ] Confirm the CLA line in your PR description.

## Style notes

- Prefer explicit types over `any`; keep the public engine surface
  (`p2p-pipeline.ts`, `manual-signaling.ts`) fully typed.
- Follow the existing file-header comment convention (`/** file.ts — purpose */`).
- Keep UI components presentational; put transport/compute logic in `engine/`.

## Questions?

Open an issue or start a discussion — maintainers are happy to help newcomers
find a good first issue.
