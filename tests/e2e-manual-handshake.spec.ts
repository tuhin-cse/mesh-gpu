import { test, expect, type Page } from '@playwright/test';

/**
 * E2E coverage for the air-gapped MeshGPU pipeline:
 *
 * 1. The Option-3 manual SDP exchange (no signaling server): the host exports
 *    a compressed offer, the joiner exports a compressed answer, and the host
 *    applies it — all client-side.
 * 2. Exact Float32Array tensor streaming over the in-memory mock transport
 *    (the documented headless path, since real ICE host candidates cannot
 *    connect between two isolated browser contexts in CI sandboxes).
 */

const TENSOR_VALUES: readonly number[] = [0.123, 0.456, 0.789];

// ---------------------------------------------------------------------------
// Page-side API surface exposed by `window.p2pPipeline` (dev builds).
// ---------------------------------------------------------------------------

interface TensorFrameLike {
  data: ArrayBuffer;
  dtype: number;
}

interface SignalingLike {
  connect(): Promise<unknown>;
  on(event: string, handler: (payload: { signal: unknown }) => void): () => void;
}

interface PeerLinkLike {
  open: boolean;
  connect(): Promise<void>;
  handleSignal(signal: unknown): Promise<void>;
  sendTensor(frame: unknown): boolean;
  close(): void;
}

interface ManualPeerConnectionLike {
  createOfferPayload(): Promise<string>;
  acceptOfferAndCreateAnswerPayload(offerBase64: string): Promise<string>;
  acceptAnswerPayload(answerBase64: string): Promise<void>;
  pc: { remoteDescription: { sdp?: string } | null };
  close(): void;
}

interface DecodedPayload {
  kind: 'offer' | 'answer';
  peerId: string;
  sdp: string;
  candidates: unknown[];
}

interface PipelineApi {
  PeerLink: new (options: {
    signaling: SignalingLike;
    remotePeerId: string;
    initiator: boolean;
    createPeerConnection?: unknown;
    callbacks?: {
      onTensor?: (link: unknown, frame: TensorFrameLike) => void;
    };
  }) => PeerLinkLike;
  ManualPeerConnection: new (options: { peerId: string }) => ManualPeerConnectionLike;
  decodeManualPayload(encoded: string): DecodedPayload;
  createSignalingAndTransport(options: {
    roomId: string;
    peerId: string;
  }): {
    signaling: unknown;
    createPeerConnection: unknown;
  };
  makeForwardFrame(
    shape: number[],
    data: Float32Array,
    meta: { seqId: number; tokenIndex: number; fromLayer: number; toLayer: number },
  ): unknown;
  frameAsFloat32(frame: TensorFrameLike): Float32Array;
}

async function waitForPipelineHook(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const w = window as unknown as { p2pPipeline?: { PipelineNode?: unknown } };
    return Boolean(w.p2pPipeline?.PipelineNode);
  });
}

// ---------------------------------------------------------------------------
// Test 1 — manual SDP exchange (no server, no network reachability required)
// ---------------------------------------------------------------------------

test('manual offer/answer payloads compress, decode and apply with no server', async ({
  page,
}) => {
  await page.goto('/');
  await waitForPipelineHook(page);

  const summary = await page.evaluate(async () => {
    const w = window as unknown as { p2pPipeline: PipelineApi };
    const host = new w.p2pPipeline.ManualPeerConnection({ peerId: 'host' });
    const joiner = new w.p2pPipeline.ManualPeerConnection({ peerId: 'joiner' });

    const offer = await host.createOfferPayload();
    const offerDecoded = w.p2pPipeline.decodeManualPayload(offer);

    const answer = await joiner.acceptOfferAndCreateAnswerPayload(offer);
    const answerDecoded = w.p2pPipeline.decodeManualPayload(answer);

    await host.acceptAnswerPayload(answer);
    const remoteDescriptionSet = Boolean(host.pc.remoteDescription?.sdp);

    host.close();
    joiner.close();

    return {
      offerKind: offerDecoded.kind,
      offerPeerId: offerDecoded.peerId,
      offerHasCandidate: offerDecoded.sdp.includes('a=candidate'),
      offerCandidateCount: offerDecoded.candidates.length,
      answerKind: answerDecoded.kind,
      answerPeerId: answerDecoded.peerId,
      answerHasCandidate: answerDecoded.sdp.includes('a=candidate'),
      remoteDescriptionSet,
    };
  });

  expect(summary.offerKind).toBe('offer');
  expect(summary.offerPeerId).toBe('host');
  expect(summary.offerHasCandidate).toBe(true);
  expect(summary.offerCandidateCount).toBeGreaterThan(0);
  expect(summary.answerKind).toBe('answer');
  expect(summary.answerPeerId).toBe('joiner');
  expect(summary.answerHasCandidate).toBe(true);
  expect(summary.remoteDescriptionSet).toBe(true);
});

// ---------------------------------------------------------------------------
// Test 2 — exact Float32Array tensor streaming over the in-memory transport
// ---------------------------------------------------------------------------

test('mock transport streams exact Float32Array tensor between two peers', async ({
  page,
}) => {
  await page.goto('/?mockP2P=true');
  await waitForPipelineHook(page);

  const result = await page.evaluate(async (values: number[]) => {
    const w = window as unknown as { p2pPipeline: PipelineApi };
    const roomId = `e2e-manual-${Date.now()}`;

    const { signaling: sigA, createPeerConnection: makeA } =
      w.p2pPipeline.createSignalingAndTransport({ roomId, peerId: 'a' });
    const { signaling: sigB, createPeerConnection: makeB } =
      w.p2pPipeline.createSignalingAndTransport({ roomId, peerId: 'b' });

    const receivedValues: number[][] = [];
    const linkB = new w.p2pPipeline.PeerLink({
      signaling: sigB as SignalingLike,
      remotePeerId: 'a',
      initiator: false,
      createPeerConnection: makeB,
      callbacks: {
        onTensor: (_link, frame) => {
          receivedValues.push(Array.from(w.p2pPipeline.frameAsFloat32(frame)));
        },
      },
    });
    const linkA = new w.p2pPipeline.PeerLink({
      signaling: sigA as SignalingLike,
      remotePeerId: 'b',
      initiator: true,
      createPeerConnection: makeA,
    });

    (sigA as SignalingLike).on('signal', (payload) => {
      void linkA.handleSignal(payload.signal).catch(() => undefined);
    });
    (sigB as SignalingLike).on('signal', (payload) => {
      void linkB.handleSignal(payload.signal).catch(() => undefined);
    });

    await (sigB as SignalingLike).connect();
    await (sigA as SignalingLike).connect();
    await linkA.connect();

    const deadline = Date.now() + 5000;
    while (!linkA.open && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }

    const frame = w.p2pPipeline.makeForwardFrame(
      [values.length],
      new Float32Array(values),
      { seqId: 1, tokenIndex: 0, fromLayer: 0, toLayer: 28 },
    );
    const sent = linkA.sendTensor(frame);

    while (receivedValues.length === 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }

    linkA.close();
    linkB.close();

    return { sent, received: receivedValues[0] ?? [] };
  }, TENSOR_VALUES as number[]);

  expect(result.sent, 'tensor should be accepted by the data channel').toBe(true);
  expect(result.received, 'tensor payload should match exactly after streaming').toEqual(
    Array.from(new Float32Array(TENSOR_VALUES as number[])),
  );
});

