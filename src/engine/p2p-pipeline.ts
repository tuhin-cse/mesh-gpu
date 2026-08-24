/**
 * p2p-pipeline.ts
 *
 * WebRTC DataChannel binary tensor streaming + pipeline forwarding for MeshGPU.
 *
 * The pipeline shards a model's transformer layers across peers:
 *
 *   Tab A (layers [0,8)) --hidden-state--> Tab B (layers [8,16)) --> ... --> Tab N
 *
 * Hidden-state tensors stream as packed binary frames over a dedicated,
 * unordered / no-retransmit DataChannel (lowest possible latency). Control
 * messages (stage assignment, output tokens, latency pings) travel over a
 * separate ordered, reliable DataChannel.
 */

import type { SignalingClientLike, SignalPayload } from './signaling';
import {
  DEFAULT_MAX_CHUNK_BYTES,
  FrameReassembler,
  TensorDType,
  TensorFrameKind,
  decodeFrame,
  encodeTensor,
  frameAsFloat32,
} from './tensor-wire';
import type { FrameMeta, TensorFrame } from './tensor-wire';
import { sanitizeTensor } from './webgpu-node';
import { isMockP2PEnabled, MockSignaling } from './mock-transport';
import type { DataChannelLike, PeerConnectionFactory, PeerConnectionLike } from './mock-transport';
import {
  MANUAL_RTC_CONFIG,
  ManualPeerConnection,
  decodeManualPayload,
  encodeManualPayload,
} from './manual-signaling';

declare global {
  interface Window {
    /** Dev/test hook: exposes the pipeline API for Playwright evaluation. */
    p2pPipeline?: Record<string, unknown>;
    /** Dev/test hook: the most recently created RTCPeerConnection. */
    peerConnection?: unknown;
    /** Force relay-only ICE (strict NAT staging verification). */
    FORCE_ICE_RELAY?: boolean;
  }
}

/**
 * Whether to force relay-only ICE (`iceTransportPolicy: 'relay'`) to strictly
 * block direct host/srflx candidate selection. Enabled via `?forceRelay=true`
 * or `window.FORCE_ICE_RELAY = true` — used by the staging relay E2E suite.
 */
export function resolveForceRelay(): boolean {
  if (typeof window !== 'undefined') {
    try {
      if (new URLSearchParams(window.location.search).get('forceRelay') === 'true') return true;
    } catch {
      // Ignore malformed search params.
    }
    if (window.FORCE_ICE_RELAY === true) return true;
  }
  return false;
}

/** Create a real RTCPeerConnection, bridged to the transport-agnostic surface. */
export function defaultPeerConnectionFactory(): PeerConnectionFactory {
  return (_remotePeerId, config) => {
    const effectiveConfig: RTCConfiguration = resolveForceRelay()
      ? { ...config, iceTransportPolicy: 'relay' }
      : config;
    const pc = new RTCPeerConnection(effectiveConfig);
    if (typeof window !== 'undefined') {
      window.peerConnection = pc;
    }
    return pc as unknown as PeerConnectionLike;
  };
}

/**
 * Instantiate the signaling + transport for this process: the real WebSocket
 * signaling + RTCPeerConnection, or the in-memory mock when `?mockP2P=true`
 * (or `VITE_USE_MOCK_P2P=true`) is set — useful for headless tests where ICE
 * cannot gather candidates.
 */
export function createSignalingAndTransport(options: {
  url?: string;
  roomId: string;
  peerId: string;
}): { signaling: SignalingClientLike; createPeerConnection: PeerConnectionFactory } {
  if (isMockP2PEnabled()) {
    const signaling = new MockSignaling({ roomId: options.roomId, peerId: options.peerId });
    return {
      signaling,
      createPeerConnection: (remotePeerId) => signaling.createPeerConnection(remotePeerId),
    };
  }
  throw new Error(
    'WebSocket signaling has been removed — use the manual handshake flow '
    + '(PipelineNode.initiateHostConnection / initiateJoinerConnection) instead.',
  );
}

// ---------------------------------------------------------------------------
// Tensor wire format
//
// Lives in tensor-wire.ts: a 32-byte framed format with chunking (so
// prefill-sized tensors survive a DataChannel) and f16 payloads (so they cost
// half as much to move). Re-exported here because the pipeline is what most
// callers import.
// ---------------------------------------------------------------------------

export {
  DEFAULT_MAX_CHUNK_BYTES,
  FrameReassembler,
  TENSOR_HEADER_BYTES,
  TENSOR_MAGIC,
  TENSOR_VERSION,
  TensorDType,
  TensorFrameKind,
  bytesPerElement,
  decodeFrame,
  elementCount,
  encodeFrame,
  encodeTensor,
  frameAsFloat32,
} from './tensor-wire';
export type { FrameMeta, TensorFrame } from './tensor-wire';

/** A contiguous, half-open layer range: [start, end). */
export interface StageRange {
  start: number;
  end: number;
}

// ---------------------------------------------------------------------------
// Control messages (JSON over the reliable DataChannel)
// ---------------------------------------------------------------------------

export type ControlMessage =
  | { type: 'ping'; sentAt: number }
  | { type: 'pong'; sentAt: number; receivedAt: number }
  | { type: 'token'; seqId: number; tokenIndex: number; token: number }
  | { type: 'stage'; stage: StageRange | null }
  | { type: 'capacity'; hostableLayers: number; totalLayers: number; vramBytes: number | null }
  | { type: 'benchmark'; throughput: number; gflops: number; bandwidthGiBps: number }
  /**
   * A downstream stage could not process a sequence. Replaces the old
   * `tensor-retransmit`: with a reliable tensor channel a bad or unprocessable
   * payload will not become good on a second sending, so the originator needs
   * to learn the sequence is dead rather than wait out its timeout.
   */
  | { type: 'sequence-failed'; seqId: number; reason: string };

// ---------------------------------------------------------------------------
// Stage execution
// ---------------------------------------------------------------------------

export interface ExecContext {
  seqId: number;
  tokenIndex: number;
}

export interface OutputToken {
  token: number;
  logits?: Float32Array;
}

/**
 * Executes a contiguous layer range on a hidden-state tensor. The real
 * implementation will wrap @mlc-ai/web-llm / ONNX Runtime Web; the interface
 * keeps the pipeline transport fully decoupled from the compute backend.
 */
export interface StageExecutor {
  execute(
    stage: StageRange,
    input: Float32Array,
    context: ExecContext,
  ): Float32Array | Promise<Float32Array>;

  /** Produce the sampled token for the final stage. Falls back to argmax if omitted. */
  finalize?(
    stage: StageRange,
    hidden: Float32Array,
    context: ExecContext,
  ): OutputToken | Promise<OutputToken>;
}

/**
 * Identity executor: returns the input unchanged. Exists to exercise the full
 * transport + forwarding path without a model download; swap for a real
 * executor once the compute backend is wired in.
 */
export class IdentityExecutor implements StageExecutor {
  execute(_stage: StageRange, input: Float32Array, _context: ExecContext): Float32Array {
    return input;
  }

  finalize(_stage: StageRange, hidden: Float32Array, _context: ExecContext): OutputToken {
    let best = 0;
    for (let i = 1; i < hidden.length; i += 1) {
      if (hidden[i] > hidden[best]) best = i;
    }
    return { token: best, logits: hidden };
  }
}

// ---------------------------------------------------------------------------
// Peer link: one RTCPeerConnection + two negotiated DataChannels
// ---------------------------------------------------------------------------

interface ChannelSpec {
  id: number;
  label: string;
  ordered: boolean;
}

const CONTROL_CHANNEL: ChannelSpec = { id: 0, label: 'meshgpu-control', ordered: true };
/**
 * Unordered but *reliable*.
 *
 * The tensor channel used to be no-retransmit, on the theory that dropping a
 * late tensor beats waiting for it. That is wrong for a forward pass: a lost
 * hidden state has no replacement, so the sequence simply hangs. At ~7 KiB per
 * token in f16, letting SCTP retransmit costs almost nothing.
 *
 * Ordering stays off deliberately — with several sequences in flight for
 * microbatching, head-of-line blocking on one would stall all the others.
 */
const TENSOR_CHANNEL: ChannelSpec = { id: 1, label: 'meshgpu-tensor', ordered: false };

export const DEFAULT_RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

/** Fall back to relay routing if direct P2P hasn't connected within this window. */
export const ICE_FALLBACK_MS = 3000;

/**
 * Pause sending once the SCTP send buffer passes this, and resume when it
 * drains below it. Previously frames over this threshold were dropped and the
 * caller never found out; now the sender waits.
 */
const MAX_BUFFERED_BYTES = 4 * 1024 * 1024;

/** Give up on a send if the peer's buffer never drains within this window. */
const DRAIN_TIMEOUT_MS = 30_000;

/** How often to re-check `bufferedAmount` while waiting for it to drain. */
const DRAIN_POLL_MS = 25;

/** Liveness heartbeat: probe every second; drop a peer after no traffic for 3s. */
export const HEARTBEAT_INTERVAL_MS = 1000;
export const HEARTBEAT_TIMEOUT_MS = 3000;

export interface PeerLinkCallbacks {
  onOpen?: (link: PeerLink) => void;
  onClose?: (link: PeerLink) => void;
  onControl?: (link: PeerLink, message: ControlMessage) => void;
  onTensor?: (link: PeerLink, frame: TensorFrame) => void;
  onError?: (link: PeerLink, error: Error) => void;
  /** Fired when the link switches to relay-only routing (strict-NAT fallback). */
  onRelayFallback?: (link: PeerLink) => void;
}

export interface PeerLinkChannels {
  pc: PeerConnectionLike;
  control: DataChannelLike;
  tensor: DataChannelLike;
}

export interface PeerLinkOptions {
  /** Signaling bus (mock mode). Omit for manual QR/copy-paste handshakes. */
  signaling?: SignalingClientLike;
  remotePeerId: string;
  /** True if this side creates the offer (deterministic: smaller peerId initiates). */
  initiator: boolean;
  rtcConfig?: RTCConfiguration;
  /** Transport factory; defaults to a real RTCPeerConnection. */
  createPeerConnection?: PeerConnectionFactory;
  /** Pre-negotiated channels (manual mode) — skips internal PC/channel creation. */
  channels?: PeerLinkChannels;
  callbacks?: PeerLinkCallbacks;
}

export class PeerLink {
  readonly remotePeerId: string;
  readonly initiator: boolean;

  private readonly signaling?: SignalingClientLike;
  private readonly baseRtcConfig: RTCConfiguration;
  private readonly createPeerConnection: PeerConnectionFactory;
  private readonly callbacks: PeerLinkCallbacks;

  private pc: PeerConnectionLike;
  private controlChannel: DataChannelLike;
  private tensorChannel: DataChannelLike;

  private channelsOpen = 0;
  private closed = false;
  private relayForced = false;
  private fallbackTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingCandidates: RTCIceCandidateInit[] = [];
  private readonly pendingLatency = new Map<number, (rttMs: number) => void>();
  /** Chunks arrive out of order; this holds them until each frame is whole. */
  private readonly reassembler = new FrameReassembler();

  constructor(options: PeerLinkOptions) {
    this.remotePeerId = options.remotePeerId;
    this.initiator = options.initiator;
    this.signaling = options.signaling;
    this.callbacks = options.callbacks ?? {};
    this.baseRtcConfig = options.rtcConfig ?? DEFAULT_RTC_CONFIG;
    this.createPeerConnection = options.createPeerConnection ?? defaultPeerConnectionFactory();

    if (options.channels) {
      this.pc = options.channels.pc;
      this.controlChannel = options.channels.control;
      this.tensorChannel = options.channels.tensor;
    } else {
      this.pc = this.createPeerConnection(this.remotePeerId, this.baseRtcConfig);
      this.controlChannel = this.createChannel(CONTROL_CHANNEL);
      this.tensorChannel = this.createChannel(TENSOR_CHANNEL);
    }
    this.wire();
  }

  get open(): boolean {
    return this.channelsOpen === 2 && !this.closed;
  }

  /** True once relay-only routing has been forced (strict-NAT fallback). */
  get usedRelay(): boolean {
    return this.relayForced;
  }

  /** Initiator only: create and send an offer. */
  async connect(): Promise<void> {
    await this.sendOffer();
  }

  /** Manual (offline) host flow: apply the joiner's answer to complete the link. */
  async applyRemoteAnswer(answerBase64: string): Promise<void> {
    const answer = decodeManualPayload(answerBase64);
    if (answer.kind !== 'answer') {
      throw new Error('expected an answer payload');
    }
    await this.pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
    await this.addRemoteCandidates(answer.candidates);
  }

  private async addRemoteCandidates(candidates: RTCIceCandidateInit[]): Promise<void> {
    for (const candidate of candidates) {
      if (!candidate.candidate) continue;
      // The bundled SDP already carries these candidates; duplicates are ignored.
      await this.pc.addIceCandidate(candidate).catch(() => undefined);
    }
  }

  /** Handle an offer/answer/ICE candidate/relay-restart relayed by the server. */
  async handleSignal(signal: SignalPayload): Promise<void> {
    if (signal.relayRestart) {
      this.relayForced = true;
      this.cancelFallbackTimer();
      this.recreate(true);
      if (this.initiator) await this.sendOffer();
      return;
    }

    if (signal.description) {
      // If we already committed to relay but the PC hasn't been rebuilt, do it
      // before applying the description (handles out-of-order delivery).
      if (this.relayForced && !this.isRelayPc()) this.recreate(true);
      await this.pc.setRemoteDescription(signal.description);
      await this.flushCandidates();
      if (signal.description.type === 'offer') {
        const answer = await this.pc.createAnswer();
        await this.pc.setLocalDescription(answer);
        this.sendDescription(this.pc.localDescription);
      }
    } else if (signal.candidate) {
      if (this.pc.remoteDescription) {
        await this.pc.addIceCandidate(signal.candidate).catch(() => undefined);
      } else {
        this.pendingCandidates.push(signal.candidate);
      }
    }
  }

  sendControl(message: ControlMessage): void {
    if (this.controlChannel.readyState === 'open') {
      this.controlChannel.send(JSON.stringify(message));
    }
  }

  /**
   * Send a hidden state, chunking it and waiting whenever the send buffer is
   * full. Resolves false if the link closed or the buffer never drained —
   * either way the caller learns, rather than losing the tensor silently.
   */
  async sendTensor(
    shape: readonly number[],
    values: Float32Array,
    meta: FrameMeta,
    options: { dtype?: TensorDType.F32 | TensorDType.F16; maxChunkBytes?: number } = {},
  ): Promise<boolean> {
    const chunks = encodeTensor(shape, values, meta, {
      dtype: options.dtype ?? TensorDType.F16,
      maxChunkBytes: options.maxChunkBytes ?? DEFAULT_MAX_CHUNK_BYTES,
    });

    for (const chunk of chunks) {
      if (this.closed || this.tensorChannel.readyState !== 'open') return false;
      if (!(await this.awaitDrain())) return false;
      this.tensorChannel.send(chunk);
    }
    return true;
  }

  /** Send one pre-encoded chunk. Used to satisfy a retransmit request. */
  sendRawChunk(chunk: ArrayBuffer): boolean {
    if (this.closed || this.tensorChannel.readyState !== 'open') return false;
    this.tensorChannel.send(chunk);
    return true;
  }

  /**
   * Wait until the send buffer has room. `bufferedamountlow` is not on the
   * transport-agnostic channel surface (the in-memory mock has no such event),
   * so this polls — cheap, since it only runs when the buffer is actually full.
   */
  private async awaitDrain(): Promise<boolean> {
    if (this.tensorChannel.bufferedAmount <= MAX_BUFFERED_BYTES) return true;

    const deadline = Date.now() + DRAIN_TIMEOUT_MS;
    while (this.tensorChannel.bufferedAmount > MAX_BUFFERED_BYTES) {
      if (this.closed || this.tensorChannel.readyState !== 'open') return false;
      if (Date.now() > deadline) {
        this.callbacks.onError?.(
          this,
          new Error(`send buffer to ${this.remotePeerId} did not drain in ${DRAIN_TIMEOUT_MS}ms`),
        );
        return false;
      }
      await new Promise((resolve) => setTimeout(resolve, DRAIN_POLL_MS));
    }
    return true;
  }

  /** Round-trip time in milliseconds to this peer (null on timeout). */
  measureLatency(timeoutMs = 5000): Promise<number | null> {
    const sentAt = performance.now();
    return new Promise<number | null>((resolve) => {
      let settled = false;
      const done = (rttMs: number | null): void => {
        if (settled) return;
        settled = true;
        this.pendingLatency.delete(sentAt);
        resolve(rttMs);
      };
      this.pendingLatency.set(sentAt, (rttMs) => done(rttMs));
      this.sendControl({ type: 'ping', sentAt });
      setTimeout(() => done(null), timeoutMs);
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.cancelFallbackTimer();
    this.controlChannel.close();
    this.tensorChannel.close();
    this.pc.close();
    this.pendingLatency.clear();
    this.reassembler.clear();
    this.callbacks.onClose?.(this);
  }

  private createChannel(channel: ChannelSpec): DataChannelLike {
    // No maxRetransmits: unordered delivery, but SCTP still guarantees arrival.
    return this.pc.createDataChannel(channel.label, {
      negotiated: true,
      id: channel.id,
      ordered: channel.ordered,
    });
  }

  private wire(): void {
    this.pc.onicecandidate = (event) => {
      if (event.candidate && this.signaling) {
        this.signaling.sendSignal(this.remotePeerId, { candidate: event.candidate.toJSON() });
      }
    };

    this.pc.oniceconnectionstatechange = () => this.onIceStateChange();

    this.pc.onconnectionstatechange = () => {
      const state = this.pc.connectionState;
      if (state === 'closed' || state === 'failed') this.close();
    };

    this.controlChannel.onopen = () => this.onChannelOpen();
    this.tensorChannel.onopen = () => this.onChannelOpen();

    this.controlChannel.onclose = () => this.close();
    this.controlChannel.onerror = () => this.close();

    this.controlChannel.onmessage = (event) => {
      let message: ControlMessage;
      try {
        message = JSON.parse(String(event.data)) as ControlMessage;
      } catch {
        return;
      }
      this.onControlMessage(message);
    };

    this.tensorChannel.onmessage = (event) => {
      let complete: TensorFrame | null;
      try {
        complete = this.reassembler.push(decodeFrame(event.data as ArrayBuffer));
      } catch (err) {
        this.callbacks.onError?.(this, toError(err));
        return;
      }
      // Null means more chunks of this frame are still in flight.
      if (complete) this.callbacks.onTensor?.(this, complete);
    };
  }

  private onIceStateChange(): void {
    const state = this.pc.iceConnectionState;
    switch (state) {
      case 'checking':
        this.startFallbackTimer();
        break;
      case 'connected':
      case 'completed':
        this.cancelFallbackTimer();
        break;
      case 'failed':
        this.cancelFallbackTimer();
        if (!this.relayForced) this.fallbackToRelay();
        break;
      default:
        break;
    }
  }

  private startFallbackTimer(): void {
    if (this.fallbackTimer !== null || this.relayForced) return;
    this.fallbackTimer = setTimeout(() => {
      this.fallbackTimer = null;
      const state = this.pc.iceConnectionState;
      if (state !== 'connected' && state !== 'completed') {
        this.fallbackToRelay();
      }
    }, ICE_FALLBACK_MS);
  }

  private cancelFallbackTimer(): void {
    if (this.fallbackTimer !== null) {
      clearTimeout(this.fallbackTimer);
      this.fallbackTimer = null;
    }
  }

  private fallbackToRelay(): void {
    if (this.relayForced || this.closed || !this.signaling) return;
    this.relayForced = true;
    this.cancelFallbackTimer();
    this.recreate(true);
    // Ask the peer to switch to relay too, then (re)start negotiation.
    this.signaling.sendSignal(this.remotePeerId, { relayRestart: true });
    this.callbacks.onRelayFallback?.(this);
    if (this.initiator) {
      void this.sendOffer().catch((err) => this.callbacks.onError?.(this, toError(err)));
    }
  }

  private recreate(relayOnly: boolean): void {
    this.cancelFallbackTimer();

    // Detach handlers so closing the old channels/PC doesn't tear down the link.
    this.controlChannel.onopen = null;
    this.controlChannel.onclose = null;
    this.controlChannel.onerror = null;
    this.controlChannel.onmessage = null;
    this.tensorChannel.onopen = null;
    this.tensorChannel.onmessage = null;
    this.pc.onicecandidate = null;
    this.pc.oniceconnectionstatechange = null;
    this.pc.onconnectionstatechange = null;

    this.controlChannel.close();
    this.tensorChannel.close();
    this.pc.close();
    this.channelsOpen = 0;
    this.pendingCandidates = [];

    const rtcConfig: RTCConfiguration = {
      ...this.baseRtcConfig,
      iceTransportPolicy: relayOnly ? 'relay' : 'all',
    };
    this.pc = this.createPeerConnection(this.remotePeerId, rtcConfig);
    this.controlChannel = this.createChannel(CONTROL_CHANNEL);
    this.tensorChannel = this.createChannel(TENSOR_CHANNEL);
    this.wire();
  }

  private isRelayPc(): boolean {
    return this.pc.getConfiguration().iceTransportPolicy === 'relay';
  }

  private async sendOffer(): Promise<void> {
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.sendDescription(this.pc.localDescription);
  }

  private onChannelOpen(): void {
    this.channelsOpen += 1;
    if (this.channelsOpen === 2) this.callbacks.onOpen?.(this);
  }

  private onControlMessage(message: ControlMessage): void {
    // Every inbound control frame is a liveness signal for the mesh heartbeat.
    this.callbacks.onControl?.(this, message);

    switch (message.type) {
      case 'ping':
        this.sendControl({ type: 'pong', sentAt: message.sentAt, receivedAt: performance.now() });
        break;
      case 'pong': {
        const resolve = this.pendingLatency.get(message.sentAt);
        if (resolve) {
          this.pendingLatency.delete(message.sentAt);
          resolve(performance.now() - message.sentAt);
        }
        break;
      }
      default:
        break;
    }
  }

  private sendDescription(description: RTCSessionDescriptionInit | null): void {
    if (!description || !this.signaling) return;
    this.signaling.sendSignal(this.remotePeerId, {
      description: { type: description.type, sdp: description.sdp ?? '' },
    });
  }

  private async flushCandidates(): Promise<void> {
    const pending = this.pendingCandidates;
    this.pendingCandidates = [];
    for (const candidate of pending) {
      await this.pc.addIceCandidate(candidate).catch(() => undefined);
    }
  }
}

// ---------------------------------------------------------------------------
// Pipeline mesh: manages PeerLinks for every peer in the room
// ---------------------------------------------------------------------------

export interface MeshCallbacks {
  onPeerConnected?: (peerId: string) => void;
  onPeerDisconnected?: (peerId: string) => void;
  onControl?: (peerId: string, message: ControlMessage) => void;
  onTensor?: (peerId: string, frame: TensorFrame) => void;
  onError?: (peerId: string, error: Error) => void;
}

export interface PipelineMeshOptions {
  /** Signaling bus (mock mode). Omit for manual QR/copy-paste handshakes. */
  signaling?: SignalingClientLike;
  rtcConfig?: RTCConfiguration;
  createPeerConnection?: PeerConnectionFactory;
  callbacks?: MeshCallbacks;
}

export class PipelineMesh {
  readonly signaling?: SignalingClientLike;

  private readonly rtcConfig?: RTCConfiguration;
  private readonly createPeerConnection: PeerConnectionFactory;
  private readonly callbacks: MeshCallbacks;
  private readonly links = new Map<string, PeerLink>();
  private readonly lastSeen = new Map<string, number>();
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: PipelineMeshOptions) {
    this.signaling = options.signaling;
    this.rtcConfig = options.rtcConfig;
    this.createPeerConnection = options.createPeerConnection ?? defaultPeerConnectionFactory();
    this.callbacks = options.callbacks ?? {};

    if (this.signaling) {
      this.signaling.on('peer-joined', (payload) => this.ensureLink(payload.peerId));
      this.signaling.on('peer-left', (payload) => this.removeLink(payload.peerId));
      this.signaling.on('signal', (payload) => {
        const link = this.ensureLink(payload.fromPeerId);
        if (link) {
          link
            .handleSignal(payload.signal)
            .catch((err) => this.callbacks.onError?.(payload.fromPeerId, toError(err)));
        }
      });
    }
  }

  /** Connect to the signaling server and establish links to existing peers. */
  async join(): Promise<void> {
    if (!this.signaling) return;
    const welcome = await this.signaling.connect();
    for (const peerId of welcome.peers) this.ensureLink(peerId);
    this.startHeartbeat();
  }

  /**
   * Attach a manually-negotiated connection (offline QR/copy-paste mode).
   * Replaces any existing link to the same peer.
   */
  attachManualConnection(
    peerId: string,
    connection: ManualPeerConnection,
    initiator: boolean,
  ): PeerLink {
    const existing = this.links.get(peerId);
    if (existing) {
      existing.close();
      this.links.delete(peerId);
    }

    const link = new PeerLink({
      remotePeerId: peerId,
      initiator,
      rtcConfig: this.resolveRtcConfig(),
      createPeerConnection: this.createPeerConnection,
      channels: {
        pc: connection.pc as unknown as PeerConnectionLike,
        control: connection.controlChannel as unknown as DataChannelLike,
        tensor: connection.tensorChannel as unknown as DataChannelLike,
      },
      callbacks: this.linkCallbacks(peerId),
    });

    this.links.set(peerId, link);
    this.startHeartbeat();
    return link;
  }

  /** Look up an existing link by peer id (manual or signaling-driven). */
  getLink(peerId: string): PeerLink | null {
    return this.links.get(peerId) ?? null;
  }

  get connectedPeers(): string[] {
    return [...this.links.keys()];
  }

  sendControl(peerId: string, message: ControlMessage): void {
    this.links.get(peerId)?.sendControl(message);
  }

  sendTensorTo(
    peerId: string,
    shape: readonly number[],
    values: Float32Array,
    meta: FrameMeta,
  ): Promise<boolean> {
    const link = this.links.get(peerId);
    if (!link) return Promise.resolve(false);
    return link.sendTensor(shape, values, meta);
  }

  broadcastControl(message: ControlMessage): void {
    for (const link of this.links.values()) link.sendControl(message);
  }

  measureLatency(peerId: string): Promise<number | null> | null {
    return this.links.get(peerId)?.measureLatency() ?? null;
  }

  close(): void {
    this.stopHeartbeat();
    for (const link of this.links.values()) link.close();
    this.links.clear();
    this.lastSeen.clear();
    this.signaling?.disconnect();
  }

  private ensureLink(peerId: string): PeerLink | null {
    if (!this.signaling) return null;
    if (peerId === this.signaling.currentPeerId) return null;
    const existing = this.links.get(peerId);
    if (existing) return existing;

    const link = new PeerLink({
      signaling: this.signaling,
      remotePeerId: peerId,
      // Deterministic initiator rule avoids simultaneous-offer glare.
      initiator: this.signaling.currentPeerId < peerId,
      rtcConfig: this.resolveRtcConfig(),
      createPeerConnection: this.createPeerConnection,
      callbacks: this.linkCallbacks(peerId),
    });

    this.links.set(peerId, link);
    if (link.initiator) {
      link.connect().catch((err) => this.callbacks.onError?.(peerId, toError(err)));
    }
    return link;
  }

  private linkCallbacks(peerId: string): PeerLinkCallbacks {
    return {
      onOpen: () => {
        this.lastSeen.set(peerId, performance.now());
        this.callbacks.onPeerConnected?.(peerId);
      },
      onClose: () => this.removeLink(peerId),
      onControl: (_link, message) => {
        this.lastSeen.set(peerId, performance.now());
        this.callbacks.onControl?.(peerId, message);
      },
      onTensor: (_link, frame) => {
        this.lastSeen.set(peerId, performance.now());
        this.callbacks.onTensor?.(peerId, frame);
      },
      onError: (_link, error) => this.callbacks.onError?.(peerId, error),
      onRelayFallback: () => this.callbacks.onError?.(peerId, new Error(`relay fallback (${peerId})`)),
    };
  }

  private removeLink(peerId: string): void {
    const link = this.links.get(peerId);
    if (!link) return;
    this.links.delete(peerId);
    this.lastSeen.delete(peerId);
    link.close();
    this.callbacks.onPeerDisconnected?.(peerId);
  }

  private resolveRtcConfig(): RTCConfiguration {
    if (this.rtcConfig) return this.rtcConfig;
    const iceServers = this.signaling?.currentIceServers;
    if (iceServers && iceServers.length > 0) return { iceServers: [...iceServers] };
    return DEFAULT_RTC_CONFIG;
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer !== null) return;
    this.heartbeatTimer = setInterval(() => this.tickHeartbeat(), HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /** Probe every open link; drop peers that have been silent too long. */
  private tickHeartbeat(): void {
    const now = performance.now();
    for (const [peerId, link] of this.links) {
      if (!link.open) continue;
      const lastSeen = this.lastSeen.get(peerId) ?? now;
      if (now - lastSeen > HEARTBEAT_TIMEOUT_MS) {
        this.removeLink(peerId); // dead peer → orphan layers get reassigned
        continue;
      }
      link.sendControl({ type: 'ping', sentAt: now });
    }
  }
}

// ---------------------------------------------------------------------------
// Pipeline node: executes a stage and forwards hidden state to the next peer
// ---------------------------------------------------------------------------

/** What a node just handed to the next stage. */
export interface ForwardInfo {
  seqId: number;
  tokenIndex: number;
  shape: number[];
  fromLayer: number;
  toLayer: number;
}

export interface OutputTokenMeta {
  seqId: number;
  tokenIndex: number;
  fromPeerId: string;
}

export interface PipelineNodeCallbacks {
  onPeerConnected?: (peerId: string) => void;
  onPeerDisconnected?: (peerId: string) => void;
  onStageChanged?: (stage: StageRange | null) => void;
  onToken?: (token: number, meta: OutputTokenMeta) => void;
  onForwarded?: (info: ForwardInfo) => void;
  onError?: (peerId: string, error: Error) => void;
}

export interface PipelineNodeOptions {
  /** Signaling bus (mock mode). Omit for manual QR/copy-paste handshakes. */
  signaling?: SignalingClientLike;
  /** Explicit peer id (manual mode). Defaults to the signaling id or a UUID. */
  peerId?: string;
  executor?: StageExecutor;
  createPeerConnection?: PeerConnectionFactory;
  callbacks?: PipelineNodeCallbacks;
  /**
   * Sequences this node may have in flight at once. Values above 1 are what
   * make pipeline parallelism worth anything: at batch 1 only one stage is
   * busy at a time, so utilisation is 1/N. Several concurrent sequences keep
   * every stage fed.
   */
  maxConcurrent?: number;
  /** Fail a sequence that produces no token within this window. */
  runTimeoutMs?: number;
}

/** Events emitted for external observers (e.g. the dynamic scheduler). */
export type PipelineNodeEvent =
  | { type: 'peer-connected'; peerId: string }
  | { type: 'peer-disconnected'; peerId: string }
  | { type: 'control'; peerId: string; message: ControlMessage };

interface InFlightEntry {
  hidden: Float32Array;
  shape: number[];
  tokenIndex: number;
  sentTo: string;
}

interface PendingRun {
  resolve: (token: number) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * Default concurrent sequences. Four keeps a short pipeline busy without
 * letting queued work pile up faster than a browser tab can drain it.
 */
const DEFAULT_MAX_CONCURRENT = 4;

/**
 * A sequence that has produced no token in this long is not coming back — a
 * peer died, or a stage is wedged. Without this the forward pass hangs
 * forever, which is exactly what the old no-retransmit channel did on a drop.
 */
const DEFAULT_RUN_TIMEOUT_MS = 60_000;

function generatePeerId(): string {
  const id =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `node-${id.slice(0, 8)}`;
}

export class PipelineNode {
  readonly mesh: PipelineMesh;
  readonly executor: StageExecutor;

  private readonly selfPeerId: string;
  private readonly callbacks: PipelineNodeCallbacks;
  private readonly topology = new Map<string, StageRange>();
  private readonly listeners = new Set<(event: PipelineNodeEvent) => void>();
  private readonly inFlight = new Map<number, InFlightEntry>();
  /** Sequences this node started and is still waiting on a token for. */
  private readonly pendingRuns = new Map<number, PendingRun>();
  /** Sequences waiting for a slot because maxConcurrent is reached. */
  private readonly admissionQueue: Array<() => void> = [];
  /**
   * Slots taken, counted synchronously at admission.
   *
   * This cannot be derived from `pendingRuns.size`: a run is recorded there
   * only after `await acquireSlot()` resumes, so several concurrent callers
   * would all see an empty map and admit themselves at once.
   */
  private activeSlots = 0;
  private readonly maxConcurrent: number;
  private readonly runTimeoutMs: number;
  private stage: StageRange | null = null;
  private seqCounter = 0;

  constructor(options: PipelineNodeOptions) {
    this.selfPeerId = options.signaling?.currentPeerId ?? options.peerId ?? generatePeerId();
    this.executor = options.executor ?? new IdentityExecutor();
    this.callbacks = options.callbacks ?? {};
    this.maxConcurrent = Math.max(1, options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT);
    this.runTimeoutMs = options.runTimeoutMs ?? DEFAULT_RUN_TIMEOUT_MS;

    this.mesh = new PipelineMesh({
      signaling: options.signaling,
      createPeerConnection: options.createPeerConnection,
      callbacks: {
        onPeerConnected: (peerId) => {
          this.announceStage(peerId);
          this.emitEvent({ type: 'peer-connected', peerId });
          this.callbacks.onPeerConnected?.(peerId);
        },
        onPeerDisconnected: (peerId) => {
          this.topology.delete(peerId);
          this.emitEvent({ type: 'peer-disconnected', peerId });
          this.callbacks.onPeerDisconnected?.(peerId);
        },
        onControl: (peerId, message) => {
          this.emitEvent({ type: 'control', peerId, message });
          void this.handleControl(peerId, message);
        },
        onTensor: (peerId, frame) => {
          void this.handleForward(peerId, frame);
        },
        onError: (peerId, error) => {
          this.callbacks.onError?.(peerId, error);
        },
      },
    });
  }

  get peerId(): string {
    return this.selfPeerId;
  }

  get currentStage(): StageRange | null {
    return this.stage ? { ...this.stage } : null;
  }

  get peers(): string[] {
    return this.mesh.connectedPeers;
  }

  /** Subscribe to peer lifecycle + control events. Returns an unsubscribe fn. */
  subscribe(listener: (event: PipelineNodeEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Join the room and open links to existing peers. */
  join(): Promise<void> {
    return this.mesh.join();
  }

  /**
   * Manual (offline) host flow: gather a complete offer, bundle it, and export
   * the compressed Base64 payload for QR/copy-paste transfer to the joiner.
   */
  async initiateHostConnection(remotePeerId: string): Promise<string> {
    const connection = new ManualPeerConnection({ peerId: this.selfPeerId });
    try {
      const offer = await connection.createOfferPayload();
      this.mesh.attachManualConnection(remotePeerId, connection, true);
      return offer;
    } catch (err) {
      connection.close();
      throw err;
    }
  }

  /**
   * Manual (offline) joiner flow: accept a host offer, gather the answer, and
   * export its compressed Base64 payload for transfer back to the host.
   */
  async initiateJoinerConnection(remotePeerId: string, offerBase64: string): Promise<string> {
    const connection = new ManualPeerConnection({ peerId: this.selfPeerId });
    try {
      const answer = await connection.acceptOfferAndCreateAnswerPayload(offerBase64);
      this.mesh.attachManualConnection(remotePeerId, connection, false);
      return answer;
    } catch (err) {
      connection.close();
      throw err;
    }
  }

  /**
   * Manual (offline) host flow: apply the joiner's answer to the pending link
   * established by `initiateHostConnection`, opening the direct P2P channel.
   */
  async applyRemoteAnswer(remotePeerId: string, answerBase64: string): Promise<void> {
    const link = this.mesh.getLink(remotePeerId);
    if (!link) {
      throw new Error(
        `no pending handshake link for peer "${remotePeerId}" — generate an offer first`,
      );
    }
    await link.applyRemoteAnswer(answerBase64);
  }

  close(): void {
    this.mesh.close();
  }

  /** Send a control message to a single peer. */
  sendControlTo(peerId: string, message: ControlMessage): void {
    this.mesh.sendControl(peerId, message);
  }

  /** Broadcast a control message to all connected peers. */
  broadcastControl(message: ControlMessage): void {
    this.mesh.broadcastControl(message);
  }

  /** Assign this node's layer range and broadcast it to the pipeline. */
  setStage(stage: StageRange | null): void {
    if (stage && (stage.start < 0 || stage.end <= stage.start)) {
      throw new Error(`invalid stage range [${stage.start}, ${stage.end})`);
    }
    this.stage = stage ? { ...stage } : null;
    if (this.stage) this.topology.set(this.selfPeerId, this.stage);
    else this.topology.delete(this.selfPeerId);
    this.broadcastStage();
    this.callbacks.onStageChanged?.(this.stage);
  }

  /** Known peer -> stage mapping (for the topology visualizer). */
  getTopology(): Array<{ peerId: string; stage: StageRange }> {
    return [...this.topology.entries()]
      .filter(([peerId]) => peerId !== this.selfPeerId)
      .map(([peerId, stage]) => ({ peerId, stage: { ...stage } }));
  }

  /**
   * Re-route in-flight hidden states after a topology change (e.g. a peer
   * disconnected and the scheduler rebalanced). Frames previously sent to a
   * now-dead peer are re-forwarded to the current next hop, or finalized
   * locally if this node became the last stage.
   */
  retryPending(): void {
    if (!this.stage) return;
    const next = this.nextHop();

    for (const [seqId, entry] of this.inFlight) {
      if (!next) {
        // This node became the last stage — finish the sequence here.
        this.inFlight.delete(seqId);
        void this.finalizeStage(this.stage, entry.hidden, seqId, entry.tokenIndex)
          .then((token) => this.deliverToken(seqId, entry.tokenIndex, token))
          .catch((err) => this.failRun(seqId, toError(err)));
        continue;
      }

      if (entry.sentTo !== next.peerId) {
        entry.sentTo = next.peerId;
        void this.forwardTo(next, seqId, entry.tokenIndex, entry.shape, entry.hidden);
      }
    }
  }

  /**
   * Start a forward pass as the first stage.
   *
   * Resolves with the output token once the last stage produces it — whether
   * that is this node or a peer several hops away. Rejects if the sequence
   * produces nothing within the run timeout, so a dead peer surfaces as an
   * error instead of a promise that never settles.
   *
   * Several calls may be in flight at once, up to `maxConcurrent`. That is the
   * point: concurrent sequences are what keep every stage busy.
   */
  async run(
    input: Float32Array,
    shape: readonly number[],
    options: { tokenIndex?: number } = {},
  ): Promise<number> {
    if (!this.stage) throw new Error('this node has no stage assigned');
    if (this.stage.start !== 0) {
      throw new Error('only the first pipeline stage (start = 0) can initiate a forward pass');
    }

    await this.acquireSlot();

    const seqId = ++this.seqCounter;
    const tokenIndex = options.tokenIndex ?? 0;

    const settled = new Promise<number>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRuns.delete(seqId);
        this.inFlight.delete(seqId);
        this.releaseSlot();
        reject(new Error(`sequence ${seqId} produced no token within ${this.runTimeoutMs}ms`));
      }, this.runTimeoutMs);
      this.pendingRuns.set(seqId, { resolve, reject, timer });
    });

    try {
      const hidden = await this.executeStage(this.stage, input, seqId, tokenIndex);
      await this.forwardOrFinalize(seqId, tokenIndex, shape, hidden);
    } catch (err) {
      this.failRun(seqId, toError(err));
    }

    return settled;
  }

  /** Sequences this node has started and not yet resolved. */
  get inFlightCount(): number {
    return this.pendingRuns.size;
  }

  private acquireSlot(): Promise<void> {
    if (this.activeSlots < this.maxConcurrent) {
      this.activeSlots += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.admissionQueue.push(() => {
        this.activeSlots += 1;
        resolve();
      });
    });
  }

  private releaseSlot(): void {
    this.activeSlots = Math.max(0, this.activeSlots - 1);
    // Handing the slot straight to the next waiter keeps the count balanced.
    this.admissionQueue.shift()?.();
  }

  /** Settle a sequence this node started, and let a queued one through. */
  private deliverToken(seqId: number, tokenIndex: number, token: number): void {
    this.mesh.broadcastControl({ type: 'token', seqId, tokenIndex, token });
    this.callbacks.onToken?.(token, { seqId, tokenIndex, fromPeerId: this.selfPeerId });
    this.settleRun(seqId, token);
  }

  private settleRun(seqId: number, token: number): void {
    const pending = this.pendingRuns.get(seqId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingRuns.delete(seqId);
    this.releaseSlot();
    pending.resolve(token);
  }

  private failRun(seqId: number, error: Error): void {
    const pending = this.pendingRuns.get(seqId);
    if (!pending) {
      // Not our sequence — we are a middle stage, so report and move on.
      this.callbacks.onError?.(this.selfPeerId, error);
      return;
    }
    clearTimeout(pending.timer);
    this.pendingRuns.delete(seqId);
    this.inFlight.delete(seqId);
    this.releaseSlot();
    pending.reject(error);
  }

  private async handleForward(peerId: string, frame: TensorFrame): Promise<void> {
    if (frame.kind !== TensorFrameKind.Forward) return;
    if (!this.stage) return;
    if (frame.fromLayer !== this.stage.start) return; // not destined for this stage

    const input = frameAsFloat32(frame);
    const expectedLength = frame.shape.reduce((acc, dim) => acc * dim, 1);
    const check = sanitizeTensor(input, expectedLength);
    if (!check.ok) {
      // The tensor channel is reliable now, so a bad payload means corruption
      // or a version mismatch rather than a dropped chunk. Retransmitting the
      // same bytes would not help; tell the sender so the sequence can fail.
      this.callbacks.onError?.(peerId, new Error(`tensor rejected: ${check.reason}`));
      this.mesh.sendControl(peerId, {
        type: 'sequence-failed',
        seqId: frame.seqId,
        reason: check.reason,
      });
      return;
    }

    try {
      const hidden = await this.executeStage(this.stage, input, frame.seqId, frame.tokenIndex);
      await this.forwardOrFinalize(frame.seqId, frame.tokenIndex, frame.shape, hidden);
    } catch (err) {
      const error = toError(err);
      this.mesh.sendControl(peerId, {
        type: 'sequence-failed',
        seqId: frame.seqId,
        reason: error.message,
      });
      this.callbacks.onError?.(this.selfPeerId, error);
    }
  }

  private async forwardOrFinalize(
    seqId: number,
    tokenIndex: number,
    shape: readonly number[],
    hidden: Float32Array,
  ): Promise<void> {
    const next = this.nextHop();

    if (!next) {
      const token = await this.finalizeStage(this.stage!, hidden, seqId, tokenIndex);
      this.deliverToken(seqId, tokenIndex, token);
      return;
    }

    // Keep the hidden state so the sequence can be re-routed if the next hop
    // dies before it answers.
    this.inFlight.set(seqId, {
      hidden,
      shape: [...shape],
      tokenIndex,
      sentTo: next.peerId,
    });
    await this.forwardTo(next, seqId, tokenIndex, shape, hidden);
  }

  private async forwardTo(
    next: { peerId: string; stage: StageRange },
    seqId: number,
    tokenIndex: number,
    shape: readonly number[],
    hidden: Float32Array,
  ): Promise<void> {
    const meta: FrameMeta = {
      seqId,
      tokenIndex,
      fromLayer: next.stage.start,
      toLayer: next.stage.end,
    };

    const sent = await this.mesh.sendTensorTo(next.peerId, shape, hidden, meta);
    if (!sent) {
      // The old code ignored this and reported success; a full or closed
      // channel silently ate the tensor.
      this.failRun(seqId, new Error(`could not send sequence ${seqId} to ${next.peerId}`));
      return;
    }
    this.callbacks.onForwarded?.({ ...meta, shape: [...shape] });
  }

  private async executeStage(
    stage: StageRange,
    input: Float32Array,
    seqId: number,
    tokenIndex: number,
  ): Promise<Float32Array> {
    return this.executor.execute(stage, input, { seqId, tokenIndex });
  }

  private async finalizeStage(
    stage: StageRange,
    hidden: Float32Array,
    seqId: number,
    tokenIndex: number,
  ): Promise<number> {
    if (this.executor.finalize) {
      const output = await this.executor.finalize(stage, hidden, { seqId, tokenIndex });
      return output.token;
    }
    // Default: argmax over the hidden state as a stand-in for logits.
    let best = 0;
    for (let i = 1; i < hidden.length; i += 1) {
      if (hidden[i] > hidden[best]) best = i;
    }
    return best;
  }

  private nextHop(): { peerId: string; stage: StageRange } | null {
    if (!this.stage) return null;
    for (const [peerId, stage] of this.topology.entries()) {
      if (peerId === this.selfPeerId) continue;
      if (stage.start === this.stage.end && stage.end > stage.start) {
        return { peerId, stage };
      }
    }
    return null;
  }

  private handleControl(peerId: string, message: ControlMessage): void {
    switch (message.type) {
      case 'stage':
        if (message.stage) this.topology.set(peerId, message.stage);
        else this.topology.delete(peerId);
        break;
      case 'token':
        this.inFlight.delete(message.seqId);
        this.callbacks.onToken?.(message.token, {
          seqId: message.seqId,
          tokenIndex: message.tokenIndex,
          fromPeerId: peerId,
        });
        // If we started this sequence, the last stage's token is what our
        // run() promise has been waiting for.
        this.settleRun(message.seqId, message.token);
        break;
      case 'sequence-failed':
        this.failRun(
          message.seqId,
          new Error(`stage on ${peerId} failed sequence ${message.seqId}: ${message.reason}`),
        );
        break;
      default:
        break;
    }
  }

  private announceStage(peerId: string): void {
    this.mesh.sendControl(peerId, { type: 'stage', stage: this.stage });
  }

  private broadcastStage(): void {
    this.mesh.broadcastControl({ type: 'stage', stage: this.stage });
  }

  private emitEvent(event: PipelineNodeEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

function toError(err: unknown): Error {
  return err instanceof Error ? err : new Error(String(err));
}

// ---------------------------------------------------------------------------
// Dev/test hook
//
// Expose the pipeline API on the global scope in non-production builds so the
// Playwright E2E suite can drive peers programmatically from page.evaluate().
// Placed at the bottom so all class/const declarations above are initialised.
// ---------------------------------------------------------------------------

if (typeof window !== 'undefined' && import.meta.env.DEV) {
  window.p2pPipeline = {
    createSignalingAndTransport,
    defaultPeerConnectionFactory,
    resolveForceRelay,
    isMockP2PEnabled,
    PeerLink,
    PipelineMesh,
    PipelineNode,
    IdentityExecutor,
    encodeTensor,
    decodeFrame,
    frameAsFloat32,
    FrameReassembler,
    TensorDType,
    TensorFrameKind,
    DEFAULT_RTC_CONFIG,
    ICE_FALLBACK_MS,
    ManualPeerConnection,
    encodeManualPayload,
    decodeManualPayload,
    MANUAL_RTC_CONFIG,
  };
}

// Transformer internals get their own hook so the GPU kernels can be checked
// against the reference implementation in a real browser.
if (typeof window !== 'undefined' && import.meta.env.DEV) {
  void Promise.all([import('./transformer'), import('./transformer-gpu')]).then(
    ([reference, gpu]) => {
      (window as unknown as { meshTransformer?: unknown }).meshTransformer = {
        ...reference,
        GpuTransformerStage: gpu.GpuTransformerStage,
      };
    },
  );
}
