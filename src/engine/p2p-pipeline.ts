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
//   bytes 0..3    uint32  magic  ("MGPU" = 0x4d475055)
//   byte  4       uint8   version (1)
//   byte  5       uint8   frame kind
//   byte  6       uint8   dtype
//   byte  7       uint8   rank (0..255)
//   bytes 8..11   uint32  seqId
//   bytes 12..15  uint32  tokenIndex
//   bytes 16..19  uint32  fromLayer (start of the destination stage)
//   bytes 20..23  uint32  toLayer   (end of the destination stage)
//   bytes 24..27  uint32  payload byte length
//   bytes 28..31  uint32  reserved (0)
//   bytes 32..    uint32[rank] shape dims
//   ...           payload (Float32Array hidden state, little-endian)
// ---------------------------------------------------------------------------

export const TENSOR_MAGIC = 0x4d475055;
export const TENSOR_VERSION = 1;
export const TENSOR_HEADER_BYTES = 32;

export enum TensorDType {
  F32 = 1,
  F16 = 2,
  I32 = 3,
  U8 = 4,
}

export enum TensorFrameKind {
  /** Hidden-state tensor flowing forward through the pipeline. */
  Forward = 1,
  /** Reserved for future binary token frames (tokens currently use control msgs). */
  OutputToken = 2,
}

/** A contiguous, half-open layer range: [start, end). */
export interface StageRange {
  start: number;
  end: number;
}

export interface TensorFrame {
  kind: TensorFrameKind;
  dtype: TensorDType;
  shape: number[];
  seqId: number;
  tokenIndex: number;
  fromLayer: number;
  toLayer: number;
  /** Raw payload bytes (hidden state for F32 frames). */
  data: ArrayBuffer;
}

export const TensorCodec = {
  encode(frame: TensorFrame): ArrayBuffer {
    const rank = frame.shape.length;
    const header = new ArrayBuffer(TENSOR_HEADER_BYTES + rank * 4);
    const view = new DataView(header);

    view.setUint32(0, TENSOR_MAGIC, true);
    view.setUint8(4, TENSOR_VERSION);
    view.setUint8(5, frame.kind);
    view.setUint8(6, frame.dtype);
    view.setUint8(7, rank);
    view.setUint32(8, frame.seqId >>> 0, true);
    view.setUint32(12, frame.tokenIndex >>> 0, true);
    view.setUint32(16, frame.fromLayer >>> 0, true);
    view.setUint32(20, frame.toLayer >>> 0, true);
    view.setUint32(24, frame.data.byteLength >>> 0, true);
    view.setUint32(28, 0, true);

    for (let i = 0; i < rank; i += 1) {
      view.setUint32(TENSOR_HEADER_BYTES + i * 4, frame.shape[i] >>> 0, true);
    }

    const out = new ArrayBuffer(header.byteLength + frame.data.byteLength);
    new Uint8Array(out).set(new Uint8Array(header), 0);
    new Uint8Array(out).set(new Uint8Array(frame.data), header.byteLength);
    return out;
  },

  decode(buffer: ArrayBuffer): TensorFrame {
    if (buffer.byteLength < TENSOR_HEADER_BYTES) {
      throw new Error('tensor frame too small');
    }
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== TENSOR_MAGIC) {
      throw new Error('bad tensor frame magic');
    }
    const version = view.getUint8(4);
    if (version !== TENSOR_VERSION) {
      throw new Error(`unsupported tensor frame version ${version}`);
    }

    const kind = view.getUint8(5) as TensorFrameKind;
    const dtype = view.getUint8(6) as TensorDType;
    const rank = view.getUint8(7);
    const seqId = view.getUint32(8, true);
    const tokenIndex = view.getUint32(12, true);
    const fromLayer = view.getUint32(16, true);
    const toLayer = view.getUint32(20, true);
    const payloadBytes = view.getUint32(24, true);

    const shape: number[] = [];
    for (let i = 0; i < rank; i += 1) {
      shape.push(view.getUint32(TENSOR_HEADER_BYTES + i * 4, true));
    }

    const offset = TENSOR_HEADER_BYTES + rank * 4;
    if (offset + payloadBytes !== buffer.byteLength) {
      throw new Error('tensor payload length mismatch');
    }

    return {
      kind,
      dtype,
      shape,
      seqId,
      tokenIndex,
      fromLayer,
      toLayer,
      data: buffer.slice(offset, offset + payloadBytes),
    };
  },
};

/** Build a forward hidden-state frame from a Float32Array. */
export function makeForwardFrame(
  shape: readonly number[],
  data: Float32Array,
  meta: { seqId: number; tokenIndex: number; fromLayer: number; toLayer: number },
): TensorFrame {
  const payload = new ArrayBuffer(data.byteLength);
  new Float32Array(payload).set(data);
  return {
    kind: TensorFrameKind.Forward,
    dtype: TensorDType.F32,
    shape: [...shape],
    seqId: meta.seqId,
    tokenIndex: meta.tokenIndex,
    fromLayer: meta.fromLayer,
    toLayer: meta.toLayer,
    data: payload,
  };
}

/** Access a forward frame's payload as F32, validating its dtype. */
export function frameAsFloat32(frame: TensorFrame): Float32Array {
  if (frame.dtype !== TensorDType.F32) {
    throw new Error(`expected F32 tensor, got dtype ${frame.dtype}`);
  }
  return new Float32Array(frame.data);
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
  | { type: 'tensor-retransmit'; seqId: number; tokenIndex: number };

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
// Unordered with no retransmits = UDP-like, lowest-latency tensor path.
const TENSOR_CHANNEL: ChannelSpec = { id: 1, label: 'meshgpu-tensor', ordered: false };

export const DEFAULT_RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

/** Fall back to relay routing if direct P2P hasn't connected within this window. */
export const ICE_FALLBACK_MS = 3000;

/** Drop tensor sends once the SCTP send buffer exceeds this many bytes. */
const MAX_BUFFERED_BYTES = 16 * 1024 * 1024;

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
  private fallbackTimer: number | null = null;
  private pendingCandidates: RTCIceCandidateInit[] = [];
  private readonly pendingLatency = new Map<number, (rttMs: number) => void>();

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

  /** Send a packed tensor frame. Returns false on backpressure/drop. */
  sendTensor(frame: TensorFrame): boolean {
    if (this.tensorChannel.readyState !== 'open' || this.closed) return false;
    if (this.tensorChannel.bufferedAmount > MAX_BUFFERED_BYTES) return false;
    this.tensorChannel.send(TensorCodec.encode(frame));
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
    this.callbacks.onClose?.(this);
  }

  private createChannel(channel: ChannelSpec): DataChannelLike {
    const init: RTCDataChannelInit = {
      negotiated: true,
      id: channel.id,
      ordered: channel.ordered,
    };
    if (!channel.ordered) init.maxRetransmits = 0;
    return this.pc.createDataChannel(channel.label, init);
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
      let frame: TensorFrame;
      try {
        frame = TensorCodec.decode(event.data as ArrayBuffer);
      } catch (err) {
        this.callbacks.onError?.(this, toError(err));
        return;
      }
      this.callbacks.onTensor?.(this, frame);
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
  private heartbeatTimer: number | null = null;

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

  sendTensorTo(peerId: string, frame: TensorFrame): boolean {
    return this.links.get(peerId)?.sendTensor(frame) ?? false;
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
  onForwarded?: (frame: TensorFrame) => void;
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

const FRAME_BUFFER_LIMIT = 64;

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
  private readonly sentFrames = new Map<number, TensorFrame>();
  private readonly inFlight = new Map<number, InFlightEntry>();
  private stage: StageRange | null = null;
  private seqCounter = 0;

  constructor(options: PipelineNodeOptions) {
    this.selfPeerId = options.signaling?.currentPeerId ?? options.peerId ?? generatePeerId();
    this.executor = options.executor ?? new IdentityExecutor();
    this.callbacks = options.callbacks ?? {};

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
        void this.finalizeStage(this.stage, entry.hidden, seqId, entry.tokenIndex)
          .then((token) => {
            this.mesh.broadcastControl({
              type: 'token',
              seqId,
              tokenIndex: entry.tokenIndex,
              token,
            });
            this.callbacks.onToken?.(token, {
              seqId,
              tokenIndex: entry.tokenIndex,
              fromPeerId: this.selfPeerId,
            });
          })
          .catch((err) => this.callbacks.onError?.(this.selfPeerId, toError(err)));
        this.inFlight.delete(seqId);
        continue;
      }

      if (entry.sentTo !== next.peerId) {
        const frame = makeForwardFrame(entry.shape, entry.hidden, {
          seqId,
          tokenIndex: entry.tokenIndex,
          fromLayer: next.stage.start,
          toLayer: next.stage.end,
        });
        this.mesh.sendTensorTo(next.peerId, frame);
        this.sentFrames.set(seqId, frame);
        entry.sentTo = next.peerId;
        this.callbacks.onForwarded?.(frame);
      }
    }
  }

  /**
   * Start a forward pass as the first stage. Returns the output token if this
   * node is also the last stage, otherwise null (the token is produced and
   * broadcast by the final stage).
   */
  async run(
    input: Float32Array,
    shape: readonly number[],
    options: { tokenIndex?: number } = {},
  ): Promise<number | null> {
    if (!this.stage) throw new Error('this node has no stage assigned');
    if (this.stage.start !== 0) {
      throw new Error('only the first pipeline stage (start = 0) can initiate a forward pass');
    }

    const seqId = ++this.seqCounter;
    const tokenIndex = options.tokenIndex ?? 0;
    const hidden = await this.executeStage(this.stage, input, seqId, tokenIndex);
    return this.forwardOrFinalize(seqId, tokenIndex, shape, hidden);
  }

  private async handleForward(peerId: string, frame: TensorFrame): Promise<void> {
    if (frame.kind !== TensorFrameKind.Forward) return;
    if (!this.stage) return;
    if (frame.fromLayer !== this.stage.start) return; // not destined for this stage

    const input = frameAsFloat32(frame);
    const expectedLength = frame.shape.reduce((acc, dim) => acc * dim, 1);
    const check = sanitizeTensor(input, expectedLength);
    if (!check.ok) {
      this.callbacks.onError?.(peerId, new Error(`tensor rejected: ${check.reason}`));
      this.mesh.sendControl(peerId, {
        type: 'tensor-retransmit',
        seqId: frame.seqId,
        tokenIndex: frame.tokenIndex,
      });
      return;
    }

    const hidden = await this.executeStage(this.stage, input, frame.seqId, frame.tokenIndex);
    await this.forwardOrFinalize(frame.seqId, frame.tokenIndex, frame.shape, hidden);
  }

  private async forwardOrFinalize(
    seqId: number,
    tokenIndex: number,
    shape: readonly number[],
    hidden: Float32Array,
  ): Promise<number | null> {
    const next = this.nextHop();

    if (!next) {
      const token = await this.finalizeStage(this.stage!, hidden, seqId, tokenIndex);
      this.mesh.broadcastControl({ type: 'token', seqId, tokenIndex, token });
      this.callbacks.onToken?.(token, {
        seqId,
        tokenIndex,
        fromPeerId: this.selfPeerId,
      });
      return token;
    }

    const frame = makeForwardFrame(shape, hidden, {
      seqId,
      tokenIndex,
      fromLayer: next.stage.start,
      toLayer: next.stage.end,
    });
    this.mesh.sendTensorTo(next.peerId, frame);
    this.rememberFrame(seqId, frame);
    this.inFlight.set(seqId, {
      hidden,
      shape: [...shape],
      tokenIndex,
      sentTo: next.peerId,
    });
    this.callbacks.onForwarded?.(frame);
    return null;
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
        this.sentFrames.delete(message.seqId);
        this.inFlight.delete(message.seqId);
        this.callbacks.onToken?.(message.token, {
          seqId: message.seqId,
          tokenIndex: message.tokenIndex,
          fromPeerId: peerId,
        });
        break;
      case 'tensor-retransmit': {
        const frame = this.sentFrames.get(message.seqId);
        if (frame) this.mesh.sendTensorTo(peerId, frame);
        break;
      }
      default:
        break;
    }
  }

  private rememberFrame(seqId: number, frame: TensorFrame): void {
    this.sentFrames.set(seqId, frame);
    if (this.sentFrames.size > FRAME_BUFFER_LIMIT) {
      const oldest = this.sentFrames.keys().next().value;
      if (oldest !== undefined) this.sentFrames.delete(oldest);
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
    TensorCodec,
    makeForwardFrame,
    frameAsFloat32,
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
