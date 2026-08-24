/**
 * mock-transport.ts
 *
 * In-memory WebRTC transport for headless testing. Mimics the
 * RTCPeerConnection + RTCDataChannel surface used by `p2p-pipeline.ts` so the
 * full tensor pipeline, layer sharding and tensor codecs can be exercised
 * without real ICE candidate gathering. Multiple mock peers chain together in
 * a single browser window (or Node process) via a module-level signaling bus
 * keyed by roomId.
 *
 * Enable with `?mockP2P=true` in the URL, or `VITE_USE_MOCK_P2P=true` in the
 * Vite env. Simulated loopback latency defaults to 5 ms and can be overridden
 * with `VITE_MOCK_P2P_DELAY_MS`.
 */

import type {
  SignalPayload,
  SignalingClientLike,
  SignalingEventMap,
  SignalingEventType,
  SignalingWelcome,
} from './signaling';

// ---------------------------------------------------------------------------
// Transport surface shared with p2p-pipeline.ts
// ---------------------------------------------------------------------------

export interface DataChannelLike {
  readonly label: string;
  readonly readyState: string;
  readonly bufferedAmount: number;
  send(data: string | ArrayBuffer): void;
  close(): void;
  onopen: (() => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null;
}

export interface PeerConnectionLike {
  onicecandidate: ((event: { candidate: { toJSON(): object } | null }) => void) | null;
  oniceconnectionstatechange: (() => void) | null;
  onconnectionstatechange: (() => void) | null;
  readonly iceConnectionState: string;
  readonly connectionState: string;
  readonly localDescription: RTCSessionDescriptionInit | null;
  readonly remoteDescription: RTCSessionDescriptionInit | null;
  createDataChannel(label: string, init: RTCDataChannelInit): DataChannelLike;
  createOffer(): Promise<RTCSessionDescriptionInit>;
  createAnswer(): Promise<RTCSessionDescriptionInit>;
  setLocalDescription(description: RTCSessionDescriptionInit): Promise<void>;
  setRemoteDescription(description: RTCSessionDescriptionInit): Promise<void>;
  addIceCandidate(candidate: object): Promise<void>;
  getConfiguration(): { iceTransportPolicy: 'all' | 'relay' };
  close(): void;
}

export type PeerConnectionFactory = (
  remotePeerId: string,
  config: RTCConfiguration,
) => PeerConnectionLike;

// ---------------------------------------------------------------------------
// Feature flag
// ---------------------------------------------------------------------------

const DEFAULT_DELAY_MS = 5;

function viteEnv(): Record<string, string> | undefined {
  try {
    return (import.meta as unknown as { env?: Record<string, string> }).env;
  } catch {
    return undefined;
  }
}

/** Whether to route the pipeline through the in-memory mock transport. */
export function isMockP2PEnabled(): boolean {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    if (params.get('mockP2P') === 'true') return true;
  }
  const env = viteEnv();
  if (env?.VITE_USE_MOCK_P2P === 'true') return true;
  return false;
}

function readDelayMs(): number {
  const env = viteEnv();
  if (env?.VITE_MOCK_P2P_DELAY_MS) {
    const value = Number(env.VITE_MOCK_P2P_DELAY_MS);
    if (Number.isFinite(value) && value >= 0) return value;
  }
  return DEFAULT_DELAY_MS;
}

// ---------------------------------------------------------------------------
// Mock DataChannel
// ---------------------------------------------------------------------------

class MockDataChannel implements DataChannelLike {
  readonly label: string;
  readonly id: number;
  readyState = 'connecting';
  bufferedAmount = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null;

  private remote: MockDataChannel | null = null;
  private readonly delayMs: number;

  constructor(label: string, id: number, delayMs: number) {
    this.label = label;
    this.id = id;
    this.delayMs = delayMs;
  }

  send(data: string | ArrayBuffer): void {
    if (this.readyState !== 'open' || !this.remote) return;
    if (this.delayMs <= 0) this.remote.deliver(data);
    else setTimeout(() => this.remote?.deliver(data), this.delayMs);
  }

  close(): void {
    if (this.readyState === 'closed') return;
    this.readyState = 'closed';
    this.onclose?.();
  }

  markOpen(): void {
    if (this.readyState === 'open') return;
    this.readyState = 'open';
    this.onopen?.();
  }

  setRemote(remote: MockDataChannel): void {
    this.remote = remote;
  }

  private deliver(data: string | ArrayBuffer): void {
    if (this.readyState === 'open') this.onmessage?.({ data });
  }
}

// ---------------------------------------------------------------------------
// Mock RTCPeerConnection
// ---------------------------------------------------------------------------

class MockPeerConnection implements PeerConnectionLike {
  onicecandidate: ((event: { candidate: { toJSON(): object } | null }) => void) | null = null;
  oniceconnectionstatechange: (() => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;

  localDescription: RTCSessionDescriptionInit | null = null;
  remoteDescription: RTCSessionDescriptionInit | null = null;
  iceConnectionState = 'new';
  connectionState = 'new';

  readonly channels = new Map<number, MockDataChannel>();
  private closed = false;

  constructor(
    private readonly localPeerId: string,
    private readonly remotePeerId: string,
    private readonly roomId: string,
    private readonly delayMs: number,
    private readonly bus: typeof MockSignalingBus,
  ) {}

  createDataChannel(label: string, init: RTCDataChannelInit): DataChannelLike {
    const channel = new MockDataChannel(label, init.id ?? 0, this.delayMs);
    this.channels.set(channel.id, channel);
    return channel;
  }

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    return { type: 'offer', sdp: '' };
  }

  async createAnswer(): Promise<RTCSessionDescriptionInit> {
    return { type: 'answer', sdp: '' };
  }

  async setLocalDescription(description: RTCSessionDescriptionInit): Promise<void> {
    this.localDescription = description;
  }

  async setRemoteDescription(description: RTCSessionDescriptionInit): Promise<void> {
    this.remoteDescription = description;
    // Wire this link's channels to the remote peer's channels and open them.
    this.bus.wire(this.roomId, this.localPeerId, this.remotePeerId);
    this.setIceState('connected');
    this.setConnectionState('connected');
  }

  async addIceCandidate(_candidate: object): Promise<void> {
    // In-memory transport: ICE candidates are unnecessary.
  }

  getConfiguration(): { iceTransportPolicy: 'all' | 'relay' } {
    return { iceTransportPolicy: 'all' };
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const channel of this.channels.values()) channel.close();
    this.setConnectionState('closed');
  }

  private setIceState(state: string): void {
    if (this.iceConnectionState === state) return;
    this.iceConnectionState = state;
    this.oniceconnectionstatechange?.();
  }

  private setConnectionState(state: string): void {
    if (this.connectionState === state) return;
    this.connectionState = state;
    this.onconnectionstatechange?.();
  }
}

// ---------------------------------------------------------------------------
// Mock signaling bus
// ---------------------------------------------------------------------------

class MockSignalingBus {
  private static readonly rooms = new Map<string, Map<string, MockSignaling>>();

  static join(signaling: MockSignaling): string[] {
    let room = this.rooms.get(signaling.roomId);
    if (!room) {
      room = new Map();
      this.rooms.set(signaling.roomId, room);
    }
    const existing = [...room.keys()];
    room.set(signaling.peerId, signaling);
    return existing;
  }

  static leave(signaling: MockSignaling): void {
    const room = this.rooms.get(signaling.roomId);
    room?.delete(signaling.peerId);
    if (room && room.size === 0) this.rooms.delete(signaling.roomId);
  }

  static peer(roomId: string, peerId: string): MockSignaling | undefined {
    return this.rooms.get(roomId)?.get(peerId);
  }

  static relay(from: MockSignaling, toPeerId: string, signal: SignalPayload): void {
    const target = this.peer(from.roomId, toPeerId);
    if (target) target.deliverSignal(from.peerId, signal);
  }

  static wire(roomId: string, localPeerId: string, remotePeerId: string): void {
    const local = this.peer(roomId, localPeerId);
    const remote = this.peer(roomId, remotePeerId);
    if (!local || !remote) return;

    const localPc = local.links.get(remotePeerId);
    const remotePc = remote.links.get(localPeerId);
    if (!localPc || !remotePc) return;

    for (const [id, channel] of localPc.channels) {
      const remoteChannel = remotePc.channels.get(id);
      if (!remoteChannel) continue;
      channel.setRemote(remoteChannel);
      remoteChannel.setRemote(channel);
      channel.markOpen();
      remoteChannel.markOpen();
    }
  }
}

// ---------------------------------------------------------------------------
// Mock signaling (drop-in implementation of SignalingClientLike)
// ---------------------------------------------------------------------------

export interface MockSignalingOptions {
  roomId: string;
  peerId: string;
  delayMs?: number;
}

export class MockSignaling implements SignalingClientLike {
  readonly roomId: string;
  readonly peerId: string;
  readonly links = new Map<string, MockPeerConnection>();

  private peers: string[] = [];
  private readonly delayMs: number;
  private readonly listeners = new Map<SignalingEventType, Set<(payload: unknown) => void>>();

  constructor(options: MockSignalingOptions) {
    this.roomId = options.roomId;
    this.peerId = options.peerId;
    this.delayMs = options.delayMs ?? readDelayMs();
  }

  get currentPeerId(): string {
    return this.peerId;
  }

  get currentIceServers(): readonly RTCIceServer[] {
    return [];
  }

  get knownPeers(): readonly string[] {
    return this.peers;
  }

  on<E extends SignalingEventType>(
    event: E,
    handler: (payload: SignalingEventMap[E]) => void,
  ): () => void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    const listener = handler as unknown as (payload: unknown) => void;
    set.add(listener);
    return () => {
      set?.delete(listener);
    };
  }

  async connect(): Promise<SignalingWelcome> {
    const existing = MockSignalingBus.join(this);
    this.peers = existing;
    for (const other of existing) {
      MockSignalingBus.peer(this.roomId, other)?.emit('peer-joined', {
        roomId: this.roomId,
        peerId: this.peerId,
      });
    }
    return { roomId: this.roomId, peerId: this.peerId, peers: existing, iceServers: [] };
  }

  sendSignal(targetPeerId: string, signal: SignalPayload): void {
    MockSignalingBus.relay(this, targetPeerId, signal);
  }

  disconnect(): void {
    MockSignalingBus.leave(this);
    for (const other of this.peers) {
      MockSignalingBus.peer(this.roomId, other)?.emit('peer-left', {
        roomId: this.roomId,
        peerId: this.peerId,
      });
    }
    this.peers = [];
    this.emit('close', undefined);
  }

  /** Create + register the in-memory PC for a link to `remotePeerId`. */
  createPeerConnection(remotePeerId: string): MockPeerConnection {
    const pc = new MockPeerConnection(
      this.peerId,
      remotePeerId,
      this.roomId,
      this.delayMs,
      MockSignalingBus,
    );
    this.links.set(remotePeerId, pc);
    return pc;
  }

  /** Deliver a relayed signal to this peer (called by the bus). */
  deliverSignal(fromPeerId: string, signal: SignalPayload): void {
    this.emit('signal', { fromPeerId, signal });
  }

  private emit<E extends SignalingEventType>(event: E, payload: SignalingEventMap[E]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const listener of set) {
      (listener as (payload: SignalingEventMap[E]) => void)(payload);
    }
  }
}
