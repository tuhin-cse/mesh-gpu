/**
 * signaling.ts
 *
 * Transport-agnostic signaling contracts for the MeshGPU pipeline.
 *
 * The WebSocket signaling server has been removed: handshakes now happen via
 * manual SDP exchange (`manual-signaling.ts`). This module retains only the
 * structural interfaces shared with the in-memory mock transport used by
 * headless tests and the mock mesh demo.
 */

/** A relayed WebRTC handshake payload: an SDP description or an ICE candidate. */
export interface SignalPayload {
  description?: RTCSessionDescriptionInit;
  candidate?: RTCIceCandidateInit;
  /** Ask the peer to restart ICE in relay-only mode (strict-NAT fallback). */
  relayRestart?: boolean;
}

export interface SignalingWelcome {
  roomId: string;
  peerId: string;
  peers: string[];
  iceServers: RTCIceServer[];
}

export interface SignalingEventMap {
  open: void;
  welcome: SignalingWelcome;
  'peer-joined': { roomId: string; peerId: string };
  'peer-left': { roomId: string; peerId: string };
  signal: { fromPeerId: string; signal: SignalPayload };
  error: { message: string };
  close: void;
}

export type SignalingEventType = keyof SignalingEventMap;

/** Structural contract shared by the in-memory mock signaling bus. */
export interface SignalingClientLike {
  readonly currentPeerId: string;
  readonly currentIceServers: readonly RTCIceServer[];
  connect(): Promise<SignalingWelcome>;
  sendSignal(targetPeerId: string, signal: SignalPayload): void;
  disconnect(): void;
  on<E extends SignalingEventType>(
    event: E,
    handler: (payload: SignalingEventMap[E]) => void,
  ): () => void;
}
