/**
 * manual-signaling.ts
 *
 * Option 3 — Manual WebRTC SDP exchange. Zero servers, zero WebSockets: the
 * two peers hand a single compressed, Base64-encoded payload (the complete SDP
 * + every locally-gathered ICE candidate) to each other via QR codes or
 * copy-paste.
 *
 *   Host:   createOfferPayload()                     → offer Base64
 *   Joiner: acceptOfferAndCreateAnswerPayload(offer)  → answer Base64
 *   Host:   acceptAnswerPayload(answer)               → direct P2P link up
 *
 * The payload is a `{ version, kind, peerId, sdp, candidates }` JSON object
 * compressed with `lz-string` and exported as Base64, so a full LAN handshake
 * fits comfortably inside a single QR code.
 */

import { compressToBase64, decompressFromBase64 } from 'lz-string';

/** Bump when the payload schema changes (never parse an unknown version). */
export const MANUAL_PAYLOAD_VERSION = 1;

/**
 * Air-gapped ICE configuration: host candidates only. No STUN, no TURN — the
 * DataChannel runs directly over the local Wi-Fi / Ethernet subnet.
 */
export const MANUAL_RTC_CONFIG: RTCConfiguration = { iceServers: [] };

/**
 * How long to wait for `iceGatheringState === 'complete'` before falling back
 * to whatever local (host) candidates have been gathered so far.
 */
export const ICE_GATHER_TIMEOUT_MS = 2000;

/** DataChannel ids shared by both peers (must match the pipeline's channel spec). */
export const MANUAL_CONTROL_CHANNEL_ID = 0;
export const MANUAL_TENSOR_CHANNEL_ID = 1;

export interface ManualSignalPayload {
  version: number;
  kind: 'offer' | 'answer';
  /** The peer id of the side that produced this payload. */
  peerId: string;
  sdp: string;
  candidates: RTCIceCandidateInit[];
}

// ---------------------------------------------------------------------------
// Codec: JSON payload <-> compressed Base64
// ---------------------------------------------------------------------------

export function encodeManualPayload(payload: ManualSignalPayload): string {
  return compressToBase64(JSON.stringify(payload));
}

export function decodeManualPayload(encoded: string): ManualSignalPayload {
  const json = decompressFromBase64(encoded.trim());
  if (!json) {
    throw new Error('handshake payload failed to decompress — not a valid lz-string Base64 blob');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new Error('handshake payload is not valid JSON');
  }
  if (!isManualSignalPayload(parsed)) {
    throw new Error('handshake payload is malformed (unexpected shape)');
  }
  if (parsed.version !== MANUAL_PAYLOAD_VERSION) {
    throw new Error(`unsupported handshake payload version ${parsed.version}`);
  }
  return parsed;
}

function isManualSignalPayload(value: unknown): value is ManualSignalPayload {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    record.version === MANUAL_PAYLOAD_VERSION &&
    (record.kind === 'offer' || record.kind === 'answer') &&
    typeof record.peerId === 'string' &&
    typeof record.sdp === 'string' &&
    Array.isArray(record.candidates)
  );
}

// ---------------------------------------------------------------------------
// ICE gathering helper
// ---------------------------------------------------------------------------

/**
 * Resolve once ICE gathering is complete, or after `timeoutMs` — whichever
 * comes first — so callers can bundle a complete (or best-effort) SDP payload.
 */
export function waitForIceGathering(
  pc: RTCPeerConnection,
  timeoutMs = ICE_GATHER_TIMEOUT_MS,
): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();

  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      pc.removeEventListener('icegatheringstatechange', onStateChange);
      clearTimeout(timer);
      resolve();
    };
    const onStateChange = (): void => {
      if (pc.iceGatheringState === 'complete') finish();
    };
    const timer = setTimeout(finish, timeoutMs);
    pc.addEventListener('icegatheringstatechange', onStateChange);
  });
}

/**
 * Best-effort detection of this machine's local IPv4 addresses by running a
 * short-lived, STUN-free RTCPeerConnection and parsing its host candidates.
 * Used purely for the UI "local IP pairing" readout.
 */
export function detectLocalIPs(timeoutMs = 1500): Promise<string[]> {
  return new Promise<string[]>((resolve) => {
    const ips = new Set<string>();
    let settled = false;
    let pc: RTCPeerConnection | null = null;

    const finish = (): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        pc?.close();
      } catch {
        // Ignore teardown errors.
      }
      resolve([...ips]);
    };

    const timer = setTimeout(finish, timeoutMs);

    try {
      pc = new RTCPeerConnection(MANUAL_RTC_CONFIG);
    } catch {
      finish();
      return;
    }

    pc.onicecandidate = (event) => {
      if (!event.candidate) {
        finish();
        return;
      }
      const ip = extractIPv4(event.candidate.candidate);
      if (ip) ips.add(ip);
    };

    // A DataChannel forces host-candidate gathering without needing a remote peer.
    try {
      pc.createDataChannel('mesh-ip-probe');
    } catch {
      finish();
      return;
    }
    pc.createOffer()
      .then((offer) => pc?.setLocalDescription(offer))
      .catch(() => finish());
  });
}

function extractIPv4(candidateLine: string): string | null {
  const match = /(?:[0-9]{1,3}\.){3}[0-9]{1,3}/.exec(candidateLine);
  return match ? match[0] : null;
}

// ---------------------------------------------------------------------------
// Manual peer connection: one RTCPeerConnection + two negotiated DataChannels
// ---------------------------------------------------------------------------

export interface ManualPeerConnectionOptions {
  peerId: string;
  rtcConfig?: RTCConfiguration;
}

export class ManualPeerConnection {
  readonly peerId: string;
  readonly pc: RTCPeerConnection;
  readonly controlChannel: RTCDataChannel;
  readonly tensorChannel: RTCDataChannel;

  private readonly candidates: RTCIceCandidateInit[] = [];
  private closed = false;

  constructor(options: ManualPeerConnectionOptions) {
    this.peerId = options.peerId;
    this.pc = new RTCPeerConnection(options.rtcConfig ?? MANUAL_RTC_CONFIG);

    if (typeof window !== 'undefined') {
      window.peerConnection = this.pc;
    }

    // Negotiated DataChannels: both peers agree on the ids up front, so the
    // channels surface without any renegotiation after the SDP exchange.
    this.controlChannel = this.pc.createDataChannel('meshgpu-control', {
      negotiated: true,
      id: MANUAL_CONTROL_CHANNEL_ID,
      ordered: true,
    });
    this.tensorChannel = this.pc.createDataChannel('meshgpu-tensor', {
      negotiated: true,
      id: MANUAL_TENSOR_CHANNEL_ID,
      ordered: false,
      maxRetransmits: 0,
    });

    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.candidates.push(event.candidate.toJSON());
      }
    };
  }

  /** Host: gather the offer and export it as a compressed Base64 payload. */
  async createOfferPayload(): Promise<string> {
    this.assertOpen();
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    await waitForIceGathering(this.pc);
    return this.exportPayload('offer');
  }

  /** Joiner: accept a host offer, gather the answer, export it as Base64. */
  async acceptOfferAndCreateAnswerPayload(offerBase64: string): Promise<string> {
    this.assertOpen();
    const offer = decodeManualPayload(offerBase64);
    if (offer.kind !== 'offer') {
      throw new Error('expected an offer payload, received an answer');
    }
    await this.pc.setRemoteDescription({ type: 'offer', sdp: offer.sdp });
    await this.addCandidates(offer.candidates);

    const answer = await this.pc.createAnswer();
    await this.pc.setLocalDescription(answer);
    await waitForIceGathering(this.pc);
    return this.exportPayload('answer');
  }

  /** Host: apply the joiner's answer to establish the direct P2P link. */
  async acceptAnswerPayload(answerBase64: string): Promise<void> {
    this.assertOpen();
    const answer = decodeManualPayload(answerBase64);
    if (answer.kind !== 'answer') {
      throw new Error('expected an answer payload, received an offer');
    }
    await this.pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
    await this.addCandidates(answer.candidates);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.controlChannel.close();
      this.tensorChannel.close();
      this.pc.close();
    } catch {
      // Best-effort teardown.
    }
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('manual peer connection is closed');
  }

  private exportPayload(kind: ManualSignalPayload['kind']): string {
    return encodeManualPayload({
      version: MANUAL_PAYLOAD_VERSION,
      kind,
      peerId: this.peerId,
      sdp: this.pc.localDescription?.sdp ?? '',
      candidates: [...this.candidates],
    });
  }

  private async addCandidates(candidates: RTCIceCandidateInit[]): Promise<void> {
    for (const candidate of candidates) {
      if (!candidate.candidate) continue;
      // Duplicates are harmless — the browser dedupes them silently.
      await this.pc.addIceCandidate(candidate).catch(() => undefined);
    }
  }
}
