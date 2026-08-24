/**
 * QRHandshakeModal.tsx
 *
 * Primary peer-connection workflow for air-gapped MeshGPU: the host renders a
 * room-offer QR code (or copies the Base64 text), the joiner scans/pastes it
 * and returns a guest-answer QR code. No server is involved — the WebRTC
 * offer/answer payloads travel by camera or clipboard alone.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { QRCodeSVG } from 'qrcode.react';
import { Html5Qrcode } from 'html5-qrcode';

import { detectLocalIPs } from '../engine/manual-signaling';

const SCANNER_ID = 'meshgpu-qr-scanner';

export interface QRHandshakeModalProps {
  open: boolean;
  onClose: () => void;
  /** True once the DataChannels are open between host and joiner. */
  connected: boolean;
  /** This node's currently assigned WebGPU layer range (scheduler). */
  stage: { start: number; end: number } | null;
  /** Live round-trip latency to the remote peer (null = not measured yet). */
  latencyMs: number | null;
  /** Host: create the node and return the compressed offer payload. */
  onHostStart: () => Promise<string>;
  /** Host: apply the joiner's answer payload. */
  onHostApplyAnswer: (answerBase64: string) => Promise<void>;
  /** Joiner: accept the host offer and return the answer payload. */
  onJoinerStart: (offerBase64: string) => Promise<string>;
  /** Tear down the connection and return to a clean slate. */
  onReset: () => void;
}

export function QRHandshakeModal(props: QRHandshakeModalProps) {
  const { open, onClose, connected, stage, latencyMs, onHostStart, onHostApplyAnswer, onJoinerStart, onReset } = props;

  const [tab, setTab] = useState<'host' | 'joiner'>('host');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offer, setOffer] = useState<string | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [offerInput, setOfferInput] = useState('');
  const [answerInput, setAnswerInput] = useState('');
  const [copied, setCopied] = useState<'offer' | 'answer' | null>(null);
  const [scannerOn, setScannerOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [localIps, setLocalIps] = useState<string[]>([]);
  const appliedRef = useRef(false);

  // Detect local IPs (from host ICE candidates) whenever the modal opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    detectLocalIPs().then((ips) => {
      if (!cancelled) setLocalIps(ips);
    });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const generateOffer = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setOffer(await onHostStart());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [onHostStart]);

  const applyAnswer = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setBusy(true);
      setError(null);
      try {
        await onHostApplyAnswer(trimmed);
        setAnswerInput(trimmed);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [onHostApplyAnswer],
  );

  const acceptOffer = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setBusy(true);
      setError(null);
      try {
        setAnswer(await onJoinerStart(trimmed));
        setOfferInput(trimmed);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [onJoinerStart],
  );

  // Route a decoded QR to the active tab's flow.
  const handleScanned = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || appliedRef.current) return;
      appliedRef.current = true;
      setScannerOn(false);
      if (tab === 'host') {
        void applyAnswer(trimmed);
      } else {
        void acceptOffer(trimmed);
      }
    },
    [tab, applyAnswer, acceptOffer],
  );

  // Camera scanner lifecycle.
  useEffect(() => {
    if (!open || !scannerOn) return;
    let cancelled = false;
    setCameraError(null);
    appliedRef.current = false;

    const scanner = new Html5Qrcode(SCANNER_ID);
    scanner
      .start(
        { facingMode: 'environment' },
        { fps: 10, qrbox: { width: 220, height: 220 } },
        (decodedText) => {
          if (!cancelled) handleScanned(decodedText);
        },
        () => {
          // Per-frame miss — ignore.
        },
      )
      .catch((err: unknown) => {
        if (cancelled) return;
        setCameraError(err instanceof Error ? err.message : 'Camera unavailable or permission denied');
        setScannerOn(false);
      });

    return () => {
      cancelled = true;
      void scanner.stop().catch(() => undefined);
      scanner.clear();
    };
  }, [open, scannerOn, handleScanned]);

  const copy = useCallback(async (text: string, which: 'offer' | 'answer') => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      window.setTimeout(() => {
        setCopied((prev) => (prev === which ? null : prev));
      }, 2000);
    } catch {
      setError('Clipboard unavailable — select and copy the text manually.');
    }
  }, []);

  const switchTab = (next: 'host' | 'joiner') => {
    setTab(next);
    setError(null);
    setCameraError(null);
    setScannerOn(false);
    setOffer(null);
    setAnswer(null);
    setOfferInput('');
    setAnswerInput('');
    setCopied(null);
    appliedRef.current = false;
  };

  const handleReset = () => {
    onReset();
    setOffer(null);
    setAnswer(null);
    setOfferInput('');
    setAnswerInput('');
    setError(null);
    setScannerOn(false);
    appliedRef.current = false;
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-zinc-700 bg-zinc-900 p-6 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">Manual QR Handshake</h2>
            <p className="mt-1 text-xs text-zinc-500">
              100% offline — offer/answer SDP payloads travel via QR or clipboard only.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg border border-zinc-700 px-2 py-1 text-sm text-zinc-400 transition hover:bg-zinc-800"
          >
            ✕
          </button>
        </div>

        {/* Live connection status */}
        <div className="mt-4 grid gap-2 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-xs sm:grid-cols-4">
          <StatusCell label="Link" value={connected ? 'Connected' : busy ? 'Negotiating…' : 'Idle'} />
          <StatusCell
            label="Local IP"
            value={localIps.length > 0 ? localIps.join(', ') : '—'}
          />
          <StatusCell
            label="Latency"
            value={latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : '—'}
          />
          <StatusCell
            label="Layer stage"
            value={stage ? `[${stage.start}, ${stage.end})` : '—'}
          />
        </div>

        {/* Role tabs */}
        <div className="mt-4 flex flex-wrap gap-2">
          <TabButton active={tab === 'host'} onClick={() => switchTab('host')} label="Host (room owner)" />
          <TabButton active={tab === 'joiner'} onClick={() => switchTab('joiner')} label="Joiner (guest)" />
          {connected ? (
            <button
              onClick={handleReset}
              className="ml-auto rounded-lg border border-rose-800 px-3 py-1.5 text-xs text-rose-300 transition hover:bg-rose-950"
            >
              Disconnect
            </button>
          ) : null}
        </div>

        {tab === 'host' ? (
          <HostTab
            busy={busy}
            offer={offer}
            answerInput={answerInput}
            copied={copied}
            onGenerate={generateOffer}
            onAnswerInput={setAnswerInput}
            onApplyAnswer={() => void applyAnswer(answerInput)}
            onCopyOffer={() => offer && void copy(offer, 'offer')}
          />
        ) : (
          <JoinerTab
            busy={busy}
            offerInput={offerInput}
            answer={answer}
            copied={copied}
            onOfferInput={setOfferInput}
            onAcceptOffer={() => void acceptOffer(offerInput)}
            onCopyAnswer={() => answer && void copy(answer, 'answer')}
          />
        )}

        {/* Shared camera scanner (rendered so html5-qrcode can mount it). */}
        <div className="mt-4">
          <button
            onClick={() => setScannerOn((prev) => !prev)}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 transition hover:bg-zinc-800"
          >
            {scannerOn ? 'Stop scanning' : 'Scan with camera'}
          </button>
          {cameraError ? (
            <p className="mt-2 text-xs text-amber-400">
              Camera unavailable — paste the payload text below instead. ({cameraError})
            </p>
          ) : null}
        </div>
        <div
          id={SCANNER_ID}
          className={`mt-2 overflow-hidden rounded-lg bg-zinc-950 ${scannerOn ? 'h-52 w-full' : 'hidden'}`}
        />

        {error ? (
          <p className="mt-3 rounded-lg border border-rose-900 bg-rose-950/40 px-3 py-2 text-xs text-rose-300">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Presentational subcomponents
// ---------------------------------------------------------------------------

function StatusCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</span>
      <span className="mt-0.5 break-words text-zinc-200">{value}</span>
    </div>
  );
}

function TabButton({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
        active
          ? 'bg-emerald-600 text-white'
          : 'border border-zinc-700 text-zinc-300 hover:bg-zinc-800'
      }`}
    >
      {label}
    </button>
  );
}

function PayloadQR({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
      <QRCodeSVG
        value={value}
        size={220}
        level="L"
        marginSize={2}
        fgColor="#e4e4e7"
        bgColor="#18181b"
      />
      <span className="text-xs text-zinc-500">{label}</span>
    </div>
  );
}

function PayloadText({ value, onChange, label, readOnly }: {
  value: string;
  onChange?: (value: string) => void;
  label: string;
  readOnly?: boolean;
}) {
  return (
    <label className="block text-sm">
      <span className="text-zinc-400">{label}</span>
      <textarea
        value={value}
        readOnly={readOnly}
        onChange={(event) => onChange?.(event.target.value)}
        placeholder="Paste the compressed Base64 payload here…"
        className="mt-1 h-28 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-200 outline-none focus:border-emerald-500"
      />
    </label>
  );
}

function CopyButton({ onClick, copied }: { onClick: () => void; copied: boolean }) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 transition hover:bg-zinc-800"
    >
      {copied ? 'Copied ✓' : 'Copy'}
    </button>
  );
}

function HostTab({
  busy,
  offer,
  answerInput,
  copied,
  onGenerate,
  onAnswerInput,
  onApplyAnswer,
  onCopyOffer,
}: {
  busy: boolean;
  offer: string | null;
  answerInput: string;
  copied: 'offer' | 'answer' | null;
  onGenerate: () => void;
  onAnswerInput: (value: string) => void;
  onApplyAnswer: () => void;
  onCopyOffer: () => void;
}) {
  return (
    <div className="mt-4 space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onGenerate}
          disabled={busy}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50"
        >
          {busy ? 'Gathering ICE…' : 'Generate room offer'}
        </button>
        <span className="text-xs text-zinc-500">Share the offer with the guest, then apply their answer.</span>
      </div>

      {offer ? (
        <div className="grid gap-4 sm:grid-cols-[220px_1fr]">
          <PayloadQR value={offer} label="Room offer (scan on the guest)" />
          <div className="space-y-2">
            <PayloadText value={offer} readOnly label="Offer payload (Base64)" />
            <CopyButton onClick={onCopyOffer} copied={copied === 'offer'} />
          </div>
        </div>
      ) : null}

      <div className="space-y-2 border-t border-zinc-800 pt-4">
        <PayloadText
          value={answerInput}
          onChange={onAnswerInput}
          label="Paste/scan guest answer"
        />
        <button
          onClick={onApplyAnswer}
          disabled={busy || !answerInput.trim()}
          className="rounded-lg border border-emerald-700 px-4 py-2 text-sm text-emerald-300 transition hover:bg-emerald-950 disabled:opacity-50"
        >
          {busy ? 'Applying…' : 'Apply guest answer'}
        </button>
      </div>
    </div>
  );
}

function JoinerTab({
  busy,
  offerInput,
  answer,
  copied,
  onOfferInput,
  onAcceptOffer,
  onCopyAnswer,
}: {
  busy: boolean;
  offerInput: string;
  answer: string | null;
  copied: 'offer' | 'answer' | null;
  onOfferInput: (value: string) => void;
  onAcceptOffer: () => void;
  onCopyAnswer: () => void;
}) {
  return (
    <div className="mt-4 space-y-4">
      <div className="space-y-2">
        <PayloadText value={offerInput} onChange={onOfferInput} label="Paste/scan host offer" />
        <button
          onClick={onAcceptOffer}
          disabled={busy || !offerInput.trim()}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50"
        >
          {busy ? 'Generating answer…' : 'Accept offer & generate answer'}
        </button>
      </div>

      {answer ? (
        <div className="grid gap-4 border-t border-zinc-800 pt-4 sm:grid-cols-[220px_1fr]">
          <PayloadQR value={answer} label="Guest answer (scan on the host)" />
          <div className="space-y-2">
            <PayloadText value={answer} readOnly label="Answer payload (Base64)" />
            <CopyButton onClick={onCopyAnswer} copied={copied === 'answer'} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
