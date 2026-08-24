import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import {
  GiB,
  ML_RELEVANT_FEATURES,
  MODEL_PROFILES,
  WebGPUNode,
  benchmarkWebGPU,
  computeThroughputScore,
  formatBytes,
} from './engine/webgpu-node';
import type { GPULimitsSummary, LayerAllocation, ModelProfile, WebGPUInfo } from './engine/webgpu-node';
import { IdentityExecutor, PipelineNode, createSignalingAndTransport } from './engine/p2p-pipeline';
import { Scheduler } from './engine/scheduler';
import type { Assignment, PeerInfo } from './engine/scheduler';
import { isMockP2PEnabled } from './engine/mock-transport';
import { VramGauge } from './components/VramGauge';
import { TopologyGraph } from './components/TopologyGraph';
import type { TopologyEdge, TopologyNode } from './components/TopologyGraph';
import { ChatUI } from './components/ChatUI';
import { ContributeCard } from './components/ContributeCard';
import { QRHandshakeModal } from './components/QRHandshakeModal';

type InspectState =
  | { status: 'loading' }
  | { status: 'ready'; info: WebGPUInfo }
  | { status: 'error'; message: string };

export default function App() {
  const [node, setNode] = useState<WebGPUNode | null>(null);
  const [state, setState] = useState<InspectState>({ status: 'loading' });
  const [profileId, setProfileId] = useState<string>(MODEL_PROFILES[0].id);
  const [headroom, setHeadroom] = useState<number>(0.2);

  useEffect(() => {
    let cancelled = false;
    let createdNode: WebGPUNode | null = null;

    WebGPUNode.create()
      .then((created) => {
        if (cancelled) {
          created.dispose();
          return;
        }
        createdNode = created;
        setNode(created);
        setState({ status: 'ready', info: created.info });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
          });
        }
      });

    return () => {
      cancelled = true;
      createdNode?.dispose();
    };
  }, []);

  const info = state.status === 'ready' ? state.info : null;

  const allocation: LayerAllocation | null = useMemo(
    () => (node ? node.allocateFor(profileId, headroom) : null),
    [node, profileId, headroom],
  );

  const selectedProfile: ModelProfile =
    MODEL_PROFILES.find((profile) => profile.id === profileId) ?? MODEL_PROFILES[0];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl space-y-6 px-6 py-10">
        <header className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Mesh<span className="text-emerald-400">GPU</span>
            </h1>
            <p className="mt-1 max-w-xl text-sm text-zinc-400">
              Browser-native GPU mesh — WebGPU capability probing, serverless WebRTC pairing and
              a peer-to-peer tensor transport. Sharded inference is not implemented yet.
            </p>
            <span
              className="mt-3 inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300"
              title="The QR/Base64 handshake uses no signaling server and an empty ICE server list, so pairing and tensor traffic stay on your subnet."
            >
              QR pairing is serverless — traffic stays on your LAN
            </span>
          </div>
          {info ? (
            <Badge ok={info.adapterAcquired}>
              {info.adapterAcquired ? 'WebGPU ready' : 'WebGPU unavailable'}
            </Badge>
          ) : state.status === 'loading' ? (
            <Badge ok={false}>Probing GPU…</Badge>
          ) : null}
        </header>

        {state.status === 'loading' && <Card>Requesting WebGPU adapter…</Card>}

        {state.status === 'error' && (
          <Card>
            <h2 className="text-sm font-medium text-rose-400">Failed to initialise WebGPU</h2>
            <p className="mt-1 text-sm text-zinc-400">{state.message}</p>
          </Card>
        )}

        {info && (
          <>
            <AdapterCard info={info} />
            <div className="grid gap-6 lg:grid-cols-2">
              <VramCard info={info} />
              <FeaturesCard info={info} />
            </div>
            <LimitsGrid limits={info.limits} />
            <AllocatorCard
              profileId={profileId}
              headroom={headroom}
              allocation={allocation}
              onProfileChange={setProfileId}
              onHeadroomChange={setHeadroom}
            />
          </>
        )}

        <PipelineCard
          profile={selectedProfile}
          allocation={allocation}
          vramBytes={info?.estimatedVRAMBytes ?? null}
          device={info?.device ?? null}
        />

        <ContributeCard />

        <ChatUI />

        {isMockP2PEnabled() ? <MockMeshDemo /> : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Presentational components (refactored into /components in a later step)
// ---------------------------------------------------------------------------

function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
      {title ? <h2 className="mb-3 text-sm font-medium text-zinc-300">{title}</h2> : null}
      {children}
    </section>
  );
}

function Badge({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <span
      className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs font-medium ${
        ok
          ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
          : 'border-amber-500/40 bg-amber-500/10 text-amber-300'
      }`}
    >
      {children}
    </span>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="mt-0.5 break-words text-zinc-200">{value}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="mt-1 text-lg font-medium text-zinc-100">{value}</div>
    </div>
  );
}

function AdapterCard({ info }: { info: WebGPUInfo }) {
  const adapter = info.adapterInfo;
  return (
    <Card title="GPU Adapter">
      <div className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2">
        <Row label="Device" value={adapter?.device ?? '—'} />
        <Row label="Architecture" value={adapter?.architecture ?? '—'} />
        <Row label="Vendor" value={adapter?.vendor ?? '—'} />
        <Row label="Description" value={adapter?.description ?? '—'} />
      </div>
    </Card>
  );
}

function VramCard({ info }: { info: WebGPUInfo }) {
  return (
    <Card title="VRAM Pool Estimate">
      <div className="text-3xl font-semibold text-emerald-300">
        {info.estimatedVRAMBytes ? formatBytes(info.estimatedVRAMBytes) : 'Unknown'}
      </div>
      <p className="mt-1 text-xs text-zinc-500">
        Source:{' '}
        <span className="text-zinc-300">{sourceLabel(info.vramSource)}</span>
      </p>
      <p className="mt-3 text-xs leading-relaxed text-zinc-500">{info.vramNote}</p>
      <p className="mt-2 text-xs leading-relaxed text-zinc-600">
        Browsers do not expose dedicated VRAM directly; MeshGPU parses the adapter description
        against a device database and reserves headroom for activations.
      </p>
    </Card>
  );
}

function FeaturesCard({ info }: { info: WebGPUInfo }) {
  const supported = new Set<string>(info.mlRelevantFeatures);
  return (
    <Card title="ML-Relevant WebGPU Features">
      <ul className="space-y-2">
        {ML_RELEVANT_FEATURES.map((feature) => (
          <li key={feature} className="flex items-center justify-between text-sm">
            <code className="font-mono text-xs text-zinc-300">{feature}</code>
            {supported.has(feature) ? (
              <span className="text-emerald-400">✓</span>
            ) : (
              <span className="text-zinc-600">✗</span>
            )}
          </li>
        ))}
      </ul>
    </Card>
  );
}

function LimitsGrid({ limits }: { limits: GPULimitsSummary | null }) {
  if (!limits) return null;

  const workgroupSize =
    limits.maxComputeWorkgroupSizeX !== null
      ? `${limits.maxComputeWorkgroupSizeX} × ${limits.maxComputeWorkgroupSizeY} × ${limits.maxComputeWorkgroupSizeZ}`
      : null;

  const entries: Array<{ label: string; value: string }> = [
    { label: 'maxBufferSize', value: formatLimit(limits.maxBufferSize) },
    {
      label: 'maxStorageBufferBindingSize',
      value: formatLimit(limits.maxStorageBufferBindingSize),
    },
    {
      label: 'maxStorageBuffersPerShaderStage',
      value: formatCount(limits.maxStorageBuffersPerShaderStage),
    },
    {
      label: 'maxComputeWorkgroupStorageSize',
      value: formatLimit(limits.maxComputeWorkgroupStorageSize),
    },
    {
      label: 'maxComputeWorkgroupSize',
      value: workgroupSize ?? '—',
    },
    {
      label: 'maxComputeInvocationsPerWorkgroup',
      value: formatCount(limits.maxComputeInvocationsPerWorkgroup),
    },
    {
      label: 'maxComputeWorkgroupsPerDimension',
      value: formatCount(limits.maxComputeWorkgroupsPerDimension),
    },
    { label: 'maxBindGroups', value: formatCount(limits.maxBindGroups) },
    { label: 'maxTextureDimension2D', value: formatCount(limits.maxTextureDimension2D) },
  ];

  function formatLimit(bytes: number | null): string {
    if (bytes === null) return '—';
    return formatBytes(bytes);
  }

  return (
    <Card title="Adapter Limits">
      <div className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
        {entries.map((entry) => (
          <Row key={entry.label} label={entry.label} value={entry.value} />
        ))}
      </div>
    </Card>
  );
}

function formatCount(value: number | null): string {
  return value === null ? '—' : value.toLocaleString();
}

function AllocatorCard({
  profileId,
  headroom,
  allocation,
  onProfileChange,
  onHeadroomChange,
}: {
  profileId: string;
  headroom: number;
  allocation: LayerAllocation | null;
  onProfileChange: (id: string) => void;
  onHeadroomChange: (value: number) => void;
}) {
  const profile = MODEL_PROFILES.find((p) => p.id === profileId) ?? MODEL_PROFILES[0];

  return (
    <Card title="Pipeline Stage Allocator">
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="text-zinc-400">Model</span>
          <select
            value={profileId}
            onChange={(event) => onProfileChange(event.target.value)}
            className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500"
          >
            {MODEL_PROFILES.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </label>

        <label className="block text-sm">
          <span className="text-zinc-400">Headroom · {(headroom * 100).toFixed(0)}%</span>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.05}
            value={headroom}
            onChange={(event) => onHeadroomChange(Number(event.target.value))}
            className="mt-3 w-full accent-emerald-500"
          />
        </label>
      </div>

      {allocation ? (
        <div className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
          <Stat
            label="Layers this node hosts"
            value={`${allocation.hostableLayers} / ${profile.layerCount}`}
          />
          <Stat label="Weights per layer" value={formatBytes(allocation.bytesPerLayer)} />
          <Stat
            label="Full model (fp16)"
            value={formatBytes(allocation.totalModelVRAMBytes)}
          />
        </div>
      ) : null}

      {allocation ? (
        <p className="mt-3 text-xs text-zinc-500">{allocation.reason}</p>
      ) : null}
    </Card>
  );
}

function sourceLabel(source: WebGPUInfo['vramSource']): string {
  switch (source) {
    case 'description-parse':
      return 'device database match';
    case 'unified-memory-heuristic':
      return 'unified memory heuristic';
    case 'none':
      return 'none';
  }
}

function PipelineCard({
  profile,
  allocation,
  vramBytes,
  device,
}: {
  profile: ModelProfile;
  allocation: LayerAllocation | null;
  vramBytes: number | null;
  device: GPUDevice | null;
}) {
  const nodeRef = useRef<PipelineNode | null>(null);
  const schedulerRef = useRef<Scheduler | null>(null);

  // Always-current props for scheduler closures (avoid stale captures).
  const profileRef = useRef(profile);
  profileRef.current = profile;
  const allocationRef = useRef(allocation);
  allocationRef.current = allocation;
  const vramBytesRef = useRef(vramBytes);
  vramBytesRef.current = vramBytes;
  const deviceRef = useRef(device);
  deviceRef.current = device;

  const [status, setStatus] = useState<'offline' | 'connecting' | 'connected' | 'error'>('offline');
  const [role, setRole] = useState<'host' | 'joiner' | null>(null);
  const [showHandshake, setShowHandshake] = useState(false);
  const [peers, setPeers] = useState<string[]>([]);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [peerInfo, setPeerInfo] = useState<PeerInfo[]>([]);
  const [ownStage, setOwnStage] = useState<{ start: number; end: number } | null>(null);
  const [latencies, setLatencies] = useState<Record<string, number | null>>({});
  const [loopbackFps, setLoopbackFps] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const tokensRef = useRef<number[]>([]);

  const pushLog = useCallback((line: string) => {
    setLogs((prev) => [...prev.slice(-99), `${new Date().toLocaleTimeString()} ${line}`]);
  }, []);

  const teardownNode = useCallback(() => {
    schedulerRef.current?.dispose();
    schedulerRef.current = null;
    nodeRef.current?.close();
    nodeRef.current = null;
    setPeers([]);
    setAssignments([]);
    setPeerInfo([]);
    setOwnStage(null);
    setLatencies({});
    setLoopbackFps(0);
    tokensRef.current = [];
  }, []);

  const createNode = useCallback(
    (nextRole: 'host' | 'joiner'): PipelineNode => {
      teardownNode();

      const node = new PipelineNode({
        peerId: nextRole,
        executor: new IdentityExecutor(),
        callbacks: {
          onPeerConnected: (id) => {
            setPeers((prev) => (prev.includes(id) ? prev : [...prev, id]));
            setStatus('connected');
            pushLog(`peer connected: ${id}`);
          },
          onPeerDisconnected: (id) => {
            setPeers((prev) => prev.filter((p) => p !== id));
            setLatencies((prev) => {
              const next = { ...prev };
              delete next[id];
              return next;
            });
            setStatus('offline');
            setRole(null);
            pushLog(`peer disconnected: ${id} — rebalancing`);
          },
          onStageChanged: (stage) => {
            pushLog(
              stage ? `auto-assigned stage: [${stage.start}, ${stage.end})` : 'stage cleared',
            );
          },
          onToken: (_token, meta) => {
            const now = performance.now();
            const stamps = tokensRef.current;
            stamps.push(now);
            const cutoff = now - 5000;
            while (stamps.length > 0 && stamps[0] < cutoff) stamps.shift();
            setLoopbackFps(stamps.length / 5);
            pushLog(`loopback frame returned (seq ${meta.seqId}) from ${meta.fromPeerId}`);
          },
          onForwarded: (frame) => {
            pushLog(`forwarded tensor → layers [${frame.fromLayer}, ${frame.toLayer})`);
          },
          onError: (id, error) => {
            pushLog(`error (${id}): ${error.message}`);
          },
        },
      });
      nodeRef.current = node;

      const scheduler = new Scheduler({
        node,
        totalLayers: profileRef.current.layerCount,
        getHostableLayers: () => allocationRef.current?.hostableLayers ?? 0,
        getVramBytes: () => vramBytesRef.current,
        getBenchmark: async () => {
          const gpuDevice = deviceRef.current;
          if (!gpuDevice) throw new Error('no WebGPU device');
          const bench = await benchmarkWebGPU(gpuDevice, 2000);
          return {
            throughput: computeThroughputScore(bench),
            gflops: bench.gflops,
            bandwidthGiBps: bench.bandwidthGiBps,
          };
        },
        onAssignmentsChanged: (next) => {
          setAssignments(next);
          setOwnStage(node.currentStage);
          setPeerInfo(schedulerRef.current?.getPeerInfo() ?? []);
        },
      });
      schedulerRef.current = scheduler;
      scheduler.start();
      void scheduler.runBenchmark();
      return node;
    },
    [pushLog, teardownNode],
  );

  const hostStart = useCallback(async (): Promise<string> => {
    setStatus('connecting');
    setRole('host');
    try {
      const node = createNode('host');
      const offer = await node.initiateHostConnection('joiner');
      pushLog('room offer generated — share via QR or copy');
      return offer;
    } catch (err) {
      setStatus('error');
      pushLog(`offer failed: ${err instanceof Error ? err.message : String(err)}`);
      throw err;
    }
  }, [createNode, pushLog]);

  const hostApplyAnswer = useCallback(
    async (answer: string): Promise<void> => {
      const node = nodeRef.current;
      if (!node) throw new Error('no active node — generate a room offer first');
      await node.applyRemoteAnswer('joiner', answer);
      pushLog('guest answer applied — awaiting DataChannel open');
    },
    [pushLog],
  );

  const joinerStart = useCallback(
    async (offer: string): Promise<string> => {
      setStatus('connecting');
      setRole('joiner');
      try {
        const node = createNode('joiner');
        const answer = await node.initiateJoinerConnection('host', offer);
        pushLog('guest answer generated — share back to the host');
        return answer;
      } catch (err) {
        setStatus('error');
        pushLog(`answer failed: ${err instanceof Error ? err.message : String(err)}`);
        throw err;
      }
    },
    [createNode, pushLog],
  );

  const resetConnection = useCallback(() => {
    teardownNode();
    setStatus('offline');
    setRole(null);
    pushLog('disconnected — peer link closed');
  }, [pushLog, teardownNode]);

  const runTest = useCallback(async () => {
    const node = nodeRef.current;
    if (!node) return;
    const input = new Float32Array(16).map((_, i) => i);
    try {
      const token = await node.run(input, [1, 16]);
      if (token !== null) pushLog(`loopback completed locally (argmax index ${token})`);
    } catch (err) {
      pushLog(`run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [pushLog]);

  // Keep the scheduler in sync with model / headroom changes.
  useEffect(() => {
    const scheduler = schedulerRef.current;
    if (!scheduler || status !== 'connected') return;
    scheduler.setTotalLayers(profile.layerCount);
    scheduler.refreshCapacity();
  }, [profile, allocation, status]);

  // Poll per-peer round-trip latency while connected.
  useEffect(() => {
    if (status !== 'connected') return;
    const interval = window.setInterval(() => {
      const node = nodeRef.current;
      if (!node) return;
      for (const id of node.peers) {
        const result = node.mesh.measureLatency(id);
        if (result) {
          result.then((rtt) => setLatencies((prev) => ({ ...prev, [id]: rtt })));
        }
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [status]);

  // Clean up on unmount.
  useEffect(
    () => () => {
      schedulerRef.current?.dispose();
      nodeRef.current?.close();
    },
    [],
  );

  const connected = status === 'connected';
  const selfPeerId = role ?? '—';
  const remotePeerId = role === 'host' ? 'joiner' : role === 'joiner' ? 'host' : null;
  const remoteLatency = remotePeerId ? latencies[remotePeerId] ?? null : null;
  const canRun = connected && ownStage !== null && ownStage.start === 0;
  const hostable = allocation?.hostableLayers ?? 0;

  const topologyNodes: TopologyNode[] = useMemo(
    () =>
      assignments.map((assignment) => {
        const info = peerInfo.find((entry) => entry.peerId === assignment.peerId);
        return {
          id: assignment.peerId,
          label: assignment.peerId === selfPeerId ? 'you' : assignment.peerId,
          layerStart: assignment.stage.start,
          layerEnd: assignment.stage.end,
          isSelf: assignment.peerId === selfPeerId,
          vramGiB: info && info.vramBytes !== null ? info.vramBytes / GiB : null,
          rttMs: assignment.peerId === selfPeerId ? null : latencies[assignment.peerId],
        };
      }),
    [assignments, peerInfo, selfPeerId, latencies],
  );

  const topologyEdges: TopologyEdge[] = useMemo(() => {
    const sorted = [...topologyNodes].sort((a, b) => a.layerStart - b.layerStart);
    const edges: TopologyEdge[] = [];
    for (let i = 0; i + 1 < sorted.length; i += 1) {
      edges.push({ from: sorted[i].id, to: sorted[i + 1].id });
    }
    return edges;
  }, [topologyNodes]);

  const usedVramBytes: number | null = useMemo(() => {
    if (!ownStage || !allocation) return null;
    return (ownStage.end - ownStage.start) * allocation.bytesPerLayer;
  }, [ownStage, allocation]);

  return (
    <Card title="P2P Transport Loopback (WebRTC) — Offline QR Handshake">
      <p className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs leading-relaxed text-amber-200/90">
        <span className="font-medium">No model runs here yet.</span> This card exercises the
        handshake, the binary tensor wire format and the layer scheduler using an identity
        executor — tensors are forwarded between peers unchanged. Sharded inference is on the
        roadmap; the single-node chat card below is the only place a real model executes.
      </p>
      <div className="flex flex-wrap items-end gap-4">
        <button
          onClick={() => setShowHandshake(true)}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500"
        >
          {connected ? 'Peer handshake' : 'Connect peers (QR)'}
        </button>

        <button
          onClick={resetConnection}
          disabled={status === 'offline'}
          className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 transition hover:bg-zinc-800 disabled:opacity-50"
        >
          Disconnect
        </button>

        <button
          onClick={runTest}
          disabled={!canRun}
          className="rounded-lg border border-emerald-700 px-4 py-2 text-sm text-emerald-300 transition hover:bg-emerald-950 disabled:opacity-50"
        >
          Run loopback tensor
        </button>

        <span className="ml-auto text-xs text-zinc-500">
          peer <code className="font-mono text-zinc-300">{selfPeerId}</code>
          <span
            className={`ml-2 inline-block h-2 w-2 rounded-full ${
              connected ? 'bg-emerald-400' : 'bg-zinc-600'
            }`}
          />
        </span>
      </div>

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-4">
        <Stat label="My capacity" value={`${hostable} / ${profile.layerCount} layers`} />
        <Stat
          label="My auto-assigned stage"
          value={ownStage ? `[${ownStage.start}, ${ownStage.end})` : '—'}
        />
        <Stat label="Connected peers" value={peers.length > 0 ? String(peers.length) : 'none'} />
        <Stat label="Loopback rate" value={`${loopbackFps.toFixed(1)} frames/s`} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[220px_1fr]">
        <VramGauge usedBytes={usedVramBytes} totalBytes={vramBytes} label="VRAM pool" />
        <div className="min-w-0">
          <TopologyGraph nodes={topologyNodes} edges={topologyEdges} />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {assignments.length > 0 ? (
          assignments.map((assignment) => (
            <span
              key={assignment.peerId}
              className={`rounded-full border px-3 py-1 text-xs font-medium ${
                assignment.peerId === selfPeerId
                  ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
                  : 'border-zinc-700 bg-zinc-900 text-zinc-300'
              }`}
            >
              {assignment.peerId === selfPeerId ? 'you' : assignment.peerId} · [
              {assignment.stage.start}, {assignment.stage.end})
            </span>
          ))
        ) : connected ? (
          <span className="text-xs text-zinc-600">
            No layers assigned — VRAM estimate unavailable or capacity is 0.
          </span>
        ) : null}
      </div>

      {status === 'error' ? (
        <p className="mt-3 text-xs text-rose-400">
          Handshake failed — reopen the peer handshake to retry (details in the event log).
        </p>
      ) : null}

      <pre className="mt-4 h-40 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 font-mono text-xs leading-relaxed text-zinc-400">
        {logs.length > 0 ? logs.join('\n') : '// event log'}
      </pre>

      <QRHandshakeModal
        open={showHandshake}
        onClose={() => setShowHandshake(false)}
        connected={connected}
        stage={ownStage}
        latencyMs={remoteLatency}
        onHostStart={hostStart}
        onHostApplyAnswer={hostApplyAnswer}
        onJoinerStart={joinerStart}
        onReset={resetConnection}
      />
    </Card>
  );
}

interface MockPeerRow {
  peerId: string;
  throughput: number;
  stage: string;
  lastEvent: string;
}

function MockMeshDemo() {
  const [count, setCount] = useState(3);
  const [running, setRunning] = useState(false);
  const [rows, setRows] = useState<MockPeerRow[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const instancesRef = useRef<Array<{ node: PipelineNode; scheduler: Scheduler }>>([]);

  const pushLog = useCallback((line: string) => {
    setLog((prev) => [...prev.slice(-49), `${new Date().toLocaleTimeString()} ${line}`]);
  }, []);

  const updateRow = useCallback((peerId: string, patch: Partial<MockPeerRow>) => {
    setRows((prev) => prev.map((row) => (row.peerId === peerId ? { ...row, ...patch } : row)));
  }, []);

  const stop = useCallback(() => {
    for (const instance of instancesRef.current) {
      instance.scheduler.dispose();
      instance.node.close();
    }
    instancesRef.current = [];
    setRows([]);
    setRunning(false);
  }, []);

  const start = useCallback(async () => {
    stop();
    setRunning(true);
    setLog([]);

    const roomId = 'mock-demo';
    const instances: Array<{ node: PipelineNode; scheduler: Scheduler }> = [];
    const initialRows: MockPeerRow[] = [];

    for (let i = 0; i < count; i += 1) {
      const peerId = `mock-${String.fromCharCode(97 + i)}`;
      const throughput = i === 1 ? 2 : 1; // second peer is 2× faster

      const { signaling, createPeerConnection } = createSignalingAndTransport({
        url: '',
        roomId,
        peerId,
      });

      const node = new PipelineNode({
        signaling,
        createPeerConnection,
        executor: new IdentityExecutor(),
        callbacks: {
          onStageChanged: (stage) => {
            updateRow(peerId, { stage: stage ? `[${stage.start}, ${stage.end})` : '—' });
          },
          onForwarded: (frame) => {
            updateRow(peerId, { lastEvent: `→ [${frame.fromLayer}, ${frame.toLayer})` });
          },
          onToken: (token) => {
            updateRow(peerId, { lastEvent: `token ${token}` });
            pushLog(`${peerId} produced token ${token}`);
          },
          onError: (id, error) => pushLog(`error (${id}): ${error.message}`),
        },
      });

      const scheduler = new Scheduler({
        node,
        totalLayers: 28,
        getHostableLayers: () => 28,
        getVramBytes: () => 8 * GiB,
        getBenchmark: async () => ({
          throughput,
          gflops: throughput * 1000,
          bandwidthGiBps: 200,
        }),
      });

      await node.join();
      scheduler.start();
      void scheduler.runBenchmark();

      instances.push({ node, scheduler });
      initialRows.push({ peerId, throughput, stage: '—', lastEvent: 'joined' });
    }

    instancesRef.current = instances;
    setRows(initialRows);
    pushLog(`${count} mock peers joined room "${roomId}"`);
  }, [count, pushLog, stop, updateRow]);

  const runTensor = useCallback(async () => {
    const first = instancesRef.current[0];
    if (!first) return;
    const input = new Float32Array(16).map((_, i) => i);
    try {
      const token = await first.node.run(input, [1, 16]);
      if (token !== null) pushLog(`first-stage token: ${token}`);
    } catch (err) {
      pushLog(`run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [pushLog]);

  useEffect(() => () => stop(), [stop]);

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
      <h2 className="text-sm font-medium text-zinc-300">Mock P2P Mesh (headless demo)</h2>
      <p className="mt-1 text-xs text-zinc-500">
        Chains {count} in-memory peers in this window — no signaling server or ICE. The second
        peer is 2× faster, so it should host ~2× the layers.
      </p>

      <div className="mt-3 flex flex-wrap items-end gap-4">
        <label className="block text-sm">
          <span className="text-zinc-400">Peers</span>
          <select
            value={count}
            disabled={running}
            onChange={(event) => setCount(Number(event.target.value))}
            className="mt-1 w-24 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-500 disabled:opacity-50"
          >
            {[2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={start}
          disabled={running}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50"
        >
          Start mesh
        </button>

        <button
          onClick={stop}
          disabled={!running}
          className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 transition hover:bg-zinc-800 disabled:opacity-50"
        >
          Stop
        </button>

        <button
          onClick={runTensor}
          disabled={!running}
          className="rounded-lg border border-emerald-700 px-4 py-2 text-sm text-emerald-300 transition hover:bg-emerald-950 disabled:opacity-50"
        >
          Run tensor
        </button>
      </div>

      {rows.length > 0 ? (
        <table className="mt-4 w-full text-left text-xs">
          <thead>
            <tr className="text-zinc-500">
              <th className="py-1 pr-3 font-medium">Peer</th>
              <th className="py-1 pr-3 font-medium">Throughput</th>
              <th className="py-1 pr-3 font-medium">Stage</th>
              <th className="py-1 font-medium">Last event</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.peerId} className="border-t border-zinc-800">
                <td className="py-1 pr-3 font-mono text-zinc-200">{row.peerId}</td>
                <td className="py-1 pr-3 text-zinc-300">{row.throughput}×</td>
                <td className="py-1 pr-3 font-mono text-emerald-300">{row.stage}</td>
                <td className="py-1 font-mono text-zinc-400">{row.lastEvent}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      <pre className="mt-3 h-32 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 font-mono text-xs leading-relaxed text-zinc-400">
        {log.length > 0 ? log.join('\n') : '// mock mesh log'}
      </pre>
    </section>
  );
}
