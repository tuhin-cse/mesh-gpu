/**
 * scheduler.ts
 *
 * Dynamic pipeline scheduler for MeshGPU. Decentralised and deterministic:
 * every peer exchanges its layer capacity, then applies the *same* balanced
 * assignment algorithm, so no leader election is required and all peers
 * converge on identical, contiguous layer ranges.
 *
 * When a peer joins, its capacity is incorporated and stages are rebalanced.
 * When a peer disconnects abruptly (tab closed, network drop), its orphaned
 * layers are reassigned to the remaining peers.
 */

import type {
  ControlMessage,
  PipelineNode,
  PipelineNodeEvent,
  StageRange,
} from './p2p-pipeline';

export interface Assignment {
  peerId: string;
  stage: StageRange;
}

/** A peer's advertised capabilities (used for the topology visualizer). */
export interface PeerInfo {
  peerId: string;
  hostableLayers: number;
  vramBytes: number | null;
  /** Relative compute throughput (1.0 = reference device). */
  throughput: number;
}

/** Result of the local GPU micro-benchmark. */
export interface PeerBenchmark {
  throughput: number;
  gflops: number;
  bandwidthGiBps: number;
}

/** Memory capacity + compute weight used by the balanced assignment. */
export interface PeerCapacity {
  capacity: number;
  weight: number;
}

export interface SchedulerOptions {
  node: PipelineNode;
  /** Total transformer layers in the model being sharded. */
  totalLayers: number;
  /** Returns this node's layer capacity (0 = cannot host). */
  getHostableLayers: () => number;
  /** Returns this node's estimated VRAM pool in bytes (null = unknown). */
  getVramBytes?: () => number | null;
  /** Runs the local 2s GPU micro-benchmark; returns relative throughput. */
  getBenchmark?: () => Promise<PeerBenchmark>;
  /** Called after every (re)assignment with the full sorted assignment list. */
  onAssignmentsChanged?: (assignments: Assignment[]) => void;
}

export class Scheduler {
  private readonly node: PipelineNode;
  private readonly getHostableLayers: () => number;
  private readonly getVramBytes: () => number | null;
  private readonly getBenchmark?: () => Promise<PeerBenchmark>;
  private readonly onAssignmentsChanged?: (assignments: Assignment[]) => void;

  private totalLayers: number;
  private readonly selfPeerId: string;
  private readonly capacities = new Map<string, number>();
  private readonly peerInfo = new Map<string, PeerInfo>();
  private readonly benchmarks = new Map<string, PeerBenchmark>();
  private assignments = new Map<string, StageRange>();
  private ownStage: StageRange | null = null;
  private readonly unsubscribe: () => void;
  private started = false;

  constructor(options: SchedulerOptions) {
    this.node = options.node;
    this.totalLayers = options.totalLayers;
    this.getHostableLayers = options.getHostableLayers;
    this.getVramBytes = options.getVramBytes ?? (() => null);
    this.getBenchmark = options.getBenchmark;
    this.onAssignmentsChanged = options.onAssignmentsChanged;
    this.selfPeerId = options.node.peerId;

    this.unsubscribe = this.node.subscribe((event) => this.handleNodeEvent(event));
  }

  /** Register this peer, advertise its capacity, and compute the first assignment. */
  start(): void {
    if (this.started) return;
    this.started = true;
    this.updateSelfInfo();
    this.broadcastCapacity();
    this.recompute();
  }

  /** Re-read local capacity (e.g. model/headroom changed) and rebalance. */
  refreshCapacity(): void {
    this.updateSelfInfo();
    this.broadcastCapacity();
    this.recompute();
  }

  /** Change the model size and rebalance. */
  setTotalLayers(totalLayers: number): void {
    if (totalLayers <= 0 || totalLayers === this.totalLayers) return;
    this.totalLayers = totalLayers;
    this.refreshCapacity();
  }

  /** Run the local GPU micro-benchmark, advertise it, and rebalance by speed. */
  async runBenchmark(): Promise<void> {
    if (!this.getBenchmark) return;
    let benchmark: PeerBenchmark;
    try {
      benchmark = await this.getBenchmark();
    } catch {
      return; // benchmark unavailable — keep neutral (1.0) weights
    }
    this.benchmarks.set(this.selfPeerId, benchmark);
    this.broadcastBenchmark();
    if (this.started) this.recompute();
  }

  get currentTotalLayers(): number {
    return this.totalLayers;
  }

  get currentOwnStage(): StageRange | null {
    return this.ownStage ? { ...this.ownStage } : null;
  }

  /** Full sorted assignment: [{ peerId, stage }] ordered by layer start. */
  getAssignments(): Assignment[] {
    return [...this.assignments.entries()]
      .map(([peerId, stage]) => ({ peerId, stage: { ...stage } }))
      .sort((a, b) => a.stage.start - b.stage.start);
  }

  /** Advertised capabilities for every known peer (incl. self). */
  getPeerInfo(): PeerInfo[] {
    return [...this.peerInfo.values()]
      .map((info) => ({ ...info }))
      .sort((a, b) => (a.peerId < b.peerId ? -1 : a.peerId > b.peerId ? 1 : 0));
  }

  dispose(): void {
    this.unsubscribe();
  }

  private handleNodeEvent(event: PipelineNodeEvent): void {
    switch (event.type) {
      case 'peer-connected':
        // Advertise our capacity + benchmark so the newcomer can schedule us.
        this.broadcastCapacity();
        this.broadcastBenchmark();
        break;
      case 'peer-disconnected':
        this.capacities.delete(event.peerId);
        this.peerInfo.delete(event.peerId);
        this.benchmarks.delete(event.peerId);
        if (this.started) this.recompute();
        break;
      case 'control':
        this.handleControl(event.peerId, event.message);
        break;
      default:
        break;
    }
  }

  private handleControl(peerId: string, message: ControlMessage): void {
    switch (message.type) {
      case 'capacity': {
        const hostableLayers = Math.max(0, Math.floor(message.hostableLayers));
        this.capacities.set(peerId, hostableLayers);
        const existing = this.peerInfo.get(peerId);
        this.peerInfo.set(peerId, {
          peerId,
          hostableLayers,
          vramBytes: typeof message.vramBytes === 'number' ? message.vramBytes : null,
          throughput: existing?.throughput ?? 1,
        });
        break;
      }
      case 'benchmark': {
        const throughput = Math.max(0.1, Math.min(10, message.throughput || 1));
        this.benchmarks.set(peerId, {
          throughput,
          gflops: message.gflops,
          bandwidthGiBps: message.bandwidthGiBps,
        });
        const existing = this.peerInfo.get(peerId);
        if (existing) existing.throughput = throughput;
        break;
      }
      default:
        break;
    }
    if (this.started) this.recompute();
  }

  private readCapacity(): number {
    return Math.max(0, Math.floor(this.getHostableLayers()));
  }

  private readVramBytes(): number | null {
    const bytes = this.getVramBytes();
    return typeof bytes === 'number' && bytes > 0 ? bytes : null;
  }

  private updateSelfInfo(): void {
    const benchmark = this.benchmarks.get(this.selfPeerId);
    this.capacities.set(this.selfPeerId, this.readCapacity());
    this.peerInfo.set(this.selfPeerId, {
      peerId: this.selfPeerId,
      hostableLayers: this.readCapacity(),
      vramBytes: this.readVramBytes(),
      throughput: benchmark?.throughput ?? 1,
    });
  }

  private broadcastCapacity(): void {
    this.node.broadcastControl({
      type: 'capacity',
      hostableLayers: this.readCapacity(),
      totalLayers: this.totalLayers,
      vramBytes: this.readVramBytes(),
    });
  }

  private broadcastBenchmark(): void {
    const benchmark = this.benchmarks.get(this.selfPeerId);
    if (!benchmark) return;
    this.node.broadcastControl({
      type: 'benchmark',
      throughput: benchmark.throughput,
      gflops: benchmark.gflops,
      bandwidthGiBps: benchmark.bandwidthGiBps,
    });
  }

  private capacityMap(): Map<string, PeerCapacity> {
    const map = new Map<string, PeerCapacity>();
    for (const [peerId, capacity] of this.capacities) {
      map.set(peerId, {
        capacity,
        weight: this.benchmarks.get(peerId)?.throughput ?? 1,
      });
    }
    return map;
  }

  private recompute(): void {
    this.assignments = computeAssignments(this.totalLayers, this.capacityMap());

    const mine = this.assignments.get(this.selfPeerId) ?? null;
    const changed = !sameStage(this.ownStage, mine);
    this.ownStage = mine;

    if (changed) {
      // Applying the stage broadcasts it, which keeps every peer's forwarding
      // topology (nextHop) in sync with the assignment.
      this.node.setStage(mine);
    }
    this.onAssignmentsChanged?.(this.getAssignments());
    // Re-route any in-flight hidden states orphaned by the topology change.
    this.node.retryPending();
  }
}

function sameStage(a: StageRange | null, b: StageRange | null): boolean {
  if (a === null || b === null) return a === b;
  return a.start === b.start && a.end === b.end;
}

/**
 * Deterministic, throughput-weighted balanced assignment.
 *
 * Peers are sorted by peerId for reproducible output. Each peer gets a
 * contiguous range sized proportionally to its compute throughput (so a 2×
 * faster GPU hosts ~2× the layers), capped at its memory capacity. Leftover
 * layers go to peers with spare capacity. If total capacity is less than
 * totalLayers, the tail is left unassigned (the pipeline cannot cover the
 * full model).
 */
export function computeAssignments(
  totalLayers: number,
  peers: ReadonlyMap<string, PeerCapacity>,
): Map<string, StageRange> {
  const result = new Map<string, StageRange>();

  const entries = [...peers.entries()]
    .filter(([, peer]) => peer.capacity > 0 && peer.weight > 0)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));

  if (entries.length === 0) return result;

  const totalWeight = entries.reduce((sum, [, peer]) => sum + peer.weight, 0);
  if (totalWeight <= 0) return result;

  // Phase 1: proportional to throughput, capped at each peer's capacity.
  const shares = new Map<string, number>();
  let assignedSum = 0;
  for (const [peerId, peer] of entries) {
    const share = Math.min(peer.capacity, Math.floor((totalLayers * peer.weight) / totalWeight));
    shares.set(peerId, share);
    assignedSum += share;
  }

  // Phase 2: hand leftover layers out to best preserve the throughput ratio.
  // Prefer the peer with the largest fractional remainder; if every under-full
  // peer's fractional share is exhausted (some peer hit its memory cap), fall
  // back to whichever peer still has the most spare capacity.
  let leftover = totalLayers - assignedSum;
  while (leftover > 0) {
    let bestPeer: string | null = null;
    let bestFraction = -1;
    let fallbackPeer: string | null = null;
    let bestSpare = -1;

    for (const [peerId, peer] of entries) {
      const current = shares.get(peerId) ?? 0;
      const spare = peer.capacity - current;
      if (spare <= 0) continue;
      if (spare > bestSpare) {
        bestSpare = spare;
        fallbackPeer = peerId;
      }
      const fraction = (totalLayers * peer.weight) / totalWeight - current;
      if (fraction > bestFraction) {
        bestFraction = fraction;
        bestPeer = peerId;
      }
    }

    const target = bestPeer !== null && bestFraction > 1e-9 ? bestPeer : fallbackPeer;
    if (target === null) break; // capacity shortfall
    shares.set(target, (shares.get(target) ?? 0) + 1);
    leftover -= 1;
  }

  // Emit contiguous, non-overlapping ranges in sorted order.
  let cursor = 0;
  for (const [peerId] of entries) {
    const layers = shares.get(peerId) ?? 0;
    if (layers <= 0) continue;
    result.set(peerId, { start: cursor, end: cursor + layers });
    cursor += layers;
  }

  return result;
}
