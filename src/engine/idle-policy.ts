/**
 * idle-policy.ts
 *
 * Decides whether this device is currently willing to take mesh work.
 *
 * Contributing has to be unobtrusive or people revoke it. The policy watches
 * three things the browser will actually tell us — the contribute toggle,
 * battery state, and recent human input — and reports a single `paused` flag
 * plus the reason, which the UI shows so nobody has to guess why their machine
 * went quiet.
 *
 * Note that tab visibility is deliberately *not* a pause condition: with the
 * engine on a worker thread, a backgrounded tab is the ideal contributor.
 */

export interface IdlePolicyOptions {
  /** Master switch — the user's contribute toggle. */
  enabled: boolean;
  /** Stop contributing when running on battery power. */
  pauseOnBattery: boolean;
  /** Stop contributing while the person is actively using this device. */
  pauseWhenActive: boolean;
  /** How long after the last keypress or click counts as "still active". */
  activeWindowMs?: number;
}

export interface IdleState {
  paused: boolean;
  /** Why work is paused, for display. Empty when contributing. */
  reason: string;
  onBattery: boolean;
  userActive: boolean;
}

const DEFAULT_ACTIVE_WINDOW_MS = 60_000;

type Listener = (state: IdleState) => void;

interface BatteryLike extends EventTarget {
  charging: boolean;
}

/**
 * Watches the local signals and notifies on any change to the resulting state.
 * Call `dispose()` to detach every listener.
 */
export class IdlePolicy {
  private options: Required<IdlePolicyOptions>;
  private readonly listeners = new Set<Listener>();
  private readonly cleanups: Array<() => void> = [];

  private onBattery = false;
  private lastInputAt = 0;
  private activeTimer: ReturnType<typeof setInterval> | null = null;
  private state: IdleState;

  constructor(options: IdlePolicyOptions) {
    this.options = {
      activeWindowMs: DEFAULT_ACTIVE_WINDOW_MS,
      ...options,
    };
    this.state = this.compute();

    this.watchBattery();
    this.watchInput();
  }

  get current(): IdleState {
    return { ...this.state };
  }

  /** Subscribe to state changes. Returns an unsubscribe function. */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.current);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Apply new user preferences and re-evaluate immediately. */
  update(options: Partial<IdlePolicyOptions>): void {
    this.options = { ...this.options, ...options };
    this.publish();
  }

  dispose(): void {
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups.length = 0;
    if (this.activeTimer !== null) clearInterval(this.activeTimer);
    this.activeTimer = null;
    this.listeners.clear();
  }

  private compute(): IdleState {
    const userActive =
      this.lastInputAt > 0 && Date.now() - this.lastInputAt < this.options.activeWindowMs;

    if (!this.options.enabled) {
      return { paused: true, reason: 'contributing is off', onBattery: this.onBattery, userActive };
    }
    if (this.options.pauseOnBattery && this.onBattery) {
      return { paused: true, reason: 'on battery power', onBattery: true, userActive };
    }
    if (this.options.pauseWhenActive && userActive) {
      return { paused: true, reason: 'you are using this device', onBattery: this.onBattery, userActive };
    }
    return { paused: false, reason: '', onBattery: this.onBattery, userActive };
  }

  private publish(): void {
    const next = this.compute();
    const changed =
      next.paused !== this.state.paused ||
      next.reason !== this.state.reason ||
      next.onBattery !== this.state.onBattery ||
      next.userActive !== this.state.userActive;

    this.state = next;
    if (!changed) return;
    for (const listener of this.listeners) listener(this.current);
  }

  private watchBattery(): void {
    const nav = navigator as Navigator & { getBattery?: () => Promise<BatteryLike> };
    if (typeof nav.getBattery !== 'function') return;

    void nav
      .getBattery()
      .then((battery) => {
        const sync = (): void => {
          this.onBattery = !battery.charging;
          this.publish();
        };
        battery.addEventListener('chargingchange', sync);
        this.cleanups.push(() => battery.removeEventListener('chargingchange', sync));
        sync();
      })
      .catch(() => {
        // Firefox and Safari do not expose the Battery API. Without it we
        // cannot tell, so we assume mains power rather than refusing to work.
      });
  }

  private watchInput(): void {
    const mark = (): void => {
      this.lastInputAt = Date.now();
      if (this.options.pauseWhenActive && !this.state.userActive) this.publish();
    };

    const events: Array<keyof WindowEventMap> = ['pointerdown', 'keydown', 'wheel'];
    for (const event of events) {
      window.addEventListener(event, mark, { passive: true });
      this.cleanups.push(() => window.removeEventListener(event, mark));
    }

    // The transition from "active" back to "idle" is the passage of time
    // rather than an event, so it needs a poll.
    this.activeTimer = setInterval(() => this.publish(), 5_000);
  }
}
