/**
 * registry.js
 *
 * The pool of browser tabs currently offering their GPU to the mesh.
 *
 * A worker is claimable when it has the requested model loaded, is not paused
 * by its own idle policy, and has room for another concurrent job. Selection
 * is least-loaded first, then least-recently-used, so work spreads across the
 * office instead of hammering whoever connected first.
 */

/** @typedef {(message: object) => void} SendFn */

/**
 * @typedef {object} WorkerRecord
 * @property {string} id
 * @property {SendFn} send            Deliver a message to this worker.
 * @property {string} label           Human-readable name shown in status output.
 * @property {string|null} model      Model id this worker has loaded, or null.
 * @property {boolean} paused         Worker declined work (battery, user active, toggled off).
 * @property {number} maxConcurrent   Jobs this worker will accept at once.
 * @property {number} inFlight        Jobs currently dispatched to it.
 * @property {number} lastAssignedAt  Monotonic counter for LRU tie-breaking.
 * @property {number} completed       Lifetime completed jobs (status display).
 */

export class WorkerRegistry {
  constructor() {
    /** @type {Map<string, WorkerRecord>} */
    this.workers = new Map();
    this.assignCounter = 0;
  }

  /**
   * Add a worker to the pool. It cannot receive jobs until `setModel` marks a
   * model as loaded — connecting is not the same as being ready.
   *
   * @param {{ id: string, send: SendFn, label?: string, maxConcurrent?: number }} options
   * @returns {WorkerRecord}
   */
  register({ id, send, label = 'browser', maxConcurrent = 1 }) {
    const record = {
      id,
      send,
      label,
      model: null,
      paused: false,
      maxConcurrent: Math.max(1, Math.floor(maxConcurrent)),
      inFlight: 0,
      lastAssignedAt: 0,
      completed: 0,
    };
    this.workers.set(id, record);
    return record;
  }

  /** @param {string} id */
  get(id) {
    return this.workers.get(id) ?? null;
  }

  /**
   * Mark which model this worker has loaded and can serve. Passing null takes
   * it out of rotation without disconnecting it (e.g. while switching models).
   *
   * @param {string} id
   * @param {string|null} model
   */
  setModel(id, model) {
    const worker = this.workers.get(id);
    if (worker) worker.model = model;
  }

  /**
   * Workers pause themselves — on battery, while the person is typing, or when
   * the toggle is off. In-flight jobs are allowed to finish.
   *
   * @param {string} id
   * @param {boolean} paused
   */
  setPaused(id, paused) {
    const worker = this.workers.get(id);
    if (worker) worker.paused = Boolean(paused);
  }

  /**
   * Remove a worker. Returns its record so the caller can requeue whatever it
   * was working on.
   *
   * @param {string} id
   * @returns {WorkerRecord|null}
   */
  remove(id) {
    const worker = this.workers.get(id);
    this.workers.delete(id);
    return worker ?? null;
  }

  /**
   * Reserve the best available worker for a model, incrementing its in-flight
   * count. Returns null when every capable worker is busy or paused.
   *
   * @param {string} model
   * @returns {WorkerRecord|null}
   */
  claim(model) {
    let best = null;
    for (const worker of this.workers.values()) {
      if (worker.model !== model) continue;
      if (worker.paused) continue;
      if (worker.inFlight >= worker.maxConcurrent) continue;

      if (
        best === null ||
        worker.inFlight < best.inFlight ||
        (worker.inFlight === best.inFlight && worker.lastAssignedAt < best.lastAssignedAt)
      ) {
        best = worker;
      }
    }

    if (best === null) return null;
    best.inFlight += 1;
    this.assignCounter += 1;
    best.lastAssignedAt = this.assignCounter;
    return best;
  }

  /**
   * Return capacity after a job ends.
   *
   * @param {string} id
   * @param {boolean} [succeeded]
   */
  release(id, succeeded = true) {
    const worker = this.workers.get(id);
    if (!worker) return;
    worker.inFlight = Math.max(0, worker.inFlight - 1);
    if (succeeded) worker.completed += 1;
  }

  /** Models at least one non-paused worker can serve right now. */
  availableModels() {
    const models = new Set();
    for (const worker of this.workers.values()) {
      if (worker.model && !worker.paused) models.add(worker.model);
    }
    return [...models].sort();
  }

  /** True if any worker could eventually serve this model, even if busy now. */
  canServe(model) {
    for (const worker of this.workers.values()) {
      if (worker.model === model && !worker.paused) return true;
    }
    return false;
  }

  /** Serialisable pool state for `/status`. */
  snapshot() {
    return [...this.workers.values()]
      .map((worker) => ({
        id: worker.id,
        label: worker.label,
        model: worker.model,
        paused: worker.paused,
        inFlight: worker.inFlight,
        maxConcurrent: worker.maxConcurrent,
        completed: worker.completed,
      }))
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }

  get size() {
    return this.workers.size;
  }
}
