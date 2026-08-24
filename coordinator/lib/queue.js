/**
 * queue.js
 *
 * Routes chat completion requests to whichever browser tab is free.
 *
 * This is the piece that makes data parallelism worth doing: a job is bound to
 * a worker only at dispatch time, so a tab that closes mid-generation costs one
 * retry rather than the whole request. Jobs that have already streamed output
 * to the client cannot be retried — the client has seen those tokens — so they
 * fail rather than silently duplicating text.
 */

/**
 * @typedef {object} Job
 * @property {string} id
 * @property {string} model
 * @property {object} payload           Body forwarded to the worker.
 * @property {(delta: string) => void} onChunk
 * @property {(info: { workerId: string }) => void} onDone
 * @property {(error: Error) => void} onError
 * @property {string|null} workerId     Worker currently running it.
 * @property {boolean} streamed         True once any delta reached the client.
 * @property {number} attempts
 * @property {ReturnType<typeof setTimeout>|null} timer
 */

export class JobQueue {
  /**
   * @param {object} options
   * @param {import('./registry.js').WorkerRegistry} options.registry
   * @param {number} [options.maxQueueDepth]  Reject beyond this many waiting jobs.
   * @param {number} [options.jobTimeoutMs]   Give up on a silent worker after this long.
   * @param {number} [options.maxAttempts]    Dispatch attempts before failing a job.
   */
  constructor({ registry, maxQueueDepth = 64, jobTimeoutMs = 120_000, maxAttempts = 2 }) {
    this.registry = registry;
    this.maxQueueDepth = maxQueueDepth;
    this.jobTimeoutMs = jobTimeoutMs;
    this.maxAttempts = maxAttempts;

    /** @type {Job[]} */
    this.waiting = [];
    /** @type {Map<string, Job>} */
    this.active = new Map();
  }

  /**
   * Accept a job. It dispatches immediately if a worker is free, otherwise it
   * waits. Throws if nothing can ever serve the model, or if the queue is full.
   *
   * @param {Omit<Job, 'workerId'|'streamed'|'attempts'|'timer'>} job
   */
  submit(job) {
    if (!this.registry.canServe(job.model)) {
      throw Object.assign(new Error(`no worker on this mesh has "${job.model}" loaded`), {
        status: 503,
        code: 'model_unavailable',
      });
    }
    if (this.waiting.length >= this.maxQueueDepth) {
      throw Object.assign(new Error('mesh is saturated — try again shortly'), {
        status: 429,
        code: 'queue_full',
      });
    }

    /** @type {Job} */
    const entry = { ...job, workerId: null, streamed: false, attempts: 0, timer: null };
    this.waiting.push(entry);
    this.pump();
    return entry;
  }

  /** Dispatch as many waiting jobs as there is free capacity for. */
  pump() {
    if (this.waiting.length === 0) return;

    const stillWaiting = [];
    for (const job of this.waiting) {
      const worker = this.registry.claim(job.model);
      if (!worker) {
        stillWaiting.push(job);
        continue;
      }
      this.dispatch(job, worker);
    }
    this.waiting = stillWaiting;
  }

  /**
   * @param {Job} job
   * @param {import('./registry.js').WorkerRecord} worker
   */
  dispatch(job, worker) {
    job.workerId = worker.id;
    job.attempts += 1;
    this.active.set(job.id, job);

    if (this.jobTimeoutMs > 0) {
      job.timer = setTimeout(() => {
        this.fail(job.id, new Error(`worker ${worker.id} timed out after ${this.jobTimeoutMs}ms`));
      }, this.jobTimeoutMs);
      // Never hold the process open just for a job deadline.
      job.timer.unref?.();
    }

    worker.send({
      type: 'job',
      jobId: job.id,
      model: job.model,
      payload: job.payload,
    });
  }

  /**
   * Forward a token delta to the client. Marks the job unretryable, since the
   * client has now seen partial output.
   *
   * @param {string} jobId
   * @param {string} delta
   */
  chunk(jobId, delta) {
    const job = this.active.get(jobId);
    if (!job) return;
    if (delta.length > 0) {
      job.streamed = true;
      job.onChunk(delta);
    }
    this.refreshTimeout(job);
  }

  /** @param {string} jobId */
  complete(jobId) {
    const job = this.active.get(jobId);
    if (!job) return;
    this.finish(job, true);
    job.onDone({ workerId: job.workerId ?? 'unknown' });
    this.pump();
  }

  /**
   * Fail a job. If it never produced output and has attempts left, it goes back
   * to the queue for a different worker instead of surfacing an error.
   *
   * @param {string} jobId
   * @param {Error} error
   */
  fail(jobId, error) {
    const job = this.active.get(jobId);
    if (!job) return;
    this.finish(job, false);

    if (!job.streamed && job.attempts < this.maxAttempts) {
      job.workerId = null;
      this.waiting.unshift(job); // retry ahead of newer work
      this.pump();
      return;
    }

    job.onError(error);
    this.pump();
  }

  /**
   * Handle a worker vanishing. Anything it held is failed, which requeues the
   * jobs that had not started streaming.
   *
   * @param {string} workerId
   */
  releaseWorker(workerId) {
    for (const job of [...this.active.values()]) {
      if (job.workerId === workerId) {
        this.fail(job.id, new Error(`worker ${workerId} disconnected mid-request`));
      }
    }
    this.pump();
  }

  /** @param {string} jobId */
  cancel(jobId) {
    const job = this.active.get(jobId);
    if (job) {
      const worker = job.workerId ? this.registry.get(job.workerId) : null;
      worker?.send({ type: 'cancel', jobId });
      this.finish(job, false);
      return;
    }
    this.waiting = this.waiting.filter((entry) => entry.id !== jobId);
  }

  /**
   * Clear a job's dispatch state and hand its slot back to the worker.
   *
   * @param {Job} job
   * @param {boolean} succeeded
   */
  finish(job, succeeded) {
    if (job.timer) {
      clearTimeout(job.timer);
      job.timer = null;
    }
    this.active.delete(job.id);
    if (job.workerId) this.registry.release(job.workerId, succeeded);
  }

  /**
   * A worker that is still streaming is still alive — push the deadline out so
   * long generations are not killed for being long.
   *
   * @param {Job} job
   */
  refreshTimeout(job) {
    if (!job.timer || this.jobTimeoutMs <= 0) return;
    clearTimeout(job.timer);
    job.timer = setTimeout(() => {
      this.fail(job.id, new Error(`worker ${job.workerId} stopped responding mid-stream`));
    }, this.jobTimeoutMs);
    job.timer.unref?.();
  }

  get stats() {
    return { waiting: this.waiting.length, active: this.active.size };
  }
}
