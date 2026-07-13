/**
 * ProcessMetricsWorker
 *
 * Periodic sampler for CPU and memory usage of running detached shell processes.
 * Delegates all `ps` invocation and parsing to the shared `ps-tree` module.
 *
 * Uses `onProjectionsSettled` to lazily start the sampling daemon — this fires
 * after every event once projections settle, on both fresh and resumed sessions.
 * The daemon loop is started exactly once; subsequent settled calls are no-ops.
 *
 * Each cycle:
 *   1. Reads DetachedProcessProjection across all forks for running PIDs.
 *   2. If none running, skips sampling.
 *   3. Publishes a `shell_process_metrics` event with tree-level aggregates.
 */

import { Effect, Schedule, Cause } from 'effect'
import { Worker, type PublishFn, type WorkerReadFn } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { DetachedProcessProjection } from '../projections/detached-process'
import { sampleMetrics, type ProcessMetricsSample } from '../process/ps-tree'

export type { ProcessMetricsSample }

const SAMPLE_INTERVAL = '5 seconds'

// Safety net: ensures the daemon loop is forked exactly once
let samplerStarted = false

// ── Worker ────────────────────────────────────────────────────────────

export const ProcessMetricsWorker = Worker.define<AppEvent>()({
  name: 'ProcessMetricsWorker',

  onProjectionsSettled: ({ publish, read }) =>
    Effect.gen(function* () {
      if (samplerStarted) return
      samplerStarted = true
      yield* Effect.fork(
        Effect.repeat(
          sampleAndPublish(publish, read),
          Schedule.spaced(SAMPLE_INTERVAL),
        ),
      )
    }),
})

// ── Sampling Loop ─────────────────────────────────────────────────────

const sampleAndPublish = (
  publish: PublishFn<AppEvent>,
  read: WorkerReadFn<AppEvent>,
) =>
  Effect.gen(function* () {
    const forkedState = yield* read.allForks(DetachedProcessProjection)

    const runningPids: number[] = []
    for (const [, fork] of forkedState) {
      for (const [pid, proc] of fork.processes) {
        if (proc.status === 'running') {
          runningPids.push(pid)
        }
      }
    }

    if (runningPids.length === 0) {
      return
    }

    const samples = yield* sampleMetrics(runningPids)

    if (samples.length === 0) {
      return
    }

    yield* publish({
      type: 'shell_process_metrics',
      forkId: null,
      samples,
    })
  }).pipe(
    Effect.catchAllCause((cause) =>
      Effect.gen(function* () {
        logger.warn(
          { cause: Cause.pretty(cause) },
          '[ProcessMetricsWorker] Sample cycle failed, skipping',
        )
      }),
    ),
  )
