/**
 * DetachedProcessProjection (Forked)
 *
 * Tracks detached shell process lifecycle from shell_process_registered +
 * shell_process_exited events. Used for replay, TUI display, and context
 * injection (reminders).
 */

import { Projection } from '@magnitudedev/event-core'
import type { ForkedState } from '@magnitudedev/event-core'
import { Schema, Option } from 'effect'
import type { AppEvent } from '../events'

const TrackedProcessSchema = Schema.Struct({
  pid: Schema.Number,
  command: Schema.String,
  forkId: Schema.NullOr(Schema.String),
  ownerAgentId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  startedAt: Schema.Number,
  stdoutPath: Schema.String,
  stderrPath: Schema.String,
  status: Schema.Literal('running', 'completed', 'killed'),
  exitCode: Schema.NullOr(Schema.Number),
  cpuPercent: Schema.NullOr(Schema.Number),
  rssBytes: Schema.NullOr(Schema.Number),
  lastMetricsAt: Schema.NullOr(Schema.Number),
  peakCpuPercent: Schema.NullOr(Schema.Number),
  peakRssBytes: Schema.NullOr(Schema.Number),
})
export type TrackedProcess = typeof TrackedProcessSchema.Type

export const DetachedProcessStateSchema = Schema.Struct({
  processes: Schema.ReadonlyMap({ key: Schema.Number, value: TrackedProcessSchema }),
})
export type DetachedProcessState = typeof DetachedProcessStateSchema.Type

// ── Helpers ──────────────────────────────────────────────────────────

// Note: Unix-centric. Node child_process uses 128+signal convention.
// 137 = SIGKILL, 143 = SIGTERM, 130 = SIGINT.
function isKillExitCode(code: number): boolean {
  return code === 137 || code === 143 || code === 130
}

function markForkProcessesKilled(
  state: ForkedState<DetachedProcessState>,
  targetForkId: string,
): ForkedState<DetachedProcessState> {
  let mutated = false
  const nextForks = new Map(state.forks)
  for (const [forkId, fork] of state.forks) {
    const processes = new Map(fork.processes)
    let forkMutated = false
    for (const [pid, proc] of processes) {
      if (proc.forkId === targetForkId && proc.status === 'running') {
        processes.set(pid, { ...proc, status: 'killed', exitCode: 137 })
        forkMutated = true
      }
    }
    if (forkMutated) {
      nextForks.set(forkId, { processes })
      mutated = true
    }
  }
  if (!mutated) return state
  return { ...state, forks: nextForks }
}

// ── Projection ───────────────────────────────────────────────────────

export const DetachedProcessProjection = Projection.defineForked<AppEvent>()({
  name: 'DetachedProcess',
  forkState: DetachedProcessStateSchema,

  initialFork: { processes: new Map<number, TrackedProcess>() },

  eventHandlers: {
    turn_outcome: ({ event, fork }) => {
      // Clear completed/killed processes at turn end — they've been shown once.
      const processes = new Map(fork.processes)
      let mutated = false
      for (const [pid, proc] of processes) {
        if (proc.status !== 'running') {
          processes.delete(pid)
          mutated = true
        }
      }
      if (!mutated) return fork
      return { processes }
    },

    shell_process_registered: ({ event, fork }) => {
      const processes = new Map(fork.processes)
      processes.set(event.pid, {
        pid: event.pid,
        command: event.command,
        forkId: event.forkId,
        ownerAgentId: event.ownerAgentId ? Option.some(event.ownerAgentId) : Option.none(),
        startedAt: event.startedAt,
        stdoutPath: event.stdoutPath,
        stderrPath: event.stderrPath,
        status: 'running',
        exitCode: null,
        cpuPercent: null,
        rssBytes: null,
        lastMetricsAt: null,
        peakCpuPercent: null,
        peakRssBytes: null,
      })
      return { processes }
    },

    shell_process_exited: ({ event, fork }) => {
      const proc = fork.processes.get(event.pid)
      if (!proc) return fork

      const processes = new Map(fork.processes)
      processes.set(event.pid, {
        ...proc,
        status: isKillExitCode(event.exitCode) ? 'killed' : 'completed',
        exitCode: event.exitCode,
      })
      return { processes }
    },

    shell_process_metrics: ({ event, fork }) => {
      if (event.samples.length === 0) return fork
      const processes = new Map(fork.processes)
      let mutated = false
      for (const sample of event.samples) {
        const proc = processes.get(sample.pid)
        if (!proc || proc.status !== 'running') continue
        const peakCpu = proc.peakCpuPercent != null ? Math.max(proc.peakCpuPercent, sample.cpuPercent) : sample.cpuPercent
        const peakRss = proc.peakRssBytes != null ? Math.max(proc.peakRssBytes, sample.rssBytes) : sample.rssBytes
        processes.set(sample.pid, {
          ...proc,
          cpuPercent: sample.cpuPercent,
          rssBytes: sample.rssBytes,
          lastMetricsAt: sample.timestamp,
          peakCpuPercent: peakCpu,
          peakRssBytes: peakRss,
        })
        mutated = true
      }
      if (!mutated) return fork
      return { processes }
    },
  },

  globalEventHandlers: {
    agent_killed: ({ event, state }) => markForkProcessesKilled(state, event.forkId),

    worker_user_killed: ({ event, state }) => markForkProcessesKilled(state, event.forkId),
  },
})
