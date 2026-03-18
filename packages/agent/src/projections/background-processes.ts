import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

export interface BackgroundProcessState {
  readonly pid: number
  readonly command: string
  readonly status: 'running' | 'exited' | 'killed'
  readonly startedAt: number
  readonly exitCode: number | null
  readonly signal: string | null
  readonly demoted: boolean
  readonly stdoutFilePath: string | null
  readonly stderrFilePath: string | null
  readonly totalStdoutLines: number
  readonly totalStderrLines: number
  readonly newStdoutLines: number
  readonly newStderrLines: number
  readonly unreadStdout: string
  readonly unreadStderr: string
}

export type BackgroundProcessesState = Map<string | 'root', Map<number, BackgroundProcessState>>

const toForkKey = (forkId: string | null): string | 'root' => forkId ?? 'root'

const countLines = (value: string): number =>
  value.length === 0 ? 0 : (value.match(/\n/g)?.length ?? 0)

const getOrCreateForkProcesses = (
  state: BackgroundProcessesState,
  forkId: string | null,
): Map<number, BackgroundProcessState> => {
  const key = toForkKey(forkId)
  const existing = state.get(key)
  if (existing) return existing
  const created = new Map<number, BackgroundProcessState>()
  state.set(key, created)
  return created
}

export const getProcessesForFork = (
  state: BackgroundProcessesState,
  forkId: string | null,
): Map<number, BackgroundProcessState> => state.get(toForkKey(forkId)) ?? new Map()

export const BackgroundProcessesProjection = Projection.define<AppEvent, BackgroundProcessesState>()({
  name: 'BackgroundProcesses',

  initial: new Map(),

  eventHandlers: {
    background_process_registered: ({ event, state }) => {
      const next = new Map(state)
      const forkProcesses = new Map(getOrCreateForkProcesses(next, event.forkId))

      forkProcesses.set(event.pid, {
        pid: event.pid,
        command: event.command,
        status: 'running',
        startedAt: event.startedAt,
        exitCode: null,
        signal: null,
        demoted: false,
        stdoutFilePath: null,
        stderrFilePath: null,
        totalStdoutLines: countLines(event.initialStdout),
        totalStderrLines: countLines(event.initialStderr),
        newStdoutLines: countLines(event.initialStdout),
        newStderrLines: countLines(event.initialStderr),
        unreadStdout: event.initialStdout,
        unreadStderr: event.initialStderr,
      })

      next.set(toForkKey(event.forkId), forkProcesses)
      return next
    },

    background_process_output: ({ event, state }) => {
      const next = new Map(state)
      const forkKey = toForkKey(event.forkId)
      const existingForkProcesses = next.get(forkKey)
      if (!existingForkProcesses) return state

      const forkProcesses = new Map(existingForkProcesses)
      const existing = forkProcesses.get(event.pid)
      if (!existing) return state

      if (event.mode === 'inline') {
        forkProcesses.set(event.pid, {
          ...existing,
          unreadStdout: existing.unreadStdout + event.stdoutChunk,
          unreadStderr: existing.unreadStderr + event.stderrChunk,
          totalStdoutLines: existing.totalStdoutLines + countLines(event.stdoutChunk),
          totalStderrLines: existing.totalStderrLines + countLines(event.stderrChunk),
          newStdoutLines: existing.newStdoutLines + countLines(event.stdoutChunk),
          newStderrLines: existing.newStderrLines + countLines(event.stderrChunk),
        })
      } else {
        forkProcesses.set(event.pid, {
          ...existing,
          unreadStdout: event.stdoutChunk,
          unreadStderr: event.stderrChunk,
          totalStdoutLines: existing.totalStdoutLines + event.stdoutLines,
          totalStderrLines: existing.totalStderrLines + event.stderrLines,
          newStdoutLines: existing.newStdoutLines + event.stdoutLines,
          newStderrLines: existing.newStderrLines + event.stderrLines,
        })
      }

      next.set(forkKey, forkProcesses)
      return next
    },

    background_process_demoted: ({ event, state }) => {
      const next = new Map(state)
      const forkKey = toForkKey(event.forkId)
      const existingForkProcesses = next.get(forkKey)
      if (!existingForkProcesses) return state

      const forkProcesses = new Map(existingForkProcesses)
      const existing = forkProcesses.get(event.pid)
      if (!existing) return state

      forkProcesses.set(event.pid, {
        ...existing,
        demoted: true,
        stdoutFilePath: event.stdoutFilePath,
        stderrFilePath: event.stderrFilePath,
      })

      next.set(forkKey, forkProcesses)
      return next
    },

    background_process_exited: ({ event, state }) => {
      const next = new Map(state)
      const forkKey = toForkKey(event.forkId)
      const existingForkProcesses = next.get(forkKey)
      if (!existingForkProcesses) return state

      const forkProcesses = new Map(existingForkProcesses)
      const existing = forkProcesses.get(event.pid)
      if (!existing) return state

      forkProcesses.set(event.pid, {
        ...existing,
        status: event.status,
        exitCode: event.exitCode,
        signal: event.signal,
        unreadStdout: existing.unreadStdout + event.stdoutTail,
        unreadStderr: existing.unreadStderr + event.stderrTail,
      })

      next.set(forkKey, forkProcesses)
      return next
    },

    observations_captured: ({ event, state }) => {
      const next = new Map(state)
      const forkKey = toForkKey(event.forkId)
      const existingForkProcesses = next.get(forkKey)
      if (!existingForkProcesses) return state

      const forkProcesses = new Map<number, BackgroundProcessState>()
      for (const [pid, process] of existingForkProcesses.entries()) {
        if (process.status === 'running') {
          forkProcesses.set(pid, {
            ...process,
            unreadStdout: '',
            unreadStderr: '',
            newStdoutLines: 0,
            newStderrLines: 0,
          })
        }
      }

      next.set(forkKey, forkProcesses)
      return next
    },
  },
})