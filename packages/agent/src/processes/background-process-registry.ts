import { Context, Effect } from 'effect'
import fs from 'fs'
import os from 'os'
import path from 'path'
import type { ChildProcess } from 'child_process'
import type {
  AppEvent,
  BackgroundProcessAutoKilled,
  BackgroundProcessDemoted,
  BackgroundProcessExited,
  BackgroundProcessOutput,
  BackgroundProcessPromoted,
  BackgroundProcessRegistered,
} from '../events'
import type {
  BackgroundProcessRecord,
  RegisterProcessInput,
} from './types'

import { AUTO_KILL_BUFFER_S, DEMOTION_THRESHOLD_CHARS, TAIL_MAX_CHARS, SHUTDOWN_GRACE_MS, OUTPUT_DIR_NAME } from './constants'

const OUTPUT_DIR = path.join(os.homedir(), OUTPUT_DIR_NAME)

export interface BackgroundProcessRegistry {
  readonly register: (input: RegisterProcessInput) => Effect.Effect<void>
  readonly flush: (forkId: string | null) => Effect.Effect<void>
  readonly listByFork: (forkId: string | null) => Effect.Effect<BackgroundProcessRecord[]>
  readonly getByPid: (pid: number) => Effect.Effect<BackgroundProcessRecord | undefined>
  readonly promote: (pid: number) => Effect.Effect<{ success: true } | { success: false; reason: 'not_found' | 'already_exited' | 'already_background' }>
  readonly cleanupFork: (forkId: string | null) => Effect.Effect<void>
  readonly shutdownAll: () => Effect.Effect<void>
}

export class BackgroundProcessRegistryTag extends Context.Tag('BackgroundProcessRegistry')<
  BackgroundProcessRegistryTag,
  BackgroundProcessRegistry
>() {}

export function make(
  publish: (event: AppEvent) => void,
): BackgroundProcessRegistry {
  const records = new Map<number, BackgroundProcessRecord>()
  const finalized = new Set<number>()

  const ensureOutputDir = () => {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }

  const countLines = (value: string): number =>
    value.length === 0 ? 0 : (value.match(/\n/g)?.length ?? 0)

  const computeTail = (buffer: string, maxChars: number): string => {
    const lines = buffer.split('\n')
    const result: string[] = []
    let chars = 0
    for (let i = lines.length - 1; i >= 0; i--) {
      const lineChars = lines[i]!.length + 1
      if (chars + lineChars > maxChars && result.length > 0) break
      result.unshift(lines[i]!)
      chars += lineChars
    }
    return result.join('\n')
  }

  const appendBuffer = (
    record: BackgroundProcessRecord,
    stream: 'stdout' | 'stderr',
    chunk: string,
  ) => {
    if (chunk.length === 0) return
    if (stream === 'stdout') {
      record.stdoutBuffer += chunk
    } else {
      record.stderrBuffer += chunk
    }
  }

  const clearAutoKillTimer = (record: BackgroundProcessRecord) => {
    if (record.autoKillTimer) {
      clearTimeout(record.autoKillTimer)
      record.autoKillTimer = undefined
    }
  }

  const cleanupFiles = (record: BackgroundProcessRecord) => {
    for (const filePath of [record.stdoutFilePath, record.stderrFilePath]) {
      if (!filePath) continue
      try {
        fs.rmSync(filePath, { force: true })
      } catch {
        // best effort cleanup
      }
    }
  }

  const flushRecord = (record: BackgroundProcessRecord) => {
    const stdoutBuffer = record.stdoutBuffer
    const stderrBuffer = record.stderrBuffer
    const totalChars = stdoutBuffer.length + stderrBuffer.length

    if (totalChars === 0) return

    if (!record.demoted && totalChars > DEMOTION_THRESHOLD_CHARS) {
      ensureOutputDir()
      const stdoutFilePath = path.join(OUTPUT_DIR, `${record.pid}-stdout.log`)
      const stderrFilePath = path.join(OUTPUT_DIR, `${record.pid}-stderr.log`)
      fs.writeFileSync(stdoutFilePath, stdoutBuffer)
      fs.writeFileSync(stderrFilePath, stderrBuffer)
      record.demoted = true
      record.stdoutFilePath = stdoutFilePath
      record.stderrFilePath = stderrFilePath

      const demotedEvent: BackgroundProcessDemoted = {
        type: 'background_process_demoted',
        forkId: record.forkId,
        pid: record.pid,
        stdoutFilePath,
        stderrFilePath,
      }
      publish(demotedEvent)

      const outputEvent: BackgroundProcessOutput = {
        type: 'background_process_output',
        forkId: record.forkId,
        pid: record.pid,
        mode: 'tail',
        stdoutChunk: computeTail(stdoutBuffer, TAIL_MAX_CHARS),
        stderrChunk: computeTail(stderrBuffer, TAIL_MAX_CHARS),
        stdoutLines: countLines(stdoutBuffer),
        stderrLines: countLines(stderrBuffer),
      }
      publish(outputEvent)
    } else if (!record.demoted) {
      const outputEvent: BackgroundProcessOutput = {
        type: 'background_process_output',
        forkId: record.forkId,
        pid: record.pid,
        mode: 'inline',
        stdoutChunk: stdoutBuffer,
        stderrChunk: stderrBuffer,
      }
      publish(outputEvent)
    } else {
      ensureOutputDir()
      const stdoutFilePath = record.stdoutFilePath ?? path.join(OUTPUT_DIR, `${record.pid}-stdout.log`)
      const stderrFilePath = record.stderrFilePath ?? path.join(OUTPUT_DIR, `${record.pid}-stderr.log`)
      fs.appendFileSync(stdoutFilePath, stdoutBuffer)
      fs.appendFileSync(stderrFilePath, stderrBuffer)
      record.stdoutFilePath = stdoutFilePath
      record.stderrFilePath = stderrFilePath

      const outputEvent: BackgroundProcessOutput = {
        type: 'background_process_output',
        forkId: record.forkId,
        pid: record.pid,
        mode: 'tail',
        stdoutChunk: computeTail(stdoutBuffer, TAIL_MAX_CHARS),
        stderrChunk: computeTail(stderrBuffer, TAIL_MAX_CHARS),
        stdoutLines: countLines(stdoutBuffer),
        stderrLines: countLines(stderrBuffer),
      }
      publish(outputEvent)
    }

    record.stdoutBuffer = ''
    record.stderrBuffer = ''
  }

  const finalizeExit = (
    record: BackgroundProcessRecord,
    exitCode: number | null,
    signal: string | null,
  ) => {
    if (finalized.has(record.pid)) return
    finalized.add(record.pid)

    clearAutoKillTimer(record)

    record.status = 'exited'
    record.exitCode = exitCode
    record.signal = signal

    flushRecord(record)

    const event: BackgroundProcessExited = {
      type: 'background_process_exited',
      forkId: record.forkId,
      pid: record.pid,
      exitCode,
      signal,
      status: signal ? 'killed' : 'exited',
    }
    publish(event)
  }

  const attachStreamListener = (
    record: BackgroundProcessRecord,
    stream: 'stdout' | 'stderr',
    childStream: ChildProcess['stdout'] | ChildProcess['stderr'] | null,
  ) => {
    childStream?.on('data', (chunk: Buffer | string) => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8')
      appendBuffer(record, stream, text)
    })
  }

  const registerRecord = (input: RegisterProcessInput) => {
    const record: BackgroundProcessRecord = {
      pid: input.pid,
      forkId: input.forkId,
      turnId: input.turnId,
      command: input.command,
      reason: input.reason,
      startedAt: input.startedAt,
      child: input.child,
      timeoutSeconds: input.timeoutSeconds,
      status: 'running',
      exitCode: null,
      signal: null,
      stdoutBuffer: '',
      stderrBuffer: '',
      demoted: false,
      stdoutFilePath: null,
      stderrFilePath: null,
    }

    records.set(record.pid, record)

    if (record.reason === 'timeout_exceeded' && record.timeoutSeconds !== undefined) {
      record.autoKillTimer = setTimeout(() => {
        if (record.status !== 'running') {
          clearAutoKillTimer(record)
          return
        }

        try {
          record.child.kill('SIGTERM')
        } catch {
          // ignore
        }

        const autoKilledEvent: BackgroundProcessAutoKilled = {
          type: 'background_process_auto_killed',
          forkId: record.forkId,
          pid: record.pid,
          command: record.command,
        }
        publish(autoKilledEvent)
        clearAutoKillTimer(record)
      }, (record.timeoutSeconds + AUTO_KILL_BUFFER_S) * 1000)
    }

    const registeredEvent: BackgroundProcessRegistered = {
      type: 'background_process_registered',
      forkId: record.forkId,
      pid: record.pid,
      command: record.command,
      reason: record.reason,
      sourceTurnId: record.turnId,
      startedAt: record.startedAt,
      initialStdout: input.initialStdout,
      initialStderr: input.initialStderr,
    }
    publish(registeredEvent)

    attachStreamListener(record, 'stdout', input.child.stdout)
    attachStreamListener(record, 'stderr', input.child.stderr)

    input.child.once('exit', (code, signal) => {
      finalizeExit(record, code, signal)
    })

    input.child.once('close', (code, signal) => {
      finalizeExit(record, code, signal)
    })

    input.child.once('error', () => {
      finalizeExit(record, record.exitCode, record.signal)
    })

    if (input.child.exitCode !== null || input.child.signalCode !== null) {
      finalizeExit(record, input.child.exitCode, input.child.signalCode)
    }
  }

  const terminateRecord = async (record: BackgroundProcessRecord) => {
    if (record.status !== 'running') return
    clearAutoKillTimer(record)
    try {
      record.child.kill('SIGTERM')
    } catch {
      // ignore
    }

    await new Promise<void>((resolve) => {
      const killTimer = setTimeout(() => {
        try {
          if (record.status === 'running') {
            record.child.kill('SIGKILL')
          }
        } catch {
          // ignore
        }
        resolve()
      }, SHUTDOWN_GRACE_MS)

      const finish = () => {
        clearTimeout(killTimer)
        resolve()
      }

      record.child.once('close', finish)
      record.child.once('exit', finish)
    })
  }

  return {
    register: (input) => Effect.sync(() => {
      registerRecord(input)
    }),

    flush: (forkId) => Effect.sync(() => {
      for (const record of records.values()) {
        if (record.forkId !== forkId || record.status !== 'running') continue
        flushRecord(record)
      }
    }),

    listByFork: (forkId) => Effect.sync(() =>
      Array.from(records.values()).filter(record => record.forkId === forkId)
    ),

    getByPid: (pid) => Effect.sync(() => records.get(pid)),

    promote: (pid) => Effect.sync(() => {
      const record = records.get(pid)
      if (!record) {
        return { success: false as const, reason: 'not_found' as const }
      }
      if (record.status !== 'running') {
        return { success: false as const, reason: 'already_exited' as const }
      }
      if (record.reason === 'background') {
        return { success: false as const, reason: 'already_background' as const }
      }

      record.reason = 'background'
      clearAutoKillTimer(record)

      const promotedEvent: BackgroundProcessPromoted = {
        type: 'background_process_promoted',
        forkId: record.forkId,
        pid: record.pid,
      }
      publish(promotedEvent)
      return { success: true as const }
    }),

    cleanupFork: (forkId) => Effect.promise(async () => {
      const owned = Array.from(records.values()).filter(record => record.forkId === forkId)
      await Promise.all(owned.map(terminateRecord))
    }),

    shutdownAll: () => Effect.promise(async () => {
      const existing = Array.from(records.values())
      await Promise.all(existing.map(terminateRecord))
      for (const record of existing) {
        cleanupFiles(record)
      }
      records.clear()
    }),
  }
}