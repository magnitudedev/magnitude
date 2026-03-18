import { describe, expect, test } from 'bun:test'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import { BackgroundProcessesProjection, getProcessesForFork, type BackgroundProcessesState } from '../background-processes'
import type { AppEvent } from '../../events'

const makeState = async (events: AppEvent[]) => {
  const projectionBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    projectionBusLayer,
    Layer.provide(BackgroundProcessesProjection.Layer, projectionBusLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* ProjectionBusTag<AppEvent>()
    const projection = yield* BackgroundProcessesProjection.Tag

    for (const [index, event] of events.entries()) {
      yield* bus.processEvent({
        ...event,
        timestamp: index + 1,
      })
    }

    return yield* projection.get
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as unknown as Effect.Effect<BackgroundProcessesState, never, never>
  )
}

describe('BackgroundProcessesProjection', () => {
  test('background_process_registered creates a process entry with correct initial state', async () => {
    const state = await makeState([{
      type: 'background_process_registered',
      forkId: 'fork-a',
      pid: 123,
      command: 'npm run dev',
      sourceTurnId: 'turn-1',
      startedAt: 1000,
      initialStdout: 'boot\n',
      initialStderr: '',
    }])

    const process = getProcessesForFork(state, 'fork-a').get(123)
    expect(process).toBeDefined()
    expect(process).toMatchObject({
      pid: 123,
      command: 'npm run dev',
      status: 'running',
      startedAt: 1000,
      demoted: false,
      unreadStdout: 'boot\n',
      unreadStderr: '',
      totalStdoutLines: 1,
      newStdoutLines: 1,
    })
  })

  test('background_process_output (inline) appends to unread stdout/stderr and line counts', async () => {
    const state = await makeState([
      {
        type: 'background_process_registered',
        forkId: 'fork-a',
        pid: 123,
        command: 'cmd',
        sourceTurnId: 'turn-1',
        startedAt: 1000,
        initialStdout: 'a\n',
        initialStderr: 'b\n',
      },
      {
        type: 'background_process_output',
        forkId: 'fork-a',
        pid: 123,
        mode: 'inline',
        stdoutChunk: 'c\n',
        stderrChunk: 'd\n',
      },
    ] satisfies AppEvent[])

    const process = getProcessesForFork(state, 'fork-a').get(123)!
    expect(process.unreadStdout).toBe('a\nc\n')
    expect(process.unreadStderr).toBe('b\nd\n')
    expect(process.totalStdoutLines).toBe(2)
    expect(process.newStdoutLines).toBe(2)
  })

  test('background_process_demoted + tail output updates demoted metadata and replaces unread', async () => {
    const state = await makeState([
      {
        type: 'background_process_registered',
        forkId: 'fork-a',
        pid: 123,
        command: 'cmd',
        sourceTurnId: 'turn-1',
        startedAt: 1000,
        initialStdout: '',
        initialStderr: '',
      },
      {
        type: 'background_process_demoted',
        forkId: 'fork-a',
        pid: 123,
        stdoutFilePath: '/tmp/stdout.log',
        stderrFilePath: '/tmp/stderr.log',
      },
      {
        type: 'background_process_output',
        forkId: 'fork-a',
        pid: 123,
        mode: 'tail',
        stdoutChunk: 'tail text\n',
        stderrChunk: '',
        stdoutLines: 3,
        stderrLines: 0,
      },
    ] satisfies AppEvent[])

    const process = getProcessesForFork(state, 'fork-a').get(123)!
    expect(process.demoted).toBe(true)
    expect(process.stdoutFilePath).toBe('/tmp/stdout.log')
    expect(process.totalStdoutLines).toBe(3)
    expect(process.newStdoutLines).toBe(3)
    expect(process.unreadStdout).toBe('tail text\n')
  })

  test('background_process_exited updates status, exitCode, signal', async () => {
    const state = await makeState([
      {
        type: 'background_process_registered',
        forkId: 'fork-a',
        pid: 123,
        command: 'cmd',
        sourceTurnId: 'turn-1',
        startedAt: 1000,
        initialStdout: '',
        initialStderr: '',
      },
      {
        type: 'background_process_exited',
        forkId: 'fork-a',
        pid: 123,
        exitCode: 0,
        signal: null,
        status: 'exited',

      },
    ] satisfies AppEvent[])

    const process = getProcessesForFork(state, 'fork-a').get(123)!
    expect(process.status).toBe('exited')
    expect(process.exitCode).toBe(0)
    expect(process.signal).toBeNull()
    expect(process.unreadStdout).toBe('')
  })

  test('observations_captured clears unread fields and removes exited processes', async () => {
    const state = await makeState([
      {
        type: 'background_process_registered',
        forkId: 'fork-a',
        pid: 1,
        command: 'run',
        sourceTurnId: 'turn-1',
        startedAt: 1000,
        initialStdout: 'running',
        initialStderr: '',
      },
      {
        type: 'background_process_registered',
        forkId: 'fork-a',
        pid: 2,
        command: 'done',
        sourceTurnId: 'turn-1',
        startedAt: 1000,
        initialStdout: '',
        initialStderr: '',
      },
      {
        type: 'background_process_exited',
        forkId: 'fork-a',
        pid: 2,
        exitCode: 0,
        signal: null,
        status: 'exited',

      },
      {
        type: 'observations_captured',
        forkId: 'fork-a',
        turnId: 'turn-2',
        parts: [],
      },
    ] satisfies AppEvent[])

    const fork = getProcessesForFork(state, 'fork-a')
    expect(fork.get(1)?.unreadStdout).toBe('')
    expect(fork.get(1)?.newStdoutLines).toBe(0)
    expect(fork.has(2)).toBe(false)
  })
})