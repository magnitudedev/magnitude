import { describe, expect, test } from 'bun:test'
import { Effect, Layer, Ref } from 'effect'
import { Fork } from '@magnitudedev/event-core'
import { noopToolContext } from '@magnitudedev/tools'
import { shellTool } from '../shell'
import { WorkingDirectoryTag } from '../../execution/working-directory'
import { ToolReminderTag } from '../../execution/tool-reminder'
import { ToolExecutionContextTag } from '../../execution/tool-execution-context'
import { BackgroundProcessRegistryTag, type BackgroundProcessRegistry } from '../../processes/background-process-registry'

const { ForkContext } = Fork

const makeRegistry = (): BackgroundProcessRegistry => ({
  register: () => Effect.void,
  flush: () => Effect.void,
  listByFork: () => Effect.succeed([]),
  getByPid: () => Effect.succeed(undefined),
  promote: () => Effect.succeed({ success: false as const, reason: 'not_found' as const }),
  cleanupFork: () => Effect.void,
  shutdownAll: () => Effect.void,
})

const makeTestLayer = (remindersRef: Ref.Ref<string[]>) =>
  Layer.mergeAll(
    Layer.succeed(WorkingDirectoryTag, { cwd: '/tmp', workspacePath: '/tmp/workspace' }),
    Layer.succeed(ForkContext, { forkId: 'fork-test' }),
    Layer.succeed(ToolExecutionContextTag, { turnId: 'turn-test' }),
    Layer.succeed(BackgroundProcessRegistryTag, makeRegistry()),
    Layer.succeed(ToolReminderTag, {
      add: (text: string) => Ref.update(remindersRef, (items) => [...items, text]),
    }),
  )

function runPromise<A, E, R>(effect: Effect.Effect<A, E, R>) {
  return Effect.runPromise(effect as unknown as Effect.Effect<A, E, never>)
}

describe('shell background behavior', () => {
  test('fast command returns mode completed with stdout, exitCode=0', async () => {
    const remindersRef = await Effect.runPromise(Ref.make<string[]>([]))

    const result = await Effect.runPromise(
      shellTool.execute({ command: 'echo hello' }, noopToolContext).pipe(
        Effect.provide(makeTestLayer(remindersRef)),
      ) as unknown as Effect.Effect<
        { readonly stdout: string; readonly stderr: string; readonly mode: 'completed'; readonly exitCode: number } |
        { readonly stdout: string; readonly stderr: string; readonly mode: 'detached'; readonly pid: number; readonly reason: 'background' | 'timeout_exceeded' },
        unknown,
        never
      >,
    )

    expect(result.mode).toBe('completed')
    if (result.mode !== 'completed') throw new Error('expected completed')
    expect(result.stdout).toBe('hello\n')
    expect(result.exitCode).toBe(0)
    expect(await Effect.runPromise(Ref.get(remindersRef))).toEqual([])
  })

  test('non-zero exit returns mode completed with exitCode=1', async () => {
    const remindersRef = await Effect.runPromise(Ref.make<string[]>([]))

    const result = await Effect.runPromise(
      shellTool.execute({ command: 'exit 1' }, noopToolContext).pipe(
        Effect.provide(makeTestLayer(remindersRef)),
      ) as unknown as Effect.Effect<
        { readonly stdout: string; readonly stderr: string; readonly mode: 'completed'; readonly exitCode: number } |
        { readonly stdout: string; readonly stderr: string; readonly mode: 'detached'; readonly pid: number; readonly reason: 'background' | 'timeout_exceeded' },
        unknown,
        never
      >,
    )

    expect(result.mode).toBe('completed')
    if (result.mode !== 'completed') throw new Error('expected completed')
    expect(result.exitCode).toBe(1)
    expect(result.stdout).toBe('')
    expect(result.stderr).toBe('')
    expect(await Effect.runPromise(Ref.get(remindersRef))).toEqual([])
  })
})