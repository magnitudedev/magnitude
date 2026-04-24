import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'
import { Fs } from '../../services/fs'
import { createVirtualFs, createVirtualFsLayer } from '../virtual-fs'

describe('virtual fs layer', () => {
  it.effect('readText and writeFile work', () =>
    Effect.gen(function* () {
      const files = createVirtualFs({ 'src/index.ts': 'export const x = 1' })
      const layer = createVirtualFsLayer(files, '/repo', '/workspace')

      const before = yield* Effect.flatMap(Fs, (fs) => fs.readText('src/index.ts')).pipe(
        Effect.provide(layer),
      )
      expect(before).toBe('export const x = 1')

      yield* Effect.flatMap(Fs, (fs) => fs.writeFile('output.txt', 'content')).pipe(
        Effect.provide(layer),
      )

      const after = yield* Effect.flatMap(Fs, (fs) => fs.readText('output.txt')).pipe(
        Effect.provide(layer),
      )
      expect(after).toBe('content')
    }),
  )

  it.effect('walk and search work', () =>
    Effect.gen(function* () {
      const files = createVirtualFs({
        'src/a.ts': 'alpha\nbeta',
        'src/b.ts': 'gamma\nalphabet',
      })
      const layer = createVirtualFsLayer(files, '/repo', '/workspace')

      const walked = yield* Effect.flatMap(Fs, (fs) => fs.walk('src')).pipe(
        Effect.provide(layer),
      )
      expect(walked.some((entry) => entry.relativePath === 'a.ts')).toBe(true)

      const matches = yield* Effect.flatMap(Fs, (fs) =>
        fs.search({ pattern: 'alpha', searchPath: 'src', limit: 10 }),
      ).pipe(Effect.provide(layer))
      expect(matches).toEqual([
        { file: 'a.ts', match: '1|alpha' },
        { file: 'b.ts', match: '2|alphabet' },
      ])
    }),
  )
})

describe('virtual fs integration with harness', () => {
  it.live('Agent reads seeded file', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<magnitude:invoke tool="read">\n<magnitude:parameter name="path">src/index.ts</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>' }, null)

      yield* harness.user('read file')
      const completed = yield* harness.wait.turnCompleted(null)
      const observation = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolObservation' && e.event.tagName === 'read',
      )

      expect(completed.outcome._tag).toBe('Completed')
      if (observation.event._tag !== 'ToolObservation') {
        throw new Error('Expected ToolObservation')
      }
      expect(observation.event.content).toHaveLength(1)
      expect(observation.event.content[0]).toEqual(expect.objectContaining({ type: 'text' }))
      if (observation.event.content[0]?.type !== 'text') {
        throw new Error('Expected text observation content')
      }
      expect(observation.event.content[0].text).toContain('export const x = 1')
    }).pipe(
      Effect.provide(
        TestHarnessLive({
          files: { 'src/index.ts': 'export const x = 1' },
        }),
      ),
    )
  )

  it.live('Agent writes file visible in h.files', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next(
        { xml: '<magnitude:invoke tool="write">\n<magnitude:parameter name="path">output.txt</magnitude:parameter>\n<magnitude:parameter name="content">content</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>' },
        null,
      )

      yield* harness.user('write file')
      const completed = yield* harness.wait.turnCompleted(null)
      const toolEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.outcome._tag).toBe('Completed')
      if (toolEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(toolEnded.event.result._tag).toBe('Success')
      expect(harness.files.get('output.txt')).toBe('content')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})