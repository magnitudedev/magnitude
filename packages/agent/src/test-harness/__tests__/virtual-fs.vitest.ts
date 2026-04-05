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
      yield* harness.script.next({ xml: '<actions><read path="src/index.ts"/></actions><idle/>' }, null)

      yield* harness.user('read file')
      const completed = yield* harness.wait.event('turn_completed', (e) => e.toolCalls.length > 0)

      expect(completed.result.success).toBe(true)
      const readCall = completed.toolCalls.find(
        (c) => c.toolKey === 'fileRead' || (c.group === 'fs' && c.toolName === 'read'),
      )
      expect(readCall?.result.status).toBe('success')
      if (readCall?.result.status === 'success') {
        expect(readCall.result.output).toBe('export const x = 1')
      }
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
        { xml: '<actions><write path="output.txt">content</write></actions><idle/>' },
        null,
      )

      yield* harness.user('write file')
      const completed = yield* harness.wait.event('turn_completed', (e) => e.toolCalls.length > 0)

      expect(completed.result.success).toBe(true)
      const writeCall = completed.toolCalls.find(
        (c) => c.toolKey === 'fileWrite' || (c.group === 'fs' && c.toolName === 'write'),
      )
      expect(writeCall?.result.status).toBe('success')
      expect(harness.files.get('output.txt')).toBe('content')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})