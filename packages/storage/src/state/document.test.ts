import { BunFileSystem, BunPath } from '@effect/platform-bun'
import * as FileSystem from '@effect/platform/FileSystem'
import { Deferred, Effect, Fiber, Layer, Schema } from 'effect'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { makeStateDocument } from './document'

const ValueSchema = Schema.Struct({ value: Schema.Number })
const base = Layer.mergeAll(BunFileSystem.layer, BunPath.layer)

describe('state document', () => {
  let directory: string
  let path: string

  beforeEach(async () => {
    directory = await mkdtemp(join(tmpdir(), 'magnitude-state-document-'))
    path = join(directory, 'state.json')
  })

  afterEach(async () => {
    await rm(directory, { recursive: true, force: true })
  })

  const run = <A, E>(effect: Effect.Effect<A, E, never>) => Effect.runPromise(effect)
  const make = () => makeStateDocument({
    path,
    schema: ValueSchema,
    initial: () => ({ value: 0 }),
    equivalence: Schema.equivalence(ValueSchema),
  }).pipe(Effect.provide(base))

  it('creates missing state and persists before publication', async () => {
    const result = await run(Effect.gen(function* () {
      const state = yield* make()
      const before = yield* state.get
      const updated = yield* state.update((current) => ({ value: current.value + 1 }))
      const after = yield* state.get
      return { before, updated, after }
    }))

    expect(result).toEqual({
      before: { value: 0 },
      updated: { value: 1 },
      after: { value: 1 },
    })
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ value: 1 })
  })

  it('rereads under the path lock before every mutation', async () => {
    const state = await run(make())
    await writeFile(path, JSON.stringify({ value: 10 }))
    await run(state.update((current) => ({ value: current.value + 1 })))
    expect(await run(state.get)).toEqual({ value: 11 })
  })

  it('publishes resident state before observing interruption after the atomic rename', async () => {
    await writeFile(path, `${JSON.stringify({ value: 0 }, null, 2)}\n`)
    const result = await run(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const renamed = yield* Deferred.make<void>()
      const delayedFs: FileSystem.FileSystem = {
        ...fs,
        rename: (from, to) => fs.rename(from, to).pipe(
          Effect.tap(() => Deferred.succeed(renamed, undefined)),
          Effect.zipRight(Effect.sleep('50 millis')),
        ),
      }
      const state = yield* makeStateDocument({
        path,
        schema: ValueSchema,
        initial: () => ({ value: 0 }),
        equivalence: Schema.equivalence(ValueSchema),
      }).pipe(Effect.provideService(FileSystem.FileSystem, delayedFs))
      const fiber = yield* state.update(() => ({ value: 1 })).pipe(Effect.fork)
      yield* Deferred.await(renamed)
      const exit = yield* Fiber.interrupt(fiber)
      const resident = yield* state.get
      return { exit, resident }
    }).pipe(Effect.provide(base)))

    expect(result.exit._tag).toBe('Failure')
    expect(result.resident).toEqual({ value: 1 })
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ value: 1 })
  })

  it('preserves malformed input before installing the default', async () => {
    await writeFile(path, '{')
    const state = await run(make())
    expect(await run(state.get)).toEqual({ value: 0 })
    await expect(
      Array.fromAsync(new Bun.Glob('state.json.corrupt-*').scan(directory)),
    ).resolves.toHaveLength(1)
  })
})
