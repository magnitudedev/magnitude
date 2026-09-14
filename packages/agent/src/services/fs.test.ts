import { afterAll, beforeAll, describe, expect, mock, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve } from 'path'
import { Effect, Exit } from 'effect'

// Point the Fs service at the vendored ripgrep binary instead of the embedded
// payload, which is only resolvable from a compiled release build.
const vendoredRg = resolve(import.meta.dir, '../../../ripgrep/bin', process.platform === 'win32' ? 'rg.exe' : 'rg')
mock.module('@magnitudedev/ripgrep', () => ({
  resolveRgPath: async () => vendoredRg,
}))

const { Fs, FsLive } = await import('./fs')

const search = (pattern: string, searchPath: string) =>
  Effect.runPromiseExit(
    Effect.gen(function* () {
      const fs = yield* Fs
      return yield* fs.search({ pattern, searchPath, limit: 50 })
    }).pipe(Effect.provide(FsLive)),
  )

describe('Fs.search', () => {
  let dir: string

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), 'fs-search-'))
    writeFileSync(join(dir, 'a.txt'), 'hello world\n')
  })

  afterAll(() => {
    rmSync(dir, { recursive: true, force: true })
  })

  test('returns matches for a valid pattern', async () => {
    const exit = await search('hello', dir)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value).toEqual([{ file: 'a.txt', match: '1|hello world' }])
    }
  })

  test('returns no matches without failing when nothing matches', async () => {
    const exit = await search('absent', dir)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) expect(exit.value).toEqual([])
  })

  test('fails with the ripgrep error instead of reporting no matches for an invalid regex', async () => {
    const exit = await search('(', dir)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      const message = JSON.stringify(exit.cause, (_key, value) => value instanceof Error ? value.message : value)
      expect(message).toContain('regex parse error')
    }
  })
})
