import { afterEach, describe, expect, it } from 'vitest'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { Effect } from 'effect'
import { Fs, FsLive, FsSearchError, makeFsLive } from '../src/services/fs'

const temporaryDirectories: string[] = []

function byteStream(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text))
      controller.close()
    },
  })
}

function fakeRgProcess(options: {
  readonly stdout?: string
  readonly stderr?: string
  readonly exitCode?: number
  readonly hangs?: boolean
}) {
  let exitCode: number | null = options.hangs ? null : (options.exitCode ?? 0)
  let resolveExit: (code: number) => void = () => {}
  let killCount = 0
  const exited = options.hangs
    ? new Promise<number>((resolve) => { resolveExit = resolve })
    : Promise.resolve(exitCode as number)
  const pendingStream = () => new ReadableStream<Uint8Array>({})
  const proc = {
    stdout: options.hangs ? pendingStream() : byteStream(options.stdout ?? ''),
    stderr: options.hangs ? pendingStream() : byteStream(options.stderr ?? ''),
    exited,
    get exitCode() { return exitCode },
    kill() {
      killCount += 1
      if (exitCode === null) {
        exitCode = 143
        resolveExit(exitCode)
      }
    },
  }
  return { proc, killCount: () => killCount }
}

async function fixture(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'magnitude-rg-search-'))
  temporaryDirectories.push(directory)
  return directory
}

function search(
  params: { pattern: string; searchPath: string; glob?: string; limit: number },
  layer = FsLive,
) {
  return Effect.runPromise(Effect.gen(function* () {
    const fs = yield* Fs
    return yield* fs.search(params)
  }).pipe(Effect.provide(layer)))
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) =>
    rm(directory, { recursive: true, force: true }),
  ))
})

describe('FsLive search', () => {
  async function processFailureMessage(stderr: string): Promise<string> {
    const directory = await fixture()
    const process = fakeRgProcess({ stderr, exitCode: 2 })
    const layer = makeFsLive({
      resolveRgPath: async () => 'fake-rg',
      spawnRg: () => process.proc,
    })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const fs = yield* Fs
      return yield* fs.search({ pattern: 'anything', searchPath: directory, limit: 50 })
    }).pipe(Effect.provide(layer), Effect.either))
    if (result._tag === 'Right') throw new Error('Expected search to fail')
    if (!(result.left instanceof FsSearchError)) throw result.left
    return result.left.message
  }

  it('returns matches after ripgrep exits naturally', async () => {
    const directory = await fixture()
    await writeFile(join(directory, 'models.ts'), 'const model = "sonnet"\n')
    await expect(search({ pattern: 'sonnet', searchPath: directory, limit: 50 })).resolves.toEqual([
      { file: 'models.ts', match: '1|const model = "sonnet"' },
    ])
  })

  it('treats ripgrep exit code 1 as an empty successful result', async () => {
    const directory = await fixture()
    await writeFile(join(directory, 'models.ts'), 'const model = "sonnet"\n')
    await expect(search({ pattern: 'not-present', searchPath: directory, limit: 50 })).resolves.toEqual([])
  })

  it('stops at the requested match limit and returns partial results', async () => {
    const directory = await fixture()
    await writeFile(join(directory, 'many.txt'), Array.from({ length: 200 }, (_, index) => `match-${index}`).join('\n'))
    await expect(search({ pattern: 'match-', searchPath: directory, limit: 7 })).resolves.toHaveLength(7)
  })

  it('preserves a concise diagnostic for ripgrep process failures', async () => {
    const directory = await fixture()
    await expect(search({ pattern: '(', searchPath: directory, limit: 50 })).rejects.toThrow(
      /Ripgrep exited with code 2: rg: regex parse error: … \[truncated\]/,
    )
  })

  it('preserves a short single-line process diagnostic', async () => {
    await expect(processFailureMessage('rg: permission denied\n')).resolves.toBe(
      'Ripgrep exited with code 2: rg: permission denied',
    )
  })

  it('surfaces only the first nonempty stderr line', async () => {
    const message = await processFailureMessage(
      '\n  \nrg: first unreadable path\nrg: second unreadable path\n',
    )
    expect(message).toBe('Ripgrep exited with code 2: rg: first unreadable path … [truncated]')
    expect(message).not.toContain('second unreadable')
  })

  it('caps an oversized stderr line including the truncation marker', async () => {
    const prefix = 'Ripgrep exited with code 2: '
    const message = await processFailureMessage(`rg: ${'x'.repeat(1_000)}\n`)
    const detail = message.slice(prefix.length)
    expect(detail).toHaveLength(300)
    expect(detail).toMatch(/ … \[truncated\]$/)
  })

  it('omits detail for whitespace-only stderr', async () => {
    await expect(processFailureMessage('\n \t\n')).resolves.toBe('Ripgrep exited with code 2')
  })

  it('returns parsed matches when ripgrep also reports an unreadable path', async () => {
    const directory = await fixture()
    const match = JSON.stringify({
      type: 'match',
      data: {
        path: { text: join(directory, 'models.ts') },
        lines: { text: 'const model = "sonnet"\n' },
        line_number: 1,
      },
    })
    const process = fakeRgProcess({
      stdout: `${match}\n`,
      stderr: 'rg: unreadable: Permission denied (os error 13)\n',
      exitCode: 2,
    })

    const layer = makeFsLive({
      resolveRgPath: async () => 'fake-rg',
      spawnRg: () => process.proc,
    })
    await expect(search({ pattern: 'sonnet', searchPath: directory, limit: 50 }, layer)).resolves.toEqual([
      { file: 'models.ts', match: '1|const model = "sonnet"' },
    ])
  })

  it('times out, terminates, and reaps a stuck process', async () => {
    const directory = await fixture()
    const process = fakeRgProcess({ hangs: true })

    const layer = makeFsLive({
      resolveRgPath: async () => 'fake-rg',
      spawnRg: () => process.proc,
      searchTimeoutMs: 1_000,
    })
    await expect(search({ pattern: 'anything', searchPath: directory, limit: 50 }, layer)).rejects.toThrow(
      'Search timed out after 1s',
    )

    expect(process.killCount()).toBeGreaterThan(0)
  })
})
