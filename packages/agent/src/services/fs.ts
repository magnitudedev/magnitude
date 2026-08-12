import { readFile, stat } from 'fs/promises'
import { relative, resolve } from 'path'
import { Context, Data, Duration, Effect, Fiber, Layer, Option } from 'effect'
import { resolveRgPath } from '@magnitudedev/ripgrep'
import { walk } from '../util/walk'
import { resolveFileRefPath } from '../scratchpad/file-ref-resolution'

export class FsError extends Data.TaggedError('FsError')<{
  readonly operation: string
  readonly path: string
  readonly cause: unknown
}> {}

export class FsSearchError extends Data.TaggedError('FsSearchError')<{
  readonly reason: 'spawn' | 'read' | 'process' | 'timeout'
  readonly path: string
  readonly message: string
}> {}

export type FsWalkEntry = {
  readonly fullPath: string
  readonly relativePath: string
  readonly name: string
  readonly type: 'file' | 'dir'
  readonly depth: number
}

export type FsSearchMatch = {
  readonly file: string
  readonly match: string
}

const SEARCH_TIMEOUT = 5_000
const TERMINATE_GRACE = 500
const KILL_GRACE = 1_000
// Retained while draining the child stream; only a much smaller summary is model-visible.
const STDERR_LIMIT = 8 * 1024
const STDERR_DIAGNOSTIC_LIMIT = 300
const STDERR_TRUNCATED_MARKER = ' … [truncated]'

type RgProcess = {
  readonly stdout: ReadableStream<Uint8Array>
  readonly stderr: ReadableStream<Uint8Array>
  readonly exited: Promise<number>
  readonly exitCode: number | null
  readonly kill: (signal?: number) => void
}

function summarizeStderr(stderr: string): string {
  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
  const firstLine = lines[0]
  if (!firstLine) return ''

  const truncated = lines.length > 1 || firstLine.length > STDERR_DIAGNOSTIC_LIMIT
  if (!truncated) return firstLine

  const prefixLimit = STDERR_DIAGNOSTIC_LIMIT - STDERR_TRUNCATED_MARKER.length
  return `${firstLine.slice(0, prefixLimit).trimEnd()}${STDERR_TRUNCATED_MARKER}`
}

function readBoundedText(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  limit: number,
): Effect.Effect<string, FsSearchError> {
  return Effect.tryPromise({
    try: async () => {
      const decoder = new TextDecoder()
      let text = ''
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          if (text.length < limit) text += decoder.decode(value, { stream: true }).slice(0, limit - text.length)
        }
        if (text.length < limit) text += decoder.decode().slice(0, limit - text.length)
        return text
      } finally {
        reader.releaseLock()
      }
    },
    catch: (cause) => new FsSearchError({
      reason: 'read',
      path: 'stderr',
      message: `Failed to read ripgrep stderr: ${String(cause)}`,
    }),
  })
}

function cancelReader(reader: ReadableStreamDefaultReader<Uint8Array>): Effect.Effect<void> {
  return Effect.promise(async () => {
    try {
      await reader.cancel()
    } catch {
      // The stream may already be closed or errored.
    }
  }).pipe(Effect.timeout(250), Effect.ignore)
}

function awaitExit(proc: RgProcess, duration: number): Effect.Effect<boolean> {
  return Effect.promise(() => proc.exited).pipe(
    Effect.as(true),
    Effect.timeoutOption(duration),
    Effect.map(Option.isSome),
  )
}

function terminateAndReap(proc: RgProcess): Effect.Effect<void> {
  return Effect.gen(function* () {
    if (proc.exitCode !== null) return
    yield* Effect.sync(() => proc.kill())
    if (yield* awaitExit(proc, TERMINATE_GRACE)) return
    if (proc.exitCode === null) yield* Effect.sync(() => proc.kill(9))
    yield* awaitExit(proc, KILL_GRACE)
  }).pipe(Effect.ignore)
}

function consumeRgStdout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  pattern: string,
  searchPath: string,
  limit: number,
): Effect.Effect<{ readonly matches: readonly FsSearchMatch[]; readonly limited: boolean }, FsSearchError> {
  return Effect.tryPromise({
    try: async () => {
      const matches: FsSearchMatch[] = []
      const decoder = new TextDecoder()
      let buffer = ''
      let limited = false

      const processLine = (line: string): void => {
        if (!line.trim()) return
        try {
          const msg = JSON.parse(line)
          if (msg.type !== 'match') return
          const data = msg.data
          matches.push({
            file: relative(searchPath, data.path.text),
            match: `${data.line_number}|${data.lines.text.replace(/\n$/, '')}`,
          })
          limited = matches.length >= limit
        } catch {
          // Ripgrep's JSON stream may contain an incomplete final line after termination.
        }
      }

      try {
        while (!limited) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let newlineIndex = buffer.indexOf('\n')
          while (newlineIndex !== -1 && !limited) {
            processLine(buffer.slice(0, newlineIndex))
            buffer = buffer.slice(newlineIndex + 1)
            newlineIndex = buffer.indexOf('\n')
          }
        }
        if (!limited) {
          buffer += decoder.decode()
          if (buffer) processLine(buffer)
        }
        return { matches, limited }
      } finally {
        reader.releaseLock()
      }
    },
    catch: (cause) => new FsSearchError({
      reason: 'read',
      path: searchPath,
      message: `Failed to read ripgrep output for ${pattern}: ${String(cause)}`,
    }),
  })
}

function rgSearch(
  pattern: string,
  searchPath: string,
  glob: string | undefined,
  limit: number,
  options: {
    readonly resolveRgPath: () => Promise<string>
    readonly spawnRg: (command: readonly string[]) => RgProcess
    readonly timeoutMs: number
  },
): Effect.Effect<readonly FsSearchMatch[], FsSearchError> {
  return Effect.scoped(Effect.gen(function* () {
    const rgPath = yield* Effect.tryPromise({
      try: options.resolveRgPath,
      catch: (cause) => new FsSearchError({
        reason: 'spawn', path: searchPath, message: `Failed to resolve ripgrep: ${String(cause)}`,
      }),
    })
    const args = ['--json', '--line-number', '--max-columns', '500', '--max-columns-preview', '-e', pattern]
    if (glob) args.push('--glob', glob)
    args.push(searchPath)

    return yield* Effect.acquireUseRelease(
      Effect.try({
        try: () => {
          const proc = options.spawnRg([rgPath, ...args])
          return {
            proc,
            stdoutReader: proc.stdout.getReader(),
            stderrReader: proc.stderr.getReader(),
          }
        },
        catch: (cause) => new FsSearchError({
          reason: 'spawn', path: searchPath, message: `Failed to start ripgrep: ${String(cause)}`,
        }),
      }),
      ({ proc, stdoutReader, stderrReader }) => Effect.gen(function* () {
        const stderrFiber = yield* Effect.forkScoped(readBoundedText(stderrReader, STDERR_LIMIT))
        const output = yield* consumeRgStdout(stdoutReader, pattern, searchPath, limit)

        if (output.limited) yield* terminateAndReap(proc)
        const exitCode = output.limited ? proc.exitCode : yield* Effect.promise(() => proc.exited)
        const stderr = yield* Fiber.join(stderrFiber)

        if (!output.limited && output.matches.length === 0 && exitCode !== 0 && exitCode !== 1) {
          const detail = summarizeStderr(stderr)
          return yield* new FsSearchError({
            reason: 'process',
            path: searchPath,
            message: detail.length > 0
              ? `Ripgrep exited with code ${exitCode}: ${detail}`
              : `Ripgrep exited with code ${exitCode}`,
          })
        }
        return output.matches
      }),
      ({ proc, stdoutReader, stderrReader }) => Effect.gen(function* () {
        if (proc.exitCode === null) yield* Effect.sync(() => proc.kill())
        yield* Effect.all(
          [cancelReader(stdoutReader), cancelReader(stderrReader)],
          { concurrency: 'unbounded' },
        )
        yield* terminateAndReap(proc)
      }),
    )
  })).pipe(
    Effect.timeoutFail({
      duration: options.timeoutMs,
      onTimeout: () => new FsSearchError({
        reason: 'timeout',
        path: searchPath,
        message: `Search timed out after ${Duration.format(Duration.millis(options.timeoutMs))} — try a more specific pattern or glob filter`,
      }),
    }),
  )
}

function tryFs<A>(operation: string, path: string, fn: () => Promise<A>): Effect.Effect<A, FsError> {
  return Effect.tryPromise({
    try: fn,
    catch: (cause) => new FsError({ operation, path, cause }),
  })
}

export class Fs extends Context.Tag('Fs')<Fs, {
  readonly readFile: (path: string) => Effect.Effect<Buffer, FsError>
  readonly readText: (path: string) => Effect.Effect<string, FsError>
  readonly writeFile: (path: string, content: string | Uint8Array) => Effect.Effect<void, FsError>
  readonly stat: (path: string) => Effect.Effect<{ readonly isDirectory: () => boolean; readonly isFile: () => boolean }, FsError>
  readonly exists: (path: string) => Effect.Effect<boolean, FsError>
  readonly walk: (rootPath: string, options?: { readonly maxDepth?: number; readonly respectGitignore?: boolean }) => Effect.Effect<readonly FsWalkEntry[], FsError>
  readonly search: (params: { readonly pattern: string; readonly searchPath: string; readonly glob?: string; readonly limit: number }) => Effect.Effect<readonly FsSearchMatch[], FsError | FsSearchError>
}>() {}

export interface FsLiveOptions {
  readonly resolveRgPath?: () => Promise<string>
  readonly spawnRg?: (command: readonly string[]) => RgProcess
  readonly searchTimeoutMs?: number
}

export function makeFsLive(options: FsLiveOptions = {}) {
  const searchOptions = {
    resolveRgPath: options.resolveRgPath ?? resolveRgPath,
    spawnRg: options.spawnRg ?? ((command) => Bun.spawn([...command], { stdout: 'pipe', stderr: 'pipe' })),
    timeoutMs: options.searchTimeoutMs ?? SEARCH_TIMEOUT,
  }
  return Layer.succeed(Fs, {
    readFile: (path) => tryFs('readFile', path, async () => await readFile(path)),
    readText: (path) => tryFs('readText', path, async () => await readFile(path, 'utf8')),
    // Bun.write auto-creates parent directories; node fs.writeFile does not
    writeFile: (path, content) => tryFs('writeFile', path, async () => { await Bun.write(path, content) }),
    stat: (path) => tryFs('stat', path, async () => await stat(path)),
    exists: (path) => tryFs('exists', path, async () => {
      try {
        await stat(path)
        return true
      } catch {
        return false
      }
    }),
    walk: (rootPath, options) =>
      tryFs('walk', rootPath, async () => {
        const entries = await walk(rootPath, rootPath, 0, options?.maxDepth, null, {
          respectGitignore: options?.respectGitignore ?? true,
        })
        return entries.map((entry) => ({
          fullPath: entry.fullPath,
          relativePath: entry.relativePath,
          name: entry.name,
          type: entry.type,
          depth: entry.depth,
        }))
      }),
    search: ({ pattern, searchPath, glob, limit }) =>
      rgSearch(pattern, searchPath, glob, limit, searchOptions),
  })
}

export const FsLive = makeFsLive()

export function resolveFsPath(path: string, cwd: string, scratchpadPath: string): string {
  const resolved = resolveFileRefPath(path, cwd, scratchpadPath)
  return resolved ? resolved.resolvedPath : resolve(cwd, path)
}
