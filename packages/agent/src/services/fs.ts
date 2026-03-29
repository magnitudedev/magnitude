import { readFile, writeFile, stat } from 'fs/promises'
import { relative, resolve } from 'path'
import { Context, Data, Effect, Layer } from 'effect'
import { resolveRgPath } from '@magnitudedev/ripgrep'
import { walk } from '../util/walk'
import { resolveFileRefPath } from '../workspace/file-ref-resolution'

export class FsError extends Data.TaggedError('FsError')<{
  readonly operation: string
  readonly path: string
  readonly cause: unknown
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

async function rgSearch(
  pattern: string,
  searchPath: string,
  glob: string | undefined,
  limit: number,
): Promise<readonly FsSearchMatch[]> {
  const args = [
    '--json',
    '--line-number',
    '--max-columns', '500',
    '--max-columns-preview',
    '-e', pattern,
  ]

  if (glob) {
    args.push('--glob', glob)
  }

  args.push(searchPath)

  const rgPath = await resolveRgPath()
  const proc = Bun.spawn([rgPath, ...args], {
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const matches: FsSearchMatch[] = []
  const decoder = new TextDecoder()
  const reader = proc.stdout.getReader()

  let buffer = ''
  let timedOut = false
  let stoppedEarly = false

  const processLine = (line: string): void => {
    if (!line.trim()) return

    try {
      const msg = JSON.parse(line)
      if (msg.type === 'match') {
        const data = msg.data
        const filePath = relative(searchPath, data.path.text)
        const lineNum = data.line_number
        const lineText = data.lines.text.replace(/\n$/, '')
        matches.push({
          file: filePath,
          match: `${lineNum}|${lineText}`,
        })
        if (matches.length >= limit) {
          stoppedEarly = true
          proc.kill()
        }
      }
    } catch {
      // Ignore malformed lines
    }
  }

  const timeout = setTimeout(() => {
    timedOut = true
    proc.kill()
  }, 5000)

  try {
    while (!stoppedEarly) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      let newlineIndex = buffer.indexOf('\n')
      while (newlineIndex !== -1) {
        const line = buffer.slice(0, newlineIndex)
        buffer = buffer.slice(newlineIndex + 1)
        processLine(line)
        if (stoppedEarly) break
        newlineIndex = buffer.indexOf('\n')
      }
    }

    if (!stoppedEarly) {
      buffer += decoder.decode()
      if (buffer) processLine(buffer)
    }

    await proc.exited

    if (timedOut) {
      throw new Error('Search timed out after 5s — try a more specific pattern or glob filter')
    }

    return matches
  } finally {
    clearTimeout(timeout)
    if (!timedOut && !stoppedEarly) {
      proc.kill()
    }
    reader.releaseLock()
  }
}

function tryFs<A>(operation: string, path: string, effect: Effect.Effect<A>): Effect.Effect<A, FsError> {
  return Effect.catchAllDefect(effect, (cause) =>
    Effect.fail(new FsError({ operation, path, cause }))
  )
}

export class Fs extends Context.Tag('Fs')<Fs, {
  readonly readFile: (path: string) => Effect.Effect<Buffer, FsError>
  readonly readText: (path: string) => Effect.Effect<string, FsError>
  readonly writeFile: (path: string, content: string | Uint8Array) => Effect.Effect<void, FsError>
  readonly stat: (path: string) => Effect.Effect<{ readonly isDirectory: () => boolean; readonly isFile: () => boolean }, FsError>
  readonly walk: (rootPath: string, options?: { readonly maxDepth?: number; readonly respectGitignore?: boolean }) => Effect.Effect<readonly FsWalkEntry[], FsError>
  readonly search: (params: { readonly pattern: string; readonly searchPath: string; readonly glob?: string; readonly limit: number }) => Effect.Effect<readonly FsSearchMatch[], FsError>
}>() {}

export const FsLive = Layer.succeed(Fs, {
  readFile: (path) => tryFs('readFile', path, Effect.promise(() => readFile(path))),
  readText: (path) => tryFs('readText', path, Effect.promise(() => readFile(path, 'utf8'))),
  writeFile: (path, content) => tryFs('writeFile', path, Effect.promise(() => writeFile(path, content))),
  stat: (path) => tryFs('stat', path, Effect.promise(() => stat(path))),
  walk: (rootPath, options) =>
    tryFs('walk', rootPath, Effect.promise(() =>
      walk(rootPath, rootPath, 0, options?.maxDepth, null, {
        respectGitignore: options?.respectGitignore ?? true,
      }).then((entries) =>
        entries.map((entry) => ({
          fullPath: entry.fullPath,
          relativePath: entry.relativePath,
          name: entry.name,
          type: entry.type,
          depth: entry.depth,
        }))
      )
    )),
  search: ({ pattern, searchPath, glob, limit }) =>
    tryFs('search', searchPath, Effect.promise(() => rgSearch(pattern, searchPath, glob, limit))),
})

export function resolveFsPath(path: string, cwd: string, workspacePath: string): string {
  const resolved = resolveFileRefPath(path, cwd, workspacePath)
  return resolved ? resolved.resolvedPath : resolve(cwd, path)
}
