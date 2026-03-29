import { Effect, Layer } from 'effect'
import { isAbsolute, join, normalize, relative, resolve, sep } from 'path'
import { Fs, FsError, type FsSearchMatch, type FsWalkEntry } from '../services/fs'

export function createVirtualFs(seed?: Record<string, string>): Map<string, string> {
  return new Map(Object.entries(seed ?? {}))
}

function normalizePath(path: string): string {
  return path.replace(/^\.?\//, '').replace(/\/+/g, '/').replace(/\/$/, '')
}


function normalizeFsPath(path: string): string {
  return normalize(path)
}

function toAbsolutePath(path: string, cwd: string): string {
  return isAbsolute(path) ? normalizeFsPath(path) : normalizeFsPath(resolve(cwd, path))
}

function buildAbsoluteFileMap(files: Map<string, string>, cwd: string): Map<string, string> {
  const absoluteFiles = new Map<string, string>()
  for (const [path, content] of files.entries()) {
    const normalizedVirtualPath = normalizePath(path)
    const absolutePath = toAbsolutePath(normalizedVirtualPath, cwd)
    absoluteFiles.set(absolutePath, content)
  }
  return absoluteFiles
}

function isDirectoryPath(path: string, absoluteFiles: Map<string, string>): boolean {
  const prefix = `${path}${sep}`
  for (const filePath of absoluteFiles.keys()) {
    if (filePath.startsWith(prefix)) return true
  }
  return false
}

function toAbsoluteVirtualPath(path: string, cwd: string, workspacePath: string): string {
  const normalized = path.replace(/\\/g, '/')
  if (normalized.startsWith('$M/')) {
    return normalize(resolve(workspacePath, normalized.slice('$M/'.length)))
  }
  if (normalized.startsWith('${M}/')) {
    return normalize(resolve(workspacePath, normalized.slice('${M}/'.length)))
  }
  return toAbsolutePath(path, cwd)
}

function toRelativeVirtualKey(absolutePath: string, cwd: string): string {
  const relPath = relative(cwd, absolutePath)
  return normalizePath(relPath)
}

function matchesVirtualGlob(path: string, glob: string): boolean {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`)
  return regex.test(path)
}

export function createVirtualFsLayer(files: Map<string, string>, cwd: string, workspacePath: string): Layer.Layer<Fs> {
  const absoluteFiles = buildAbsoluteFileMap(files, cwd)

  const readFromMap = (path: string): string => {
    const absolutePath = toAbsoluteVirtualPath(path, cwd, workspacePath)
    const content = absoluteFiles.get(absolutePath)
    if (content === undefined) {
      throw new Error(`ENOENT: no such file or directory, open '${path}'`)
    }
    return content
  }

  return Layer.succeed(Fs, {
    readFile: (path) =>
      Effect.try({
        try: () => Buffer.from(readFromMap(path), 'utf8'),
        catch: (cause) => new FsError({ operation: 'readFile', path, cause }),
      }),

    readText: (path) =>
      Effect.try({
        try: () => readFromMap(path),
        catch: (cause) => new FsError({ operation: 'readText', path, cause }),
      }),

    writeFile: (path, content) =>
      Effect.try({
        try: () => {
          const absolutePath = toAbsoluteVirtualPath(path, cwd, workspacePath)
          const textContent = typeof content === 'string' ? content : Buffer.from(content).toString('utf8')
          absoluteFiles.set(absolutePath, textContent)
          const relativeKey = toRelativeVirtualKey(absolutePath, cwd)
          files.set(relativeKey, textContent)
        },
        catch: (cause) => new FsError({ operation: 'writeFile', path, cause }),
      }),

    stat: (path) =>
      Effect.try({
        try: () => {
          const absolutePath = toAbsoluteVirtualPath(path, cwd, workspacePath)
          if (absoluteFiles.has(absolutePath)) {
            return { isDirectory: () => false, isFile: () => true }
          }
          if (isDirectoryPath(absolutePath, absoluteFiles)) {
            return { isDirectory: () => true, isFile: () => false }
          }
          throw new Error(`ENOENT: no such file or directory, stat '${path}'`)
        },
        catch: (cause) => new FsError({ operation: 'stat', path, cause }),
      }),

    walk: (rootPath, options) =>
      Effect.try({ try: () => {
        const absoluteRoot = toAbsoluteVirtualPath(rootPath, cwd, workspacePath)
        const rootPrefix = `${absoluteRoot}${sep}`
        const maxDepth = options?.maxDepth
        const entries = new Map<string, FsWalkEntry>()

        for (const filePath of absoluteFiles.keys()) {
          if (filePath !== absoluteRoot && !filePath.startsWith(rootPrefix)) continue

          const relFile = relative(absoluteRoot, filePath)
          if (!relFile || relFile.startsWith('..')) continue

          const fileDepth = relFile.split(sep).length
          if (maxDepth !== undefined && fileDepth > maxDepth) continue

          const fileName = filePath.split(sep).at(-1) ?? filePath
          entries.set(filePath, {
            fullPath: filePath,
            relativePath: relFile,
            name: fileName,
            type: 'file',
            depth: fileDepth,
          })

          const parts = relFile.split(sep)
          let current = absoluteRoot
          for (let i = 0; i < parts.length - 1; i++) {
            current = join(current, parts[i] ?? '')
            const relDir = relative(absoluteRoot, current)
            const depth = relDir ? relDir.split(sep).length : 0
            if (maxDepth !== undefined && depth > maxDepth) continue
            if (!entries.has(current)) {
              entries.set(current, {
                fullPath: current,
                relativePath: relDir,
                name: parts[i] ?? current,
                type: 'dir',
                depth,
              })
            }
          }
        }

        return [...entries.values()].sort((a, b) => a.fullPath.localeCompare(b.fullPath))
      }, catch: (cause) => new FsError({ operation: 'walk', path: rootPath, cause }) }),

    search: ({ pattern, searchPath, glob, limit }) =>
      Effect.try({ try: () => {
        const absoluteSearchPath = toAbsoluteVirtualPath(searchPath, cwd, workspacePath)
        const searchPrefix = `${absoluteSearchPath}${sep}`
        const matches: FsSearchMatch[] = []
        const regex = new RegExp(pattern)

        for (const [filePath, content] of absoluteFiles.entries()) {
          if (filePath !== absoluteSearchPath && !filePath.startsWith(searchPrefix)) continue

          const fileRelativeToSearch = relative(absoluteSearchPath, filePath)
          if (glob && !matchesVirtualGlob(fileRelativeToSearch, glob)) continue

          const lines = content.split('\n')
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i] ?? ''
            regex.lastIndex = 0
            if (!regex.test(line)) continue
            matches.push({
              file: fileRelativeToSearch,
              match: `${i + 1}|${line}`,
            })
            if (matches.length >= limit) return matches
          }
        }

        return matches
      }, catch: (cause) => new FsError({ operation: 'search', path: searchPath, cause }) }),
  })
}