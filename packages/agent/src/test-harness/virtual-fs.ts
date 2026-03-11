import { validateAndApply } from '../util/edit'

type FsReadInput = {
  path: string
  offset?: number
  limit?: number
}

type FsWriteInput = {
  path: string
  content: string
}

type EditInput = {
  path: string
  oldString: string
  newString: string
  replaceAll?: boolean
}

type FsTreeInput = {
  path: string
  options?: {
    recursive?: boolean
    maxDepth?: number
    gitignore?: boolean
  }
}

type FsSearchInput = {
  pattern: string
  path?: string
  glob?: string
  limit?: number
  options?: {
    path?: string
    glob?: string
    limit?: number
  }
}

type TreeEntry = {
  path: string
  name: string
  type: 'file' | 'dir'
  depth: number
}

type SearchMatch = {
  file: string
  match: string
}

export function createVirtualFs(seed?: Record<string, string>): Map<string, string> {
  return new Map(Object.entries(seed ?? {}))
}

function normalizePath(path: string): string {
  return path.replace(/^\.?\//, '').replace(/\/+/g, '/').replace(/\/$/, '')
}

function matchesGlob(path: string, glob: string): boolean {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`)
  return regex.test(path)
}

export function createFsReadHandler(files: Map<string, string>) {
  return (input: unknown): string => {
    const { path, offset, limit } = input as FsReadInput
    const normalized = normalizePath(path)
    const content = files.get(normalized)
    if (content === undefined) {
      throw new Error(`Failed to read ${path}`)
    }

    const lines = content.split('\n')
    const startLine = offset ?? 1
    const maxLines = limit ?? 2000

    if (startLine < 1) {
      throw new Error('offset must be >= 1')
    }

    if (startLine > lines.length) {
      throw new Error(`offset ${startLine} exceeds total lines ${lines.length}`)
    }

    const startIdx = startLine - 1
    const endIdx = startIdx + maxLines
    const slice = lines.slice(startIdx, endIdx)

    const remaining = lines.length - endIdx
    let result = slice.join('\n')
    if (remaining > 0) {
      result += `\n... (${remaining} more lines remaining. Use offset=${startLine + maxLines} to continue reading.)`
    }

    return result
  }
}

export function createFsWriteHandler(files: Map<string, string>) {
  return (input: unknown): void => {
    const { path, content } = input as FsWriteInput
    files.set(normalizePath(path), content)
  }
}

export function createEditHandler(files: Map<string, string>) {
  return (input: unknown): string => {
    const { path, oldString, newString, replaceAll } = input as EditInput
    const normalized = normalizePath(path)
    const content = files.get(normalized)
    if (content === undefined) {
      throw new Error(`Failed to read ${path}`)
    }

    const applied = validateAndApply(content, oldString, newString, replaceAll ?? false)
    files.set(normalized, applied.result)

    if (applied.replaceCount > 1) {
      return `Replaced ${applied.replaceCount} occurrences in ${path}`
    }
    if (applied.addedLines.length === 0) {
      return `Deleted ${applied.removedLines.length} line(s) from ${path}`
    }
    return `Replaced ${applied.removedLines.length} line(s) with ${applied.addedLines.length} line(s) in ${path}`
  }
}

export function createFsTreeHandler(files: Map<string, string>) {
  return (input: unknown): TreeEntry[] => {
    const { path, options } = input as FsTreeInput
    const base = normalizePath(path)
    const recursive = options?.recursive ?? true
    const maxDepth = options?.maxDepth
    const dirSet = new Set<string>()
    const entries = new Map<string, TreeEntry>()

    for (const filePathRaw of files.keys()) {
      const filePath = normalizePath(filePathRaw)
      if (base && filePath !== base && !filePath.startsWith(`${base}/`)) continue

      const rel = base ? (filePath === base ? '' : filePath.slice(base.length + 1)) : filePath
      if (!rel) continue

      const parts = rel.split('/')
      const fileDepth = parts.length

      if (recursive === false && fileDepth > 1) continue
      if (maxDepth !== undefined && fileDepth > maxDepth) continue

      let acc = ''
      for (let i = 0; i < parts.length - 1; i++) {
        acc = acc ? `${acc}/${parts[i]}` : parts[i]!
        dirSet.add(acc)
      }
    }

    for (const dirRel of dirSet) {
      const name = dirRel.split('/').at(-1) ?? dirRel
      const depth = dirRel.split('/').length
      const fullPath = base ? `${base}/${dirRel}` : dirRel
      entries.set(fullPath, { path: fullPath, name, type: 'dir', depth })
    }

    for (const filePathRaw of files.keys()) {
      const filePath = normalizePath(filePathRaw)
      if (base && filePath !== base && !filePath.startsWith(`${base}/`)) continue

      const rel = base ? (filePath === base ? '' : filePath.slice(base.length + 1)) : filePath
      if (!rel) continue

      const parts = rel.split('/')
      const depth = parts.length
      if (recursive === false && depth > 1) continue
      if (maxDepth !== undefined && depth > maxDepth) continue

      entries.set(filePath, {
        path: filePath,
        name: parts.at(-1) ?? filePath,
        type: 'file',
        depth,
      })
    }

    return [...entries.values()].sort((a, b) => a.path.localeCompare(b.path))
  }
}

export function createFsSearchHandler(files: Map<string, string>) {
  return (input: unknown): SearchMatch[] => {
    const { pattern, path, glob, limit, options } = input as FsSearchInput
    const resolvedPath = normalizePath(path ?? options?.path ?? '')
    const resolvedGlob = glob ?? options?.glob
    const resolvedLimit = limit ?? options?.limit ?? 50

    const regex = new RegExp(pattern)
    const matches: SearchMatch[] = []

    for (const [filePathRaw, content] of files.entries()) {
      const filePath = normalizePath(filePathRaw)

      if (resolvedPath && filePath !== resolvedPath && !filePath.startsWith(`${resolvedPath}/`)) {
        continue
      }

      const scopedPath = resolvedPath ? filePath.slice(resolvedPath.length).replace(/^\//, '') : filePath
      if (resolvedGlob && !matchesGlob(scopedPath, resolvedGlob)) {
        continue
      }

      const lines = content.split('\n')
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i] ?? ''
        if (!regex.test(line)) continue
        matches.push({ file: filePath, match: `${i + 1}|${line}` })
        if (matches.length >= resolvedLimit) return matches
      }
    }

    return matches
  }
}

export function createDefaultToolOverrides(files: Map<string, string>): Record<string, (input: unknown) => unknown> {
  return {
    'fs-read': createFsReadHandler(files),
    'fs-write': createFsWriteHandler(files),
    edit: createEditHandler(files),
    'fs-tree': createFsTreeHandler(files),
    'fs-search': createFsSearchHandler(files),
    shell: () => ({ stdout: '', stderr: '', exitCode: 0 }),
    'web-fetch': (input: unknown) => {
      const { url } = input as { url: string }
      return { url, content: '' }
    },
    'web-search': () => ({ text: '', sources: [] as Array<{ title: string; url: string }> }),
  }
}