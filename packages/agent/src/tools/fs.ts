/**
 * Filesystem Tools
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { ToolEmitTag } from '../execution/tool-emit'
import { join, relative, resolve } from 'path'
import { resolveRgPath } from '@magnitudedev/ripgrep'
import { walk } from '../util/walk'
import { createDefaultIgnore } from '../util/gitignore'
import { validateAndApply, toEditDiff } from '../util/edit'
import { WorkingDirectoryTag } from '../execution/working-directory'

// =============================================================================
// Errors
// =============================================================================

const FsError = ToolErrorSchema('FsError', {})
type FsError = { readonly _tag: 'FsError'; readonly message: string }

function fsError(message: string): FsError {
  return { _tag: 'FsError', message }
}

// =============================================================================
// fs.read()
// =============================================================================

export const readTool = createTool({
  name: 'read',
  group: 'fs',
  description: 'Read file content as string. Supports line-range reads via optional offset (1-indexed start line) and limit (max lines, default 2000). Use this instead of running cat, head, tail, or less in the shell.',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({
      description: 'Relative path to a file from cwd. Use tree instead for directories'
    }),
    offset: Schema.optional(Schema.Number).annotations({
      description: '1-indexed start line (default: 1)'
    }),
    limit: Schema.optional(Schema.Number).annotations({
      description: 'Max lines to return (default: 2000)'
    }),
  }),
  outputSchema: Schema.String,
  errorSchema: FsError,
  argMapping: ['path', 'offset', 'limit'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['path', 'offset', 'limit'], selfClosing: true },
    xmlOutput: { type: 'tag' },
  } as const,

  execute: ({ path, offset, limit }) => Effect.gen(function* () {
    const { cwd } = yield* WorkingDirectoryTag
    const fullPath = resolve(cwd, path)
    const file = Bun.file(fullPath)
    const content = yield* Effect.tryPromise({
      try: () => file.text(),
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to read ${path}`),
    })

    const lines = content.split('\n')
    const startLine = offset ?? 1
    const maxLines = limit ?? 2000

    if (startLine < 1) {
      return yield* Effect.fail(fsError('offset must be >= 1'))
    }

    if (startLine > lines.length) {
      return yield* Effect.fail(fsError(`offset ${startLine} exceeds total lines ${lines.length}`))
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
  })
})

// =============================================================================
// fs.write()
// =============================================================================

export const writeTool = createTool({
  name: 'write',
  group: 'fs',
  description: 'Write content to file. Use this instead of running echo, tee, or heredocs in the shell.',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({
      description: 'Relative path from cwd'
    }),
    content: Schema.String.annotations({
      description: 'File content to write'
    })
  }),
  outputSchema: Schema.Void,
  errorSchema: FsError,
  argMapping: ['path', 'content'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['path'], body: 'content' },
    xmlOutput: { type: 'tag' },
  } as const,

  execute: ({ path, content }) => Effect.gen(function* () {
    const emit = yield* ToolEmitTag
    const { cwd } = yield* WorkingDirectoryTag
    yield* Effect.tryPromise({
      try: async () => {
        const fullPath = resolve(cwd, path)
        await Bun.write(fullPath, content)
      },
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to write ${path}`),
    })
    const linesWritten = content.split('\n').length
    yield* emit.emit({ type: 'write_stats', path, linesWritten })
  })
})

// =============================================================================
// edit() — string find-replace using <old>/<new> child tags
// =============================================================================

export const editTool = createTool({
  name: 'edit',
  description: 'Edit a file by replacing exact text. The <old> content must match the file exactly. Read the file first. Use this instead of running sed, perl, or awk in the shell.',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({
      description: 'Relative path from cwd'
    }),
    oldString: Schema.String.annotations({
      description: 'Exact text to find in the file'
    }),
    newString: Schema.String.annotations({
      description: 'Replacement text'
    }),
    replaceAll: Schema.optional(Schema.Boolean.annotations({
      description: 'Replace all occurrences instead of requiring uniqueness'
    })),
  }),
  outputSchema: Schema.String,
  errorSchema: FsError,
  argMapping: ['path', 'oldString', 'newString', 'replaceAll'],
  bindings: {
    xmlInput: {
      type: 'tag',
      attributes: ['path', 'replaceAll'],
      childTags: [
        { tag: 'old', field: 'oldString' },
        { tag: 'new', field: 'newString' },
      ],
    },
    xmlOutput: { type: 'tag' },
  } as const,

  execute: ({ path, oldString, newString, replaceAll }) => Effect.gen(function* () {
    const emit = yield* ToolEmitTag
    const { cwd } = yield* WorkingDirectoryTag
    const fullPath = resolve(cwd, path)

    // Read current file
    const content = yield* Effect.tryPromise({
      try: () => Bun.file(fullPath).text(),
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to read ${path}`),
    })

    // Validate and apply
    let applied
    try {
      applied = validateAndApply(content, oldString, newString, replaceAll ?? false)
    } catch (e) {
      return yield* Effect.fail(fsError(e instanceof Error ? e.message : String(e)))
    }

    // Write result to disk
    yield* Effect.tryPromise({
      try: () => Bun.write(fullPath, applied.result),
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to write ${path}`),
    })

    // Emit diff for UI
    const diff = toEditDiff(applied)
    yield* emit.emit({ type: 'edit_diff', path, diffs: [diff] })

    // Build summary
    if (applied.replaceCount > 1) {
      return `Replaced ${applied.replaceCount} occurrences in ${path}`
    }
    if (applied.addedLines.length === 0) {
      return `Deleted ${applied.removedLines.length} line(s) from ${path}`
    }
    return `Replaced ${applied.removedLines.length} line(s) with ${applied.addedLines.length} line(s) in ${path}`
  }),
})

// =============================================================================
// fs.tree()
// =============================================================================

const TreeEntry = Schema.Struct({
  path: Schema.String,
  name: Schema.String,
  type: Schema.Literal('file', 'dir'),
  depth: Schema.Number
})

type TreeEntry = Schema.Schema.Type<typeof TreeEntry>

export const treeTool = createTool({
  name: 'tree',
  group: 'fs',
  description: 'List directory structure with optional gitignore filtering. Use this instead of running ls, find, or tree in the shell.',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({
      description: 'Relative path from cwd'
    }),
    options: Schema.optional(Schema.Struct({
      recursive: Schema.optional(Schema.Boolean.annotations({
        description: 'Include subdirectories (default: true)'
      })),
      maxDepth: Schema.optional(Schema.Number.annotations({
        description: 'Maximum depth to traverse'
      })),
      gitignore: Schema.optional(Schema.Boolean.annotations({
        description: 'Respect .gitignore patterns (default: true)'
      }))
    }))
  }),
  outputSchema: Schema.Array(TreeEntry),
  errorSchema: FsError,
  argMapping: ['path', 'options'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['path'], selfClosing: true },
    xmlOutput: { type: 'tag', items: { tag: 'entry', attributes: ['path', 'name', 'type', 'depth'] } },
  } as const,

  execute: ({ path, options }) => Effect.gen(function* () {
    const { cwd } = yield* WorkingDirectoryTag
    return yield* Effect.tryPromise({
      try: async () => {
        const fullPath = resolve(cwd, path)
        const respectGitignore = options?.gitignore ?? true
        const maxDepth = options?.maxDepth

        const ignore = respectGitignore ? createDefaultIgnore() : null
        const entries = await walk(fullPath, fullPath, 0, maxDepth, ignore, respectGitignore)

        return entries.map(entry => ({
          path: entry.relativePath,
          name: entry.name,
          type: entry.type,
          depth: entry.depth
        }))
      },
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to list ${path}`),
    })
  })
})

// =============================================================================
// fs.search()
// =============================================================================

const SearchMatch = Schema.Struct({
  file: Schema.String,
  match: Schema.String
})

type SearchMatch = Schema.Schema.Type<typeof SearchMatch>

export async function searchFiles(
  pattern: string,
  searchPath: string,
  globPattern: string | undefined,
  limit: number
): Promise<SearchMatch[]> {
  const args = [
    '--json',
    '--line-number',
    '--max-columns', '500',
    '--max-columns-preview',
    '-e', pattern,
  ]

  if (globPattern) {
    args.push('--glob', globPattern)
  }

  args.push(searchPath)

  const rgPath = await resolveRgPath()
  const proc = Bun.spawn([rgPath, ...args], {
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const stdout = await new Response(proc.stdout).text()
  await proc.exited

  const matches: SearchMatch[] = []

  for (const line of stdout.split('\n')) {
    if (!line.trim()) continue
    try {
      const msg = JSON.parse(line)
      if (msg.type === 'match') {
        const data = msg.data
        const filePath = relative(searchPath, data.path.text)
        const lineNum = data.line_number
        const lineText = data.lines.text.replace(/\n$/, '')
        matches.push({
          file: filePath,
          match: `${lineNum}|${lineText}`
        })
        if (matches.length >= limit) break
      }
    } catch {
      continue
    }
  }

  return matches
}

export const searchTool = createTool({
  name: 'search',
  group: 'fs',
  description: 'Search file contents with regex. Use this instead of running grep, rg, or ag in the shell — it uses ripgrep under the hood.',
  inputSchema: Schema.Struct({
    pattern: Schema.String.annotations({
      description: 'Regex pattern to search for'
    }),
    path: Schema.optional(Schema.String.annotations({
      description: 'Directory to search in (default: cwd)'
    })),
    glob: Schema.optional(Schema.String.annotations({
      description: 'Glob pattern to filter files (e.g., "*.ts")'
    })),
    limit: Schema.optional(Schema.Number.annotations({
      description: 'Maximum number of matches to return (default: 50)'
    })),
    // Backward compatibility for non-XML callers still sending options object
    options: Schema.optional(Schema.Struct({
      path: Schema.optional(Schema.String),
      glob: Schema.optional(Schema.String),
      limit: Schema.optional(Schema.Number),
    })),
  }),
  outputSchema: Schema.Array(SearchMatch),
  errorSchema: FsError,
  argMapping: ['pattern', 'path', 'glob', 'limit', 'options'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['pattern', 'path', 'glob', 'limit'], selfClosing: true },
    xmlOutput: { type: 'tag', items: { tag: 'item', attributes: ['file'], body: 'match' } },
  } as const,

  execute: ({ pattern, path, glob, limit, options }) => Effect.gen(function* () {
    const { cwd } = yield* WorkingDirectoryTag
    return yield* Effect.tryPromise({
      try: async () => {
        const resolvedPath = path ?? options?.path
        const resolvedGlob = glob ?? options?.glob
        const resolvedLimit = limit ?? options?.limit ?? 50

        const searchPath = resolvedPath
          ? resolve(cwd, resolvedPath)
          : cwd

        return await searchFiles(pattern, searchPath, resolvedGlob, resolvedLimit)
      },
      catch: (e) => fsError(e instanceof Error ? e.message : `Search failed for ${pattern}`),
    })
  })
})

// =============================================================================
// Filesystem Tools Group
// =============================================================================

export const fsTools = [readTool, writeTool, editTool, treeTool, searchTool]
