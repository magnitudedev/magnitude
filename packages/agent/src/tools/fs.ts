/**
 * Filesystem Tools
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema, ToolImageSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { relative, resolve } from 'path'
import { resolveRgPath } from '@magnitudedev/ripgrep'
import { walk } from '../util/walk'
import { createDefaultIgnore } from '../util/gitignore'
import { validateAndApply } from '../util/edit'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { readImageFileForModel } from '../util/read-image-file'
import { expandWorkspacePath } from '../workspace'

// =============================================================================
// Errors
// =============================================================================

type FsError = { readonly _tag: 'FsError'; readonly message: string }

function fsError(message: string): FsError {
  return { _tag: 'FsError', message }
}

const FsErrorSchema = ToolErrorSchema('FsError', {})

// =============================================================================
// fs.read()
// =============================================================================

export const readTool = defineTool({
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
  errorSchema: FsErrorSchema,
  label: (input) => input.path ? `Reading ${input.path}` : 'Reading file...',
  execute: ({ path, offset, limit }, _ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    const expandedPath = expandWorkspacePath(path, workspacePath)
    const fullPath = resolve(cwd, expandedPath)
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

export const readXmlBinding = defineXmlBinding(readTool, {
  input: {
    attributes: [
      { field: 'path', attr: 'path' },
      { field: 'offset', attr: 'offset' },
      { field: 'limit', attr: 'limit' },
    ],
  },
  output: {},
} as const)

// =============================================================================
// fs.write()
// =============================================================================

export const writeTool = defineTool({
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
  errorSchema: FsErrorSchema,
  emissionSchema: Schema.Struct({
    type: Schema.Literal('write_stats'),
    path: Schema.String,
    linesWritten: Schema.Number,
  }),
  label: (input) => input.path ? `Writing ${input.path}` : 'Writing file...',
  execute: ({ path, content }, ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    const expandedPath = expandWorkspacePath(path, workspacePath)
    yield* Effect.tryPromise({
      try: async () => {
        const fullPath = resolve(cwd, expandedPath)
        await Bun.write(fullPath, content)
      },
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to write ${path}`),
    })
    const linesWritten = content.split('\n').length
    yield* ctx.emit({ type: 'write_stats', path, linesWritten })
  })
})

export const writeXmlBinding = defineXmlBinding(writeTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
    body: 'content',
  },
  output: {},
} as const)

// =============================================================================
// edit() — string find-replace using <old>/<new> child tags
// =============================================================================

export const editTool = defineTool({
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
  errorSchema: FsErrorSchema,
  emissionSchema: Schema.Struct({
    type: Schema.Literal('file_edit_base_content'),
    path: Schema.String,
    baseContent: Schema.String,
  }),
  stream: {
    initial: { emitted: false },
    onInput: (input, state, ctx) => Effect.gen(function* () {
      if (state.emitted) return state
      const path = input.path
      if (!path || !path.isFinal) return state
      const { cwd, workspacePath } = yield* WorkingDirectoryTag
      const expandedPath = expandWorkspacePath(path.value, workspacePath)
      const fullPath = resolve(cwd, expandedPath)
      const content = yield* Effect.tryPromise({
        try: () => Bun.file(fullPath).text(),
        catch: () => new Error('read failed'),
      }).pipe(Effect.catchAll(() => Effect.succeed(undefined)))
      if (content == null) return state
      yield* ctx.emit({ type: 'file_edit_base_content', path: path.value, baseContent: content })
      return { emitted: true }
    }),
  },
  label: (input) => input.path ? `Editing ${input.path}` : 'Editing file...',
  execute: ({ path, oldString, newString, replaceAll }, _ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    const expandedPath = expandWorkspacePath(path, workspacePath)
    const fullPath = resolve(cwd, expandedPath)

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

export const editXmlBinding = defineXmlBinding(editTool, {
  input: {
    attributes: [
      { field: 'path', attr: 'path' },
      { field: 'replaceAll', attr: 'replaceAll' },
    ],
    childTags: [
      { tag: 'old', field: 'oldString' },
      { tag: 'new', field: 'newString' },
    ],
  },
  output: {},
} as const)

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

export const treeTool = defineTool({
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
  errorSchema: FsErrorSchema,
  label: (input) => input.path ? `Listing ${input.path}` : 'Listing directory...',
  execute: ({ path, options }, _ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    const expandedPath = expandWorkspacePath(path, workspacePath)
    return yield* Effect.tryPromise({
      try: async () => {
        const fullPath = resolve(cwd, expandedPath)
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

export const treeXmlBinding = defineXmlBinding(treeTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
  },
  output: {
    items: {
      tag: 'entry',
      attributes: [
        { attr: 'path', field: 'path' },
        { attr: 'name', field: 'name' },
        { attr: 'type', field: 'type' },
        { attr: 'depth', field: 'depth' },
      ],
    },
  },
} as const)

// =============================================================================
// fs.search()
// =============================================================================

const SearchMatch = Schema.Struct({
  file: Schema.String,
  match: Schema.String
})

type SearchMatch = Schema.Schema.Type<typeof SearchMatch>

export async function grepFiles(
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

  const matches: SearchMatch[] = []
  const decoder = new TextDecoder()
  const reader = proc.stdout.getReader()

  let buffer = ''
  let timedOut = false
  let stoppedEarly = false

  const processLine = (line: string) => {
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
          match: `${lineNum}|${lineText}`
        })
        if (matches.length >= limit) {
          stoppedEarly = true
          proc.kill()
        }
      }
    } catch {
      // Ignore malformed lines from rg output
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
      if (buffer) {
        processLine(buffer)
      }
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

export const grepTool = defineTool({
  name: 'grep',
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
  errorSchema: FsErrorSchema,
  label: (input) => input.pattern ? `Searching for ${input.pattern}` : 'Searching files...',
  execute: ({ pattern, path, glob, limit, options }, _ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    return yield* Effect.tryPromise({
      try: async () => {
        const resolvedPath = expandWorkspacePath(path ?? options?.path ?? '', workspacePath) || undefined
        const resolvedGlob = glob ?? options?.glob
        const resolvedLimit = limit ?? options?.limit ?? 50

        const searchPath = resolvedPath
          ? resolve(cwd, resolvedPath)
          : cwd

        return await grepFiles(pattern, searchPath, resolvedGlob, resolvedLimit)
      },
      catch: (e) => fsError(e instanceof Error ? e.message : `Search failed for ${pattern}`),
    })
  })
})

export const grepXmlBinding = defineXmlBinding(grepTool, {
  input: {
    attributes: [
      { field: 'pattern', attr: 'pattern' },
      { field: 'path', attr: 'path' },
      { field: 'glob', attr: 'glob' },
      { field: 'limit', attr: 'limit' },
    ],
  },
  output: {
    items: {
      tag: 'item',
      attributes: [{ attr: 'file', field: 'file' }],
      body: 'match',
    },
  },
} as const)

// =============================================================================
// fs.view()
// =============================================================================

export const viewTool = defineTool({
  name: 'view',
  group: 'fs',
  description: 'Read an image file and return it as image output for visual inspection. Supports PNG, JPEG, WebP, GIF, and SVG files.',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({
      description: 'Relative path to an image file from cwd'
    }),
  }),
  outputSchema: ToolImageSchema,
  errorSchema: FsErrorSchema,
  label: (input) => input.path ? `Viewing ${input.path}` : 'Viewing image...',
  execute: ({ path: filePath }, _ctx) => Effect.gen(function* () {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag
    const expandedPath = expandWorkspacePath(filePath, workspacePath)
    const fullPath = resolve(cwd, expandedPath)

    return yield* Effect.tryPromise({
      try: () => readImageFileForModel(fullPath),
      catch: (e) => fsError(e instanceof Error ? e.message : `Failed to read image: ${filePath}`),
    })
  })
})

export const viewXmlBinding = defineXmlBinding(viewTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
  },
  output: {},
} as const)

// =============================================================================
// Filesystem Tools Group
// =============================================================================

export const fsTools = [readTool, writeTool, editTool, treeTool, grepTool, viewTool]

export const fsXmlBindings = [
  readXmlBinding,
  writeXmlBinding,
  editXmlBinding,
  treeXmlBinding,
  grepXmlBinding,
  viewXmlBinding,
]
