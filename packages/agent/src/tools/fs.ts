/**
 * Filesystem Tools
 */

import { Effect, Option, Schema } from 'effect'
import { defineHarnessTool, StreamValidationError } from '@magnitudedev/harness'
import { resolve } from 'path'
import { validateAndApply } from '../util/edit'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { captureContextImageFromFile } from '../util/capture-context-image'
import { ContextImageResultSchema } from '../content'
import { expandScratchpadPath } from '@magnitudedev/scratchpad'
import { Fs, resolveFsPath } from '../services/fs'
import { ToolErrorSchema } from './errors'

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

export const readTool = defineHarnessTool({
  definition: {
    name: 'read',
    description: 'Read file text content. Use this instead of running cat, head, tail, or less in the shell. Large files are automatically truncated. Read the full file when reading for understanding or editing. Only use offset/limit for partial reads when tracing specific symbols or locating related files. Do not re-read files that are already in your context window.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path to a file from cwd. Use $M/ prefix for scratchpad path. Use tree instead for directories'
      }),
      offset: Schema.optionalWith(Schema.Number, { default: () => 1, exact: true }).annotations({
        description: '1-indexed start line (default: 1)'
      }),
      limit: Schema.optionalWith(Schema.Number, { default: () => 2000, exact: true }).annotations({
        description: 'Max lines to return (default: 2000)'
      }),
    }),
    outputSchema: Schema.String,
  },
  errorSchema: FsErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.path?.isFinal) return {}
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const fs = yield* Fs
      const { path: expandedPath } = expandScratchpadPath(input.path.value, scratchpadPath)
      const fullPath = resolve(cwd, expandedPath)
      const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!exists) {
        return yield* new StreamValidationError({ message: `File not found: ${input.path.value}` })
      }
      return {}
    }),
  },
  execute: ({ path, offset, limit }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const { path: expandedPath } = expandScratchpadPath(path, scratchpadPath)
    const fullPath = resolve(cwd, expandedPath)
    const content = yield* fs.readText(fullPath).pipe(
      Effect.mapError(() => fsError(`Failed to read ${path}`))
    )

    const lines = content.split('\n')
    const startLine = offset
    const maxLines = limit

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
  }),
})

// =============================================================================
// fs.write()
// =============================================================================

export const writeTool = defineHarnessTool({
  definition: {
    name: 'write',
    description: 'Write content to file, completely replacing any existing content. Read file in full if already exists before overwriting. Use this instead of running echo, tee, or heredocs in the shell. For partial edits, use the edit tool instead.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path from cwd. Use $M/ prefix for scratchpad path.'
      }),
      content: Schema.String.annotations({
        description: 'File content to write'
      })
    }),
    outputSchema: Schema.Struct({}),
  },
  errorSchema: FsErrorSchema,
  emissionSchema: Schema.Struct({
    type: Schema.Literal('write_stats'),
    path: Schema.String,
    linesWritten: Schema.Number,
  }),
  execute: ({ path, content }, ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const { path: expandedPath } = expandScratchpadPath(path, scratchpadPath)
    const fullPath = resolve(cwd, expandedPath)
    yield* fs.writeFile(fullPath, content).pipe(
      Effect.mapError(() => fsError(`Failed to write ${path}`))
    )
    const linesWritten = content.split('\n').length
    yield* ctx.emit({ type: 'write_stats', path, linesWritten })
    return {}
  }),
})

// =============================================================================
// edit() — string find-replace
// =============================================================================

export const editTool = defineHarnessTool({
  definition: {
    name: 'edit',
    description: 'Edit a file by replacing exact text. The "old" parameter content must match the file exactly. Read the file first. Use this instead of running sed, perl, or awk in the shell. If an edit touches >50% of file content, write is more efficient.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path from cwd. Use $M/ prefix for scratchpad path.'
      }),
      old: Schema.String.annotations({
        description: 'Exact text to find in the file'
      }),
      new: Schema.String.annotations({
        description: 'Replacement text'
      }),
      replaceAll: Schema.optionalWith(Schema.Boolean.annotations({
        description: 'Replace all occurrences instead of requiring uniqueness'
      }), { default: () => false, exact: true }),
    }),
    outputSchema: Schema.String,
  },
  errorSchema: FsErrorSchema,
  emissionSchema: Schema.Struct({
    type: Schema.Literal('file_edit_base_content'),
    path: Schema.String,
    baseContent: Schema.String,
  }),
  stream: {
    initial: { emitted: false },
    onInput: (input, state: { emitted: boolean }, ctx) => Effect.gen(function* () {
      const path = input.path

      // --- Validation: path must exist ---
      if (path?.isFinal && !state.emitted) {
        const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
        const fs = yield* Fs
        const { path: expandedPath } = expandScratchpadPath(path.value, scratchpadPath)
        const fullPath = resolve(cwd, expandedPath)

        const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
        if (!exists) {
          return yield* new StreamValidationError({
            message: `File not found: ${path.value}`,
          })
        }

        // Emit base content for preview diffs (only once, after path validated)
        const content = yield* fs.readText(fullPath).pipe(Effect.option)
        if (Option.isSome(content)) {
          yield* ctx.emit({ type: 'file_edit_base_content', path: path.value, baseContent: content.value })
        }
        return { emitted: true }
      }

      // --- Validation: old text must be found in file ---
      if (path?.isFinal && input.old?.isFinal && state.emitted) {
        const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
        const fs = yield* Fs
        const { path: expandedPath } = expandScratchpadPath(path.value, scratchpadPath)
        const fullPath = resolve(cwd, expandedPath)
        const content = yield* fs.readText(fullPath).pipe(Effect.catchAll(() => Effect.succeed('')))

        if (!content.includes(input.old.value)) {
          return yield* new StreamValidationError({
            message: `Text not found in file. Read the file first to get exact content.`,
          })
        }
      }

      return state
    }),
  },
  execute: ({ path, old: oldStr, new: newStr, replaceAll }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const { path: expandedPath } = expandScratchpadPath(path, scratchpadPath)
    const fullPath = resolve(cwd, expandedPath)

    const content = yield* fs.readText(fullPath).pipe(
      Effect.mapError(() => fsError(`Failed to read ${path}`))
    )

    const applied = yield* Effect.try({
      try: () => validateAndApply(content, oldStr, newStr, replaceAll),
      catch: (e) => fsError(e instanceof Error ? e.message : String(e)),
    })

    yield* fs.writeFile(fullPath, applied.result).pipe(
      Effect.mapError(() => fsError(`Failed to write ${path}`))
    )

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

export const treeTool = defineHarnessTool({
  definition: {
    name: 'tree',
    description: 'List directory structure with optional gitignore filtering. Use this instead of running ls, find, or tree in the shell.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path from cwd. Use $M/ prefix for scratchpad path.'
      }),
      recursive: Schema.optionalWith(Schema.Boolean.annotations({
        description: 'Include subdirectories (default: true)'
      }), { default: () => true, exact: true }),
      maxDepth: Schema.optionalWith(Schema.Number.annotations({
        description: 'Maximum depth to traverse'
      }), { as: 'Option', exact: true }),
      gitignore: Schema.optionalWith(Schema.Boolean.annotations({
        description: 'Respect .gitignore patterns (default: true)'
      }), { default: () => true, exact: true }),
    }),
    outputSchema: Schema.Array(TreeEntry),
  },
  errorSchema: FsErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.path?.isFinal) return {}
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const fs = yield* Fs
      const { path: expandedPath } = expandScratchpadPath(input.path.value, scratchpadPath)
      const fullPath = resolve(cwd, expandedPath)
      const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!exists) {
        return yield* new StreamValidationError({ message: `Path not found: ${input.path.value}` })
      }
      return {}
    }),
  },
  execute: ({ path, gitignore, maxDepth }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const { path: expandedPath } = expandScratchpadPath(path, scratchpadPath)
    const fullPath = resolve(cwd, expandedPath)
    const respectGitignore = gitignore

    const walkOptions = Option.match(maxDepth, {
      onNone: () => ({ respectGitignore }),
      onSome: (maxDepth) => ({ maxDepth, respectGitignore }),
    })
    const entries = yield* fs.walk(fullPath, walkOptions).pipe(
      Effect.mapError(() => fsError(`Failed to list ${path}`))
    )

    return entries.map(entry => ({
      path: entry.relativePath,
      name: entry.name,
      type: entry.type,
      depth: entry.depth
    }))
  }),
})

// =============================================================================
// fs.search()
// =============================================================================

const SearchMatch = Schema.Struct({
  file: Schema.String,
  match: Schema.String
})

type SearchMatch = Schema.Schema.Type<typeof SearchMatch>

export const grepTool = defineHarnessTool({
  definition: {
    name: 'grep',
    description: 'Search file contents with regex. Use this instead of running grep, rg, or ag in the shell — it uses ripgrep under the hood.',
    inputSchema: Schema.Struct({
      pattern: Schema.String.annotations({
        description: 'Regex pattern to search for'
      }),
      path: Schema.optionalWith(Schema.String.annotations({
        description: 'Directory to search in (default: cwd). Use $M/ prefix for scratchpad path.'
      }), { as: 'Option', exact: true }),
      glob: Schema.optionalWith(Schema.String.annotations({
        description: 'Glob pattern to filter files (e.g., "*.ts")'
      }), { as: 'Option', exact: true }),
      limit: Schema.optionalWith(Schema.Number.annotations({
        description: 'Maximum number of matches to return (default: 50)'
      }), { default: () => 50, exact: true }),
    }),
    outputSchema: Schema.Array(SearchMatch),
  },
  errorSchema: FsErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      const pathValue = input.path?.value
      if (typeof pathValue !== 'string' || !input.path?.isFinal) return {}
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const fs = yield* Fs
      const { path: expandedPath } = expandScratchpadPath(pathValue, scratchpadPath)
      const fullPath = resolve(cwd, expandedPath)
      const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!exists) {
        return yield* new StreamValidationError({ message: `Path not found: ${pathValue}` })
      }
      return {}
    }),
  },
  execute: ({ pattern, path, glob, limit }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const { path: resolvedPath } = expandScratchpadPath(Option.getOrElse(path, () => ''), scratchpadPath)
    const resolvedLimit = limit

    const searchPath = resolvedPath.length > 0
      ? resolve(cwd, resolvedPath)
      : cwd

    const searchParams = Option.match(glob, {
      onNone: () => ({ pattern, searchPath, limit: resolvedLimit }),
      onSome: (glob) => ({ pattern, searchPath, glob, limit: resolvedLimit }),
    })

    return yield* fs.search(searchParams).pipe(
      Effect.mapError(() => fsError(`Search failed for ${pattern}`))
    )
  }),
})

// =============================================================================
// fs.view()
// =============================================================================

export const viewTool = defineHarnessTool({
  definition: {
    name: 'view',
    description: 'Read an image file and return it as image output for visual inspection. Supports PNG, JPEG, WebP, GIF, and SVG files.',
    inputSchema: Schema.Struct({
      path: Schema.String.annotations({
        description: 'Relative path to an image file from cwd. Use $M/ prefix for scratchpad path.'
      }),
    }),
    outputSchema: ContextImageResultSchema,
  },
  errorSchema: FsErrorSchema,
  stream: {
    initial: {},
    onInput: (input, _state, _ctx) => Effect.gen(function* () {
      if (!input.path?.isFinal) return {}
      const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
      const fs = yield* Fs
      const { path: expandedPath } = expandScratchpadPath(input.path.value, scratchpadPath)
      const fullPath = resolve(cwd, expandedPath)
      const exists = yield* fs.exists(fullPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!exists) {
        return yield* new StreamValidationError({ message: `File not found: ${input.path.value}` })
      }
      return {}
    }),
  },
  execute: ({ path: filePath }, _ctx) => Effect.gen(function* () {
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag
    const fs = yield* Fs
    const fullPath = resolveFsPath(filePath, cwd, scratchpadPath)

    yield* fs.readFile(fullPath).pipe(
      Effect.mapError(() => fsError(`Failed to read image: ${filePath}`))
    )

    return yield* captureContextImageFromFile({
        absolutePath: fullPath,
        logicalPath: filePath,
      }).pipe(
        Effect.map((image) => ({ _tag: 'ContextImageResult' as const, image })),
        Effect.mapError((error) => fsError(error.message)),
      )
  }),
})

// =============================================================================
// Filesystem Tools Group
// =============================================================================

export const fsTools = [readTool, writeTool, editTool, treeTool, grepTool, viewTool]
