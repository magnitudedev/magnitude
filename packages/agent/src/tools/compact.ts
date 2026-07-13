import { Effect, Option, Ref, Schema } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import { resolve } from 'path'
import * as fs from 'fs/promises'
import { CompactionContextTag } from '../compaction/context'
import { WorkingDirectoryTag } from '../execution/working-directory'
import { expandScratchpadPath } from '@magnitudedev/scratchpad'
import { COMPACT_MAX_FILES, COMPACT_MAX_FILE_CHARS, CHARS_PER_TOKEN_LOWER } from '../constants'
import { ToolErrorSchema } from './errors'

const CompactErrorSchema = ToolErrorSchema('CompactError', {})

export const compactTool = defineHarnessTool({
  definition: {
    name: 'compact',
    description: 'System tool for context compaction. Do not call directly — the system will instruct you when to use it.',
    inputSchema: Schema.Struct({
      summary: Schema.String.annotations({
        description: 'What happened: decisions made, work completed, current state, user instructions and preferences, work in progress. Be specific: file paths, function names, error messages, architectural decisions, user requirements. Enough for your future self to continue without re-reading the conversation.',
      }),
      reflection: Schema.String.annotations({
        description: 'What went wrong, incorrect assumptions, approaches that failed, what to do differently. Not what happened — what your future self should change. Name the reasoning traps so your future self avoids them.',
      }),
      files: Schema.optionalWith(Schema.Array(Schema.String).annotations({
        description: 'File paths to read and preserve verbatim in future context. Use for source code being actively edited, configs, or content that cannot survive summarization. Max 10 files.',
      }), { default: () => [], exact: true }),
    }),
    outputSchema: Schema.Struct({
      status: Schema.Literal('ok'),
      filesRead: Schema.Number,
      budgetUsed: Schema.Number,
      budgetTotal: Schema.Number,
    }),
  },
  errorSchema: CompactErrorSchema,
  execute: (input, _ctx) => Effect.gen(function* () {
    const ctxOption = yield* Effect.serviceOption(CompactionContextTag)

    if (Option.isNone(ctxOption)) {
      return yield* Effect.fail({
        _tag: 'CompactError' as const,
        message: 'compact() is a system tool that can only be called during context compaction. Do not call it directly.',
      })
    }

    const ctx = ctxOption.value
    const { cwd, scratchpadPath } = yield* WorkingDirectoryTag

    const charBudget = ctx.maxPayloadTokens * CHARS_PER_TOKEN_LOWER
    let remaining = charBudget - input.summary.length - input.reflection.length

    const filePaths = input.files.slice(0, COMPACT_MAX_FILES)
    const fileResults: Array<{ path: string; content: string }> = []
    let filesRead = 0

    for (const filePath of filePaths) {
      if (remaining <= 0) break
      const { path: expandedPath } = expandScratchpadPath(filePath, scratchpadPath)
      const fullPath = resolve(cwd, expandedPath)
      const readResult = yield* Effect.tryPromise({
        try: () => fs.readFile(fullPath, 'utf-8'),
        catch: (e) => e instanceof Error ? e.message : String(e),
      }).pipe(Effect.either)

      if (readResult._tag === 'Right') {
        const raw = readResult.right
        const limit = Math.min(COMPACT_MAX_FILE_CHARS, remaining)
        if (raw.length <= limit) {
          // Below limit — include full content
          remaining -= raw.length
          fileResults.push({ path: filePath, content: raw })
          filesRead++
        } else {
          // Above limit — just reference, don't include content
          fileResults.push({ path: filePath, content: `[${raw.length} chars — read file as needed]` })
          filesRead++
        }
      } else {
        // Skip files that can't be read — just log and continue
        yield* Effect.logWarning(`[compact] Failed to read file ${filePath}: ${readResult.left}`)
      }
    }

    const totalUsed = charBudget - remaining

    yield* Ref.set(ctx.resultRef, {
      summary: input.summary,
      reflection: input.reflection,
      files: fileResults,
    })

    return {
      status: 'ok' as const,
      filesRead,
      budgetUsed: totalUsed,
      budgetTotal: charBudget,
    }
  }),
})
