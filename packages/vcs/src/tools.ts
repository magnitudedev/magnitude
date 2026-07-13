/**
 * VCS Tools
 *
 * Tool definitions and state models for the shadow VCS checkpoint system.
 * Owned by the VCS package — real layer provides these, no-op layer returns empty.
 */

import { Effect, Option, Schema } from "effect"
import { BaseStateSchema, defineHarnessTool, defineStateModel, type ToolkitEntry } from "@magnitudedev/harness"
import { ShadowVcs } from "./service"
import { selectorToRestoreScope } from "./path-selector"
import type { VcsFailure } from "./errors"
import type { PointInTime } from "./types"

// ---------------------------------------------------------------------------
// Error schema
// ---------------------------------------------------------------------------

const CheckpointErrorSchema = Schema.TaggedStruct('CheckpointError', {
  message: Schema.String,
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map a VCS Effect to the tool's expected schema, catching VcsFailure. */
function runVcs<A>(
  eff: Effect.Effect<A, VcsFailure, ShadowVcs>,
): Effect.Effect<A, { _tag: 'CheckpointError'; message: string }, ShadowVcs> {
  return eff.pipe(
    Effect.mapError((e) =>
      ({
        _tag: 'CheckpointError' as const,
        message: String(e),
      }),
    ),
  )
}

/**
 * Parse an HH:MM:SS string into a Date (using today's date in the given
 * timezone). Returns Option.none() if the format is invalid.
 */
function parseTimeString(timeStr: string, timezone: string): Option.Option<Date> {
  const match = timeStr.match(/^(\d{2}):(\d{2}):(\d{2})$/)
  if (!match) return Option.none()

  const [, hh, mm, ss] = match
  const hour = Number(hh)
  const minute = Number(mm)
  const second = Number(ss)

  if (hour > 23 || minute > 59 || second > 59) return Option.none()

  // Build a Date in the session timezone (or local if none)
  const now = new Date()
  const datePart = new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    ...(timezone ? { timeZone: timezone } : {}),
  }).format(now)

  // datePart is like "2026-05-28"
  const iso = `${datePart}T${hh}:${mm}:${ss}`

  if (timezone) {
    // Parse in the target timezone by appending the offset
    const offsetMs = getTimezoneOffsetMs(iso, timezone)
    if (Option.isNone(offsetMs)) return Option.none()
    return Option.some(new Date(new Date(iso + 'Z').getTime() - offsetMs.value))
  }

  return Option.some(new Date(iso))
}

/**
 * Get the timezone offset in milliseconds for a given ISO-ish datetime string
 * in the specified timezone. Used to convert a local-time string to UTC.
 */
function getTimezoneOffsetMs(isoLocal: string, timezone: string): Option.Option<number> {
  try {
    const dt = new Date(isoLocal)
    // Use Intl to format the same instant in the target timezone
    const formatter = new Intl.DateTimeFormat('en-GB', {
      timeZone: timezone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
    const parts = formatter.formatToParts(dt)
    const get = (type: string) => parts.find((p) => p.type === type)?.value
    const yyyy = get('year')
    const MM = get('month')
    const dd = get('day')
    const hh = get('hour')
    const mm = get('minute')
    const ss = get('second')
    if (!yyyy || !MM || !dd || !hh || !mm || !ss) return Option.none()

    const localInTz = new Date(`${yyyy}-${MM}-${dd}T${hh}:${mm}:${ss}`)
    return Option.some(dt.getTime() - localInTz.getTime())
  } catch {
    return Option.none()
  }
}

/**
 * Build a PointInTime from a parsed Date.
 */
function pointInTimeFromDate(when: Date): PointInTime {
  return { kind: 'time' as const, when }
}

// ---------------------------------------------------------------------------
// checkpoint_rollback tool
// ---------------------------------------------------------------------------

export const checkpointRollbackTool = defineHarnessTool({
  definition: {
    name: 'checkpoint_rollback',
    description:
      'Roll back changes you made since a turn boundary. Operates on your private checkpoint system, not the user\'s git repository. Use the exact HH:MM:SS timestamp from a --- separator in your conversation. The glob parameter is required to prevent accidentally reverting unrelated changes.',
    inputSchema: Schema.Struct({
      since: Schema.String.annotations({
        description: 'HH:MM:SS timestamp from a --- separator, e.g. "01:35:00"',
      }),
      glob: Schema.String.annotations({
        description: 'Glob pattern for files to roll back, e.g. "src/auth*", "**/*.test.ts", "package.json". Required.',
      }),
    }),
    outputSchema: Schema.Struct({
      diff: Schema.String.annotations({
        description: 'Unified diff of what was rolled back',
      }),
      files: Schema.Array(Schema.Struct({
        path: Schema.String,
        status: Schema.Literal('added', 'deleted', 'modified'),
      })),
      fileCount: Schema.Number,
      additions: Schema.Number,
      deletions: Schema.Number,
    }),
  },
  errorSchema: CheckpointErrorSchema,
  execute: ({ since, glob }) =>
    Effect.gen(function* () {
      const vcs = yield* ShadowVcs
      const timezone = vcs.timezone

      const when = parseTimeString(since, timezone)
      if (Option.isNone(when)) {
        return yield* Effect.fail({
          _tag: 'CheckpointError' as const,
          message:
            'Invalid since timestamp. Use the exact timestamp from the --- separator (format: HH:MM:SS).',
        })
      }

      const point = pointInTimeFromDate(when.value)
      const scope = selectorToRestoreScope(glob)

      // Get the diff BEFORE restoring so we know what's being rolled back
      const diffBefore = yield* runVcs(
        vcs.diffWorking({ against: point, pathFilter: glob }),
      )

      const fullDiff = diffBefore.files.map((f) => f.diff).join('\n')
      const plusLines = fullDiff.split('\n').filter(l => l.startsWith('+') && !l.startsWith('+++')).length
      const minusLines = fullDiff.split('\n').filter(l => l.startsWith('-') && !l.startsWith('---')).length

      // Now perform the restore
      yield* runVcs(
        vcs.restore({ to: point, scope }),
      )

      return {
        diff: fullDiff,
        files: diffBefore.files.map(f => ({ path: f.path, status: f.status as 'added' | 'deleted' | 'modified' })),
        fileCount: diffBefore.files.length,
        additions: plusLines,
        deletions: minusLines,
      }
    }),
})

// ---------------------------------------------------------------------------
// checkpoint_changes tool
// ---------------------------------------------------------------------------

export const checkpointChangesTool = defineHarnessTool({
  definition: {
    name: 'checkpoint_changes',
    description:
      'Show what you changed since a turn boundary. Operates on your private checkpoint system, not the user\'s git repository. Use the exact HH:MM:SS timestamp from a --- separator in your conversation. Optionally scope to a glob pattern.',
    inputSchema: Schema.Struct({
      since: Schema.String.annotations({
        description: 'HH:MM:SS timestamp from a --- separator, e.g. "01:35:00"',
      }),
      glob: Schema.optionalWith(
        Schema.String.annotations({
          description: 'Optional glob pattern to scope the diff, e.g. "src/auth*"',
        }),
        { as: 'Option', exact: true },
      ),
    }),
    outputSchema: Schema.Struct({
      diff: Schema.String.annotations({
        description: 'Unified diff of all changes since the checkpoint',
      }),
      files: Schema.Array(Schema.Struct({
        path: Schema.String,
        status: Schema.Literal('added', 'deleted', 'modified'),
      })),
      fileCount: Schema.Number,
      additions: Schema.Number,
      deletions: Schema.Number,
    }),
  },
  errorSchema: CheckpointErrorSchema,
  execute: ({ since, glob }) =>
    Effect.gen(function* () {
      const vcs = yield* ShadowVcs
      const timezone = vcs.timezone

      const when = parseTimeString(since, timezone)
      if (Option.isNone(when)) {
        return yield* Effect.fail({
          _tag: 'CheckpointError' as const,
          message:
            'Invalid since timestamp. Use the exact timestamp from the --- separator (format: HH:MM:SS).',
        })
      }

      const point = pointInTimeFromDate(when.value)

      const diff = yield* runVcs(
        vcs.diffWorking(Option.match(glob, {
          onNone: () => ({ against: point }),
          onSome: (pathFilter) => ({ against: point, pathFilter }),
        })),
      )

      const fullDiff = diff.files.map((f) => f.diff).join('\n')
      const plusLines = fullDiff.split('\n').filter(l => l.startsWith('+') && !l.startsWith('+++')).length
      const minusLines = fullDiff.split('\n').filter(l => l.startsWith('-') && !l.startsWith('---')).length

      return {
        diff: fullDiff,
        files: diff.files.map(f => ({ path: f.path, status: f.status as 'added' | 'deleted' | 'modified' })),
        fileCount: diff.files.length,
        additions: plusLines,
        deletions: minusLines,
      }
    }),
})

// =============================================================================
// State models
// =============================================================================

const CheckpointFileSchema = Schema.Struct({
  path: Schema.String,
  status: Schema.Literal('added', 'deleted', 'modified'),
})

export const CheckpointRollbackStateSchema = Schema.extend(
  BaseStateSchema,
  Schema.Struct({
    since: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    glob: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    diff: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    files: Schema.optionalWith(Schema.Array(CheckpointFileSchema), { as: 'Option', exact: true }),
    fileCount: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    additions: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    deletions: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  }),
)
export type CheckpointRollbackState = typeof CheckpointRollbackStateSchema.Type

const rollbackInitial: Omit<CheckpointRollbackState, 'phase' | 'errorMessage'> = {
  since: Option.none(),
  glob: Option.none(),
  diff: Option.none(),
  files: Option.none(),
  fileCount: Option.none(),
  additions: Option.none(),
  deletions: Option.none(),
}

export const checkpointRollbackModel = defineStateModel(checkpointRollbackTool)({
  state: CheckpointRollbackStateSchema,
  initial: rollbackInitial,
  reduce: (state, event): CheckpointRollbackState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'since') {
          return { ...state, since: Option.some(Option.getOrElse(state.since, () => '') + event.delta) }
        }
        if (event.field === 'glob') {
          return { ...state, glob: Option.some(Option.getOrElse(state.glob, () => '') + event.delta) }
        }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          since: typeof event.input.since === 'string' ? Option.some(event.input.since) : state.since,
          glob: typeof event.input.glob === 'string' ? Option.some(event.input.glob) : state.glob,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const out = event.result.output
            return {
              ...state,
              phase: 'completed',
              diff: Option.some(out.diff),
              files: Option.some(out.files),
              fileCount: Option.some(out.fileCount),
              additions: Option.some(out.additions),
              deletions: Option.some(out.deletions),
            }
          }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error' }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})

// ---------------------------------------------------------------------------

export const CheckpointChangesStateSchema = Schema.extend(
  BaseStateSchema,
  Schema.Struct({
    since: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    glob: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    diff: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    files: Schema.optionalWith(Schema.Array(CheckpointFileSchema), { as: 'Option', exact: true }),
    fileCount: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    additions: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    deletions: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  }),
)
export type CheckpointChangesState = typeof CheckpointChangesStateSchema.Type

const changesInitial: Omit<CheckpointChangesState, 'phase' | 'errorMessage'> = {
  since: Option.none(),
  glob: Option.none(),
  diff: Option.none(),
  files: Option.none(),
  fileCount: Option.none(),
  additions: Option.none(),
  deletions: Option.none(),
}

export const checkpointChangesModel = defineStateModel(checkpointChangesTool)({
  state: CheckpointChangesStateSchema,
  initial: changesInitial,
  reduce: (state, event): CheckpointChangesState => {
    switch (event._tag) {
      case 'ToolInputStarted':
        return { ...state, phase: 'streaming' }
      case 'ToolInputFieldChunk':
        if (event.field === 'since') {
          return { ...state, since: Option.some(Option.getOrElse(state.since, () => '') + event.delta) }
        }
        if (event.field === 'glob') {
          return { ...state, glob: Option.some(Option.getOrElse(state.glob, () => '') + event.delta) }
        }
        return state
      case 'ToolInputReady':
        return state
      case 'ToolExecutionStarted':
        return {
          ...state,
          phase: 'executing',
          since: typeof event.input.since === 'string' ? Option.some(event.input.since) : state.since,
          glob: typeof event.input.glob === 'string' ? Option.some(event.input.glob) : state.glob,
        }
      case 'ToolExecutionEnded': {
        switch (event.result._tag) {
          case 'Success': {
            const out = event.result.output
            return {
              ...state,
              phase: 'completed',
              diff: Option.some(out.diff),
              files: Option.some(out.files),
              fileCount: Option.some(out.fileCount),
              additions: Option.some(out.additions),
              deletions: Option.some(out.deletions),
            }
          }
          case 'Error':
            return { ...state, phase: 'error' }
          case 'Denied':
            return { ...state, phase: 'rejected' }
          case 'Interrupted':
            return { ...state, phase: 'interrupted' }
          default:
            return state
        }
      }
      case 'ToolInputRejected':
        return { ...state, phase: 'error' }
      case 'ToolEmission':
      case 'ToolInputFieldComplete':
      default:
        return state
    }
  },
})

// ---------------------------------------------------------------------------
// VcsToolEntry export
// ---------------------------------------------------------------------------

export interface VcsToolEntry {
  readonly key: 'checkpointRollback' | 'checkpointChanges'
  readonly tool: ToolkitEntry
}

export function getVcsToolEntries(): ReadonlyArray<VcsToolEntry> {
  return [
    { key: 'checkpointRollback', tool: { tool: checkpointRollbackTool, state: checkpointRollbackModel } },
    { key: 'checkpointChanges', tool: { tool: checkpointChangesTool, state: checkpointChangesModel } },
  ]
}
