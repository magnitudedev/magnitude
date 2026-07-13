/**
 * Result Persistence — persist tool results to files for retroactive disclosure.
 *
 * File path convention: {resultsDir}/{turnId}_{callId}.json
 * (flat directory, no subdirectories)
 */

import { mkdir } from 'fs/promises'
import { join } from 'path'
import { Data, Effect, Schema } from 'effect'

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

export class PersistError extends Data.TaggedError('PersistError')<{
  readonly operation: 'persist' | 'load' | 'has' | 'ensureDir'
  readonly path: string
  readonly cause: unknown
}> {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function resultPath(resultsDir: string, turnId: string, callId: string): string {
  return join(resultsDir, `${turnId}_${callId}.json`)
}

function ensureDir(dir: string): Effect.Effect<void, PersistError> {
  return Effect.tryPromise({
    try: () => mkdir(dir, { recursive: true }).then(() => undefined),
    catch: (cause) => new PersistError({ operation: 'ensureDir', path: dir, cause }),
  })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Persist a tool result as JSON to {resultsDir}/{turnId}_{callId}.json.
 */
export const persistResult = (
  output: unknown,
  turnId: string,
  callId: string,
  resultsDir: string,
): Effect.Effect<void, PersistError> =>
  Effect.gen(function* () {
    yield* ensureDir(resultsDir)
    const filePath = resultPath(resultsDir, turnId, callId)
    const json = yield* Schema.encodeUnknown(Schema.parseJson({ space: 2 }))(output).pipe(
      Effect.mapError((cause) => new PersistError({ operation: 'persist', path: filePath, cause })),
    )
    yield* Effect.tryPromise({
      try: () => Bun.write(filePath, json),
      catch: (cause) => new PersistError({ operation: 'persist', path: filePath, cause }),
    })
  })

/**
 * Load a previously persisted result.
 * Fails with PersistError if the file does not exist or is not valid JSON.
 */
export const loadResult = (
  turnId: string,
  callId: string,
  resultsDir: string,
): Effect.Effect<unknown, PersistError> => {
  const filePath = resultPath(resultsDir, turnId, callId)
  return Effect.tryPromise({
    try: async () => {
      const text = await Bun.file(filePath).text()
      return text
    },
    catch: (cause) => new PersistError({ operation: 'load', path: filePath, cause }),
  }).pipe(
    Effect.flatMap((text) =>
      Schema.decodeUnknown(Schema.parseJson())(text).pipe(
        Effect.mapError((cause) => new PersistError({ operation: 'load', path: filePath, cause })),
      ),
    ),
  )
}

/**
 * Check whether a persisted result exists for the given turn/call IDs.
 */
export const hasResult = (
  turnId: string,
  callId: string,
  resultsDir: string,
): Effect.Effect<boolean> => {
  const filePath = resultPath(resultsDir, turnId, callId)
  return Effect.tryPromise({
    try: () => Bun.file(filePath).exists(),
    catch: () => false as never,
  })
}
