import * as FileSystem from '@effect/platform/FileSystem'
import type { PlatformError } from '@effect/platform/Error'
import * as Path from '@effect/platform/Path'
import { randomUUID } from 'node:crypto'
import { Effect, Either, Schema, SubscriptionRef, type Equivalence } from 'effect'

import { makeStorageIo } from '../io/storage'
import {
  readRecoverableStructuredFile,
  StructuredFileEncodeFailed,
  writeStructuredFileAtomic,
  writeTextFileAtomic,
} from '../io/structured-file'
import type { StateHandle } from './contracts'

export class StateDocumentInvalid extends Schema.TaggedError<StateDocumentInvalid>()(
  'StateDocumentInvalid',
  {
    path: Schema.String,
    reason: Schema.String,
  },
) {}

export type StateDocumentError = PlatformError | StructuredFileEncodeFailed | StateDocumentInvalid

const safeMessage = (value: unknown): string => String(value).slice(0, 2_000)

export const makeStateDocument = <A, I>(options: {
  readonly path: string
  readonly schema: Schema.Schema<A, I>
  readonly initial: () => A
  readonly equivalence: Equivalence.Equivalence<A>
}): Effect.Effect<
  StateHandle<A, StateDocumentError>,
  StateDocumentError,
  FileSystem.FileSystem | Path.Path
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const io = yield* makeStorageIo()
  const initial = options.initial()
  const validatedInitial = yield* Schema.validate(options.schema)(initial).pipe(Effect.either)
  if (Either.isLeft(validatedInitial)) {
    return yield* Effect.dieMessage(
      `Invalid default for state document ${options.path}: ${safeMessage(validatedInitial.left)}`,
    )
  }

  const write = (value: A) => writeStructuredFileAtomic(
    options.path,
    options.schema,
    value,
    { mode: 0o600 },
  )

  const backupPath = (): string => {
    const timestamp = new Date().toISOString().replaceAll(/[-:.]/g, '')
    return `${options.path}.corrupt-${timestamp}-${randomUUID()}`
  }

  const preserve = (originalText: string) => Effect.gen(function* () {
    const path = backupPath()
    yield* writeTextFileAtomic(path, originalText, { mode: 0o600 })
    return path
  })

  const readUnlocked: Effect.Effect<A, StateDocumentError> = Effect.gen(function* () {
    const result = yield* readRecoverableStructuredFile(options.path, options.schema, {
      rootDefault: () => validatedInitial.right,
      preserveExcessProperties: false,
    })
    if (result._tag === 'Missing') {
      yield* write(validatedInitial.right)
      return validatedInitial.right
    }
    if (result._tag === 'Unrecoverable') {
      return yield* new StateDocumentInvalid({ path: options.path, reason: result.reason })
    }
    if (result._tag === 'Malformed') {
      const preservedAt = yield* preserve(result.originalText)
      yield* write(validatedInitial.right)
      yield* Effect.logWarning('Recovered malformed state document').pipe(
        Effect.annotateLogs({ path: options.path, preservedAt, reason: result.reason.slice(0, 1_000) }),
      )
      return validatedInitial.right
    }
    const canonicalText = yield* Schema.encodeUnknown(
      Schema.parseJson(options.schema, { space: 2 }),
    )(result.value).pipe(
      Effect.map((text) => text.endsWith('\n') ? text : `${text}\n`),
      Effect.mapError((error) => new StructuredFileEncodeFailed({
        path: options.path,
        reason: safeMessage(error),
      })),
    )
    if (result.recovery.recovered || canonicalText !== result.originalText) {
      const preservedAt = result.recovery.resetRoot
        ? yield* preserve(result.originalText)
        : undefined
      yield* write(result.value)
      if (result.recovery.recovered) {
        yield* Effect.logWarning('Recovered invalid state document values').pipe(
          Effect.annotateLogs({
            path: options.path,
            resetRoot: result.recovery.resetRoot,
            removedPaths: result.recovery.removedPaths
              .map((parts) => parts.map(String).join('.'))
              .join(','),
            ...(preservedAt === undefined ? {} : { preservedAt }),
          }),
        )
      }
    }
    return result.value
  }).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
  )

  const initialState = yield* io.withPathLock(options.path, readUnlocked)
  const state = yield* SubscriptionRef.make(initialState)

  const modify: StateHandle<A, StateDocumentError>['modify'] = (transition) =>
    io.withPathLock(options.path, Effect.gen(function* () {
      const current = yield* readUnlocked
      const [result, next] = transition(current)
      const validated = yield* Schema.validate(options.schema)(next).pipe(
        Effect.mapError((error) => new StateDocumentInvalid({
          path: options.path,
          reason: safeMessage(error),
        })),
      )
      yield* Effect.uninterruptible(Effect.gen(function* () {
        if (!options.equivalence(current, validated)) yield* write(validated)
        const resident = yield* SubscriptionRef.get(state)
        if (!options.equivalence(resident, validated)) yield* SubscriptionRef.set(state, validated)
      }))
      return result
    }).pipe(Effect.provideService(FileSystem.FileSystem, fs)))

  return {
    get: SubscriptionRef.get(state),
    changes: state.changes,
    modify,
    update: (transition) => modify((current) => {
      const next = transition(current)
      return [next, next] as const
    }),
  }
})
