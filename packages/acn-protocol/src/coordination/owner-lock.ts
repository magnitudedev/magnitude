import type { PlatformError } from "@effect/platform/Error"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Effect, Exit, Layer, Option, Schema, Scope } from "effect"
import { AcnProcessStoreInvalid, AcnProcessStoreUnavailable, type AcnProcessStoreError } from "./errors"
import { AcnOwnerRecordSchema, type AcnOwnerRecord } from "./schemas"
import { SqliteMutex } from "./sqlite-mutex"

const reason = (error: PlatformError): string | undefined =>
  error._tag === "SystemError" ? error.reason : undefined

const unavailable = (operation: string, path: string, error: unknown) =>
  new AcnProcessStoreUnavailable({ operation, path, message: String(error) })

const bytesEqual = (left: Uint8Array, right: Uint8Array): boolean =>
  left.length === right.length && left.every((byte, index) => byte === right[index])

const decodeRecord = (path: string, bytes: Uint8Array) =>
  Schema.decodeUnknown(Schema.parseJson(AcnOwnerRecordSchema))(
    new TextDecoder().decode(bytes),
  ).pipe(Effect.mapError((error) => new AcnProcessStoreInvalid({
    path,
    message: `owner record is malformed: ${String(error)}`,
  })))

export type AcnOwnerObservation =
  | { readonly _tag: "Unlocked" }
  | { readonly _tag: "Publishing" }
  | { readonly _tag: "Locked"; readonly owner: AcnOwnerRecord }

export interface AcnOwnerLockHandle {
  readonly previous: Option.Option<AcnOwnerRecord>
  readonly publish: (record: AcnOwnerRecord) => Effect.Effect<void, AcnProcessStoreError>
  readonly close: Effect.Effect<void>
}

export interface AcnOwnerLock {
  readonly tryAcquire: Effect.Effect<
    Option.Option<AcnOwnerLockHandle>,
    AcnProcessStoreError,
    Scope.Scope
  >
  readonly observe: Effect.Effect<AcnOwnerObservation, AcnProcessStoreError>
}

export const AcnOwnerLock = Context.GenericTag<AcnOwnerLock>(
  "@magnitudedev/acn-protocol/coordination/AcnOwnerLock",
)

export const makeAcnOwnerLock = (
  dataDirectory: string,
): Effect.Effect<AcnOwnerLock, never, FileSystem.FileSystem | Path.Path | SqliteMutex> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const paths = yield* Path.Path
    const mutexes = yield* SqliteMutex
    const directory = paths.join(dataDirectory, "acn")
    const mutexPath = paths.join(directory, "owner-lock.sqlite")
    const recordPath = paths.join(directory, "owner.json")

    interface OwnerMutexLease {
      readonly release: Effect.Effect<void>
    }

    const ensureMutex = fs.makeDirectory(directory, { recursive: true }).pipe(
      Effect.zipRight(mutexes.initialize(mutexPath).pipe(
        Effect.mapError((error) => unavailable("initialize-owner-lock", mutexPath, error)),
      )),
      Effect.mapError((error) => error instanceof AcnProcessStoreUnavailable
        ? error
        : unavailable("create-owner-directory", directory, error)),
    )

    const acquireMutex = (): Effect.Effect<OwnerMutexLease | null, AcnProcessStoreError> =>
      Effect.gen(function* () {
        const mutexScope = yield* Scope.make()
        const acquired = yield* mutexes.tryAcquire(mutexPath).pipe(
          Effect.provideService(Scope.Scope, mutexScope),
          Effect.mapError((error) => unavailable("lock-owner", mutexPath, error)),
          Effect.onError(() => Scope.close(mutexScope, Exit.void)),
        )
        if (acquired) return { release: Scope.close(mutexScope, Exit.void) }
        yield* Scope.close(mutexScope, Exit.void)
        return null
      })

    const readPublishedRecord = fs.readFile(recordPath).pipe(
      Effect.map(Option.some),
      Effect.catchAll((error) => reason(error) === "NotFound"
        ? Effect.succeed(Option.none<Uint8Array>())
        : Effect.fail(unavailable("read-owner", recordPath, error))),
    )

    const readOptionalRecord = fs.readFile(recordPath).pipe(
      Effect.flatMap((bytes) => decodeRecord(recordPath, bytes).pipe(Effect.option)),
      Effect.catchAll((error) => reason(error) === "NotFound"
        ? Effect.succeed(Option.none<AcnOwnerRecord>())
        : Effect.fail(unavailable("read-owner", recordPath, error))),
    )

    const tryAcquire: AcnOwnerLock["tryAcquire"] = ensureMutex.pipe(
      Effect.zipRight(acquireMutex()),
      Effect.flatMap((mutex) => {
        if (mutex === null) return Effect.succeed(Option.none<AcnOwnerLockHandle>())
        return Effect.gen(function* () {
          const previous = yield* readOptionalRecord
          const publish = (record: AcnOwnerRecord) => {
            const temporary = `${recordPath}.${crypto.randomUUID()}.tmp`
            return Schema.encode(Schema.parseJson(AcnOwnerRecordSchema))(record).pipe(
              Effect.mapError((error) => new AcnProcessStoreInvalid({
                path: recordPath,
                message: String(error),
              })),
              Effect.flatMap((encoded) => fs.writeFileString(temporary, encoded, {
                flag: "wx",
                mode: 0o600,
              })),
              Effect.flatMap(() => fs.rename(temporary, recordPath)),
              Effect.mapError((error) => error instanceof AcnProcessStoreInvalid
                ? error
                : unavailable("publish-owner", recordPath, error)),
              Effect.onError(() => fs.remove(temporary).pipe(Effect.ignore)),
            )
          }
          const handle: AcnOwnerLockHandle = {
            previous,
            publish,
            close: mutex.release,
          }
          yield* Effect.addFinalizer(() => handle.close)
          return Option.some(handle)
        }).pipe(Effect.onError(() => mutex.release))
      }),
    )

    const observe: AcnOwnerLock["observe"] =
      ensureMutex.pipe(
        Effect.zipRight(acquireMutex()),
        Effect.flatMap((firstProbe) => {
          if (firstProbe !== null) {
            return firstProbe.release.pipe(
              Effect.as<AcnOwnerObservation>({ _tag: "Unlocked" }),
            )
          }
          return Effect.gen(function* () {
            const first = yield* readPublishedRecord
            if (Option.isNone(first)) return { _tag: "Publishing" as const }
            const decoded = yield* Effect.either(decodeRecord(recordPath, first.value))
            if (decoded._tag === "Left") return { _tag: "Publishing" as const }
            const secondProbe = yield* acquireMutex()
            if (secondProbe !== null) {
              yield* secondProbe.release
              return { _tag: "Unlocked" as const }
            }
            const second = yield* readPublishedRecord
            return Option.isSome(second) && bytesEqual(first.value, second.value)
              ? { _tag: "Locked" as const, owner: decoded.right }
              : { _tag: "Publishing" as const }
          })
        }),
      )

    return AcnOwnerLock.of({ tryAcquire, observe })
  })

export const AcnOwnerLockLive = (dataDirectory: string) =>
  Layer.effect(AcnOwnerLock, makeAcnOwnerLock(dataDirectory))
