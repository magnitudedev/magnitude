import type { PlatformError } from "@effect/platform/Error"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Duration, Effect, Exit, Layer, Option, Scope } from "effect"
import { AcnRevisionSchema, type AcnRevision } from "../acn-revision"
import { AcnProcessStoreInvalid, AcnProcessStoreUnavailable, type AcnProcessStoreError } from "./errors"
import { SqliteMutex } from "./sqlite-mutex"

const markerName = (revision: AcnRevision): string => String(revision).padStart(20, "0")
const markerPattern = /^\d{20}$/
const developmentKeyPattern = /^[a-f0-9]{16}$/
const holderNamePattern = /^[a-f0-9]{32}\.sqlite$/
const HOLDER_ACQUISITION_RETRY_DELAY = Duration.millis(10)

const reason = (error: PlatformError): string | undefined =>
  error._tag === "SystemError" ? error.reason : undefined

const unavailable = (operation: string, path: string, error: unknown) =>
  new AcnProcessStoreUnavailable({ operation, path, message: String(error) })

const invalid = (path: string, message: string) =>
  new AcnProcessStoreInvalid({ path, message })

export interface DevelopmentRevisionHold {
  readonly revision: AcnRevision
  readonly close: Effect.Effect<void>
}

export interface AcnRevisionStore {
  readonly registerPublished: (revision: AcnRevision) => Effect.Effect<void, AcnProcessStoreError>
  readonly holdDevelopment: (
    revision: AcnRevision,
    key: string,
  ) => Effect.Effect<DevelopmentRevisionHold, AcnProcessStoreError, Scope.Scope>
  readonly selected: Effect.Effect<Option.Option<AcnRevision>, AcnProcessStoreError>
}

export const AcnRevisionStore = Context.GenericTag<AcnRevisionStore>(
  "@magnitudedev/acn-protocol/coordination/AcnRevisionStore",
)

export const makeAcnRevisionStore = (
  dataDirectory: string,
): Effect.Effect<AcnRevisionStore, never, FileSystem.FileSystem | Path.Path | SqliteMutex> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const mutexes = yield* SqliteMutex
    const directory = path.join(dataDirectory, "acn", "revisions")
    const holdsDirectory = path.join(dataDirectory, "acn", "development-holds")
    const ensureDirectory = fs.makeDirectory(directory, { recursive: true }).pipe(
      Effect.mapError((error) => unavailable("create-directory", directory, error)),
    )
    const at = (revision: AcnRevision) => path.join(directory, markerName(revision))
    const holdsAt = (revision: AcnRevision) => path.join(holdsDirectory, markerName(revision))

    const read = (marker: string) => fs.readFile(marker).pipe(
      Effect.mapError((error) => unavailable("read-marker", marker, error)),
    )

    const registerPublished: AcnRevisionStore["registerPublished"] = (revision) =>
      ensureDirectory.pipe(
        Effect.zipRight(fs.writeFile(at(revision), new Uint8Array(), { flag: "wx", mode: 0o600 })),
        Effect.catchAll((error) => reason(error as PlatformError) === "AlreadyExists"
          ? read(at(revision)).pipe(
              Effect.flatMap((bytes) => bytes.length === 0
                ? Effect.void
                : Effect.fail(invalid(at(revision), "published marker is not zero bytes"))),
            )
          : Effect.fail(unavailable("create-published-marker", at(revision), error))),
      )

    const holdDevelopment: AcnRevisionStore["holdDevelopment"] = (revision, key) =>
      Effect.gen(function* () {
        if (!developmentKeyPattern.test(key)) {
          return yield* invalid(at(revision), "development key must be 16 lowercase hexadecimal bytes")
        }
        yield* ensureDirectory
        const marker = at(revision)
        yield* fs.writeFileString(marker, key, { flag: "wx", mode: 0o600 }).pipe(
          Effect.catchAll((error) => reason(error) === "AlreadyExists"
            ? fs.readFileString(marker).pipe(
                Effect.mapError((readError) => unavailable("read-development-marker", marker, readError)),
                Effect.flatMap((current) => current === key
                  ? Effect.void
                  : Effect.fail(invalid(marker, "development revision belongs to another key"))),
              )
            : Effect.fail(unavailable("create-development-marker", marker, error))),
        )

        const holderDirectory = holdsAt(revision)
        yield* fs.makeDirectory(holderDirectory, { recursive: true }).pipe(
          Effect.mapError((error) => unavailable("create-development-holds", holderDirectory, error)),
        )
        const holderPath = path.join(
          holderDirectory,
          `${crypto.randomUUID().replaceAll("-", "")}.sqlite`,
        )
        yield* mutexes.initialize(holderPath).pipe(
          Effect.mapError((error) => unavailable("lock-development-holder", holderPath, error)),
        )
        const mutexScope = yield* Scope.make()
        const acquire = (): Effect.Effect<void, AcnProcessStoreError> =>
          mutexes.tryAcquire(holderPath).pipe(
            Effect.provideService(Scope.Scope, mutexScope),
            Effect.mapError((error) => unavailable("lock-development-holder", holderPath, error)),
            Effect.flatMap((acquired) => !acquired
            ? Effect.sleep(HOLDER_ACQUISITION_RETRY_DELAY).pipe(Effect.zipRight(acquire()))
            : Effect.void))
        yield* acquire().pipe(
          Effect.onError(() => Scope.close(mutexScope, Exit.void)),
        )
        const close = Scope.close(mutexScope, Exit.void).pipe(
          Effect.zipRight(fs.remove(holderPath).pipe(Effect.ignore)),
        )
        yield* Effect.addFinalizer(() => close)
        return { revision, close }
      })

    const holderActive = (
      holderPath: string,
    ): Effect.Effect<boolean, AcnProcessStoreError> =>
      Effect.scoped(mutexes.tryAcquire(holderPath)).pipe(
        Effect.mapError((error) => unavailable("probe-development-holder", holderPath, error)),
        Effect.map((acquired) => !acquired),
      )

    const developmentActive = (
      revision: AcnRevision,
    ): Effect.Effect<boolean, AcnProcessStoreError> => {
      const holderDirectory = holdsAt(revision)
      return fs.readDirectory(holderDirectory).pipe(
        Effect.catchAll((error) => reason(error) === "NotFound"
          ? Effect.succeed<readonly string[]>([])
          : Effect.fail(unavailable("enumerate-development-holders", holderDirectory, error))),
        Effect.flatMap((names) => Effect.forEach(
          names.filter((name) => holderNamePattern.test(name)),
          (name) => holderActive(path.join(holderDirectory, name)),
          { concurrency: "unbounded" },
        )),
        Effect.map((active) => active.some(Boolean)),
      )
    }

    const inspectMarker = (name: string): Effect.Effect<Option.Option<AcnRevision>, AcnProcessStoreError> =>
      Effect.gen(function* () {
        if (!markerPattern.test(name)) return Option.none()
        const value = Number(name)
        const marker = path.join(directory, name)
        if (!Number.isSafeInteger(value) || value <= 0) {
          return yield* invalid(marker, "revision filename is not a positive safe integer")
        }
        const revision = AcnRevisionSchema.make(value)
        const bytes = yield* read(marker)
        if (bytes.length === 0) return Option.some(revision)
        const key = new TextDecoder().decode(bytes)
        if (!developmentKeyPattern.test(key)) {
          return yield* invalid(marker, "marker content is neither published nor development")
        }
        return (yield* developmentActive(revision)) ? Option.some(revision) : Option.none()
      })

    const selected: AcnRevisionStore["selected"] = ensureDirectory.pipe(
      Effect.zipRight(fs.readDirectory(directory)),
      Effect.mapError((error) => unavailable("enumerate-revisions", directory, error)),
      Effect.flatMap((names) => Effect.forEach(names, inspectMarker, { concurrency: "unbounded" })),
      Effect.map((revisions) => revisions.reduce<Option.Option<AcnRevision>>(
        (greatest, revision) => Option.isNone(revision)
          ? greatest
          : Option.isNone(greatest) || revision.value > greatest.value
            ? revision
            : greatest,
        Option.none(),
      )),
    )

    return AcnRevisionStore.of({ registerPublished, holdDevelopment, selected })
  })

export const AcnRevisionStoreLive = (dataDirectory: string) =>
  Layer.effect(AcnRevisionStore, makeAcnRevisionStore(dataDirectory))
