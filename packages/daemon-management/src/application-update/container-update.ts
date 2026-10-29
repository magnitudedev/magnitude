import { FileSystem } from "@effect/platform"
import { Clock, Effect, ExecutionStrategy, Exit, Option, Schema, Scope, Stream, SubscriptionRef } from "effect"
import { randomUUID } from "node:crypto"
import { dirname, join } from "node:path"
import { isNewerVersion } from "@magnitudedev/release"
import { Version, type UpdateCheckReason } from "@magnitudedev/release/hosted-update"
import type { ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { ApplicationUpdate, ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"

export const CONTAINER_UPDATE_INSTRUCTION = "Pull the new image (docker pull magnitudedev/magnitude) and recreate the container."
export const containerUpdateNotice = (version: string) => `Magnitude ${version} is available. ${CONTAINER_UPDATE_INSTRUCTION}`
export const containerUpdateRefusal = `Magnitude in a container is updated by its image, not in place. ${CONTAINER_UPDATE_INSTRUCTION}`

const LastRun = Schema.parseJson(Schema.Struct({ version: Version }))

/**
 * A container is upgraded by starting a newer image on the same data directory, which nothing here
 * installs, so the data directory keeps the version that last ran. Starting a newer version records it
 * as `applied` from that one, for the next check to report; the same version or a downgrade records nothing.
 */
export const recordContainerUpgrade = (dataDirectory: string, version: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const store = yield* PreparedUpdateStore
  const path = join(dataDirectory, "updates", "last-run.json")
  const previous = yield* fs.readFileString(path).pipe(Effect.flatMap(Schema.decodeUnknown(LastRun)), Effect.option)
  if (Option.exists(previous, last => last.version === version)) return
  if (Option.isSome(previous) && isNewerVersion(version, previous.value.version)) {
    yield* store.recordOutcome({ outcome: "applied", version, from: Option.some(previous.value.version), reason: Option.none() })
  }
  const text = yield* Schema.encode(LastRun)({ version })
  yield* fs.makeDirectory(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = `${path}.${randomUUID()}.tmp`
  yield* fs.writeFileString(temporary, text)
  yield* fs.rename(temporary, path)
}).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "The running version could not be recorded." })))

/**
 * Updates of a Docker installation. Checks run as everywhere, which counts the installation and reports
 * outcomes, but a newer release is only announced: the image is the installation, so an update applied
 * inside the container would be lost when it is recreated.
 */
export const makeContainerApplicationUpdate = Effect.gen(function* () {
  const source = yield* ApplicationUpdateSource
  const workers = yield* Scope.fork(yield* Scope.Scope, ExecutionStrategy.sequential)
  const refused = new ApplicationUpdateFailed({ message: containerUpdateRefusal })
  const state = yield* SubscriptionRef.make<ApplicationUpdateState>({ transfer: { _tag: "Idle" }, check: { _tag: "Idle" },
    preference: { _tag: "Unavailable", message: containerUpdateRefusal } })
  const gate = yield* Effect.makeSemaphore(1)
  const settle = (f: (current: ApplicationUpdateState) => ApplicationUpdateState) =>
    SubscriptionRef.update(state, current => current.transfer._tag === "Closed" ? current : f(current))
  const check = (reason: UpdateCheckReason) => gate.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(state)
    if (current.transfer._tag === "Closed") return yield* new ApplicationUpdateFailed({ message: "Magnitude is quitting." })
    if (current.check._tag === "Checking") return
    yield* SubscriptionRef.set(state, { ...current, check: { _tag: "Checking" } })
    yield* source.check(reason).pipe(
      Effect.flatMap(candidate => Clock.currentTimeMillis.pipe(Effect.flatMap(at => settle(current => ({ ...current, check: { _tag: "Succeeded", at },
        transfer: Option.match(candidate, { onNone: () => ({ _tag: "Idle" }) as const, onSome: release => ({ _tag: "ImageAvailable", version: release.version }) as const }) }))))),
      Effect.catchAll(error => settle(current => ({ ...current, check: { _tag: "Failed", message: error.message } }))),
      Effect.catchAllDefect(() => settle(current => ({ ...current, check: { _tag: "Failed", message: "Could not check for updates." } }))),
      Effect.interruptible, Effect.forkIn(workers),
    )
  })).pipe(Effect.uninterruptible)
  const close = SubscriptionRef.update(state, (current): ApplicationUpdateState => ({ ...current, transfer: { _tag: "Closed" } })).pipe(
    Effect.zipRight(Scope.close(workers, Exit.void)), Effect.uninterruptible)
  yield* Effect.addFinalizer(() => close)
  return ApplicationUpdate.of({
    state: SubscriptionRef.get(state), changes: state.changes, check, close,
    download: refused, discard: refused, setAutoDownload: () => refused, requireReady: refused,
  })
})

/** The newer image versions an update stream announces, without repeating one that is still announced. */
export const containerImageVersions = (changes: Stream.Stream<ApplicationUpdateState>) => changes.pipe(
  Stream.map(state => state.transfer), Stream.filterMap(transfer => transfer._tag === "ImageAvailable" ? Option.some(transfer.version) : Option.none()),
  Stream.changes)
