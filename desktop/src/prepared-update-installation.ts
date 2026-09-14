import { Context, Effect, Option } from "effect"
import { isNewerVersion, isValidVersion } from "@magnitudedev/release"
import type { UpdateRelease } from "@magnitudedev/release/hosted-update"
import { PreparedUpdateStore, type PreparedUpdate } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"

export interface PreparedUpdateInstaller {
  readonly requiresAuthorization: boolean
  /** Returns after the native handoff is admitted; the retiring owner then exits. */
  readonly install: (archive: string, release: UpdateRelease, showWindow: boolean) => Effect.Effect<void, ApplicationUpdateFailed>
}
export interface UpdateInstallationIntent {
  readonly showWindow: boolean
  readonly allowAuthorizationPrompt: boolean
}
export const PreparedUpdateInstaller = Context.GenericTag<PreparedUpdateInstaller>("desktop/PreparedUpdateInstaller")

export const preparedUpdateFailure = (record: PreparedUpdate): Option.Option<string> => {
  switch (record.installation._tag) {
    case "Unattempted": return Option.none()
    case "Attempted": return Option.some("The update did not complete.")
    case "Failed": return Option.some(record.installation.reason)
  }
}

/** Native exclusion is checked first by bootstrap; this runs before any owned service starts. */
export const reconcilePreparedUpdate = (installedVersion: string) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  if (!isValidVersion(installedVersion)) return yield* new ApplicationUpdateFailed({ message: "Could not identify the installed application version." })
  yield* store.removeAbandonedTransfers
  const pending = yield* store.read
  if (Option.isNone(pending)) return pending
  if (!isNewerVersion(pending.value.release.version, installedVersion)) {
    yield* store.discard
    return Option.none<PreparedUpdate>()
  }
  return pending
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))

/** The startup and explicit-retry paths use this same verification and durable attempt barrier. */
export const installPreparedUpdate = (intent: UpdateInstallationIntent) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const installer = yield* PreparedUpdateInstaller
  if (!intent.allowAuthorizationPrompt && installer.requiresAuthorization) return "Deferred" as const
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "Download the application update before restarting." })
  const release = pending.value.release
  const archive = yield* store.verify(release).pipe(
    Effect.tapError(error => store.recordFailure(release, error.message)),
  )
  // No native invocation is possible if this write fails, including an uncertain fsync result.
  yield* store.recordAttempt(release)
  yield* installer.install(archive, release, intent.showWindow).pipe(
    Effect.catchAllDefect(() => new ApplicationUpdateFailed({ message: "The update installer could not be started." })),
    Effect.tapError(error => store.recordFailure(release, error.message)),
  )
  return "Started" as const
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
