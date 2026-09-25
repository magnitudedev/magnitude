import { Effect, Option } from "effect"
import { ApplicationUpdateFailed } from "./application-update"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import type { MacExclusiveInstallationLease } from "../desktop-native/mac-update-lease"
import { MacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { MacUpdateArchiveStager } from "../desktop-native/mac-update-staging"
import { openMacUpdateWorkspace, MacUpdateInstallationBusy } from "../desktop-native/mac-update-workspace"

const reconcilePreparation = (workspace: Option.Option.Value<Effect.Effect.Success<ReturnType<typeof openMacUpdateWorkspace>>>) => Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const pending = yield* store.read
  const recovered = yield* workspace.recover
  if (recovered._tag !== "NoTransaction") {
    if (Option.isSome(pending)) {
      if (recovered._tag === "Installed" && recovered.version === pending.value.release.version) yield* store.discard
      else if (recovered._tag === "Preserved") yield* store.recordFailure(pending.value.release, "The previous installation attempt was interrupted. Retry installation explicitly.")
    }
    yield* workspace.retire
  }
  return recovered
})

/** Reconciles an existing transaction without admitting another installation attempt. */
export const recoverMacPreparedInstallation = (bundle: string, retained?: MacExclusiveInstallationLease) => Effect.scoped(Effect.gen(function* () {
  const installation = yield* MacApplicationInstallation
  if (yield* installation.isInstalling(bundle)) return yield* new MacUpdateInstallationBusy()
  const workspace = yield* openMacUpdateWorkspace(bundle, false, retained)
  if (Option.isNone(workspace)) return { _tag: "NoTransaction" } as const
  return yield* reconcilePreparation(workspace.value)
}))

/** Finite installer only. The caller retains application admission and owns startup continuation. */
export const completeMacPreparedInstallation = (options: {
  readonly bundle: string
  readonly version: string
  readonly architecture: "arm64" | "x64"
}, retained?: MacExclusiveInstallationLease) => Effect.scoped(Effect.gen(function* () {
  const installation = yield* MacApplicationInstallation
  if (yield* installation.isInstalling(options.bundle)) return yield* new MacUpdateInstallationBusy()
  const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(options.bundle, true, retained))
  const store = yield* PreparedUpdateStore
  const cleanup = workspace.retire.pipe(Effect.catchTag("MacUpdateCleanupFailed", error => Effect.logWarning(error.message)))
  const recovered = yield* reconcilePreparation(workspace)
  if (recovered._tag !== "NoTransaction") return recovered
  const pending = yield* store.read
  if (Option.isNone(pending)) return yield* new ApplicationUpdateFailed({ message: "There is no prepared application update to install." })
  const release = pending.value.release
  const archive = yield* store.verify(release)
  yield* store.recordAttempt(release)
  return yield* Effect.gen(function* () {
    yield* workspace.clearUnpublishedStaging
    const stager = yield* MacUpdateArchiveStager
    yield* stager.stage(archive, workspace.staging, release)
    const installed = yield* workspace.exchange({ previous: options.version, replacement: release.version, architecture: options.architecture })
    if (installed._tag === "Installed") yield* store.discard
    else yield* store.recordFailure(release, "The installation did not complete. Retry installation explicitly.")
    yield* cleanup
    return installed
  }).pipe(Effect.tapError(error => store.recordFailure(release, error.message)))
}))
