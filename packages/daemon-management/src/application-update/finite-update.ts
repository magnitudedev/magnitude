import { Clock, Effect, Option, Schema } from "effect"
import { UpdateRelease } from "@magnitudedev/release/hosted-update"
import type { DesktopUpdateState } from "@magnitudedev/sdk/desktop-host"
import { PreparedUpdateStore, type PreparedUpdate } from "../desktop-native/prepared-update"
import { UpdatePreferences } from "../desktop-native/update-preferences"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"
import { acquireApplicationMaintenance } from "../desktop-native/application-owner"
import { preparedUpdateFailure } from "./prepared-update-installation"

const presentPrepared = (pending: Option.Option<PreparedUpdate>): DesktopUpdateState["transfer"] => Option.match(pending, {
  onNone: () => ({ _tag: "Idle" }),
  onSome: record => Option.match(preparedUpdateFailure(record), {
    onNone: () => ({ _tag: "Ready", version: record.release.version }),
    onSome: message => ({ _tag: "InstallationFailed", version: record.release.version, message }),
  }),
})

/** Reads persisted preparation and preference only; no source, identity, owner or timer is acquired. */
export const readPreparedUpdateState = Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const preferences = yield* UpdatePreferences
  const pending = yield* store.read
  const preference = yield* preferences.read.pipe(Effect.match({
    onSuccess: autoDownload => ({ _tag: "Known", autoDownload }) as const,
    onFailure: error => ({ _tag: "Unavailable", message: error.message }) as const,
  }))
  return { transfer: presentPrepared(pending), check: { _tag: "Idle" }, preference } satisfies DesktopUpdateState
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))

export const discardPreparedUpdate = Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  const initial = yield* readPreparedUpdateState
  if (initial.transfer._tag !== "Ready" && initial.transfer._tag !== "InstallationFailed") {
    return yield* new ApplicationUpdateFailed({ message: "There is no prepared update to discard." })
  }
  yield* store.discard
  return { ...initial, transfer: { _tag: "Idle" } } satisfies DesktopUpdateState
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))

/** Caller retains maintenance ownership. All transfer work ends before this finite command returns. */
export const runFiniteUpdatePreparation = (action: "check" | "download" | "discard") => Effect.scoped(Effect.gen(function* () {
  const store = yield* PreparedUpdateStore
  if (action === "discard") return yield* discardPreparedUpdate
  const initial = yield* readPreparedUpdateState
  if (action === "download" && initial.transfer._tag !== "Idle") return initial
  const source = yield* ApplicationUpdateSource
  const candidate = yield* source.check("manual")
  const check = { _tag: "Succeeded", at: yield* Clock.currentTimeMillis } as const
  if (initial.transfer._tag !== "Idle") return { ...initial, check }
  if (Option.isNone(candidate)) return { ...initial, check }
  if (action === "check") return { ...initial, check,
    transfer: { _tag: "Available", version: candidate.value.version, bytes: candidate.value.bytes } } satisfies DesktopUpdateState
  yield* store.removeAbandonedTransfers
  const archive = yield* source.download(candidate.value, () => Effect.void)
  yield* source.stage(archive, candidate.value)
  const saved = yield* store.read
  if (Option.isNone(saved) || !Schema.equivalence(UpdateRelease)(saved.value.release, candidate.value) || saved.value.installation._tag !== "Unattempted") {
    return yield* new ApplicationUpdateFailed({ message: "The update download did not publish a complete prepared installer." })
  }
  return { ...initial, check, transfer: presentPrepared(saved) } satisfies DesktopUpdateState
})).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))

/** Admission is nonblocking and rechecks installation after acquiring the application lock. */
export const runApplicationUpdateMaintenance = (directory: string, action: "check" | "download" | "discard") =>
  Effect.scoped(acquireApplicationMaintenance(directory).pipe(Effect.zipRight(runFiniteUpdatePreparation(action))))
