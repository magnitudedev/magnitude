import { FileSystem } from "@effect/platform"
import { Clock, Duration, Effect, Option, Schedule, Schema } from "effect"
import { join } from "node:path"
import { localModelResidency, ModelIdSchema, type ModelId } from "@magnitudedev/sdk"
import { withLocalService } from "./server-reach"

/**
 * The local models that were loaded when `serve` stopped to install an update, or that the desktop app had loaded
 * while it served in place of `serve`, so the server that starts next loads them again. A record older than
 * `freshness` belongs to a stop that never resumed and is ignored.
 */
const ResidentModels = Schema.parseJson(Schema.Struct({
  recordedAt: Schema.Number,
  // Versions before the desktop handoff recorded only updates, and wrote no reason.
  after: Schema.optionalWith(Schema.Literal("Update", "Desktop"), { default: () => "Update" as const }),
  models: Schema.Array(ModelIdSchema),
}))
type Resumption = typeof ResidentModels.Type["after"]
const freshness = Duration.minutes(15)
const recordFile = (stateDirectory: string) => join(stateDirectory, "models-after-update.json")

const residentModels = (endpoint: string) => withLocalService(endpoint, client => client.models.getCatalog({})).pipe(
  Effect.map((catalog): ReadonlyArray<ModelId> => catalog._tag === "Initializing" ? [] : catalog.models.flatMap(entry => {
    if (entry._tag !== "Local") return []
    const residency = Option.getOrUndefined(localModelResidency(entry.product))
    return residency && (residency._tag === "Requested" || residency._tag === "Loading" || residency._tag === "Ready") ? [entry.product.modelId] : []
  })))

/**
 * Records the local models the service at `endpoint` has loaded, replacing the previous record; none loaded removes
 * it. Fails when the service cannot be read, leaving the previous record.
 */
export const recordResidentModels = (endpoint: string, stateDirectory: string, after: Resumption) => Effect.gen(function* () {
  const models = yield* residentModels(endpoint)
  const fs = yield* FileSystem.FileSystem
  if (models.length === 0) return yield* fs.remove(recordFile(stateDirectory), { force: true })
  const encoded = yield* Schema.encode(ResidentModels)({ recordedAt: yield* Clock.currentTimeMillis, after, models })
  yield* fs.writeFileString(recordFile(stateDirectory), encoded)
})

/** Records the loaded local models before the service stops to install an update. */
export const recordModelsBeforeUpdate = (endpoint: string, stateDirectory: string) => recordResidentModels(endpoint, stateDirectory, "Update").pipe(
  Effect.catchAll(error => Effect.logWarning(`Could not record the loaded models before the update: ${String(error)}`)))

/**
 * Once the service is ready after an update or after the desktop app quits, loads the recorded models again and
 * removes the record. The new service may still be reading its catalog, so each load is retried for a while.
 */
export const resumeResidentModels = (endpoint: string, stateDirectory: string, notify: (line: string) => Effect.Effect<void>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const file = recordFile(stateDirectory)
  if (!(yield* fs.exists(file))) return
  const record = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decode(ResidentModels)), Effect.option)
  yield* fs.remove(file, { force: true })
  if (Option.isNone(record) || (yield* Clock.currentTimeMillis) - record.value.recordedAt > Duration.toMillis(freshness)) return
  yield* Effect.forEach(record.value.models, modelId => withLocalService(endpoint, client => client.models.load({ modelId })).pipe(
    Effect.retry(Schedule.spaced("5 seconds").pipe(Schedule.compose(Schedule.recurs(24)))),
    Effect.zipRight(notify(`Loading ${modelId} again ${occasion[record.value.after]}.`)),
    Effect.catchAll(() => notify(`Could not load ${modelId} again ${occasion[record.value.after]}; it loads on its next request.`))), { discard: true })
}).pipe(Effect.catchAll(error => Effect.logWarning(`Could not read the models to load again: ${String(error)}`)))

const occasion: Record<Resumption, string> = { Update: "after the update", Desktop: "now that the desktop app has quit" }
