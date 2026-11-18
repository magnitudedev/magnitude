import { FileSystem } from "@effect/platform"
import { Clock, Duration, Effect, Option, Schedule, Schema } from "effect"
import { join } from "node:path"
import { localModelResidency, ModelIdSchema, type ModelId } from "@magnitudedev/sdk"
import { withLocalService } from "./server-reach"

/**
 * The local models that were loaded when `serve` stopped to install an update, so the version that starts next
 * loads them again. A record older than `freshness` belongs to an update that never restarted and is ignored.
 */
const ResidentModels = Schema.parseJson(Schema.Struct({ recordedAt: Schema.Number, models: Schema.Array(ModelIdSchema) }))
const freshness = Duration.minutes(15)
const recordFile = (stateDirectory: string) => join(stateDirectory, "models-after-update.json")

const residentModels = (endpoint: string) => withLocalService(endpoint, client => client.models.getCatalog({})).pipe(
  Effect.map((catalog): ReadonlyArray<ModelId> => catalog._tag === "Initializing" ? [] : catalog.models.flatMap(entry => {
    if (entry._tag !== "Local") return []
    const residency = Option.getOrUndefined(localModelResidency(entry.product))
    return residency && (residency._tag === "Requested" || residency._tag === "Loading" || residency._tag === "Ready") ? [entry.product.modelId] : []
  })))

/** Records the loaded local models before the service stops to install an update; nothing is recorded when none is loaded. */
export const recordResidentModels = (endpoint: string, stateDirectory: string) => Effect.gen(function* () {
  const models = yield* residentModels(endpoint)
  if (models.length === 0) return
  const fs = yield* FileSystem.FileSystem
  const encoded = yield* Schema.encode(ResidentModels)({ recordedAt: yield* Clock.currentTimeMillis, models })
  yield* fs.writeFileString(recordFile(stateDirectory), encoded)
}).pipe(Effect.catchAll(error => Effect.logWarning(`Could not record the loaded models before the update: ${String(error)}`)))

/**
 * Once the service is ready after an update, loads the recorded models again and removes the record. The new
 * service may still be reading its catalog, so each load is retried for a while.
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
    Effect.zipRight(notify(`Loading ${modelId} again after the update.`)),
    Effect.catchAll(() => notify(`Could not load ${modelId} again after the update; it loads on its next request.`))), { discard: true })
}).pipe(Effect.catchAll(error => Effect.logWarning(`Could not read the models to load after the update: ${String(error)}`)))
