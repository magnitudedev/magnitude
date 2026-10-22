import * as FileSystem from "@effect/platform/FileSystem"
import { Context, Effect, Layer } from "effect"
import type { ModelLocation } from "@magnitudedev/acn-protocol"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import { locationEnabled } from "./custom-models"
import {
  hostDiscoveryEnvironment,
  scannedPathState,
  resolveAutomaticLocations,
  type DiscoveryEnvironment,
  type ResolvedAutomaticLocation,
} from "./icn/discovery-roots"

/** What Magnitude scans: automatic locations resolved on this host, then the user's folders and files. */
export interface ModelLocationsApi {
  readonly environment: DiscoveryEnvironment
  readonly resolve: Effect.Effect<readonly ResolvedAutomaticLocation[]>
  readonly list: (config: CustomModelsConfig) => Effect.Effect<readonly ModelLocation[]>
}
export class ModelLocations extends Context.Tag("ModelLocations")<ModelLocations, ModelLocationsApi>() {}

export const makeModelLocations = (
  environment: DiscoveryEnvironment,
): Layer.Layer<ModelLocations, never, FileSystem.FileSystem> => Layer.effect(ModelLocations, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const provide = Effect.provideService(FileSystem.FileSystem, fs)
  const resolve = resolveAutomaticLocations(environment).pipe(provide)
  return ModelLocations.of({
    environment,
    resolve,
    list: (config) => Effect.gen(function* () {
      const automatic = (yield* resolve).map((location): ModelLocation => ({
        kind: location.kind,
        paths: location.paths,
        enabled: locationEnabled(config, location.kind),
        automatic: true,
        state: location.state,
      }))
      const folders = yield* Effect.forEach(config.paths, (path) => scannedPathState(path).pipe(
        provide,
        Effect.map((state): ModelLocation => ({ kind: "Folder", paths: [{ path, state }], enabled: true, automatic: false, state })),
      ))
      return [...automatic, ...folders]
    }),
  })
}))

export const ModelLocationsLive = Layer.suspend(() => makeModelLocations(hostDiscoveryEnvironment()))
