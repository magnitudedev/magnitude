import * as FileSystem from "@effect/platform/FileSystem"
import { Context, Effect, Layer } from "effect"
import type { ModelLocation } from "@magnitudedev/acn-protocol"
import type { CustomModelsConfig } from "@magnitudedev/storage"
import {
  directoryState,
  hostDiscoveryEnvironment,
  resolveAutomaticLocations,
  type DiscoveryEnvironment,
  type ResolvedAutomaticLocation,
} from "./icn/discovery-roots"

/** The directories Magnitude scans: automatic locations resolved on this host, then the user's folders. */
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
        enabled: !config.disabledLocations.includes(location.kind),
        automatic: true,
        state: location.state,
      }))
      const folders = yield* Effect.forEach(config.folders, (path) => directoryState(path).pipe(
        provide,
        Effect.map((state): ModelLocation => ({ kind: "Folder", paths: [{ path, state }], enabled: true, automatic: false, state })),
      ))
      return [...automatic, ...folders]
    }),
  })
}))

export const ModelLocationsLive = Layer.suspend(() => makeModelLocations(hostDiscoveryEnvironment()))
