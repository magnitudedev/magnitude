import { Effect, Option, Schema, SynchronizedRef } from "effect"
import { ModelArtifactIndexSchema, type ModelArtifactIndex } from "./model-files"
import {
  LocalModelDiscoveredPropertiesSchema,
  type LocalModelDiscoveredProperties,
} from "./model-properties"
import {
  LlamaFitAssessmentCacheEntrySchema,
  type LlamaFitAssessmentCacheEntry,
} from "./llamacpp/fit"
import type { LlamaFitAssessmentKey, NormalizedLlamaModelPath } from "./llamacpp/identity"

export const LocalModelIndexSchema = Schema.Struct({
  artifacts: ModelArtifactIndexSchema,
  discoveredProperties: Schema.Array(LocalModelDiscoveredPropertiesSchema),
  fitAssessments: Schema.Array(LlamaFitAssessmentCacheEntrySchema),
})
export type LocalModelIndex = typeof LocalModelIndexSchema.Type

export const emptyLocalModelIndex = (): LocalModelIndex => ({
  artifacts: {
    capturedAt: new Date(),
    sets: [],
    issues: [],
  },
  discoveredProperties: [],
  fitAssessments: [],
})

export interface LocalModelIndexStoreApi {
  readonly snapshot: Effect.Effect<LocalModelIndex>
  readonly artifacts: Effect.Effect<ModelArtifactIndex>
  readonly replaceArtifacts: (artifacts: ModelArtifactIndex) => Effect.Effect<void>
  readonly discoveredProperties: (modelPath: string) => Effect.Effect<Option.Option<LocalModelDiscoveredProperties>>
  readonly putDiscoveredProperties: (properties: LocalModelDiscoveredProperties) => Effect.Effect<void>
  readonly fitAssessment: (
    modelPath: NormalizedLlamaModelPath,
    key: LlamaFitAssessmentKey,
  ) => Effect.Effect<Option.Option<LlamaFitAssessmentCacheEntry>>
  readonly putFitAssessment: (entry: LlamaFitAssessmentCacheEntry) => Effect.Effect<void>
}

export interface LocalModelIndexStoreOptions {
  readonly initialIndex: Option.Option<LocalModelIndex>
  readonly persist: (index: LocalModelIndex) => Effect.Effect<void>
}

const equivalentArtifacts = Schema.equivalence(ModelArtifactIndexSchema)
const equivalentDiscoveredProperties = Schema.equivalence(LocalModelDiscoveredPropertiesSchema)
const equivalentFitAssessment = Schema.equivalence(LlamaFitAssessmentCacheEntrySchema)

export const makeLocalModelIndexStore = (
  options: LocalModelIndexStoreOptions,
): Effect.Effect<LocalModelIndexStoreApi> => Effect.gen(function* () {
  const state = yield* SynchronizedRef.make(Option.getOrElse(options.initialIndex, emptyLocalModelIndex))

  const update = (f: (current: LocalModelIndex) => LocalModelIndex) =>
    SynchronizedRef.modifyEffect(state, (current) => {
      const next = f(current)
      return next === current
        ? Effect.succeed([undefined, current] as const)
        : options.persist(next).pipe(Effect.as([undefined, next] as const))
    })

  return {
    snapshot: SynchronizedRef.get(state),
    artifacts: SynchronizedRef.get(state).pipe(Effect.map((index) => index.artifacts)),
    replaceArtifacts: (artifacts) => update((current) =>
      equivalentArtifacts(current.artifacts, artifacts) ? current : { ...current, artifacts }),
    discoveredProperties: (modelPath) => SynchronizedRef.get(state).pipe(
      Effect.map((index) => Option.fromNullable(
        index.discoveredProperties.find((entry) => entry.modelPath === modelPath),
      )),
    ),
    putDiscoveredProperties: (properties) => update((current) => {
      const previous = current.discoveredProperties.find((entry) => entry.modelPath === properties.modelPath)
      if (previous && equivalentDiscoveredProperties(previous, properties)) return current
      return {
        ...current,
        discoveredProperties: [
          ...current.discoveredProperties.filter((entry) => entry.modelPath !== properties.modelPath),
          properties,
        ].sort((left, right) => left.modelPath.localeCompare(right.modelPath)),
      }
    }),
    fitAssessment: (modelPath, key) => SynchronizedRef.get(state).pipe(
      Effect.map((index) => Option.fromNullable(
        index.fitAssessments.find((entry) => entry.modelPath === modelPath && entry.key === key),
      )),
    ),
    putFitAssessment: (entry) => update((current) => {
      const previous = current.fitAssessments.find((candidate) => candidate.modelPath === entry.modelPath)
      if (previous && equivalentFitAssessment(previous, entry)) return current
      return {
        ...current,
        fitAssessments: [
          ...current.fitAssessments.filter((candidate) => candidate.modelPath !== entry.modelPath),
          entry,
        ].sort((left, right) => left.modelPath.localeCompare(right.modelPath)),
      }
    }),
  }
})
