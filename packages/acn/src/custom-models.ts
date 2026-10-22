import * as FileSystem from "@effect/platform/FileSystem"
import {
  Context,
  Duration,
  Effect,
  Layer,
  Option,
  Schedule,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  CustomFormModelIdSchema,
  LocalModelMutationFailed,
  parseModelId,
  sameModelId,
  type AutomaticModelLocationKind,
  type CustomFormModelId,
  type DrafterChoice,
  type HuggingFaceFormModelId,
  type DrafterReference,
  type ModelId,
} from "@magnitudedev/acn-protocol"
import {
  CustomModelsConfigSchema,
  EMPTY_CUSTOM_MODELS_CONFIG,
  GlobalStorage,
  MagnitudeConfigSchema,
  MagnitudeStorage,
  type AddedModelSetting,
  type CustomModelsConfig,
  type DrafterSetting,
  type ModelLocationName,
} from "@magnitudedev/storage"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import { nativeDrafterReference } from "./drafter-references"
import type { ResolvedAutomaticLocation } from "./icn/discovery-roots"

/**
 * Every user choice about custom models, persisted under `models` in config.json. `modify` is a
 * serialized read-modify-write: `f` sees the latest saved choices and returns its result with the next ones.
 */
export interface CustomModelsApi {
  readonly get: Effect.Effect<CustomModelsConfig>
  readonly changes: Stream.Stream<CustomModelsConfig>
  readonly modify: <A, E>(
    f: (current: CustomModelsConfig) => Effect.Effect<readonly [A, CustomModelsConfig], E>,
  ) => Effect.Effect<A, E | LocalModelMutationFailed>
}
export class CustomModels extends Context.Tag("CustomModels")<CustomModels, CustomModelsApi>() {}

const fromConfig = (models: Option.Option<CustomModelsConfig>) => Option.getOrElse(models, () => EMPTY_CUSTOM_MODELS_CONFIG)

const saveFailed = new LocalModelMutationFailed({
  code: "custom_models_save_failed",
  message: "Your model choices could not be saved. Check access to the Magnitude configuration and try again.",
  retryable: true,
})

export const CustomModelsLive: Layer.Layer<
  CustomModels,
  never,
  MagnitudeStorage | GlobalStorage | FileSystem.FileSystem
> = Layer.scoped(CustomModels, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  const globalStorage = yield* GlobalStorage
  const fs = yield* FileSystem.FileSystem
  const initial = yield* storage.config.load().pipe(
    Effect.map((config) => fromConfig(config.models)),
    Effect.catchAll((error) => Effect.logWarning("Unable to load custom model choices").pipe(
      Effect.annotateLogs({ error: String(error).slice(0, 1_000) }),
      Effect.as(EMPTY_CUSTOM_MODELS_CONFIG),
    )),
  )
  const state = yield* SubscriptionRef.make(initial)
  const writes = yield* Effect.makeSemaphore(1)
  const equivalent = Schema.equivalence(CustomModelsConfigSchema)
  const publish = (next: CustomModelsConfig) => SubscriptionRef.get(state).pipe(
    Effect.flatMap((current) => equivalent(current, next) ? Effect.void : SubscriptionRef.set(state, next)),
  )

  // A hand edit of config.json applies without a restart; an invalid edit keeps the last good choices.
  const observe = fs.readFileString(globalStorage.paths.configFile).pipe(
    Effect.flatMap((text) => Schema.decodeUnknown(Schema.parseJson(MagnitudeConfigSchema), { onExcessProperty: "preserve" })(text)),
    Effect.map((config) => Option.some(fromConfig(config.models))),
    Effect.catchAll((error) => Effect.succeed(
      error._tag === "SystemError" && error.reason === "NotFound" ? Option.some(EMPTY_CUSTOM_MODELS_CONFIG) : Option.none(),
    )),
    Effect.flatMap(Option.match({ onNone: () => Effect.void, onSome: publish })),
  )
  yield* writes.withPermits(1)(observe).pipe(
    Effect.repeat(Schedule.spaced(Duration.seconds(1))),
    Effect.catchAllCause((cause) => Effect.logError("Custom model configuration observer stopped").pipe(
      Effect.annotateLogs({ cause: String(cause).slice(0, 1_000) }),
    )),
    Effect.forkScoped,
  )

  return CustomModels.of({
    get: SubscriptionRef.get(state),
    changes: state.changes,
    modify: (f) => writes.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(state)
      const [result, next] = yield* f(current)
      if (!equivalent(current, next)) {
        yield* storage.config.update((config) => ({ ...config, models: Option.some(next) })).pipe(
          Effect.mapError(() => saveFailed),
        )
        yield* SubscriptionRef.set(state, next)
      }
      return result
    })),
  })
}))

const locationNames: Readonly<Record<AutomaticModelLocationKind, ModelLocationName>> = {
  HuggingFaceCache: "huggingface",
  LmStudio: "lmstudio",
  LlamaCpp: "llamacpp",
  Ollama: "ollama",
}

/** Whether an automatic location is scanned; one config.json doesn't mention is. */
export const locationEnabled = (config: CustomModelsConfig, kind: AutomaticModelLocationKind): boolean =>
  config.locations[locationNames[kind]] !== false

/** Turns an automatic location on or off; on is written by leaving it out. */
export const withLocationEnabled = (
  config: CustomModelsConfig,
  kind: AutomaticModelLocationKind,
  enabled: boolean,
): CustomModelsConfig => {
  const { [locationNames[kind]]: _, ...others } = config.locations
  return { ...config, locations: enabled ? others : { ...others, [locationNames[kind]]: false } }
}

/** The added model with this ID, and the key it is stored under. */
export const addedModel = (
  config: CustomModelsConfig,
  modelId: ModelId,
): Option.Option<readonly [HuggingFaceFormModelId, AddedModelSetting]> => Option.fromNullable(
  Object.entries(config.added).find(([id]) => id === modelId) as readonly [HuggingFaceFormModelId, AddedModelSetting] | undefined,
)

/** The drafter choice stored for a model, under its ID with any `local:` label. */
export const configuredDrafter = (config: CustomModelsConfig, modelId: ModelId): Option.Option<DrafterSetting> =>
  Option.fromNullable(Object.entries(config.drafters).find(([id]) => sameModelId(id, modelId))?.[1])

/** The reference a drafter setting names; `"none"` names none. */
export const drafterSettingReference = (setting: DrafterSetting): Option.Option<DrafterReference> => {
  switch (setting._tag) {
    case "None": return Option.none()
    case "BuiltIn": return Option.some({ _tag: "Embedded" })
    case "Catalog": return Option.some({ _tag: "Catalog", modelId: setting.modelId })
    case "HuggingFace": return Option.some({
      _tag: "HuggingFace",
      repository: setting.repository,
      commit: setting.revision,
      path: setting.path,
    })
    case "Local": return Option.some({ _tag: "Local", path: setting.path })
  }
}

/** How config.json spells the choice of a drafter reference, or of none. */
export const drafterSetting = (reference: Option.Option<DrafterReference>): DrafterSetting => Option.match(reference, {
  onNone: (): DrafterSetting => ({ _tag: "None" }),
  onSome: (value): DrafterSetting => {
    switch (value._tag) {
      case "Embedded": return { _tag: "BuiltIn" }
      case "Catalog": return { _tag: "Catalog", modelId: value.modelId }
      case "HuggingFace": return {
        _tag: "HuggingFace",
        repository: value.repository,
        path: value.path,
        revision: value.commit,
      }
      case "Local": return { _tag: "Local", path: value.path }
    }
  },
})

/** A model's drafter choice as clients see it: automatic when config.json names none. */
export const configuredDrafterChoice = (config: CustomModelsConfig, modelId: ModelId): DrafterChoice => Option.match(
  configuredDrafter(config, modelId),
  {
    onNone: (): DrafterChoice => ({ _tag: "Auto" }),
    onSome: (setting): DrafterChoice => Option.match(drafterSettingReference(setting), {
      onNone: (): DrafterChoice => ({ _tag: "None" }),
      onSome: (reference): DrafterChoice => ({ _tag: "Reference", reference }),
    }),
  },
)

/** Stores a model's drafter choice; automatic is stored by leaving it out. */
export const withDrafterChoice = (
  config: CustomModelsConfig,
  modelId: CustomFormModelId,
  choice: DrafterChoice,
): CustomModelsConfig => {
  const drafters = Object.fromEntries(Object.entries(config.drafters).filter(([id]) => !sameModelId(id, modelId)))
  return {
    ...config,
    drafters: choice._tag === "Auto" ? drafters : {
      ...drafters,
      [modelId]: drafterSetting(choice._tag === "None" ? Option.none() : Option.some(choice.reference)),
    },
  }
}

/** Whether a model is hidden, by its ID with any `local:` label. */
export const isHidden = (config: CustomModelsConfig, modelId: ModelId): boolean =>
  config.hidden.some((hidden) => sameModelId(hidden, modelId))

/**
 * Stored IDs renamed to the ones they now name (`canonical`: a catalog model's for one of its aliases, a listed
 * model's own for an older `local:` label of it): `hidden` entries and `drafters` keys. A drafter choice for a model
 * that is now a catalog model goes: a catalog model always serves with its catalog drafter.
 */
export const withCanonicalReferences = (
  config: CustomModelsConfig,
  canonical: (modelId: ModelId) => ModelId,
): CustomModelsConfig => {
  const hidden = [...new Set(config.hidden.map(canonical))]
  const drafters = Object.entries(config.drafters).flatMap(([id, setting]) =>
    Option.toArray(Option.map(Schema.decodeUnknownOption(CustomFormModelIdSchema)(canonical(id as ModelId)),
      (renamed) => [renamed, setting] as const)))
  return { ...config, hidden, drafters: Object.fromEntries(drafters) }
}

/**
 * The whole snapshot ICN receives, converted from config.json's choices. ICN lists hidden models apart and never
 * assesses them; a hidden catalog model is hidden by its external copies.
 */
export const localModelsConfiguration = (
  config: CustomModelsConfig,
  locations: readonly ResolvedAutomaticLocation[],
): Native.LocalModelsConfiguration => ({
  roots: [
    ...locations.flatMap((location) => location.paths.map(({ path }): Native.ModelRoot => ({
      kind: location.kind,
      path,
      enabled: locationEnabled(config, location.kind),
    }))),
    ...config.paths.map((path): Native.ModelRoot => ({ kind: "Folder", path, enabled: true })),
  ],
  added: Object.entries(config.added).flatMap(([modelId, added]): readonly Native.AddedModel[] => {
    const identity = parseModelId(modelId as ModelId)
    return identity._tag === "HuggingFace" && added !== undefined ? [{
      repository: identity.repositoryId,
      path: identity.artifactSelector,
      revision: added.revision,
      projector: added.projector,
    }] : []
  }),
  hidden: [...config.hidden],
  drafters: Object.entries(config.drafters).flatMap(([modelId, setting]): readonly Native.DrafterAttachment[] =>
    setting === undefined ? [] : [{
      modelId,
      drafter: Option.match(drafterSettingReference(setting), {
        onNone: (): Native.DrafterChoice => ({ _tag: "None" }),
        onSome: (reference): Native.DrafterChoice => ({ _tag: "Reference", reference: nativeDrafterReference(reference) }),
      }),
    }]),
})
