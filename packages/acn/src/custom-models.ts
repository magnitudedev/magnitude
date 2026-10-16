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
  CatalogFormModelIdSchema,
  LocalModelMutationFailed,
  type DrafterReference,
  type SpeculativeMethod,
} from "@magnitudedev/acn-protocol"
import {
  CustomModelsConfigSchema,
  EMPTY_CUSTOM_MODELS_CONFIG,
  GlobalStorage,
  MagnitudeConfigSchema,
  MagnitudeStorage,
  type CustomModelsConfig,
} from "@magnitudedev/storage"
import type { LocalModelsConfiguration, ModelRoot } from "@magnitudedev/icn-protocol/schemas"
import type { ResolvedAutomaticLocation } from "./icn/discovery-roots"
import { nativeDrafterReference } from "./drafter-references"

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

/**
 * The whole snapshot ICN receives. ICN lists hidden models apart and never assesses them; a drafter
 * whose binding method is not yet known is omitted until the engine reports it.
 */
export const localModelsConfiguration = (
  config: CustomModelsConfig,
  locations: readonly ResolvedAutomaticLocation[],
  methodOf: (reference: DrafterReference) => Option.Option<SpeculativeMethod>,
): LocalModelsConfiguration => ({
  roots: [
    ...locations.flatMap((location) => location.paths.map(({ path }): ModelRoot => ({
      kind: location.kind,
      path,
      enabled: !config.disabledLocations.includes(location.kind),
    }))),
    ...config.folders.map((path): ModelRoot => ({ kind: "Folder", path, enabled: true })),
  ],
  declarations: Object.entries(config.declarations).map(([modelId, declaration]) => ({
    modelId,
    origin: declaration.origin._tag === "Added"
      ? { _tag: "Added", repository: declaration.origin.repository, commit: declaration.origin.commit }
      : { _tag: "Imported", path: declaration.origin.path },
    target: [...declaration.target],
    projector: declaration.projector,
  })) as unknown as LocalModelsConfiguration["declarations"],
  hidden: config.hidden.filter((modelId) => !Schema.is(CatalogFormModelIdSchema)(modelId)),
  drafters: Object.entries(config.drafters).flatMap(([modelId, choice]): readonly unknown[] => {
    if (choice._tag !== "Reference") return [{ modelId, drafter: { _tag: choice._tag } }]
    return Option.match(methodOf(choice.reference), {
      onNone: () => [],
      onSome: (method) => [{
        modelId,
        drafter: { _tag: "Reference", method, reference: nativeDrafterReference(choice.reference) },
      }],
    })
  }) as unknown as LocalModelsConfiguration["drafters"],
})
