import { Context, Effect, Layer, Option, PubSub, Schema, Stream } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
  ModelCapabilitiesSchema,
  type LocalProviderOffering,
  type LocalProviderOfferingOrigin,
  type ModelCapabilities,
  type ModelOfferingTarget,
  type ModelOfferingTargetId,
  type ModelServingConfiguration,
  type ModelPackageEntry,
  type RecommendableModel,
  modelOfferingTargetPackageIds,
  ModelOfferingTargetSchema,
  ModelServingConfigurationSchema,
} from "@magnitudedev/protocol"
import {
  ProviderModelIdSchema,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import {
  MagnitudeStorage,
  type PersistedLocalProviderOffering,
} from "@magnitudedev/storage"
import { IcnCatalog, IcnInstalledModels } from "@magnitudedev/icn"
import {
  modelPackageFromIcn,
  packageInspectionFromIcn,
  recommendableModelFromIcn,
} from "./local-model-icn-adapter"

const failure = (operation: string, error: unknown) =>
  new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })

const capabilitySet = (
  target: ModelOfferingTarget,
  catalog: readonly RecommendableModel[],
  installed: readonly Pick<ModelPackageEntry, "package" | "inspection">[],
): ModelCapabilities => {
  const sameTarget = Schema.equivalence(ModelOfferingTargetSchema)
  const recommendation = catalog.find((model) => sameTarget(model.target, target))
  if (recommendation) return recommendation.capabilities
  const primaryPackageId = target._tag === "Package" ? target.package.id : target.target.id
  const inspection = installed.find(({ package: modelPackage }) =>
    modelPackage.id === primaryPackageId)?.inspection
  return inspection?._tag === "Inspected"
    ? inspection.capabilities
    : ModelCapabilitiesSchema.make({
        vision: false,
        tools: false,
        structuredOutput: false,
        reasoning: {
          supported: false,
          efforts: [],
          defaultEffort: Option.none(),
        },
      })
}

export interface LocalProviderOfferingsApi {
  readonly list: Effect.Effect<readonly LocalProviderOffering[], LocalInferenceError>
  readonly changes: Stream.Stream<void>
  readonly resolve: (
    providerModelId: ProviderModelId,
  ) => Effect.Effect<LocalProviderOffering, LocalInferenceError>
  readonly save: (
    modelId: ModelOfferingTargetId,
    configuration: ModelServingConfiguration,
    origin: LocalProviderOfferingOrigin,
  ) => Effect.Effect<LocalProviderOffering, LocalInferenceError>
}

export class LocalProviderOfferings extends Context.Tag("LocalProviderOfferings")<
  LocalProviderOfferings,
  LocalProviderOfferingsApi
>() {}

export const LocalProviderOfferingsLive: Layer.Layer<
  LocalProviderOfferings,
  never,
  MagnitudeStorage | IcnCatalog | IcnInstalledModels
> = Layer.effect(LocalProviderOfferings, Effect.gen(function* () {
  const storage = yield* MagnitudeStorage
  const catalog = yield* IcnCatalog
  const installed = yield* IcnInstalledModels
  const mutations = yield* PubSub.sliding<void>(16)
  const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

  const capabilitySources = Effect.all({
    catalog: catalog.get.pipe(
      Effect.flatMap(({ state }) => Effect.forEach(state.models, recommendableModelFromIcn)),
      Effect.mapError((error) => failure("read_recommendable_model_catalog_failed", error)),
    ),
    installed: installed.get.pipe(
      Effect.flatMap(({ state }) => Effect.forEach(
        state.packages,
        (entry) => Effect.all({
          package: modelPackageFromIcn(entry.package),
          inspection: packageInspectionFromIcn(entry.inspection),
        }),
      )),
      Effect.mapError((error) => failure("read_installed_model_capabilities_failed", error)),
    ),
  })

  const list: LocalProviderOfferingsApi["list"] = Effect.gen(function* () {
    const persisted = (yield* storage.config.load()).models?.localProviderOfferings ?? []
    const sources = yield* capabilitySources
    return persisted.map((offering): LocalProviderOffering => ({
      ...offering,
      capabilities: capabilitySet(offering.configuration.target, sources.catalog, sources.installed),
    }))
  }).pipe(
    Effect.mapError((error) => error instanceof LocalModelMutationFailed
      ? error
      : failure("read_local_provider_offerings_failed", error)),
  )

  const changes = Stream.mergeAll([
    Stream.fromPubSub(mutations),
    catalog.changes.pipe(Stream.map(() => undefined)),
    installed.changes.pipe(Stream.map(() => undefined)),
  ], { concurrency: "unbounded" })

  return LocalProviderOfferings.of({
    list,
    changes,
    resolve: (providerModelId) => list.pipe(Effect.flatMap((offerings) => {
      const offering = offerings.find((candidate) => candidate.providerModelId === providerModelId)
      return offering
        ? Effect.succeed(offering)
        : Effect.fail(new LocalModelMutationFailed({
            code: "local_provider_offering_not_found",
            message: `Local provider offering ${providerModelId} was not found`,
            retryable: false,
          }))
    })),
    save: (modelId, configuration, origin) => Effect.gen(function* () {
      const providerModelId = ProviderModelIdSchema.make(`local:${configuration.id}`)
      const persisted: PersistedLocalProviderOffering = {
        providerModelId,
        modelId,
        configuration,
        origin,
      }
      const configured = yield* list
      const existing = configured.find((offering) => offering.providerModelId === providerModelId)
      if (existing && !sameConfiguration(existing.configuration, configuration)) {
        return yield* new LocalModelMutationFailed({
          code: "local_provider_offering_identity_conflict",
          message: `Local provider offering ${providerModelId} conflicts with its stored identity`,
          retryable: false,
        })
      }
      yield* storage.config.upsertLocalProviderOffering(persisted).pipe(
        Effect.mapError((error) => failure("save_local_provider_offering_failed", error)),
      )
      yield* PubSub.publish(mutations, undefined)
      const offerings = yield* list
      const saved = offerings.find((offering) => offering.providerModelId === providerModelId)
      if (saved) return saved
      return yield* new LocalModelMutationFailed({
        code: "saved_local_provider_offering_unresolved",
        message: `Saved local provider offering ${providerModelId} could not be resolved`,
        retryable: true,
      })
    }),
  })
}))
