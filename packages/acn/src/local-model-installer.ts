import { Context, Effect, Layer, Option, Schema } from "effect"
import {
  LocalModelMutationFailed,
  ModelServingConfigurationSchema,
  type LocalInferenceError,
  type ModelInstallationAdmission,
  type ModelServingConfiguration,
  type ModelServingConfigurationId,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { IcnCatalog, type IcnCatalogService } from "@magnitudedev/icn"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"
import { LocalModelAssessor, type LocalModelAssessorApi } from "./local-model-assessor"
import { LocalModelPackages, type LocalModelPackagesApi } from "./local-model-packages"
import {
  RetainedConfigurationConflict,
  RetainedModelConfigurations,
  type RetainedModelConfigurationsApi,
} from "./retained-model-configurations"
import {
  LocalModelConfigurationCoordinator,
  type LocalModelConfigurationCoordinatorApi,
} from "./local-model-configuration-coordinator"

const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

const failure = (code: string, message: string, retryable = false) =>
  new LocalModelMutationFailed({ code, message, retryable })

export interface LocalModelInstallerApi {
  readonly install: (
    configurationId: ModelServingConfigurationId,
  ) => Effect.Effect<ModelInstallationAdmission, LocalInferenceError>
}

export class LocalModelInstaller extends Context.Tag("LocalModelInstaller")<
  LocalModelInstaller,
  LocalModelInstallerApi
>() {}

export const makeLocalModelInstaller = (
  retained: RetainedModelConfigurationsApi,
  packages: LocalModelPackagesApi,
  catalog: IcnCatalogService,
  coordinator: LocalModelConfigurationCoordinatorApi,
  assessor: LocalModelAssessorApi,
): Effect.Effect<LocalModelInstallerApi> => Effect.gen(function* () {
  const catalogConfiguration = (configurationId: ModelServingConfigurationId) => Effect.gen(function* () {
    if (!(yield* catalog.ready)) {
      return yield* failure(
        "local_model_catalog_not_ready",
        "The local model catalog is not ready",
        true,
      )
    }
    const snapshot = yield* catalog.get
    const models = yield* Effect.forEach(snapshot.state.models, recommendableModelFromIcn)
    return {
      revision: snapshot.revision,
      configuration: Option.fromNullable(models.find(({ configuration }) =>
        configuration.id === configurationId)?.configuration),
    }
  }).pipe(Effect.mapError((error) => error instanceof LocalModelMutationFailed
    ? error
    : failure("read_local_model_catalog_failed", String(error), true)))

  const installUnlocked = (
    configurationId: ModelServingConfigurationId,
  ): Effect.Effect<ModelInstallationAdmission, LocalInferenceError> => Effect.suspend(() =>
    Effect.gen(function* () {
      const retainedConfiguration = yield* retained.resolve(configurationId)
      const assessedConfiguration = (yield* assessor.state).get(configurationId)?.configuration
      const publishedConfiguration = (yield* catalog.ready)
        ? yield* catalogConfiguration(configurationId)
        : undefined
      if (
        Option.isSome(retainedConfiguration)
        && publishedConfiguration !== undefined
        && Option.isSome(publishedConfiguration.configuration)
        && !sameConfiguration(retainedConfiguration.value, publishedConfiguration.configuration.value)
      ) {
        return yield* failure(
          "local_model_configuration_identity_conflict",
          `Configuration ${configurationId} has conflicting retained and catalog values`,
        )
      }
      const configuration = Option.isSome(retainedConfiguration)
        ? retainedConfiguration.value
        : Option.getOrUndefined(publishedConfiguration?.configuration ?? Option.none())
          ?? assessedConfiguration
      if (configuration === undefined) {
        if (!(yield* catalog.ready)) {
          return yield* failure(
            "local_model_catalog_not_ready",
            "The local model catalog is not ready",
            true,
          )
        }
        return yield* failure(
          "local_model_configuration_not_found",
          `Local model configuration ${configurationId} was not found`,
        )
      }
      if (Option.isNone(retainedConfiguration)
        && Option.isSome(publishedConfiguration?.configuration ?? Option.none())) {
        const latest = yield* catalogConfiguration(configurationId)
        if (
          latest.revision !== publishedConfiguration?.revision
          || Option.isNone(latest.configuration)
          || !sameConfiguration(configuration, latest.configuration.value)
        ) {
          return yield* installUnlocked(configurationId)
        }
      }
      const materialized = yield* retained.materialize(configuration).pipe(
        Effect.mapError((error) => error instanceof RetainedConfigurationConflict
          ? failure("local_model_configuration_identity_conflict", error.reason)
          : failure("retain_local_model_configuration_failed", error.message, true)),
      )
      const download = yield* packages.admitBundle(materialized.bundle)
      return {
        providerModelId: ProviderModelIdSchema.make(materialized.id),
        download,
      }
    }))

  const install = (configurationId: ModelServingConfigurationId) =>
    coordinator.exclusive(installUnlocked(configurationId))

  return { install }
})

export const LocalModelInstallerLive: Layer.Layer<
  LocalModelInstaller,
  never,
  RetainedModelConfigurations | LocalModelPackages | IcnCatalog | LocalModelConfigurationCoordinator
    | LocalModelAssessor
> = Layer.effect(LocalModelInstaller, Effect.gen(function* () {
  const retained = yield* RetainedModelConfigurations
  const packages = yield* LocalModelPackages
  const catalog = yield* IcnCatalog
  const coordinator = yield* LocalModelConfigurationCoordinator
  const assessor = yield* LocalModelAssessor
  return LocalModelInstaller.of(yield* makeLocalModelInstaller(
    retained,
    packages,
    catalog,
    coordinator,
    assessor,
  ))
}))
