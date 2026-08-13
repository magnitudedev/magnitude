import { Context, Effect, Layer, Option, Schema } from "effect"
import {
  LocalModelMutationFailed,
  ModelServingConfigurationSchema,
  type LocalInferenceError,
  type LocalModelInstallationAdmission,
  type ModelServingConfigurationId,
} from "@magnitudedev/acn-protocol"
import { LocalModelPackages, type LocalModelPackagesApi } from "./local-model-packages"
import { localProviderModelId } from "./local-provider-offerings"
import {
  RetainedModelConfigurations,
  type RetainedModelConfigurationsApi,
} from "./retained-model-configurations"
import {
  LocalModelConfigurationResolver,
  type LocalModelConfigurationResolverApi,
} from "./local-model-configuration-resolver"
import {
  LocalModelConfigurationCoordinator,
  type LocalModelConfigurationCoordinatorApi,
} from "./local-model-configuration-coordinator"

const failure = (code: string, message: string, retryable = false) =>
  new LocalModelMutationFailed({ code, message, retryable })
const sameConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

export interface LocalModelInstallerApi {
  readonly install: (
    configurationId: ModelServingConfigurationId,
  ) => Effect.Effect<LocalModelInstallationAdmission, LocalInferenceError>
}

export class LocalModelInstaller extends Context.Tag("LocalModelInstaller")<
  LocalModelInstaller,
  LocalModelInstallerApi
>() {}

export const makeLocalModelInstaller = (
  retained: RetainedModelConfigurationsApi,
  packages: LocalModelPackagesApi,
  coordinator: LocalModelConfigurationCoordinatorApi,
  resolver: LocalModelConfigurationResolverApi,
): Effect.Effect<LocalModelInstallerApi> => Effect.gen(function* () {
  const installUnlocked = (
    configurationId: ModelServingConfigurationId,
  ): Effect.Effect<LocalModelInstallationAdmission, LocalInferenceError> => Effect.gen(function* () {
      const retainedConfiguration = yield* retained.resolve(configurationId)
      const configuration = Option.isSome(retainedConfiguration)
        ? retainedConfiguration.value
        : yield* resolver.resolve(configurationId).pipe(
          Effect.flatMap(Option.match({
            onNone: () => resolver.catalogReady.pipe(Effect.flatMap((ready) => ready
              ? failure(
                  "local_model_configuration_not_found",
                  `Local model configuration ${configurationId} was not found`,
                )
              : failure(
                  "local_model_catalog_not_ready",
                  "The local model catalog is not ready",
                  true,
                ))),
            onSome: (resolved) => Option.match(resolved.assessment, {
              onNone: () => failure(
                "local_model_configuration_assessing",
                `Local model configuration ${configurationId} is still being assessed`,
                true,
              ),
              onSome: (assessment) => {
                if (assessment._tag === "Fits") return Effect.succeed(resolved.configuration)
                if (assessment._tag === "Failed" || assessment._tag === "Incompatible") {
                  return Effect.fail(new LocalModelMutationFailed(assessment.failure))
                }
                if (assessment._tag === "DoesNotFit") {
                  return failure(
                    "local_model_configuration_does_not_fit",
                    `Local model configuration ${configurationId} exceeds available ${assessment.limitingResource} capacity by ${assessment.deficitBytes} bytes`,
                  )
                }
                return failure(
                  "local_model_configuration_assessing",
                  `Local model configuration ${configurationId} is still being assessed`,
                  true,
                )
              },
            }),
          })),
        )
      if (Option.isNone(retainedConfiguration)) {
        const current = yield* resolver.resolve(configurationId)
        if (
          Option.isNone(current)
          || !sameConfiguration(current.value.configuration, configuration)
          || !Option.exists(current.value.assessment, ({ _tag }) => _tag === "Fits")
        ) {
          return yield* failure(
            "local_model_configuration_changed",
            `Local model configuration ${configurationId} changed before installation`,
            true,
          )
        }
      }
      const materialized = yield* retained.materialize(configuration).pipe(
        Effect.mapError((error) =>
          failure("retain_local_model_configuration_failed", error.message, true)),
      )
      const download = yield* packages.admitBundle(materialized.bundle)
      const providerModelId = localProviderModelId(materialized.id)
      return download._tag === "AlreadyInstalled"
        ? { _tag: "AlreadyInstalled", providerModelId }
        : { _tag: "DownloadAdmitted", providerModelId, downloadId: download.downloadId }
    })

  const install = (configurationId: ModelServingConfigurationId) =>
    coordinator.exclusive(installUnlocked(configurationId))

  return { install }
})

export const LocalModelInstallerLive: Layer.Layer<
  LocalModelInstaller,
  never,
  RetainedModelConfigurations | LocalModelPackages | LocalModelConfigurationCoordinator
    | LocalModelConfigurationResolver
> = Layer.effect(LocalModelInstaller, Effect.gen(function* () {
  const retained = yield* RetainedModelConfigurations
  const packages = yield* LocalModelPackages
  const coordinator = yield* LocalModelConfigurationCoordinator
  const resolver = yield* LocalModelConfigurationResolver
  return LocalModelInstaller.of(yield* makeLocalModelInstaller(
    retained,
    packages,
    coordinator,
    resolver,
  ))
}))
