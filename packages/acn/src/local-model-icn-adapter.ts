import { Effect, Option, ParseResult, Schema } from "effect"
import type {
  ModelBundleDownload,
  CatalogIdentity,
  ServableModelBundle,
  ModelPackage,
  ModelPackageInspection,
  ModelServingConfiguration,
  RecommendableModel,
  ServingProfile,
} from "@magnitudedev/acn-protocol"
import {
  ModelBundleDownloadSchema,
  CatalogIdentitySchema,
  ServableModelBundleSchema,
  ModelPackageInspectionSchema,
  ModelPackageSchema,
  ModelServingConfigurationSchema,
  RecommendableModelSchema,
  ServingProfileSchema,
} from "@magnitudedev/acn-protocol"
import type {
  ModelDownload as NativeModelDownload,
  CatalogModel as NativeCatalogModel,
  ServableModelBundle as NativeServableModelBundle,
  ModelPackageInspection as NativeModelPackageInspection,
  ModelPackage as NativeModelPackage,
  ModelServingConfiguration as NativeModelServingConfiguration,
  ModelBundleInput,
  RecommendableModel as NativeRecommendableModel,
  ServingProfile as NativeServingProfile,
} from "@magnitudedev/icn-protocol/schemas"

export const catalogIdentityFromIcn = (
  model: Pick<NativeCatalogModel, "modelId" | "variantId">,
): Effect.Effect<CatalogIdentity, ParseResult.ParseError> =>
  Schema.decodeUnknown(CatalogIdentitySchema)({
    modelId: model.modelId,
    variantId: model.variantId,
  })

export const catalogModelDefinitionFromIcn = (
  model: NativeCatalogModel,
): Effect.Effect<RecommendableModel, ParseResult.ParseError> =>
  recommendableModelFromIcn({
    ...model,
    configuration: model.desiredConfiguration,
  })

export const catalogModelEffectiveConfigurationFromIcn = (
  model: NativeCatalogModel,
): Effect.Effect<Option.Option<ModelServingConfiguration>, ParseResult.ParseError> =>
  model.localState._tag === "Installed"
    && model.localState.installation.effectiveConfiguration._tag === "Runnable"
    ? modelServingConfigurationFromIcn(
        model.localState.installation.effectiveConfiguration.configuration,
      ).pipe(Effect.map(Option.some))
    : Effect.succeed(Option.none())

const normalizeModelPackageFromIcn = (
  modelPackage: NativeModelPackage,
) => ({
  ...modelPackage,
  files: modelPackage.files.map((file) => ({
    ...file,
    tensorStorageBytes: Option.flatMap(file.tensorStorageBytes, Option.fromNullable),
  })),
})

const normalizeBundleFromIcn = (
  bundle: NativeServableModelBundle,
) => bundle._tag === "Standalone"
  ? { ...bundle, package: normalizeModelPackageFromIcn(bundle.package) }
  : {
      ...bundle,
      target: normalizeModelPackageFromIcn(bundle.target),
      draftSource: bundle.draftSource._tag === "Embedded"
        ? bundle.draftSource
        : { ...bundle.draftSource, draft: normalizeModelPackageFromIcn(bundle.draftSource.draft) },
    }
import {
  ServableModelBundle as NativeServableModelBundleSchema,
  ModelPackage as NativeModelPackageSchema,
  ModelBundleInput as NativeModelBundleInputSchema,
} from "@magnitudedev/icn-protocol/schemas"

export const modelPackageFromIcn = (
  modelPackage: NativeModelPackage,
): Effect.Effect<ModelPackage, ParseResult.ParseError> =>
  Schema.validate(ModelPackageSchema)(normalizeModelPackageFromIcn(modelPackage))

export const modelPackageToIcn = (
  modelPackage: ModelPackage,
): Effect.Effect<NativeModelPackage, ParseResult.ParseError> =>
  Schema.encode(ModelPackageSchema)(modelPackage).pipe(
    Effect.flatMap(Schema.decodeUnknown(NativeModelPackageSchema)),
  )

export const packageInspectionFromIcn = (
  inspection: NativeModelPackageInspection,
): Effect.Effect<ModelPackageInspection, ParseResult.ParseError> =>
  Schema.validate(ModelPackageInspectionSchema)(inspection)

export const servingProfileFromIcn = (
  profile: NativeServingProfile,
): Effect.Effect<ServingProfile, ParseResult.ParseError> =>
  Schema.validate(ServingProfileSchema)(profile)

export const servingProfileToIcn = (profile: ServingProfile): NativeServingProfile => ({
  contextLength: profile.contextLength,
})

export const servableModelBundleFromIcn = (
  bundle: NativeServableModelBundle,
): Effect.Effect<ServableModelBundle, ParseResult.ParseError> =>
  Schema.validate(ServableModelBundleSchema)(normalizeBundleFromIcn(bundle))

export const servableModelBundleToIcn = (
  bundle: ServableModelBundle,
): Effect.Effect<NativeServableModelBundle, ParseResult.ParseError> =>
  Schema.encode(ServableModelBundleSchema)(bundle).pipe(
    Effect.flatMap(Schema.decodeUnknown(NativeServableModelBundleSchema)),
  )

export const modelServingConfigurationToIcn = (
  configuration: ModelServingConfiguration,
): Effect.Effect<NativeModelServingConfiguration, ParseResult.ParseError> =>
  servableModelBundleToIcn(configuration.bundle).pipe(
    Effect.map((bundle) => ({
      id: configuration.id,
      bundle,
      profile: servingProfileToIcn(configuration.profile),
    })),
  )

export const modelServingConfigurationFromIcn = (
  configuration: NativeModelServingConfiguration,
): Effect.Effect<ModelServingConfiguration, ParseResult.ParseError> =>
  Schema.validate(ModelServingConfigurationSchema)({
    ...configuration,
    bundle: normalizeBundleFromIcn(configuration.bundle),
  })

export const recommendableModelFromIcn = (
  model: NativeRecommendableModel,
): Effect.Effect<RecommendableModel, ParseResult.ParseError> =>
  modelServingConfigurationFromIcn(model.configuration).pipe(
    Effect.flatMap((configuration) => Schema.validate(RecommendableModelSchema)({
      ...model,
      configuration,
    })),
  )

export const modelDownloadFromIcn = (
  download: NativeModelDownload,
): Effect.Effect<ModelBundleDownload, ParseResult.ParseError> =>
  Schema.validate(ModelBundleDownloadSchema)({
    id: download.id,
    bundle: normalizeBundleFromIcn(download.bundle),
    state: download.state._tag === "Downloading"
      ? {
          ...download.state,
          bytesPerSecond: Option.flatMap(download.state.bytesPerSecond, Option.fromNullable),
        }
      : download.state,
  })

export const bundleToIcnInput = (
  bundle: ServableModelBundle,
  installedPackageIds: ReadonlySet<string>,
): Effect.Effect<ModelBundleInput, ParseResult.ParseError> => {
  const operand = (modelPackage: ModelPackage) =>
    installedPackageIds.has(modelPackage.id)
      ? Effect.succeed({ _tag: "Installed" as const, packageId: modelPackage.id })
      : Schema.encode(ModelPackageSchema)(modelPackage).pipe(
          Effect.map((encoded) => ({ _tag: "SourceBacked" as const, package: encoded })),
        )
  return Effect.gen(function* () {
    const input = bundle._tag === "Standalone"
      ? { _tag: "Standalone" as const, package: yield* operand(bundle.package) }
      : {
          _tag: "SpeculativeDecoding" as const,
          target: yield* operand(bundle.target),
          draftSource: bundle.draftSource._tag === "Embedded"
            ? bundle.draftSource
            : { _tag: "Separate" as const, draft: yield* operand(bundle.draftSource.draft) },
          method: bundle.method,
        }
    return yield* Schema.decodeUnknown(NativeModelBundleInputSchema)(input)
  })
}
