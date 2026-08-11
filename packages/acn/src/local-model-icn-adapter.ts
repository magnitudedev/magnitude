import { Effect, Option, ParseResult, Schema } from "effect"
import type {
  DownloadAttempt,
  ServableModelBundle,
  ModelPackage,
  ModelPackageInspection,
  ModelServingConfiguration,
  RecommendableModel,
  ServingProfile,
} from "@magnitudedev/acn-protocol"
import {
  DownloadAttemptSchema,
  ServableModelBundleSchema,
  ModelPackageInspectionSchema,
  ModelPackageSchema,
  ModelServingConfigurationSchema,
  RecommendableModelSchema,
  ServingProfileSchema,
} from "@magnitudedev/acn-protocol"
import type {
  DownloadAttempt as NativeDownloadAttempt,
  ServableModelBundle as NativeServableModelBundle,
  ModelPackageInspection as NativeModelPackageInspection,
  ModelPackage as NativeModelPackage,
  ModelServingConfiguration as NativeModelServingConfiguration,
  ModelBundleInput,
  RecommendableModel as NativeRecommendableModel,
  ServingProfile as NativeServingProfile,
} from "@magnitudedev/icn-protocol/schemas"

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
      draft: normalizeModelPackageFromIcn(bundle.draft),
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

export const downloadAttemptFromIcn = (
  attempt: NativeDownloadAttempt,
): Effect.Effect<DownloadAttempt, ParseResult.ParseError> =>
  Schema.validate(DownloadAttemptSchema)(attempt._tag === "Downloading"
    ? {
        ...attempt,
        bytesPerSecond: Option.flatMap(attempt.bytesPerSecond, Option.fromNullable),
      }
    : attempt)

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
          _tag: "SpeculativeDecodingPair" as const,
          target: yield* operand(bundle.target),
          draft: yield* operand(bundle.draft),
        }
    return yield* Schema.decodeUnknown(NativeModelBundleInputSchema)(input)
  })
}
